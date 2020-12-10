#!/usr/bin/env python
# coding: utf-8

# Process Vo data for a final size analysis by age
# Edit 17 Aug: just look at adults and children to
# Edit 25 Aug: add Cauchemez model and other updates for ONS
# Edit 7 Oct: Try to debug the sub-epidemics problem Lorenzo noticed
# Remember to atleast_2d the design matrices!

## have to seed the random number generator first otherwise we get different
## results depending on which numba options are turned on or off.
import numpy as np
np.random.seed(42)

from numba import jit, njit, prange
from numba.typed import List
import networkx as nx
import scipy.stats as st
import scipy.optimize as op
from numpy import linalg as LA
import matplotlib.pyplot as plt
from collections import namedtuple
import sys
import pathlib
import argparse
import logging
from stamford.graph import households, members

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s:%(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def feature_age_band(g, n, band=(13,30)):
    """
    Return a feature vector for the given individual according
    to whether they are in the provided age band or not. The
    vector is 1-dimensional.
    """
    age = g.nodes[n]['age']
    vec = np.zeros(1)
    vec[0] = 1 if age >= band[0] and age <= band[1] else 0
    return vec
feature_age_band.size = 1

def read_graph_data(filename, feature, **args):
    """
    Read a GraphML file and produce Y and XX.

    Y is a list containing an array for each household. The household array
      contains in turn, for each member, a 1 if that person has been infected,
      and a 0 otherwise.

    XX is a list containing a matrix for each household. The household matrix
      is of size n x m where n is the size of the household, and m is the
      number of features.

    The argument `feature` is a callable that is called as `feature(g, n, **args)`
    it takes the graph and a node id that corresponds to an individual, and
    some optional arguments that are opaquely passed through. It must return
    a feature vector for that individual.
    """
    g = nx.read_graphml(filename)

    ## these are the criteria by which to decide if an individial has been
    ## infected. any will do. this could usefully be passed in as an argument
    ## rather than specified here which would allow for different criteria
    ## or observable data.
    swabs = nx.get_node_attributes(g, "swab.test.result")
    swabscreens = nx.get_node_attributes(g, "swab.test.screening.result")
    seros = nx.get_node_attributes(g, "serology.test.result")
    seroscreens = nx.get_node_attributes(g, "serology.test.screening.result")
    ages = nx.get_node_attributes(g, "age")

    def is_positive(p):
        for d in (swabs, swabscreens, seros, seroscreens):
            result = d.get(p)
            if isinstance(result, str) and result == "positive":
                return True
        return False

    Y = List()
    XX = List()

    for hh in households(g):
        myx = np.vstack([feature(g, m, **args) for m in members(g, hh)])
        myy = np.array([1 if is_positive(m) else 0 for m in members(g, hh)])

        # Sort the household grouping by outcome
        ii = np.argsort(myy)
        myx = myx[ii, :]
        myy = myy[ii]

        Y.append(myy)
        XX.append(myx)

    return Y, XX


@njit(cache=True)
def phi(s, logtheta=0.0):
    theta = np.exp(logtheta)
    return (1.0 + theta * s) ** (-1.0 / theta)


@njit(cache=True)
def decimal_to_bit_array(d, n_digits):
    powers_of_two = 2 ** np.arange(32)[::-1]
    return ((d & powers_of_two) / powers_of_two)[-n_digits:]

@njit(fastmath=True, cache=True)
def nll(feats, outcomes, llaL, llaG, logtheta, eta, alpha, beta, gamma):
    """
    Compute the negative log-likelihood for a specific household.
    """
    ## if no infections, we can compute the log-likelihood directly
    if np.all(outcomes == 0.0):
        return np.exp(llaG) * np.sum(np.exp(alpha @ (feats.T)))

    q = np.sum(outcomes > 0)
    r = 2**q
    m = outcomes.shape[0]    ## number of household members

    # Quantities that don't vary through the sum
    Bk = np.exp(-np.exp(llaG) * np.exp(alpha @ (feats.T)))
    laM = np.exp(llaL) * np.outer(
        np.exp(beta @ (feats.T)), np.exp(gamma @ (feats.T))
    )
    laM *= m ** eta

    BB = np.zeros((r, r))  # To be the Ball matrix
    for jd in range(0, r):
        j = decimal_to_bit_array(jd, m)
        for omd in range(0, jd + 1):
            om = decimal_to_bit_array(omd, m)
            if np.all(om <= j):
                ## divide by zero encountered in double_scalars
                ## overflow encountered in double_scalars
                my_phi = phi((1-j) @ laM, logtheta)

                if np.any(
                    np.floor(np.log10(np.abs(my_phi[my_phi != 0]))) < -100
                ):
                    return np.inf

                BB[jd, omd] = 1.0 / np.prod(
                    (my_phi ** om) * (Bk ** (1 - j))
                )
    if np.all(BB == 0):
        return np.inf
    return -np.log(LA.solve(BB, np.ones(r))[-1])

@njit(parallel=True, fastmath=True, cache=True)
def mynll(x, Y, XX, fsize):
    #print(x)
    llaL = x[0]
    llaG = x[1]
    logtheta = x[2]
    eta = (4.0 / np.pi) * np.arctan(x[3])
    alpha = x[4 : (4 + fsize)]
    beta = x[(4 + fsize) : (4 + 2 * fsize)]
    gamma = x[(4 + 2 * fsize) :]

    num_households = len(Y)
    nlv = np.zeros(num_households)  # Vector of negative log likelihoods

    for i in prange(num_households):
        nlv[i] = nll(XX[i], Y[i], llaL, llaG, logtheta, eta, alpha, beta, gamma)

    # return the nll sum, with a Ridge 
    return np.sum(nlv) + np.sum(x ** 2) 

def optimise_gradient_descent(x0, bb, Y, XX, feature, nrestarts):
    def callbackF(x, x2=0.0, x3=0.0):
        log.info(
            "Evaluated at [{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}]: {:.8f}".format(
                x[0], x[1], x[2], x[3], x[4], x[5], x[6], mynll(x, Y, XX, feature.size)
            )
        )

    samples = []

    nll0 = mynll(x0, Y, XX, feature.size)
    log.info(f"Descending from: {x0} distance {nll0}")
    try:
        minimum = op.minimize(
            mynll,
            x0,
            (Y, XX, feature.size),
            method="TNC",
            callback=callbackF,
            bounds=bb,
            options={"maxiter": 10000},
        )
        log.info(f"Found minimum: {minimum.x}")
        samples.append(minimum)
    except Exception as e:
        log.error(f"Failure starting from {x0} restarting:")
        log.error(f"{e}")

    ## Find different minima from a selection of starting points.
    while len(samples) < nrestarts + 1:
        nll0 = np.nan
        try:
            while (np.isnan(nll0)) or (np.isinf(nll0)):
                xx0 = np.random.uniform(bb[:, 0], bb[:, 1])
                nll0 = mynll(xx0, Y, XX, feature.size)
            log.info(f"Descending from: {xx0} - {nll0}")
            minimum = op.minimize(
                mynll,
                xx0,
                (Y, XX, feature.size),
                bounds=bb,
                method="TNC",
                callback=callbackF,
                options={"maxiter": 1000, "ftol": 1e-9},
            )
            log.info(f"Found minimum: {minimum.x}")
            samples.append(minimum)
        except Exception as e:
            log.error(f"Failure starting from {xx0} restarting:")
            log.error(f"{e}")

    ff = np.inf * np.ones(len(samples))
    for i in range(len(samples)):  # In case of crash - XXX ww???
        if samples[i].success:
            ff[i] = samples[i].fun

    xhat = samples[ff.argmin()].x

    return xhat

def command():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nrestarts", type=int, default=1, help="Number of times to run the fitting process")
    parser.add_argument("--lower", type=int, default=13, help="Lower age cutoff")
    parser.add_argument("--upper", type=int, default=30, help="Upper age cutoff")
    parser.add_argument("graph", help="GraphML file containing network")

    args = parser.parse_args()

    feature=feature_age_band
    fargs={"band": (args.lower, args.upper) }
    log.info(f"Reading population graph from {args.graph}")
    Y, XX = read_graph_data(args.graph, feature, **fargs)
    log.info(f"Done.")
    num_households = len(Y)

    # Indicative parameters - to do, add bounds and mulitple restarts
    x0 = np.zeros(7)

    ## bounds for the optimization. the x is constrained to remain
    ## within these ranges.
    bb = np.array(
        [
            [-5.0, 0.0],
            [-5.0, 0.0],
            [-10.0, 10.0],
            [-10.0, 10.0],
            [-3.0, 3.0],
            [-3.0, 3.0],
            [-3.0, 3.0],
        ]
    )

    xhat = optimise_gradient_descent(x0, bb, Y, XX, feature, args.nrestarts)
    log.info("xhat = {}".format(xhat))

    pn = len(x0)
    delta = (
        1e-2  # This will need some tuning, but here set at sqrt(default delta in optimiser)
    )
    dx = delta * xhat
    ej = np.zeros(pn)
    ek = np.zeros(pn)
    Hinv = np.zeros((pn, pn))
    for j in range(0, pn):
        ej[j] = dx[j]
        for k in range(0, j):
            ek[k] = dx[k]
            Hinv[j, k] = (
                mynll(xhat + ej + ek, Y, XX, feature.size)
                - mynll(xhat + ej - ek, Y, XX, feature.size)
                - mynll(xhat - ej + ek, Y, XX, feature.size)
                + mynll(xhat - ej - ek, Y, XX, feature.size)
            )
            ek[k] = 0.0
        Hinv[j, j] = (
            -mynll(xhat + 2 * ej, Y, XX, feature.size)
            + 16 * mynll(xhat + ej, Y, XX, feature.size)
            - 30 * mynll(xhat, Y, XX, feature.size)
            + 16 * mynll(xhat - ej, Y, XX, feature.size)
            - mynll(xhat - 2 * ej, Y, XX, feature.size)
        )
        ej[j] = 0.0
    Hinv += np.triu(Hinv.T, 1)
    Hinv /= 4.0 * np.outer(dx, dx) + np.diag(
        8.0 * dx ** 2
    )  # TO DO: replace with a chol ...
    covmat = LA.inv(0.5 * (Hinv + Hinv.T))
    stds = np.sqrt(np.diag(covmat))

    print(
        "Baseline probability of infection from outside is {:.1f} ({:.1f},{:.1f}) %".format(
            100.0 * (1.0 - np.exp(-np.exp(xhat[1]))),
            100.0 * (1.0 - np.exp(-np.exp(xhat[1] - 1.96 * stds[1]))),
            100.0 * (1.0 - np.exp(-np.exp(xhat[1] + 1.96 * stds[1]))),
        )
    )

    # phi gets bigger as xhat[1] gets smaller and bigger as xhat[2] gets bigger
    # 'Safest' method is Monte Carlo - sample
    mymu = xhat[[0, 2, 3]]
    mySig = covmat[[0, 2, 3], :][:, [0, 2, 3]]
    m = 4000

    for k in range(2, 7):
        sarvec = np.zeros(m)
        for i in range(0, m):
            uu = np.random.multivariate_normal(mymu, mySig)
            eta = (4.0 / np.pi) * np.arctan(uu[2])
            sarvec[i] = 100.0 * (1.0 - phi(np.exp(uu[0]) * (k ** eta), uu[1]))

        eta = (4.0 / np.pi) * np.arctan(xhat[3])
        print(
            "HH size {:d} baseline pairwise infection probability is {:.1f} ({:.1f},{:.1f}) %".format(
                k,
                100.0 * (1.0 - phi(np.exp(xhat[0]) * (k ** eta), xhat[2])),
                np.percentile(sarvec, 2.5),
                np.percentile(sarvec, 97.5),
            )
        )


    print(
        "Relative external exposure for {} <= age <= {} {:.1f} ({:.1f},{:.1f}) %".format(
            args.lower, args.upper,
            100.0 * np.exp(xhat[4]),
            100.0 * np.exp(xhat[4] - 1.96 * stds[4]),
            100.0 * np.exp(xhat[4] + 1.96 * stds[4]),
        )
    )
    print(
        "Relative susceptibility for {} <= age <= {} {:.1f} ({:.1f},{:.1f}) %".format(
            args.lower, args.upper,
            100.0 * np.exp(xhat[5]),
            100.0 * np.exp(xhat[5] - 1.96 * stds[5]),
            100.0 * np.exp(xhat[5] + 1.96 * stds[5]),
        )
    )
    print(
        "Relative transmissibility for {} <= age <= {} {:.1f} ({:.1f},{:.1f}) %".format(
            args.lower, args.upper,
            100.0 * np.exp(xhat[6]),
            100.0 * np.exp(xhat[6] - 1.96 * stds[6]),
            100.0 * np.exp(xhat[6] + 1.96 * stds[6]),
        )
    )

