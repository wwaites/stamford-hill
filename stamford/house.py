#!/usr/bin/env python
# coding: utf-8

# Process Vo data for a final size analysis by age
# Edit 17 Aug: just look at adults and children to
# Edit 25 Aug: add Cauchemez model and other updates for ONS
# Edit 7 Oct: Try to debug the sub-epidemics problem Lorenzo noticed
# Remember to atleast_2d the design matrices!

import numba
import numpy as np
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

def read_graph_data(filename, cutoff=(0, 20)):
    """
    Read a GraphML file and produce Y and XX.

    Y is a list containing an array for each household. The household array
      contains in turn, for each member, a 1 if that person has been infected,
      and a 0 otherwise.

    XX is a list containing a matrix for each household. The household matrix
      is of size n x m where n is the size of the household, and m is one
      less than the number of age categories.

    It is unclear why XX should be encoded in this way as this seems very
    prone to off-by-one errors. Nevertheless, this is what the mynll()
    function below expects.
    """
    g = nx.read_graphml(filename)

    ## these are the criteria by which to decide if an individial has been
    ## infected. any will do.
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

    Y = numba.typed.List()
    XX = numba.typed.List()

    lower, upper = cutoff
    def age_of_interest(a):
        return a >= lower and a <= upper

    for hh in households(g):
        ms = list(members(g, hh))
        n = len(ms)
        age_groups = [1 if age_of_interest(ages[m]) else 0 for m in ms]
        myx = np.zeros((n, 1)) ### XXX this 1 should not be hardcoded
        myy = np.zeros(n)
        for j, a in enumerate(age_groups):
            if a > 0:
                myx[j, a-1] = 1
            if is_positive(ms[j]):
                myy[j] = 1
        Y.append(myy)
        XX.append(myx)
    return Y, XX


@numba.jit(nopython=True, cache=True)
def phi(s, logtheta=0.0):
    theta = np.exp(logtheta)
    return (1.0 + theta * s) ** (-1.0 / theta)


@numba.jit(nopython=True, cache=True)
def decimal_to_bit_array(d, n_digits):
    powers_of_two = 2 ** np.arange(32)[::-1]
    return ((d & powers_of_two) / powers_of_two)[-n_digits:]

@numba.jit(nopython=True, parallel=False, fastmath=False, cache=True)
def mynll(x, Y, XX):
    na = 1 ### XXX the maximum number identifying subpopulations
           ### probably better passed in as an argument
    if True:  # Ideally catch the linear algebra fail directly
        llaL = x[0]
        llaG = x[1]
        logtheta = x[2]
        eta = (4.0 / np.pi) * np.arctan(x[3])
        alpha = x[4 : (4 + na)]
        beta = x[(4 + na) : (4 + 2 * na)]
        gamma = x[(4 + 2 * na) :]

        num_households = len(Y)
        nlv = np.zeros(num_households)  # Vector of negative log likelihoods
        for i in range(0, num_households):
            y = Y[i]
            X = XX[i]
            if np.all(y == 0.0):
                nlv[i] = np.exp(llaG) * np.sum(np.exp(alpha @ (X.T)))
            else:
                # Sort to go zeros then ones WLOG (could do in pre-processing)
                ii = np.argsort(y)
                y = y[ii]
                X = X[ii, :]
                q = np.sum(y > 0)
                r = 2 ** q
                m = len(y)

                # Quantities that don't vary through the sum
                Bk = np.exp(-np.exp(llaG) * np.exp(alpha @ (X.T)))
                laM = np.exp(llaL) * np.outer(
                    np.exp(beta @ (X.T)), np.exp(gamma @ (X.T))
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
                nlv[i] = -np.log(LA.solve(BB, np.ones(r))[-1])
        nll = np.sum(nlv)
        # nll += 7.4*np.sum(x**2) # Add a Ridge if needed
        nll += np.sum(x ** 2)  # Add a Ridge if needed
        return nll
    else:
        # This was a try/except block but these are not supported by numba. TODO: work out and implement correct branching logic
        nll = np.inf
        return nll

def command():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nrestarts", type=int, default=1, help="Number of times to run the fitting process")
    parser.add_argument("--seed", type=int, default=46, help="Random seed")
    parser.add_argument("--lower", type=int, default=0, help="Lower age cutoff")
    parser.add_argument("--upper", type=int, default=20, help="Upper age cutoff")
    parser.add_argument("graph", help="GraphML file containing network")

    args = parser.parse_args()

    Y, XX = read_graph_data(args.graph, cutoff=(args.lower, args.upper))
    num_households = len(Y)
    na = 1

    # Indicative parameters - to do, add bounds and mulitple restarts
    x0 = np.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    mynll(x0, Y, XX)

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

    def callbackF(x, x2=0.0, x3=0.0):
        log.info(
            "Evaluated at [{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}]: {:.8f}".format(
                x[0], x[1], x[2], x[3], x[4], x[5], x[6], mynll(x, Y, XX)
            )
        )

    ## XXX the below should be restructured. why do we set the seed only after the
    ## first iteration? why do we have different options to the minimizer?
    foutstore = []
    fout = op.minimize(
        mynll,
        x0,
        (Y, XX),
        method="TNC",
        callback=callbackF,
        bounds=bb,
        options={"maxiter": 10000},
    )
    foutstore.append(fout)

    np.random.seed(args.seed)

    ## Find different minima from a selection of starting points.
    log.info("nrestarts: {}".format(args.nrestarts))
    for k in range(0, args.nrestarts):
        nll0 = np.nan
        while (np.isnan(nll0)) or (np.isinf(nll0)):
            xx0 = np.random.uniform(bb[:, 0], bb[:, 1])
            nll0 = mynll(xx0, Y, XX)
        try:
            log.info("Starting at: {} - {}".format(xx0, nll0))
            fout = op.minimize(
                mynll,
                xx0,
                (Y, XX),
                bounds=bb,
                method="TNC",
                callback=callbackF,
                options={"maxiter": 1000, "ftol": 1e-9},
            )
            log.info("Found: {}".format(fout.x))
            foutstore.append(fout)
        except:
            k -= 1

    ff = np.inf * np.ones(len(foutstore))
    for i in range(0, len(foutstore)):  # In case of crash
        if foutstore[i].success:
            ff[i] = foutstore[i].fun


    xhat = foutstore[ff.argmin()].x
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
                mynll(xhat + ej + ek, Y, XX)
                - mynll(xhat + ej - ek, Y, XX)
                - mynll(xhat - ej + ek, Y, XX)
                + mynll(xhat - ej - ek, Y, XX)
            )
            ek[k] = 0.0
        Hinv[j, j] = (
            -mynll(xhat + 2 * ej, Y, XX)
            + 16 * mynll(xhat + ej, Y, XX)
            - 30 * mynll(xhat, Y, XX)
            + 16 * mynll(xhat - ej, Y, XX)
            - mynll(xhat - 2 * ej, Y, XX)
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

