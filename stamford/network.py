__all__ = ["build_graph"]

from stamford.graph import people, household
import networkx as nx
import numpy as np
from scipy.stats import wasserstein_distance as wasserstein

def sar(g, kind):
    places = {}
    attacks = {}
    for node in g:
        if g.nodes[node]["type"] != kind:
            continue
        r, n = 0., 0.
        for c in nx.neighbors(g, node):
            n += 1.
            if g.nodes[c].get("c", "s") == "r":
                r += 1.
        if n == 0.:
            continue
        places[n] = places.get(n, 0.) + 1.
        attacks.setdefault(n, []).append(r/n)
    ##
    ## now we have a histogram of place sizes, and
    ## a list of attack rates for each size.
    ## we (1) figure out the average attack rate for
    ## each size and (2) make the weighted average
    ## for all sizes to find an overall attack rate
    ##
    N = sum(places.values())
    return sum( np.mean(attacks[n])*places[n]/N for n in places )
#    return np.mean([np.mean(attacks[n]) for n in places])

class EMSAR(object):
    def __init__(self):
        self.attacks = {}
    def data(self, g):
        types = set(g.nodes[n]["type"] for n in g)
        for t in types:
            self.attacks[t] = attack_histograms(g, t)
    def __call__(self, g, t):
        ghist = self.attacks[t]
        ahist = attack_histograms(g, t, ghist)
        dist = 0.0
        for size in ghist:
            ## get attack rate and censorship histograms
            ## for the reference data
            u, x = ghist[size]
            if size not in ahist:
                continue
            ## there is no intrinsic censorship in the
            ## simulate
            v, _ = ahist[size]
            ## now, we should censor the simulation data
            ## with the same censorship probability
#            print(u, v)
            dist += wasserstein(u/u.sum(), v/v.sum())
        return dist

def attack_histograms(g, t, ref = None):
    """
    Calculate attack rate and censorship histogram for
    settings of type t. Optionally takes reference
    histograms from which to censor the data in g.
    """
    attacks = {}
    for node in g:
        if g.nodes[node]["type"] != t:
            continue

        ns = list(nx.neighbors(g, node))
        n = len(ns)
        if n == 0:
            continue

        ## decide how many household members to censor
        if ref is not None:
            if n not in ref:
                #click.secho(f"no {t} of size {n} exists", fg="red")
                drop = 0.0
            else:
                _, cens = ref[n]
                drop = float(np.random.choice(range(n+1), p=cens/cens.sum()))
                #print(f"size {n} censoring {drop}")
        else:
            drop = 0.0
        drop = drop/n

        r, x = 0, 0
        for c in ns:
            state = g.nodes[c].get("c", "s")
            ## censor measurements probabilistically
            if state == "x" or np.random.uniform() < drop:
                ## sanity check, should not see censored data from simulation
                if state == "x" and ref is not None:
                    raise ValueError(f"should not have state {state} in simulation data")
                x += 1
            elif state == "r":
                r += 1

        if x == n: ## no serology from anyone in this household
            continue

        hists = attacks.get(n)
        if hists is None:
            att  = np.zeros(n+1)
            cens = np.zeros(n+1)
            attacks[n] = (att, cens)
        else:
            att, cens = hists
        att[r]  += 1
        cens[x] += 1

    return attacks

emsar = EMSAR()

def process_serology(g):
    """
    Mark disease state in the graph according to measured
    serology status
    """
    for p in people(g):
        h = household(g, p)
        if g.nodes[h]["enriched"]:
            continue
        if g.nodes[p]["serology"]:
            if g.nodes[p]["positive"]:
                g.nodes[p]["c"] = "r"
            else:
                g.nodes[p]["c"] = "s"
        else:
            g.nodes[p]["c"] = "x"
    return g

def scale_graph(template_graph, graph_scale, graph_interventions):
    """
    Scale the network by adding or deleting a number of households
    """
    g = template_graph.copy()
    hh = [n for n in g.nodes if g.nodes[n]["type"] == "household"]
    n = len(hh)
    m = int(graph_scale*n)
    ## we select nodes to delete without replacement because we
    ## can only chuck out a node once. when we are inflating the
    ## network, however, we can use replacement because there is
    ## no harm in principle with duplicating a node twice and
    ## indeed for values of `graph_scale` > 2 we must do so.
    replace = True if graph_scale > 1.0 else False
    for h in np.random.choice(hh, abs(n-m), replace=replace):
        ms = [i for i in nx.neighbors(g, h)]
        if graph_scale < 1.0:
            g.remove_nodes_from(ms)
            g.remove_node(h)
        else:
            nid = max(int(i) for i in g.nodes) + 1
            hid = nid
            hattrs = g.nodes[h].copy()
            hattrs["label"] = f"SH{hid}"
            g.add_node(str(hid), **hattrs)
            nid += 1
            for p in ms:
                pattrs = g.nodes[p].copy()
                pattrs["label"] = f"SP{nid}"
                g.add_node(str(nid), **pattrs)
                g.add_edge(str(nid), str(hid))
                for loc in nx.neighbors(g, p):
                    if loc == h: continue
                    g.add_edge(str(nid), loc)
                nid += 1

    for kind, scale in graph_interventions:
        places = [n for n in g.nodes if g.nodes[n]["type"] == kind]
        degrees = nx.degree(g, places)
        places = sorted(places, key=lambda p: degrees[p])
        n = len(places)
        m = int(scale*n)
        g.remove_nodes_from(places[m:])

    for kind, scale in graph_interventions:
        places = [n for n in g.nodes if g.nodes[n]["type"] == kind]

    return g

import click
from netabc.command import cli
@cli.command(name="stamford")
@click.option("--scale", "-s", type=float, default=None,
              help="Scale graph by increasing/decreasing households")
@click.option("--interventions", "-i", type=(str, float), multiple=True,
              help="Close a proportion of places of the given kind")
@click.pass_context
def command(ctx, scale, interventions):
    import inspect
    args = [v for k,v in ctx.obj.fixed.items() if k in inspect.getfullargspec(ctx.obj.graph).args]
    g = ctx.obj.graph(*args)
    g = process_serology(g)

    emsar.data(g)

    if scale is not None or len(interventions) > 0:
        ctx.obj.fixed["graph_scale"] = scale
        ctx.obj.fixed["graph_interventions"] = interventions
        ctx.obj.graph = scale_graph

    # for kind in ("household", "synagogue", "school", "yeshiva", "mikvah"):
    #     click.secho(f"secondary attack rate for places of type {kind}: {sar(g, kind)}", fg="green")
    #     dist = emsar(g, kind)
    #     assert dist == 0.0, dist
