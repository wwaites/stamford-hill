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
        ahist = attack_histograms(g, t)
        dist = 0.0
        for size in ghist:
            u = ghist[size]
            if size not in ahist:
                continue
            v = ahist[size]
#            print(u, v)
            dist += wasserstein(u/u.sum(), v/v.sum())
        return dist

def attack_histograms(g, t):
    attacks = {}
    for node in g:
        if g.nodes[node]["type"] != t:
            continue
        r, n = 0, 0
        for c in nx.neighbors(g, node):
            n += 1
            if g.nodes[c].get("c", "s") == "r":
                r += 1
        if n == 0:
            continue
        hist = attacks.get(n)
        if hist is None:
            hist = np.zeros(n+1)
            attacks[n] = hist
        hist[r] += 1
    return attacks

emsar = EMSAR()

import click
from netabc.command import cli
@cli.command(name="stamford")
@click.pass_context
def command(ctx):
    import inspect
    args = [v for k,v in ctx.obj.fixed.items() if k in inspect.getfullargspec(ctx.obj.graph).args]
    g = ctx.obj.graph(*args)

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
            g.remove_node(p)
    emsar.data(g)
    for kind in ("household", "synagogue", "school", "yeshiva", "mikvah"):
        click.secho(f"secondary attack rate for places of type {kind}: {sar(g, kind)}", fg="green")
        assert emsar(g, kind) == 0.0
