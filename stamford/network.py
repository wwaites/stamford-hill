__all__ = ["build_graph"]

from stamford.graph import people, household
import networkx as nx
import numpy as np
import sys
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
            if t == "person":
                continue
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

        # for the reference histogram, only include those with serology
        if ref is None:
            if not any([g.nodes[p]["serology"] for p in ns]):
                continue
            ## do not include enriched households either
            if t == "household" and g.nodes[node]["enriched"]:
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

def scale_graph(template_graph, graph_scale):
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
    return g

def close_graph(template_graph, graph_close):
    """
    Remove the largest places from graph
    """
    g = template_graph.copy()

    for kind, scale in graph_close:
        places = [n for n in g.nodes if g.nodes[n]["type"] == kind]
        degrees = nx.degree(g, places)
        places = sorted(places, key=lambda p: degrees[p])
        n = len(places)
        m = int(scale*n)
        g.remove_nodes_from(places[m:])

    return g

def split_graph(template_graph, graph_split):
    """
    Split large places in the network
    """
    g = template_graph.copy()

    nid = max(int(i) for i in g.nodes) + 1

    for kind, scale in graph_split:
#        click.secho(f"splitting the top {1.0-scale} places of kind {kind}", fg="green")
        places  = [n for n in g.nodes if g.nodes[n]["type"] == kind]
        degree_dict = nx.degree(g, places)
        places  = sorted(places, key=lambda p: degree_dict[p])
        degrees = [degree_dict[p] for p in places]
        pctile  = np.percentile(degrees, scale*100)
#        click.secho(f"{int(scale*100)}th %ile size is {pctile}", fg="green")

        bigplaces = [places[i] for i in range(len(degrees)) if degrees[i] > pctile]
        for bigplace in bigplaces:
#            click.secho(f"splitting size {nx.degree(g, bigplace)} place {bigplace} {g.nodes[bigplace]}", fg="yellow")
            assert g.nodes[bigplace]["bipartite"] == 1

            members   = list(nx.neighbors(g, bigplace))
            chunks    = int(len(members)/pctile) + 1
            chunksize = int(len(members)/chunks)

            for _ in range(chunks - 1):
                newplace = nid
                nid += 1
                attrs = g.nodes[bigplace].copy()
                attrs["label"] = attrs["label"] + f"-C{nid}"
                g.add_node(newplace, **attrs)
#                click.secho(f"\tmade {newplace} {g.nodes[newplace]}", fg="yellow")

                move = np.random.choice(members, chunksize, replace=False)
                for m in move:
                    g.remove_edge(m, bigplace)
                    g.add_edge(m, newplace)
                    members.remove(m)
#                click.secho(f"\t{newplace} is size {nx.degree(g, newplace)}", fg="yellow")

#            click.secho(f"\tnow {bigplace} is size {nx.degree(g, bigplace)}", fg="yellow")

    return g

import click
from netabc.command import cli
@cli.command(name="stamford")
@click.option("--scale", "-s", type=float, default=None,
              help="Scale graph by increasing/decreasing households")
@click.option("--split", "-x", type=(str, float), multiple=True,
              help="Split places greater than the given %ile degree in the original network")
@click.option("--close", "-c", type=(str, float), multiple=True,
              help="Close a proportion of places of the given kind")
@click.pass_context
def command(ctx, scale, split, close):
    import inspect
    args = [v for k,v in ctx.obj.fixed.items() if k in inspect.getfullargspec(ctx.obj.graph).args]
    g = ctx.obj.graph(*args)
    g = process_serology(g)

    emsar.data(g)

    if np.sum([scale is not None, len(split) > 0, len(close) > 0]) > 1:
        click.secho("stamford: --scale and --split are mutually exclusive", fg="red")
        sys.exit(-1)

    if scale is not None:
        ctx.obj.fixed["graph_scale"] = scale
        ctx.obj.graph = scale_graph
    if len(split) > 0:
        ctx.obj.fixed["graph_split"] = split
        ctx.obj.graph = split_graph
    if len(close) > 0:
        ctx.obj.fixed["graph_close"] = close
        ctx.obj.graph = close_graph
