__all__ = ["build_graph"]

from stamford.graph import people
import networkx as nx
import numpy as np

def build_graph(graphml, other=False):
    g = nx.read_graphml(graphml)
    if not other:
        g.remove_node("S80")
        g.remove_node("M80")
        g.remove_node("Y54")
    for node, kind in nx.get_node_attributes(g, "kind").items():
        if kind == "person":
            g.nodes[node]["bipartite"] = 0
        else:
            g.nodes[node]["bipartite"] = 1
            g.nodes[node]["label"] = kind
    return g

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

import click
from netabc.command import cli
@cli.command(name="stamford")
@click.option("-f", "--filename", required=True, help="File containing annotated graph data")
@click.pass_context
def command(ctx, filename):
    g = build_graph(filename)
    for p in people(g):
        spike = g.nodes[p].get("spike_pos")
        if spike == 1.:
            g.nodes[p]["c"] = "r"
        elif spike == 0.:
            g.nodes[p]["c"] = "s"
        else:
            g.remove_node(p)
    for kind in ("household", "shul", "yeshiva", "mikvah"):
        print(kind, sar(g, kind))
