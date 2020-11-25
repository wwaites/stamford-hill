import argparse
import networkx as nx
from networkx.algorithms import isomorphism
import sys
import numpy as np
import logging
from stamford.data import read_data, generate_graph

log = logging.getLogger(__name__)

def nodes_by_kind(g, kind):
    return [n for (n,k) in nx.get_node_attributes(g, "kind").items() if k == kind]
def households(g):
    return nodes_by_kind(g, "household")
def people(g):
    return nodes_by_kind(g, "person")
def shuls(g):
    return nodes_by_kind(g, "shul")
def yeshivas(g):
    return nodes_by_kind(g, "yeshiva")
def mikvahs(g):
    return nodes_by_kind(g, "mikvah")
def places(g):
    return { "shul": shuls(g), "yeshiva": yeshivas(g), "mikvah": mikvahs(g) }
def members(g, hh):
    """
    returns household members sorted by descending age and sex.
    """
    ages  = nx.get_node_attributes(g, "age")
    sexes = nx.get_node_attributes(g, "sex")
    return sorted(nx.neighbors(g, hh), key=lambda m: (-ages[m], 0 if sexes[m] == "male" else 1))

def household_graphs(g):
    i = 0 ## counter for synthetic shuls, yeshivas, mikvahs, etc
    ps = people(g)
    pl = places(g)
    graphs = []
    sexes = nx.get_node_attributes(g, "sex")
    for hh in households(g):
        hhg = nx.Graph()
        hhg.add_node(hh, kind="household")
        hangouts = {}
        for m in nx.neighbors(g, hh): ## first find the people of the household
            hhg.add_node(m, kind="person", sex=sexes[m])
            hhg.add_edge(m, hh)
            for p in nx.neighbors(g, m): ## next find the places that person hangs out in
                size = nx.degree(g, p)  ## hoe big is the place
                for kind in pl: ## figure out what kind of place it is
                    if p in pl[kind]:
                        if p not in hangouts: ## need to mint a new place
                            i += 1
                            hangouts[p] = i
                            hhg.add_node(i, kind=kind)
                        hhg.add_edge(m, hangouts[p])
        graphs.append(hhg)
    return graphs

def household_motifs(g):
    motifs = {}
    for hhg in household_graphs(g):
        found = False
        for k, (m, count) in motifs.items():
            GM = isomorphism.GraphMatcher(m, hhg)
            if GM.is_isomorphic():
                motifs[k] = (m, count+1)
                found = True
                break
        if not found:
            motifs[len(motifs)] = (hhg, 1)

    result = nx.Graph()
    for k, (g, c) in motifs.items():
        if c > 1:
            result = nx.disjoint_union(result, g)
    nx.set_node_attributes(result, dict((hh, { "width": 1, "height": 1 }) for hh in households(result)))
    return result

def degrees(g):
    hh_degs = [d for (_, d) in nx.degree(g, households(g))]
    shul_degs = [d for (_, d) in nx.degree(g, shuls(g))]
    yeshiva_degs = [d for (_, d) in nx.degree(g, yeshivas(g))]
    mikvah_degs = [d for (_, d) in nx.degree(g, mikvahs(g))]

    return [hh_degs, shul_degs, yeshiva_degs, mikvah_degs]

def command():
    parser = argparse.ArgumentParser("stamford_graph")
    parser.add_argument("survey", help="Data file (.zip) containing survey data downloaded from ODK")
    parser.add_argument("--maximal", action="store_true", default=False, help="Do not minimise output; store all annotations in the graph")
    parser.add_argument("--motifs", "-m", action="store_true", default=False, help="Generate a graph of household motifs")
    args = parser.parse_args()

    data = read_data(args.survey)
    g = generate_graph(data, minimal=not args.maximal)

    if args.motifs:
        g = household_motifs(g)

    for line in nx.generate_graphml(g):
        print(line)

if __name__ == '__main__':
    command()

