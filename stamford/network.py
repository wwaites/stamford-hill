__all__ = ["build_graph"]

import networkx as nx

def build_graph(graphml):
    g = nx.read_graphml(graphml)
    for node, kind in nx.get_node_attributes(g, "kind").items():
        if kind == "person":
            g.nodes[node]["bipartite"] = 0
        else:
            g.nodes[node]["bipartite"] = 1
            g.nodes[node]["label"] = kind
    return g
