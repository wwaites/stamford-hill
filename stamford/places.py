from stamford.graph import people, yeshivot, synagogues, mikvahs
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from math import sqrt

months = dict((m, i) for (i, m) in enumerate(["Feb", "Mar", "Apr", "May", "June", "July", "Aug", "Sep", "Oct", "Nov"]))

def dates(g, p):
    dc = g.nodes[p].get("dates.covid")
    if not isinstance(dc, str):
        return []
    reports = [ d.split(".") for d in dc.split(" ") ]
    return [ 2*months[month] + (0 if half == "Early" else 1) for half, month in reports ]

def placegraph(g):
    pg = nx.Graph()

    pat = nx.Graph()
    pat.add_node(0)
    pat.add_node(1, type="person")
    pat.add_node(2)
    pat.add_edge(0,1)
    pat.add_edge(1,2)

    ## these are not real places, they are "other" in the survey
    other = ('S80', 'M80', 'Y54')
    def node_match(n, p):
        return all(k in n and n[k] == p[k] for k in p)

    gm = nx.isomorphism.GraphMatcher(g, pat, node_match)
    for mapping in gm.subgraph_isomorphisms_iter():
        p1, per, p2 = [m[0] for m in sorted(mapping.items(), key=lambda m: m[1])]
        if g.nodes[p1]["type"] == "household":
            continue
        if g.nodes[p2]["type"] == "household":
            continue
        if p1 not in pg:
            pg.add_node(p1, type=g.nodes[p1]["type"], size=0, times=np.zeros(2*len(months)))
        if p2 not in pg:
            pg.add_node(p2, type=g.nodes[p2]["type"], size=0, times=np.zeros(2*len(months)))
        if (p1, p2) not in pg.edges:
            ## these correspond to "other", do not create edges
            if p1 not in other and p2 not in other:
                pg.add_edge(p1, p2, weight=0)

        pg.nodes[p1]["size"] += 1
        pg.nodes[p2]["size"] += 2
        if p1 not in other and p2 not in other:
            pg.edges[p1, p2]["weight"] += 1

        spike = g.nodes[per].get("spike_pos")
        if spike == 1.0:
            for date in dates(g, per):
                pg.nodes[p1]["times"][date] += 1
                pg.nodes[p2]["times"][date] += 1

    return pg

def draw(g, fname, step=4, pos=None):

    fig = plt.figure(figsize=(12,8))
    k = 100/sqrt(len(g))
    if pos is None:
        pos = nx.spring_layout(g, k)

    nodes = list(g.nodes)
    min_size, max_size = 50, 250
    min_line, max_line = 0.05, 5
    node_sizes = nx.get_node_attributes(g, "size")
    max_node_size = max(node_sizes.values())
    node_sizes_normed = { n: float(node_sizes[n])*(max_size-min_size)/max_node_size+min_size for n in nodes }

    node_burden = { k: float(v[step])/max_node_size for k, v in nx.get_node_attributes(g, "times").items() }
    max_burden = max( max(v/max_node_size) for v in nx.get_node_attributes(g, "times").values() )
    cmap = plt.get_cmap("rainbow")
    node_colors = { k: cmap(int(255*node_burden[k]/max_burden)) for k in nodes }

    cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap), ax=plt.axes())
    cbar.set_ticks([])

    labels = {k:k for k in g}

    edge_weights = nx.get_edge_attributes(g, "weight")
    max_edge_weight = max(edge_weights.values())
    for e in g.edges:
        width = g.edges[e]["weight"]*(max_line-min_line)/max_edge_weight+min_line
        nx.draw(g, pos=pos, nodelist=[], edgelist=[e], edge_color="grey", width=width)

    shul_nodes = shuls(g)
    nx.draw(g, pos=pos,
            edgelist=[],
            nodelist=shul_nodes,
            labels=labels, font_size=8,
            node_shape='o',
            node_size=[node_sizes_normed[n] for n in shul_nodes],
            node_color=[node_colors[n] for n in shul_nodes],
            )

    yeshiva_nodes = yeshivas(g)
    nx.draw(g, pos=pos,
            edgelist=[],
            nodelist=yeshiva_nodes,
            labels=labels, font_size=8,
            node_shape='s',
            node_size=[node_sizes_normed[n] for n in yeshiva_nodes],
            node_color=[node_colors[n] for n in yeshiva_nodes]
            )
    fig.savefig(fname)

    mikvah_nodes = mikvahs(g)
    nx.draw(g, pos=pos,
            edgelist=[],
            nodelist=mikvah_nodes,
            labels=labels, font_size=8,
            node_shape='d',
            node_size=[node_sizes_normed[s] for s in mikvah_nodes],
            node_color=[node_colors[n] for n in mikvah_nodes]
            )

    imonths = { v:k for k,v in months.items() }
    early = "Early" if step == 0 or step % 2 == 0 else "Late"
    label = f"{early} {imonths[int(step/2)]}"
    fig.text(0.1, 0.1, label)

    fig.savefig(f"{fname}-{step:04d}.png")

    return pos
