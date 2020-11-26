import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from stamford import graph
import argparse


def command():
    parser = argparse.ArgumentParser("stamford_plot")
    parser.add_argument("graph", help="Sampled population graph")
    args = parser.parse_args()

    g = nx.read_graphml(args.graph)

    hh_degs, shul_degs, yeshiva_degs, mikvah_degs = graph.degrees(g)

    fig, ax = plt.subplots(2,2, figsize=(10,10))

    fig.suptitle("Population distributions in various settings")
    ax[0][0].hist(hh_degs, density=False)
    ax[0][0].set_ylabel("Households (total {})".format(len(graph.households(g))))
    ax[0][0].set_xlabel("Size")
    ax[0][1].hist(shul_degs, density=False)
    ax[0][1].set_ylabel("Shuls (total {})".format(len(graph.shuls(g))))
    ax[0][1].set_xlabel("Size")
    ax[1][0].hist(yeshiva_degs, density=False)
    ax[1][0].set_ylabel("Yeshivas (total {})".format(len(graph.yeshivas(g))))
    ax[1][0].set_xlabel("Size")
    ax[1][1].hist(mikvah_degs, density=False)
    ax[1][1].set_ylabel("Mikvahs (total {})".format(len(graph.mikvahs(g))))
    ax[1][1].set_xlabel("Size")

    fig.savefig("degree-distributions.png")

    # shul_yeshivas = graph.correlations(g)

    # fig, ax = plt.subplots(1,1, figsize=(12,6))
    # fig.suptitle("Conditional probability of attending yeshivas given elder shul")
    # yeshivas = sorted(graph.yeshivas(g))
    # yeshiva_map = dict((y, i) for (i,y) in enumerate(yeshivas))
    # x = np.array(range(len(yeshivas)), dtype=np.float)
    # width = 1.0/(len(yeshivas))
    # for i, shul in enumerate(sorted(shul_yeshivas.keys())):
    #     y = [0]*len(yeshivas)
    #     total = sum(shul_yeshivas[shul].values())
    #     for yeshiva, mult in shul_yeshivas[shul].items():
    #         y[yeshiva_map[yeshiva]] = float(mult)/total
    #     ax.bar(x+i*width, y, width=width, label=shul)
    # ax.set_xlabel("Yeshiva number")
    # ax.legend(ncol=6,framealpha=0)
    # fig.savefig("conditional-yeshiva.png")

    scount = {}
    ages = nx.get_node_attributes(g, "age")
    symptoms = nx.get_node_attributes(g, "symptoms")
    for p in graph.people(g):
        band = scount.setdefault(int(ages[p]/5), [])
        sym = symptoms.get(p)
        if isinstance(sym, str):
            band.append(True)
        else:
            band.append(False)


    fig, ax = plt.subplots(1,1, figsize=(12,6))
    fig.suptitle("Percentage of individuals by age reporting symptoms")
    x = list(range(max(scount.keys())+1))
    y = [int(100*sum(scount.get(age, []))/len(scount.get(age, [False]))) for age in x]
    ax.bar(x, y)
    labels = ["{}-{}".format(5*age, 5*age+4) for age in x]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation="vertical")
    fig.savefig("age-reporting-symptoms.png")

    cdate = {}
    dates = nx.get_node_attributes(g, "dates.covid")
    for p in graph.people(g):
        dc = dates.get(p)
        if isinstance(dc, str):
            for d in dc.split(" "):
                count = cdate.get(d, 0)
                cdate[d] = count + 1

    months = dict((m, i) for (i, m) in enumerate(["Feb", "Mar", "Apr", "May", "June", "July", "Aug", "Sep", "Oct", "Nov"]))
    curve = np.zeros(len(months)*2)
    for d, c in cdate.items():
        half, month = d.split(".")
        half = 0 if half == "Early" else 1
        curve[2*months[month] + half] = c

    fig, ax = plt.subplots(1,1, figsize=(12,6))
    fig.suptitle("Self-reported COVID-19 cases")
    ax.bar(range(len(curve)), curve)
    ticks = range(len(curve))
    labels = ["{} {}".format(h, m) for m in months for h in ["Early", "Late"]]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation="vertical")
    plt.subplots_adjust(bottom=0.15)
    fig.savefig("self-reported-curve.png")

if __name__ == '__main__':
    command()
