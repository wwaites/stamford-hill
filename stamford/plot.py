import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from stamford import graph
from stamford.colours import LSHTM_COLOURS
import argparse

colours = list(LSHTM_COLOURS.values())

def command():
    parser = argparse.ArgumentParser("stamford_plot")
    parser.add_argument("graph", help="Sampled population graph")
    args = parser.parse_args()

    g = nx.read_graphml(args.graph)

    hh_degs, shul_degs, yeshiva_degs, mikvah_degs = graph.degrees(g)

    fig, ax = plt.subplots(2,2, figsize=(12,6))

#    fig.suptitle("Population distributions in various settings")
    ax[0][0].hist(hh_degs, bins=np.linspace(1,15,15)-0.5, density=False, rwidth=0.9, color=colours[2])
    ax[0][0].axvline(np.average(hh_degs), color=colours[1])
    ax[0][0].text(0.5, 0.9, "mean = {:.1f}".format(np.average(hh_degs)), transform=ax[0][0].transAxes)
    ax[0][0].set_ylabel("Households (total {})".format(len(graph.households(g))))
    ax[0][0].set_xlabel("Size")
    ax[0][1].hist(shul_degs, bins=15, density=False, rwidth=0.9, color=colours[2])
    ax[0][1].axvline(np.average(shul_degs), color=colours[1])
    ax[0][1].text(0.5, 0.9, "mean = {:.1f}".format(np.average(shul_degs)), transform=ax[0][1].transAxes)
    ax[0][1].set_ylabel("Shuls (total {})".format(len(graph.shuls(g))))
    ax[0][1].set_xlabel("Size")
    ax[1][0].hist(yeshiva_degs, bins=15, density=False, rwidth=0.9, color=colours[2])
    ax[1][0].axvline(np.average(yeshiva_degs), color=colours[1])
    ax[1][0].text(0.5, 0.9, "mean = {:.1f}".format(np.average(yeshiva_degs)), transform=ax[1][0].transAxes)
    ax[1][0].set_ylabel("Yeshivot (total {})".format(len(graph.yeshivas(g))))
    ax[1][0].set_xlabel("Size")
    ax[1][1].hist(mikvah_degs, bins=15, density=False, rwidth=0.9, color=colours[2])
    ax[1][1].axvline(np.average(mikvah_degs), color=colours[1])
    ax[1][1].text(0.5, 0.9, "mean = {:.1f}".format(np.average(mikvah_degs)), transform=ax[1][1].transAxes)
    ax[1][1].set_ylabel("Mikvot (total {})".format(len(graph.mikvahs(g))))
    ax[1][1].set_xlabel("Size")
    ax[1][1].set_ylim((0,18))
    ax[1][1].set_yticks([2*y for y in range(10)])

    fig.tight_layout()
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
    [ax.text(age-0.15, 2.5, "n = {}".format(len(scount.get(age, []))), rotation="vertical") for age in x]

    fig.tight_layout()
    fig.savefig("age-reporting-symptoms.png")

    ccount = {}
    dates = nx.get_node_attributes(g, "dates.covid")
    for p in graph.people(g):
        band = ccount.setdefault(int(ages[p]/5), [])
        cov = dates.get(p)
        if isinstance(cov, str):
            band.append(True)
        else:
            band.append(False)

    fig, ax = plt.subplots(1,1, figsize=(12,6))
    fig.suptitle("Percentage of individuals by age reporting COVID-19 infection dates")
    x = list(range(max(ccount.keys())+1))
    y = [int(100*sum(ccount.get(age, []))/len(ccount.get(age, [False]))) for age in x]
    ax.bar(x, y)
    labels = ["{}-{}".format(5*age, 5*age+4) for age in x]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation="vertical")
    [ax.text(age-0.15, 2.5, "n = {}".format(len(ccount.get(age, []))), rotation="vertical") for age in x]

    fig.tight_layout()
    fig.savefig("age-reporting-covid.png")

    tcount = {}
    swabs = nx.get_node_attributes(g, "swab.test.result")
    swabscreens = nx.get_node_attributes(g, "swab.test.screening.result")
    seros = nx.get_node_attributes(g, "serology.test.result")
    seroscreens = nx.get_node_attributes(g, "serology.test.screening.result")
    for p in graph.people(g):
        band = tcount.setdefault(int(ages[p]/5), [])
        positive = False
        for d in (swabs, swabscreens, seros, seroscreens):
            result = d.get(p)
            if isinstance(result, str) and result =="positive":
                positive = True
        if positive:
            band.append(True)
        else:
            band.append(False)

    fig, ax = plt.subplots(1,1, figsize=(12,6))
    fig.suptitle("Percentage of individuals by age with a positive COVID-19 test result")
    x = list(range(max(tcount.keys())+1))
    y = [int(100*sum(tcount.get(age, []))/len(tcount.get(age, [False]))) for age in x]
    ax.bar(x, y)
    labels = ["{}-{}".format(5*age, 5*age+4) for age in x]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation="vertical")
    [ax.text(age-0.15, 2.5, "n = {}".format(len(tcount.get(age, []))), rotation="vertical") for age in x]

    fig.tight_layout()
    fig.savefig("age-test-covid.png")

    cdate = {}
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
#    fig.suptitle("Self-reported COVID-19 cases")
    ax.bar(range(len(curve)), curve, color=colours[1])
    ticks = range(len(curve))
    labels = ["{} {}".format(h, m) for m in months for h in ["Early", "Late"]]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation="vertical")
    plt.subplots_adjust(bottom=0.15)

    fig.tight_layout()
    fig.savefig("self-reported-curve.png")

    cdate = {}
    syms =  nx.get_node_attributes(g, "symptoms")
    for p in graph.people(g):
        if p not in swabscreens and p not in seroscreens:
            continue
        if isinstance(syms.get(p), str):
            continue
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
    fig.suptitle("Asymptomatic screening")
    ax.bar(range(len(curve)), curve, color=colours[1])
    ticks = range(len(curve))
    labels = ["{} {}".format(h, m) for m in months for h in ["Early", "Late"]]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation="vertical")
    plt.subplots_adjust(bottom=0.15)

    fig.tight_layout()
    fig.savefig("asymptomatic-screening.png")


    symptoms = set()
    for syms in nx.get_node_attributes(g, "symptoms").values():
        if not isinstance(syms, str):
            continue
        for sym in syms.split(" "):
            symptoms.add(sym)
    for sym in symptoms:
        cdate = {}
        for p in graph.people(g):
            s = g.nodes[p]["symptoms"]
            if not isinstance(s, str) or sym not in s:
                continue
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
        fig.suptitle(f"Symptoms by date: {sym}")
        ax.bar(range(len(curve)), curve, color=colours[1])
        ticks = range(len(curve))
        labels = ["{} {}".format(h, m) for m in months for h in ["Early", "Late"]]
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation="vertical")
        plt.subplots_adjust(bottom=0.15)

        fig.tight_layout()
        fig.savefig(f"symptoms-{sym}.png")

    ## serology dates
    sdate = {}
    for test in ["serology_pos", "spike_pos", "spike_pos2", "RBD_pos", "NC_pos"]:
        for p in graph.people(g):
            pos = g.nodes[p].get(test, np.nan)
            if np.isnan(pos) or pos == 0:
                continue
            dc = dates.get(p)
            if isinstance(dc, str):
                for d in dc.split(" "):
                    count = sdate.get(d, 0)
                    sdate[d] = count + 1

        curve = np.zeros(len(months)*2)
        for d, c in sdate.items():
            half, month = d.split(".")
            half = 0 if half == "Early" else 1
            curve[2*months[month] + half] = c

        fig, ax = plt.subplots(1,1, figsize=(12,6))
        fig.suptitle(f"COVID-19 cases ({test})")
        ax.bar(range(len(curve)), curve, color=colours[1])
        ticks = range(len(curve))
        labels = ["{} {}".format(h, m) for m in months for h in ["Early", "Late"]]
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation="vertical")
        plt.subplots_adjust(bottom=0.15)

        fig.tight_layout()
        fig.savefig(f"{test}-curve.png")

if __name__ == '__main__':
    command()
