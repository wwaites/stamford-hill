import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import sys
import inspect
from stamford import graph
from netabc.utils import envelope
from netabc.command import cli
from netabc.plot import colours
import click

def household_distributions(g):
    degs = {}
    pos = {}
    for h in graph.households(g):
        if "enriched" not in g.nodes[h]:
            print(f"household {h} is neither random nor enriched")
            continue
        if g.nodes[h]["enriched"]:
            print(f"skipping enriched household {h}")
            continue
        else:
            print(f"using random household {h}")
        members = list(nx.neighbors(g,h))
        sero = False
        spike = 0
        for m in members:
            s = g.nodes[m].get("spike_pos")
            if s is None:
                continue
            sero = True
            if s == 1.0:
                spike += 1
        if sero:
            degs[h] = len(members)
            pos[h] = spike

    subplots = {}
    for h, d in degs.items():
        p = pos[h]
        if d > 10:
            d = 10
        if p > 10:
            p = 10
        subplots.setdefault(d, []).append(p)

    attacks = {}
    for d in sorted(subplots):
        dens, cases = np.histogram(subplots[d], bins=np.linspace(0,10,12), density=True)
        ar = sum(dens*cases[:11])/d
        attacks[d] = ar
        print(f"size {d}: n = {len(subplots[d])} attack rate = {ar}")
    fig, axes = plt.subplots(2,5, figsize=(10,5))
    for i in range(10):
        ax = axes[int(i/5)][i%5]
        ax.hist(subplots[i+1], bins=np.linspace(0,10,11)-0.5, rwidth=0.9, density=False)
        if i == 9:
            ax.set_title(f"size $\geq$ {i+1}, AR $\geq$ {100*attacks[i+1]:.0f}%")
        else:
            ax.set_title(f"size = {i+1}, AR $\geq$ {100*attacks[i+1]:.0f}%")
#        ax.set_ylim(0,0.6)

    for i in range(2):
        ax = axes[i][0]
        ax.set_ylabel("Number of households")
    for i in range(5):
        ax = axes[1][i]
        ax.set_xlabel("Number of infections")
    fig.tight_layout()
    fig.savefig("houshold-distributions.png")

def command():
    parser = argparse.ArgumentParser("stamford_plot")
    parser.add_argument("graph", help="Sampled population graph")
    args = parser.parse_args()

    g = nx.read_graphml(args.graph)

    household_distributions(g)

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


@cli.command(name="plot_scaled_activity")
@click.option("--observable", "-o", "observables",
              type=(str,str), multiple=True,
              help="Observable, type pairs to plot.")
@click.pass_context
def plot_scaled_activity(ctx, observables):
    """
    Plot rule activity scaled by degree
    """
    if "graph" not in ctx.obj:
        click.secho(f"No population graph specified.", fg="red")
        sys.exit(-1)

    args = [v for k,v in ctx.obj.fixed.items() if k in inspect.getfullargspec(ctx.obj.graph).args]
    g = ctx.obj.graph(*args)

    if "store" not in ctx.obj:
        click.secho(f"No data store specified. Cannot tell where to store results.", fg="red")
        sys.exit(-1)
    h5 = ctx.obj.store

#    if output is None:
    output = ctx.obj.prefix.strip("/")

    samples = [h5[k] for k in h5 if k.startswith(ctx.obj.prefix)]

    nobs = len(observables)
    fig, axes = plt.subplots(nobs,2, figsize=(10,2*nobs))
    degrees = nx.degree(g)
    data = []

    axa = axes[:,0]
    axr = axes[:,1]

    for i, (obs, kind) in enumerate(observables):
        paths = 0
        for n in g:
            if g.nodes[n]["type"] == kind:
                paths += degrees[n]
        if paths == 0:
            click.secho(f"Couldn't find any vertices of type {kind} in graph", fg="red")
            return

        abss = [s[obs].iloc[-1]/sum(s[o].iloc[-1] for o,_ in observables) for s in samples]
        axa[i].axvline(np.mean(abss))
        axa[i].hist(abss, bins=50, range=(0,0.5), density=True, color=colours[i], label=kind)
        axa[i].set_xlim(0,0.5)
        axa[i].legend()

        rels = [s[obs].iloc[-1]/paths for s in samples]
        axr[i].axvline(np.mean(rels))
        axr[i].hist(rels, bins=50, range=(0,0.5), density=True, color=colours[i], label=kind)
        axr[i].set_xlim(0,0.5)
        axr[i].legend()

    fig.tight_layout()
    fig.savefig(f"{output}-final-activity.png")
    #avg, std = envelope(samples)

if __name__ == '__main__':
    command()
