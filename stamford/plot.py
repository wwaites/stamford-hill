import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
import sys
import inspect
import ot
from stamford import graph
from stamford.network import emsar, attack_histograms
from netabc.utils import envelope
from netabc.command import cli
from netabc.plot import colours
import click

## Because of community sensitivities, we must produce plots with
## particular labels. The appropriate labels being the subject of
## ongoing discussion and debate, we use a translating dictionary
place_labels = {
    "household": "household",
    "primary": "primary school",
    "secondary": "secondary school",
    "synagogue": "place of worship",
    "mikvah": "ritual bath",
    "environment": "community",
}

def household_distributions(g, output=None):
    degs = {}
    pos = {}
    for h in graph.households(g):
        members = list(nx.neighbors(g,h))
        sero = False
        spike = 0
        if not any([g.nodes[m]["serology"] for m in members]):
            continue
        for m in members:
            if g.nodes[m]["serology"] and g.nodes[m]["positive"]:
                spike += 1

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

    if output is None:
        return subplots

    attacks = {}
    for d in sorted(subplots):
        dens, cases = np.histogram(subplots[d], bins=np.linspace(0,10,11), density=True)
        ar = sum(dens*cases[:10])/d
        attacks[d] = ar
        print(f"size {d}: n = {len(subplots[d])} attack rate = {ar}")

    fig, axes = plt.subplots(2,5, figsize=(10,5))
    for i in range(10):
        ax = axes[int(i/5)][i%5]
        if (i+1) not in subplots:
            continue
        ax.hist(subplots[i+1], bins=np.linspace(0,10,11), rwidth=0.9, density=True)
        if i == 9:
            ax.set_title(f"size $\geq$ {i+1}, AR $\geq$ {100*attacks[i+1]:.0f}%")
        else:
            ax.set_title(f"size = {i+1}, AR $\geq$ {100*attacks[i+1]:.0f}%")
#        ax.set_ylim(0,0.6)

    for i in range(2):
        ax = axes[i][0]
        ax.set_ylabel("Proportion of households")
    for i in range(5):
        ax = axes[1][i]
        ax.set_xlabel("Number of infections")
    fig.tight_layout()
    fig.savefig(f"{output}-houshold-distributions.png")

@cli.command(name="plot_stamford_data")
@click.option("--enriched", "-e", "enriched", is_flag=True, default=False,
              help="Plot only enriched households.")
@click.option("--random", "-r", "random", is_flag=True, default=False,
              help="Plot only random households.")
@click.option("--serology", "-s", "serology", is_flag=True, default=False,
              help="Plot only households with serology.")
@click.option("--output", "-o", "output", required=True,
              help="Output filename")
@click.pass_context
def plot_stamford_data(ctx, enriched, random, serology, output):
    """
    Generate plots about Stamford Hill
    """
    if "graph" not in ctx.obj:
        click.secho(f"No population graph specified.", fg="red")
        sys.exit(-1)

    args = [v for k,v in ctx.obj.fixed.items() if k in inspect.getfullargspec(ctx.obj.graph).args]
    g = ctx.obj.graph(*args)

    def is_enriched(n):
        return all([g.nodes[p]["enriched"] for p in nx.neighbors(g, n) if g.nodes[p]["type"] == "household"])
    if enriched:
        not_enriched = [n for n in g if g.nodes[n]["type"] == "person" and not is_enriched(n)]
        g.remove_nodes_from(not_enriched)
    elif random:
        not_random = [n for n in g if g.nodes[n]["type"] == "person" and is_enriched(n)]
        g.remove_nodes_from(not_random)
    if serology:
        no_serology = [n for n in g if g.nodes[n]["type"] == "person" and not g.nodes[n]["serology"]]
        g.remove_nodes_from(no_serology)

    household_distributions(g, output)

    hh_degs, primary_degs, secondary_degs, synagogue_degs, mikvah_degs = graph.degrees(g)

    fig, ax = plt.subplots(5,1, figsize=(6,15))

#    fig.suptitle("Population distributions in various settings")
    ax[0].hist(hh_degs, bins=np.linspace(1,15,15)-0.5, density=False, rwidth=0.9, color=colours[2])
    ax[0].axvline(np.average(hh_degs), color=colours[1])
    ax[0].text(0.5, 0.9, "mean = {:.1f}".format(np.average(hh_degs)), transform=ax[0].transAxes)
    ax[0].set_ylabel("Households (total {})".format(len(graph.households(g))))
    ax[0].set_xlabel("Size")

    ax[1].hist(primary_degs, bins=15, density=False, rwidth=0.9, color=colours[2])
    ax[1].axvline(np.average(primary_degs), color=colours[1])
    ax[1].text(0.5, 0.9, "mean = {:.1f}".format(np.average(primary_degs)), transform=ax[1].transAxes)
    ax[1].set_ylabel("Primaries (total {})".format(len(graph.primaries(g))))
    ax[1].set_xlabel("Size")

    ax[2].hist(secondary_degs, bins=15, density=False, rwidth=0.9, color=colours[2])
    ax[2].axvline(np.average(secondary_degs), color=colours[1])
    ax[2].text(0.5, 0.9, "mean = {:.1f}".format(np.average(secondary_degs)), transform=ax[2].transAxes)
    ax[2].set_ylabel("Secondaries (total {})".format(len(graph.secondaries(g))))
    ax[2].set_xlabel("Size")

    ax[3].hist(synagogue_degs, bins=15, density=False, rwidth=0.9, color=colours[2])
    ax[3].axvline(np.average(synagogue_degs), color=colours[1])
    ax[3].text(0.5, 0.9, "mean = {:.1f}".format(np.average(synagogue_degs)), transform=ax[3].transAxes)
    ax[3].set_ylabel("Synagogues (total {})".format(len(graph.synagogues(g))))
    ax[3].set_xlabel("Size")

    ax[4].hist(mikvah_degs, bins=15, density=False, rwidth=0.9, color=colours[2])
    ax[4].axvline(np.average(mikvah_degs), color=colours[1])
    ax[4].text(0.5, 0.9, "mean = {:.1f}".format(np.average(mikvah_degs)), transform=ax[4].transAxes)
    ax[4].set_ylabel("Mikvot (total {})".format(len(graph.mikvahs(g))))
    ax[4].set_xlabel("Size")

    fig.tight_layout()
    fig.savefig(f"{output}-degree-distributions.png")

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
    fig.savefig(f"{output}-age-reporting-symptoms.png")

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
    fig.savefig(f"{output}-age-reporting-covid.png")

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
    fig.savefig(f"{output}-age-test-covid.png")

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
    fig.savefig(f"{output}-self-reported-curve.png")

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
    fig.savefig(f"{output}-asymptomatic-screening.png")


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
        fig.savefig(f"{output}-symptoms-{sym}.png")

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
        fig.savefig(f"{output}-{test}-curve.png")


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
        if obs == "ACTE": ## special case for non-network-mediated transmission
            paths = len([n for n in g.nodes if g.nodes[n]["bipartite"] == 0]) - 1
        else:
            for n in g:
                if g.nodes[n]["type"] == kind:
                    paths += degrees[n]

        abss = [s[obs].iloc[-1]/sum(s[o].iloc[-1] for o,_ in observables) for s in samples]
        axa[i].axvline(np.mean(abss))
        axa[i].hist(abss, bins=50, range=(0,0.5), density=True, color=colours[i], alpha=0.5, edgecolor=colours[i], label=place_labels[kind])
        axa[i].set_xlim(0,0.5)
        axa[i].legend()

        if paths == 0:
            click.secho(f"Couldn't find any vertices of type {kind} in graph", fg="red")
            continue

        rels = [s[obs].iloc[-1]/paths for s in samples]
        axr[i].axvline(np.mean(rels))
        axr[i].hist(rels, bins=50, range=(0,0.5), density=True, color=colours[i], alpha=0.5, edgecolor=colours[i], label=place_labels[kind])
        axr[i].set_xlim(0,0.5)
        axr[i].legend()

    axa[0].set_title("Total share of community transmission")
    axr[0].set_title("Relative transmission risk")

    fig.tight_layout()
    fig.savefig(f"{output}-final-activity.png")
    #avg, std = envelope(samples)

def stamford_act(g, h5, prefix):
    samples = [h5[k] for k in h5 if k.startswith(prefix)]

    activities = {}
    risks = {}
    for obs, place in (("ACTH", "household"), ("ACTP", "primary"), ("ACTS", "secondary"), ("ACTG", "synagogue"), ("ACTM", "mikvah"), ("ACTE", "environment")):
        activities[place] = np.array([s[obs].iloc[-1] for s in samples])
        if place == "environment":
            count = sum(1 for n in g.nodes if g.nodes[n]["type"] == "person") - 1
        else:
            count = sum(nx.degree(g, n) for n in g.nodes if g.nodes[n]["type"] == place)
        risks[place] = activities[place]/count
    total = np.array([sum(activities[p] for p in activities)])
    activities = { p: d/total for p, d in activities.items() }

    support = np.linspace(0.0,0.49,50) + 0.005

    for p in list(activities):
        ah, _ = np.histogram(activities[p], bins=50, range=(0,0.5), density=True)
        rh, _ = np.histogram(risks[p], bins=50, range=(0,0.5), density=True)
        activities[p] = ah
        risks[p] = rh

    return support, activities, risks

@cli.command(name="write_stamford_act")
@click.option("--output", "-o", default="rule-activities.tsv", help="Output filename")
@click.option("--summary", "-s", default=False, is_flag=True, help="Write summary statistics")
@click.pass_context
def write_stamford_act(ctx, output, summary):
    """
    Stamford-hill specific simulation plots
    """
    if "graph" not in ctx.obj:
        click.secho(f"No population graph specified.", fg="red")
        sys.exit(-1)
    args = [v for k,v in ctx.obj.fixed.items() if k in inspect.getfullargspec(ctx.obj.graph).args]
    g = ctx.obj.graph(*args)

    if "store" not in ctx.obj:
        click.secho(f"No data store specified. Cannot tell where to store results.", fg="red")
        sys.exit(-1)

    support, activities, risks = stamford_act(g, ctx.obj.store, ctx.obj.prefix)

    if summary:
        cols = ["place",
                "act_mean", "act_std", "act_p25", "act_p500", "act_p975",
                "risk_mean", "risk_std", "risk_p25", "risk_p500", "risk_p975",
                ]
        rows = []
        for p in activities:
            aw   = activities[p]
            am    = (support*aw).sum()
            astd  = np.sqrt(((support-aw)**2 * support).sum())
            acs   = np.cumsum(aw)
            ap25  = 100*np.interp(2.5,  acs - 0.5*aw, support)
            ap500 = 100*np.interp(50,   acs - 0.5*aw, support)
            ap975 = 100*np.interp(97.5, acs - 0.5*aw, support)

            rw = risks[p]
            rm    = (support*rw).sum()
            rstd  = np.sqrt(((support-rw)**2 * support).sum())
            rcs   = np.cumsum(rw)
            rp25  = 100*np.interp(2.5,  rcs - 0.5*rw, support)
            rp500 = 100*np.interp(50,   rcs - 0.5*rw, support)
            rp975 = 100*np.interp(97.5, rcs - 0.5*rw, support)

            row = [p, am, astd, ap25, ap500, ap975, rm, rstd, rp25, rp500, rp975]
            rows.append(row)
        data = np.vstack(rows)

    else:
        cols = ["x"]
        rows = [support]
        for p in activities:
            cols.append(f"{p}_act")
            rows.append(activities[p])
            cols.append(f"{p}_risk")
            rows.append(risks[p])
        data = np.vstack(rows).T


    df = pd.DataFrame(data, columns=cols)
    df.to_csv(output, sep="\t", index=False)

@cli.command(name="plot_stamford_act")
@click.option("--output", "-o", default="rule-activities.png", help="Output filename")
@click.pass_context
def plot_stamford_act(ctx, output):
    """
    Stamford-hill specific simulation plots
    """
    if "graph" not in ctx.obj:
        click.secho(f"No population graph specified.", fg="red")
        sys.exit(-1)
    args = [v for k,v in ctx.obj.fixed.items() if k in inspect.getfullargspec(ctx.obj.graph).args]
    g = ctx.obj.graph(*args)

    if "store" not in ctx.obj:
        click.secho(f"No data store specified. Cannot tell where to store results.", fg="red")
        sys.exit(-1)

    support, activities, risks = stamford_act(g, ctx.obj.store, ctx.obj.prefix)

    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8))
    for i, p in enumerate(activities):
        ax1.bar(support, activities[p], width=0.01, color=colours[i], alpha=0.5, edgecolor=colours[i], label=place_labels[p])
        ax2.bar(support, risks[p], width=0.01, color=colours[i], alpha=0.5, edgecolor=colours[i])

    ax1.set_ylabel("Probability density")
    ax1.set_title("Total share of community transmission", loc="right", y=1.05)
    ax2.set_ylabel("Probability density")
    ax2.set_title("Relative transmission risk", loc="right", y=1.05)
    ax1.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.05))
    fig.tight_layout()
    fig.savefig(output)

def stamford_wass(snapshots):
    """
    Stamford-hill specific simulation plots -- Wasserstein barycentre
    """
    empirical = {}
    eh = emsar.attacks["household"]
    for size, (hist, _) in eh.items():
        if size > 10: continue
        hist = np.pad(hist, (0, 10-size))
        hist = hist/hist.sum()
        empirical[size] = hist

    attacks = {}
    for s in snapshots:
        g = nx.read_graphml(s)
        ah = attack_histograms(g, "household", eh)
        for size, (hist, _) in ah.items():
            if size > 10: continue
            hist = np.pad(hist, (0, 10-size))
            hist = hist/hist.sum()
            attacks.setdefault(size, []).append(hist)

    wass = {}
    for size in sorted(attacks):
        print(f"doing size {size}")

        A = np.vstack(attacks[size]).T
        if size == 1:
            ## fix the problem where the computation of the barycentre fails
            ## for households of size one. that's fine, we can do that case
            ## by hand
            infected = np.mean(A[1])
            bary_wass = np.zeros(11)
            bary_wass[0] = 1.0 - infected
            bary_wass[1] = infected
        else:
            ## adapted from https://pot.readthedocs.io/en/autonb/auto_examples/plot_barycenter_1D.html
            A = np.vstack(attacks[size]).T
            n, n_dists = A.shape

            # loss matrix + normalization
            M = ot.utils.dist0(n)
            M /= M.max()

            # equal weights
            weights = np.ones(n_dists)/n_dists

            # wasserstein
            reg = 1e-3
            bary_wass = ot.bregman.barycenter(A, M, reg, weights)
        wass[size] = bary_wass

    return empirical, wass

@cli.command(name="plot_stamford_wass")
@click.option("--output", "-o", default="wasserstein.png", help="Output filename")
@click.argument("snapshots", nargs=-1)
@click.pass_context
def plot_stamford_wass(ctx, output, snapshots):
    empirical, wass = stamford_wass(snapshots)
    edges = np.linspace(0,10,11)

    fig, axes = plt.subplots(2,5, figsize=(12,6))
    for size in sorted(wass):
        if size > 10: continue
        ax = axes[ int((size-1) / 5) ][ (size-1) % 5]
        bary_wass = wass[size]
        ax.bar(edges+0.25, bary_wass, width=0.4, color=colours[1], alpha=0.5, edgecolor=colours[1], label="Simulated")

    for size in sorted(empirical):
        if size > 10: continue
        ax = axes[ int((size-1) / 5) ][ (size-1) % 5]
        ax.set_title(f"size = {size}")
        ax.set_xlim(-0.5,10.5)
        ax.bar(edges-0.25, empirical[size], width=0.4, color=colours[0], alpha=0.5, edgecolor=colours[0], label="Empirical")
        if size == 1: ax.legend()

    for i in range(2):
        ax = axes[i][0]
        ax.set_ylabel("Fraction of households")
    for i in range(5):
        ax = axes[1][i]
        ax.set_xlabel("Number of infections")

    fig.tight_layout()
    fig.savefig(output)

@cli.command(name="write_stamford_wass")
@click.option("--output", "-o", default="wasserstein.tsv", help="Output filename")
@click.argument("snapshots", nargs=-1)
@click.pass_context
def write_stamford_wass(ctx, output, snapshots):
    empirical, wass = stamford_wass(snapshots)

    cols = ["count"]
    rows = [np.array(list(range(0,11)))]
    for size in range(1, 11):
        cols.append(f"e{size}")
        rows.append(empirical[size])
        cols.append(f"s{size}")
        rows.append(wass[size])

    data = np.vstack(rows).T
    df = pd.DataFrame(data, columns=cols)
    df.to_csv(output, sep="\t", index=False)

@cli.command(name="plot_stamford_cens")
@click.option("--output", "-o", default="censoring", help="Output filename")
@click.pass_context
def plot_stamford_cens(ctx, output):
    """
    Stamford-hill specific simulation plots -- Censoring
    """
    empirical = {}
    eh = emsar.attacks["household"]
    for size, (_, hist) in eh.items():
        if size > 10: continue
        hist = np.pad(hist, (0, 10-size))
        hist = hist/hist.sum()
        empirical[size] = hist

    fig, axes = plt.subplots(2,5, figsize=(12,6))
    for size in sorted(empirical):
        ax = axes[ int((size-1) / 5) ][ (size-1) % 5]
        ax.set_title(f"size = {size}")
        ax.set_xlim(-0.5,10.5)

        edges = np.linspace(0,10,11)
        ax.bar(edges-0.25, empirical[size], width=0.4, color=colours[0], alpha=0.5, edgecolor=colours[0])

    for i in range(2):
        ax = axes[i][0]
        ax.set_ylabel("Fraction of households")
    for i in range(5):
        ax = axes[1][i]
        ax.set_xlabel("Censored members")

    fig.tight_layout()
    fig.savefig(output)

@cli.command(name="plot_stamford_demo")
@click.option("--output", "-o", "output", default="stamford-demographics.png",
              help="Output filename")
@click.pass_context
def plot_stamford_demo(ctx, output):
    """
    Generate plots about Stamford Hill
    """
    if "graph" not in ctx.obj:
        click.secho(f"No population graph specified.", fg="red")
        sys.exit(-1)

    args = [v for k,v in ctx.obj.fixed.items() if k in inspect.getfullargspec(ctx.obj.graph).args]
    g = ctx.obj.graph(*args)

    fig, (axs, axa) = plt.subplots(2, len(place_labels), figsize=(10,5))

    for k, kind in enumerate(place_labels):
        if kind == "environment":
            males = [sum([g.nodes[p]["sex"] == "male" for p in g if g.nodes[p]["type"] == "person"])]
            females = [sum([g.nodes[p]["sex"] == "female" for p in g if g.nodes[p]["type"] == "person"])]
        else:
            males = [sum([g.nodes[p]["sex"] == "male" for p in nx.neighbors(g, q)]) for q in g if g.nodes[q]["type"] == kind]
            females = [sum([g.nodes[p]["sex"] == "female" for p in nx.neighbors(g, q)]) for q in g if g.nodes[q]["type"] == kind]

        ## filter out any places of zero size
        nz_males = np.array([m for i,m in enumerate(males) if m + females[i] > 0])
        nz_females = np.array([f for i,f in enumerate(females) if f + males[i] > 0])
        males, females = nz_males, nz_females
        #print(kind, males + females)

        edges = np.linspace(0,1,11)
        hist, _ = np.histogram(males/(males + females), bins=11, density=True, range=(0, 1))
        axs[k].set_title(place_labels[kind], fontdict={"fontsize": 8})
        axs[k].bar(edges, hist, width=0.1, color=colours[k], alpha=0.5, edgecolor=colours[k])
        axs[k].set_yticks([])
        if k == 0:
            axs[k].set_ylabel("Fraction male")

    for k, kind in enumerate(place_labels):
        if kind == "environment":
            ages = [g.nodes[p]["age"] for p in g if g.nodes[p]["type"] == "person"]
        else:
            ## adapted from https://pot.readthedocs.io/en/autonb/auto_examples/plot_barycenter_1D.html
            ages = []
            for q in g.nodes:
                if g.nodes[q]["type"] != kind:
                    continue
                ns = [g.nodes[p]["age"] for p in nx.neighbors(g, q)]
                if len(ns) == 0:
                    continue
                ages.append(np.mean(ns))

        hist, _ = np.histogram(ages, bins=17, density=True, range=(0,80))
        edges = np.linspace(0,80,17)
        axa[k].bar(edges, hist, width=5, color=colours[k], alpha=0.5, edgecolor=colours[k])
        axa[k].set_yticks([])
        axa[k].set_xticks([0,20,40,60,80])
        if k == 0:
            axa[k].set_ylabel("Mean age distribution")

    fig.tight_layout()
    fig.savefig(output)

## get histogram of introductions, returns a dictionary with
## keys being place type and values being a list of histograms
## of introductions
def stamford_intro(ctx, snapshots):
    ## place_labels has "environment" in place of "community"
    sources = list(place_labels)[:-1] + ["community", "init", "none"]
    intros = {}

    for s in snapshots:
        g = nx.read_graphml(s)
        for h in graph.households(g):
            members = list(nx.neighbors(g, h))
            if len(members) > 10: continue
            counts = { s: 0 for s in sources }
            for m in members:
                source = g.nodes[m].get("i", "none")
                if source == "houshold": source = "household" ## TODO remove typeo kludge
                counts[source] = counts[source] + 1

            assert sum(counts.values()) == len(members)
            if counts.get("init", 0) + counts.get("none", 0) == len(members): ## don't count households with no infections
                continue
            for s, c in counts.items():
                intros.setdefault(s, {}).setdefault(len(members), []).append(c)

    return intros

@cli.command(name="plot_stamford_intro")
@click.argument("snapshots", nargs=-1)
@click.option("--output", "-o", "output", default="stamford-intros.png",
              help="Output filename")
@click.pass_context
def plot_stamford_intro(ctx, snapshots, output):
    sources = list(place_labels)[:-1] + ["community", "init", "none"]
    intros = stamford_intro(ctx, snapshots)

    rows, cols = 3, 2
    fig, axes = plt.subplots(rows, cols, figsize=(10,7))
    bins = 10
    edges = np.linspace(1,10,bins)
    for src in range(rows*cols):
        row = src // cols
        col = src % cols
        ax = axes[row][col]

        ## list of arrays of counts by household size
        source_counts = intros[sources[src]]
        avg_counts = np.array([np.mean(source_counts[i]) for i in sorted(source_counts)])
        p5_counts = np.array([np.percentile(source_counts[i], 5) for i in sorted(source_counts)])
        p95_counts = np.array([np.percentile(source_counts[i], 95) for i in sorted(source_counts)])
        confidence = np.vstack([avg_counts-p5_counts,p95_counts-avg_counts])

        label = place_labels[sources[src]] if sources[src] != "community" else sources[src]
        ax.bar(edges, avg_counts, yerr=confidence, capsize=5, color=colours[src], alpha=0.5, edgecolor=colours[src])
        ax.set_title(label)

    for i in range(cols):
        axes[rows-1][i].set_xlabel("Household size")
    for i in range(rows):
        axes[i][0].set_ylabel("Infections")

    fig.tight_layout()
    fig.savefig(output)

@cli.command(name="write_stamford_intro")
@click.argument("snapshots", nargs=-1)
@click.option("--output", "-o", "output", default="stamford-intros.png",
              help="Output filename")
@click.pass_context
def write_stamford_intro(ctx, snapshots, output):
    """
    Write out introduction data.
    """
    sources = list(place_labels)[:-1] + ["community", "init", "none"]
    intros = stamford_intro(ctx, snapshots)
    columns = [ "count" ]
    rows = [ list(range(0,11)) ]
    for src in range(len(intros)):
        label = sources[src] if sources[src] != "community" else sources[src]
        source_counts = intros[sources[src]]
        for i in sorted(source_counts):
            hist, _ = np.histogram(source_counts[i], bins=11, range=(0,10), density=True)
            columns.append(f"{label}_{i}")
            rows.append(hist)

    df = pd.DataFrame(np.vstack(rows).T, columns=columns)
    df.to_csv(output, sep="\t", index=False)

@cli.command(name="write_stamford_dist")
@click.argument("snapshots", nargs=-1)
@click.pass_context
def write_stamford_dist(ctx, snapshots):
    """
    Stamford-hill specific simulation plots -- distance measure
    """
    distances = []
    for s in snapshots:
        g = nx.read_graphml(s)
        dist = emsar(g, "household")
        distances.append(dist)

    print(np.mean(distances), np.std(distances))
