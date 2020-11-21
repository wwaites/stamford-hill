import matplotlib.pyplot as plt
from stamford.data import read_data, generate_graph
from stamford import graph
import argparse


def command():
    parser = argparse.ArgumentParser("stamford_graph")
    parser.add_argument("survey", help="Data file (.zip) containing survey data downloaded from ODK")
    args = parser.parse_args()

    data = read_data(args.survey)
    g = generate_graph(data)

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

if __name__ == '__main__':
    command()
