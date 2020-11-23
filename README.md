# Data processing and modelling for Stamford Hill

## Introduction

This repository contains scripts and programs for processing data collected as
part of the project to understand the spread of COVID-19 in a Haredi Jewish
community in North London. There are two primary sources of data: information
about households and locations that they frequent collected by survey and
serology data. It is not possible to distribute this data for confidentiality
reasons.

The programs are therefore separated into four groups:

  * *Group 1* contains programs that produce, from the source data, anonymous
    statistics and other representations that can be distributed. These data are
    used as input to the *Group 2* programs. These are:
    * `stamford_graph` produces documents in GraphML format from the survey
       data. The attributes representing demographic and other information on
       the nodes and edges are minimised and pseudonymous identifiers for
       individuals, households and places are used.
    * `stamford_plot` produces basic demographic plots and summary information
       about the sampled population.
  * *Group 2* contains programs that must not use any identifiable information
    as input. These programs take the statistical and other representations of
    the input data about the sampled population and produce synthetic data
    representative of the entire population.
  * *Group 3* contains programs for simulations of transmission in the
    community.
  * *Group 4* contains programs for processing the output of simulations into
    summaries, plots and so forth for including in publications and other
    materials.

## Installation

This repository is not intended to be specific to any particular programming
language. We use a mix of Python, R and other languages as convenient. The
programs are intended to be run as command-line executables. This, however,
requires some language-specific setup.

### Python

The easiest is to use `anaconda` or similar software for virtual environments.
We require Python 3 throughout. A typical installation for development purposes
is done as follows:

    conda create -n stamford python=3
    conda activate stamford
    python setup.py develop

now the programs written in python will be in your shell's search path and you
will simply be able to run them.

## Programs

### stamford_graph

    usage: stamford_graph [-h] [--motifs] survey

    positional arguments:
      survey        Data file (.zip) containing survey data downloaded from ODK

    optional arguments:
      -h, --help    show this help message and exit
      --motifs, -m  Generate a graph of household motifs

Run this program like this:

    stamford_graph ./data/survey.zip > stamford.graphml
    stamford_graph -m ./data/survey.zip > stamford-motifs.graphml

The first form creates a graph of the entire community. The second produces
household motifs for each household pattern that appears more than once in the
data.

### stamford_plot

    usage: stamford_graph [-h] survey

    positional arguments:
      survey      Data file (.zip) containing survey data downloaded from ODK

    optional arguments:
      -h, --help  show this help message and exit

This program writes histograms for the numbers of households, shuls, yeshivas
and mikvas that associated with a given number of people.
