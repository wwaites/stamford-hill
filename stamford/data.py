import zipfile
import math
import datetime as dt
import networkx as nx
import numpy    as np
import pandas   as pd
import logging

log = logging.getLogger("__name__")

## constants for validation
SURVEY_START = dt.datetime(2020, 10, 19, 0, 0, 0, 0, dt.timezone.utc)
MIN_HHID = 0
MAX_HHID = 20000

def read_data(datafile):
    """
    Read the survey data. The name of the .zip file containing csv files
    is given, and a dictionary is returned with keys being the names of
    the csv file and values being Pandas DataFrame objects.
    """
    data = {}
    with zipfile.ZipFile(datafile) as z:
        for csvname in z.namelist():
            with z.open(csvname) as csvfile:
                frame = pd.read_csv(csvfile)
                if "SubmissionDate" in frame:
                    frame["SubmissionDate"] = pd.to_datetime(frame["SubmissionDate"])
                data[csvname] = frame
    return data

def _depanda(row):
    """
    Undo some of Pandas datatype fanciness which is too clever by half
    """
    cleaned = {}
    for k,v in dict(row).items():
        if isinstance(v, pd.Timestamp):
            v = str(v)
        cleaned[k] = v
    return cleaned

def generate_graph(data, minimal=True):
    """
    Construct an annotated graph from the data (see function read_data)
    """
    g = nx.Graph()

    ## keep track of household key -> household id
    hhkeys = {}
    hhids  = {}

    ## XXX don't like hard-coding the file name here, must thing of better way
    for hh in data['stamford_hill_survey.csv'].iloc:
        if hh["SubmissionDate"] < SURVEY_START:
            continue

        hhkey = hh["KEY"]
        hhid  = int(hh["household.id"])

        assert hhkey not in hh, "duplicate household key {}".format(hhkey)

        ## validate the household id
        if hhid > MAX_HHID:
            log.warn("Household record {} has household.id = {} which is outwith the ({}, {}) range".format(hhkey, hhid, MIN_HHID, MAX_HHID))
            continue

        if np.isnan(hhid):
            log.warn("Household record {} has invalid household.id".format(hhkey))
            continue

        hhid = "H%d" % hhid

        if hhid in hhids:
            log.warn("Duplicate household id {}".format(hhid))
            continue

        hhkeys[hhkey] = hhid
        hhids[hhid] = True

        ## create a household node with attributes given by the row, verbatim
        attrs = { "width": 5, "height": 5 } if minimal else _dependa(hh)
        g.add_node(hhid, kind="household", **attrs)

    ## we don't really care about 'stamford_hill_survey-hh_member_names_repeat.csv'
    ## because the names are repeated in member_info.

    ## keep track of Schools, Yeshivas, Synagogues and Mikvahs
    places = {}
    def _add_place(p, pid, key):
        place = p[key]
        if isinstance(place, str):
            for place in place.split(" "):
                if place in ('S0',): ## S0 means "no regular shul"
                    continue
                if place not in places:
                    places[place] = True
                    if place.startswith("S"):
                        kind = "shul"
                        size = 60
                    elif place.startswith("Y"):
                        kind = "yeshiva"
                        size = 40
                    elif place.startswith("M"):
                        kind = "mikvah"
                        size = 30
                    else:
                        raise ValueError("Person {} has unknown kind of place {}".format(p["KEY"], place))
                    g.add_node(place, kind=kind, weight=10, width=size, height=size)
                g.add_edge(pid, place)


    ## PARENT_KEY is a pointer back to household
    npeople = 0
    for p in data['stamford_hill_survey-hh_member_info_repeat.csv'].iloc:
        npeople += 1
        pid = p["person_ID"]
        assert isinstance(pid, str), "Person {} has invalid person ID: {}".format(p["KEY"], p["person_ID"])

        if minimal: pid = "P%d" % npeople

        if p["PARENT_KEY"] not in hhkeys:
            log.warn("Person {} belongs to nonexistent household {}".format(p["person_ID"], p["PARENT_KEY"]))
            continue
        hid = hhkeys[p["PARENT_KEY"]]

        attrs = { "age": p["hh_member_age"], "sex": p["hh_member_sex"] } if minimal else _depanda(p)
        attrs["width"] = 5*int(math.log(max(attrs["age"],2), 2))
        attrs["height"] = 5*int(math.log(max(attrs["age"],2), 2))
        g.add_node(pid, kind="person", **attrs)
        g.add_edge(pid, hid, weight=1)

        _add_place(p, pid, "school.define")
        _add_place(p, pid, "yeshiva.define")
        _add_place(p, pid, "shul-shul.define")
        _add_place(p, pid, "mikveh-mikveh.define")

    return g
