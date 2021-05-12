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
MIN_HHID     = 0
MAX_HHID     = 20000
N_HOUSEHOLDS = 374
N_SCHOOLS    = 36
N_YESHIVEH   = 14
N_SYNAGOGUES = 69
N_MIKVEH     = 35
N_PEOPLE     = 1942
N_ENRICHED   = 28
N_SEROLOGY   = 1377
N_PENRICHED  = 183
N_SRANDOM    = 1240
N_PRIMARY    = 30
N_SECONDARY  = 20

place_columns = {
    "school.define": "school",
    "yeshiva.define": "yeshiva",
    "shul-shul.define": "synagogue",
    "mikveh-mikveh.define": "mikvah",
}

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

def add_households(g, data, minimal=True):
    ## keep track of household key -> node id
    hhkeys = {}
    hhids = {}

    nrows, _ = data.shape
    log.info(f"Processing {nrows} households")
    for hh in data.iloc:
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

        if hhid in hhids:
            log.warn("Duplicate household id {}".format(hhid))

        nid = len(g)
        hhids[hhid] = nid
        hhkeys[hhkey] = nid

        ## create a household node with attributes given by the row, verbatim
        attrs = {} if minimal else _depanda(hh)

        g.add_node(nid, type="household", uuid=hhkey, bipartite=1, **attrs)

    return hhkeys

def add_places(g, data):
    places = {}
    for person in data.iloc:
        for column, kind in place_columns.items():
            place = person[column]
            if isinstance(place, str):
                for place in place.split(" "):
                    if place in ('S0', 'S80', 'M80', 'Y54'):
                        ## S0 means "no regular shul"
                        ## S80, M80, Y54 mean "other"
                        continue
                    if place not in places:
                        pid = len(g)
                        places[place] = pid
                        g.add_node(pid, type=kind, label=place, bipartite=1)
    return places

def add_people(g, places, data, minimal=True):
    for p in data.iloc:
        label = p["person_ID"]
        assert isinstance(label, str), "Person {} has invalid person ID: {}".format(p["KEY"], p["person_ID"])

        if p["PARENT_KEY"] not in places:
            log.warn("Person {} belongs to nonexistent household {}".format(p["person_ID"], p["PARENT_KEY"]))
            continue
        hid = places[p["PARENT_KEY"]]

        if minimal:
            attrs = {
                "age": p["hh_member_age"],
                "sex": p["hh_member_sex"],
            }
        else:
            attrs = _depanda(p)

        if attrs["age"] < 13:
            attrs["lifecycle"] = "child"
        else:
            attrs["lifecycle"] = "adult"

        pid = len(g)
        g.add_node(pid, type="person", label=label, bipartite=0, **attrs)
        g.add_edge(pid, hid)

        for column in place_columns:
            place = p[column]
            if isinstance(place, str):
                for place in place.split(" "):
                    plid = places.get(place)
                    if plid is not None:
                        g.add_edge(pid, plid)

def generate_graph(data, minimal=True):
    """
    Construct an annotated graph from the data (see function read_data)
    """
    g = nx.Graph()

    ## XXX don't like hard-coding the file name here, must thing of better way
    hhkeys = add_households(g, data['stamford_hill_survey.csv'], minimal)

    assert len(hhkeys) == N_HOUSEHOLDS

    ## we don't really care about 'stamford_hill_survey-hh_member_names_repeat.csv'
    ## because the names are repeated in member_info.

    places = add_places(g, data['stamford_hill_survey-hh_member_info_repeat.csv'])

    ## check that place labels are unique and the we do not have, for example,
    ## that some people are connected to a given place as both a school and
    ## a synagogue
    place_labels = set(g.nodes[p]["label"] for p in places.values())
    assert len(place_labels) == len(places)

    ## check that we have all the right numbers of places of different kinds
    n_households = len([n for n in g if g.nodes[n]["type"] == "household"])
    assert n_households == N_HOUSEHOLDS, n_households

    n_schools = len([n for n in g if g.nodes[n]["type"] == "school"])
    assert n_schools == N_SCHOOLS, n_schools

    n_yeshiveh = len([n for n in g if g.nodes[n]["type"] == "yeshiva"])
    assert n_yeshiveh == N_YESHIVEH, n_yeshiveh

    n_synagogues = len([n for n in g if g.nodes[n]["type"] == "synagogue"])
    assert n_synagogues == N_SYNAGOGUES, n_synagogues

    n_mikveh = len([n for n in g if g.nodes[n]["type"] == "mikvah"])
    assert n_mikveh == N_MIKVEH, n_mikveh

    places.update(hhkeys)

    ## PARENT_KEY is a pointer back to household
    add_people(g, places, data['stamford_hill_survey-hh_member_info_repeat.csv'])

    n_people = len([n for n in g if g.nodes[n]["type"] == "person"])
    assert n_people == N_PEOPLE, n_people

    ## no re-arrange schools and yeshiveh into primary and secondary age-bands
    for n in [n for n in g if g.nodes[n]["type"] in ("school", "yeshiva")]:
        age = np.mean([g.nodes[p]["age"] for p in nx.neighbors(g, n)])
        if age < 13: g.nodes[n]["type"] = "primary"
        else: g.nodes[n]["type"] = "secondary"

    n_primary = len([n for n in g if g.nodes[n]["type"] == "primary"])
    n_secondary = len([n for n in g if g.nodes[n]["type"] == "secondary"])
    assert n_primary == N_PRIMARY, n_primary
    assert n_secondary == N_SECONDARY, n_secondary

    return g

def augment_graph(g, datafile, minimal=True):
    """
    Augment the graph with the combined serology data
    """
    ## create an inverted map from label to node id
    lmap = dict((g.nodes[n]["label"], n) for n in g if g.nodes[n].get("label") is not None)
    ## create an inverted map from uuid to node id
    umap = dict((g.nodes[n]["uuid"], n) for n in g if g.nodes[n].get("uuid") is not None)

    df = pd.read_csv(datafile)
    fields = ["spike_pos", "spike_pos2", "swab_POS", "serology_POS", "RBD_pos", "NC_pos",
              "SARS_S", "CoV2_N", "CoV_2_NTD", "CoV_2_RBD", "CoV_2_S", "X229Es", "HKU1S", "HL63s",
              "OC43S"]

    for row in df.iloc:
        label = row["person_ID"]
        pid = lmap[label]

        hh = [n for n in nx.neighbors(g, pid) if g.nodes[n]["type"] == "household"]
        if len(hh) != 1:
            log.error(f"person {pid} ({uuid}) has wrong number of households {hh}")
        hid = hh[0]

        enriched = row["enriched"]
        if enriched == "RANDOM":
            g.nodes[hid]["enriched"] = False
        elif enriched == "ENRICHED":
            g.nodes[hid]["enriched"] = True
        else:
            log.warn(f"Weird enriched value for Person {label}: {enriched}")

        crowding = row["crowding"]
        if crowding == "Correct":
            g.nodes[hid]["crowding"] = False
        elif crowding == "Over":
            g.nodes[hid]["crowding"] = True
        else:
            log.warn(f"Weird crowding value for Person {label}: {crowding}")

        serum = row["GOTSERUM"]
        if np.isnan(serum):
            g.nodes[pid]["serology"] = False
        elif serum == 0.:
            g.nodes[pid]["serology"] = False
        else:
            g.nodes[pid]["serology"] = True
            spike = row["spike_pos"]
            if np.isnan(spike):
                log.warn(f"Weird spike_pos value for Person {label}: {spike}")
            elif spike == 0.:
                g.nodes[pid]["positive"] = False
            else:
                g.nodes[pid]["positive"] = True

        if not minimal:
            for field in fields:
                v = row[field]
                if isinstance(v, str):
                    try:
                        v = np.float64(v)
                    except ValueError:
                        log.warn(f"Weird value for Person {pid} field {field}: {v}")
                        continue
                if np.isnan(v):
                    continue
                if field.endswith("_POS"):
                    field = field.lower()
                elif field.startswith("CoV_2_"):
                    field = "CoV2_" + field[5:]
                g.nodes[pid][field] = v

    n_enriched = len([n for n in g if g.nodes[n].get("enriched")])
    assert n_enriched == N_ENRICHED, n_enriched

    p_enriched = 0
    for h in g:
        if g.nodes[h].get("enriched"):
            p_enriched += len([p for p in nx.neighbors(g, h)])
    assert p_enriched == N_PENRICHED, p_enriched

    n_serology = len([n for n in g if g.nodes[n].get("serology")])
    assert n_serology == N_SEROLOGY, n_serology

    n_srandom = 0
    for h in g:
        if g.nodes[h].get("enriched") is False:
            n_srandom += len([p for p in nx.neighbors(g, h) if g.nodes[p].get("serology")])
    assert n_srandom == N_SRANDOM, n_srandom

    return g
