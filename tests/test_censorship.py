from stamford.network import attack_histograms, process_serology, emsar
import networkx as nx
import click

G = nx.read_graphml("./data/stamford.graphml")

def test_emsar():
    g = process_serology(G.copy())
    emsar.data(g)
    for kind in ("household", "synagogue", "school", "yeshiva", "mikvah"):
        click.secho(f"secondary attack rate for places of type {kind}: {sar(g, kind)}", fg="green")
        assert emsar(g, kind) == 0.0

def test_attack():
    g = process_serology(G.copy())
    attacks = attack_histograms(g, "household")
    for size in sorted(attacks.keys()):
        print(attacks[size])
    #assert False

def test_censor():
    g = process_serology(G.copy())
    ref = attack_histograms(g, "household")
    att = attack_histograms(g, "household", ref)
    for size in sorted(att.keys()):
        rh, rc = ref[size]
        ah, ac = att[size]
        print(rc, ac)
    assert False
