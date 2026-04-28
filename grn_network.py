"""
=============================================================================
Lens GRN Explorer — NetworkX core
=============================================================================
"""

import os, sys
import pandas as pd
import networkx as nx
from collections import Counter

WONG = {
    "black":      "#000000",
    "orange":     "#E69F00",
    "sky_blue":   "#56B4E9",
    "green":      "#009E73",
    "yellow":     "#F0E442",
    "blue":       "#0072B2",
    "vermillion": "#D55E00",
    "pink":       "#CC79A7",
}

THEMES = {
    "light": {
        "bgcolor":     "#f8fafc",
        "panel_bg":    "#ffffff",
        "panel_bdr":   "#e2e8f0",
        "sidebar_bg":  "#f1f5f9",
        "sidebar_bdr": "#e2e8f0",
        "text":        "#0f172a",
        "muted":       "#64748b",
        "row_bg":      "#f8fafc",
        "topbar_bg":   "#ffffff",
        "topbar_bdr":  "#e2e8f0",
    },
    "dark": {
        "bgcolor":     "#0a0e1a",
        "panel_bg":    "#0f1525",
        "panel_bdr":   "#1e2d4a",
        "sidebar_bg":  "#0f1525",
        "sidebar_bdr": "#1e2d4a",
        "text":        "#e2e8f0",
        "muted":       "#64748b",
        "row_bg":      "#1a2540",
        "topbar_bg":   "#0f1525",
        "topbar_bdr":  "#1e2d4a",
    },
}

CONFIG = {
    "input_file":  "data/Lens_GRN_June_2016_original FOR HACKATHON - Salil Lachke.xlsx",
    "sheet_name":  "Lens_GRN_pert",
    "stage_from":   None,
    "stage_to":     None,
    "stage_single": None,
    "filter_regulator":    None,
    "filter_target":       None,
    "filter_tissue_reg":   None,
    "filter_tissue_tgt":   None,
    "relationships_include": ["activating", "inhibiting", "no_effect"],
    "max_edges":             300,
    "show_feedback_loops":   True,
}

def load_data(config):
    path = config["input_file"]
    if not os.path.exists(path):
        print("[ERROR] GRN file not found: " + path)
        sys.exit(1)
    print("[GRN] Loading: " + path)
    df = pd.read_excel(path, sheet_name=config["sheet_name"])
    df.columns = [str(c).strip() for c in df.columns]
    col_map = {
        df.columns[0]:  "sno",
        df.columns[1]:  "regulator",
        df.columns[2]:  "target",
        df.columns[4]:  "perturbation",
        df.columns[5]:  "effect",
        df.columns[6]:  "stage",
        df.columns[7]:  "context",
        df.columns[12]: "tissue_reg",
        df.columns[13]: "tissue_tgt",
        df.columns[20]: "reference",
        df.columns[21]: "pmid",
    }
    df = df.rename(columns=col_map)
    df = df[["sno","regulator","target","perturbation","effect","stage",
             "context","tissue_reg","tissue_tgt","reference","pmid"]]
    df = df.dropna(subset=["regulator","target"])
    for c in ["regulator","target","perturbation","effect","stage","context",
              "tissue_reg","tissue_tgt"]:
        df[c] = df[c].astype(str).str.strip()
    df["effect"]       = df["effect"].replace({"nan":"o","none":"o","None":"o","0":"o"})
    df["perturbation"] = df["perturbation"].replace({"nan":"-","none":"-","None":"-"})
    df["tissue_reg"]   = df["tissue_reg"].replace({"nan":"Unknown","fiber cells":"Fiber cells"})
    df["tissue_tgt"]   = df["tissue_tgt"].replace({"nan":"Unknown","fiber cells":"Fiber cells"})
    def clean_pmid(v):
        try:
            if pd.notna(v): return str(int(float(v)))
        except: pass
        return ""
    df["pmid"] = df["pmid"].apply(clean_pmid)
    df["true_relationship"] = df.apply(_compute_relationship, axis=1)
    print("[GRN] Loaded " + str(len(df)) + " edges")
    return df

def _compute_relationship(row):
    pert   = str(row["perturbation"]).strip()
    effect = str(row["effect"]).strip()
    if effect == "o": return "no_effect"
    return "activating" if pert == effect else "inhibiting"

def _find_file(path):
    if os.path.exists(path): return path
    for p in [
        path.replace(" ","_").replace("(","").replace(")",""),
        path.replace("_"," "),
    ]:
        if os.path.exists(p): return p
    return None

def load_external_data(data_sources: dict) -> dict:
    result = {}
    for key, src in data_sources.items():
        real_path = _find_file(src.get("path",""))
        if not real_path:
            print("[WARN] Not found: " + src.get("path",""))
            result[key] = {}
            continue
        print("[DATA] Loading " + src.get("label",key) + ": " + real_path)
        try:
            df = pd.read_excel(real_path)
            sym_col = src.get("symbol_col","Symbol")
            lookup  = {}
            for _, row in df.iterrows():
                sym = str(row.get(sym_col,"")).strip()
                if not sym or sym == "nan": continue
                gene_data = {}
                for section in src.get("sections",[]):
                    for display_name, excel_col in section.get("columns",{}).items():
                        v = row.get(excel_col)
                        try:
                            gene_data[display_name] = round(float(v),1) if pd.notna(v) else None
                        except:
                            gene_data[display_name] = str(v) if pd.notna(v) else None
                for meta_key, excel_col in src.get("meta_cols",{}).items():
                    v = row.get(excel_col)
                    gene_data["_meta_"+meta_key] = str(v) if (v is not None and pd.notna(v)) else ""
                lookup[sym] = gene_data
            result[key] = lookup
            print("[DATA]   -> " + str(len(lookup)) + " genes")
        except Exception as e:
            print("[WARN] Failed loading " + key + ": " + str(e))
            result[key] = {}
    return result

def stage_numeric(stage):
    s = str(stage).strip()
    if s == "Adult": return 100000.0
    try:
        if s.startswith("E"): return float(s[1:])
        if s.startswith("P"): return 1000.0 + float(s[1:])
    except ValueError: pass
    return 99999.0

def filter_data(df, config):
    original = len(df)
    df = df[df["true_relationship"].isin(config["relationships_include"])]
    if config.get("stage_single"):
        df = df[df["stage"] == config["stage_single"]]
    else:
        if config.get("stage_from"):
            df = df[df["stage"].apply(stage_numeric) >= stage_numeric(config["stage_from"])]
        if config.get("stage_to"):
            df = df[df["stage"].apply(stage_numeric) <= stage_numeric(config["stage_to"])]
    if config.get("filter_regulator"):
        df = df[df["regulator"] == config["filter_regulator"]]
    if config.get("filter_target"):
        df = df[df["target"] == config["filter_target"]]
    if config.get("filter_tissue_reg"):
        df = df[df["tissue_reg"] == config["filter_tissue_reg"]]
    if config.get("filter_tissue_tgt"):
        df = df[df["tissue_tgt"] == config["filter_tissue_tgt"]]
    if config.get("max_edges") and len(df) > config["max_edges"]:
        df = df.head(config["max_edges"])
    print("[FILTER] " + str(original) + " -> " + str(len(df)) + " edges")
    return df.reset_index(drop=True)

def build_graph(df):
    G = nx.DiGraph()
    regulators = set(df["regulator"].unique())
    targets    = set(df["target"].unique())
    for node in regulators | targets:
        G.add_node(node, is_reg=(node in regulators), is_tgt=(node in targets))
    for _, row in df.iterrows():
        reg, tgt  = row["regulator"], row["target"]
        pert, eff = row["perturbation"], row["effect"]
        stg       = row["stage"]
        rel       = row["true_relationship"]
        pmid      = str(row.get("pmid","")).strip()
        t_reg     = row.get("tissue_reg","")
        t_tgt     = row.get("tissue_tgt","")
        if G.has_edge(reg, tgt):
            G[reg][tgt]["perturbations"].append(pert)
            G[reg][tgt]["effects"].append(eff)
            G[reg][tgt]["relationships"].append(rel)
            G[reg][tgt]["stages"].append(stg)
            G[reg][tgt]["count"] += 1
            if pmid and pmid not in G[reg][tgt]["pmids"]:
                G[reg][tgt]["pmids"].append(pmid)
        else:
            G.add_edge(reg, tgt,
                perturbations=[pert], effects=[eff], relationships=[rel],
                stages=[stg], count=1, pmids=[pmid] if pmid else [],
                tissue_reg=t_reg, tissue_tgt=t_tgt)
    return G

def analyze_graph(G, config):
    results = {}
    in_deg  = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    for node in G.nodes():
        G.nodes[node]["in_degree"]    = in_deg[node]
        G.nodes[node]["out_degree"]   = out_deg[node]
        G.nodes[node]["total_degree"] = in_deg[node] + out_deg[node]
    results["hub_genes"]      = sorted(out_deg.items(), key=lambda x: x[1], reverse=True)[:10]
    results["feedback_nodes"] = set()
    results["feedback_edges"] = set()
    results["feedback_loops"] = []
    results["self_loops"]     = []
    if config.get("show_feedback_loops"):
        try:
            for i, cycle in enumerate(nx.simple_cycles(G)):
                if i > 500: break
                results["feedback_loops"].append(cycle)
                for node in cycle: results["feedback_nodes"].add(node)
                for j in range(len(cycle)):
                    results["feedback_edges"].add((cycle[j], cycle[(j+1)%len(cycle)]))
        except: pass
        results["self_loops"] = list(nx.selfloop_edges(G))
    results["components"] = list(nx.weakly_connected_components(G))
    return results

def build_cytoscape_elements(G, analysis, config, ext_data):
    elements       = []
    feedback_nodes = analysis.get("feedback_nodes", set())
    self_loop_nodes= set(e[0] for e in analysis.get("self_loops", []))
    feedback_edges = analysis.get("feedback_edges", set())

    for node in G.nodes():
        d      = G.nodes[node]
        is_reg = d.get("is_reg", False)
        is_tgt = d.get("is_tgt", False)
        deg    = d.get("total_degree", 1)

        if node in self_loop_nodes:    role_class = "selfloop"
        elif is_reg and is_tgt:        role_class = "both"
        elif is_reg:                   role_class = "regulator"
        else:                          role_class = "target"

        classes = role_class
        if node in feedback_nodes: classes += " feedback"

        elements.append({
            "data": {
                "id":        node,
                "label":     node,
                "role":      role_class,
                "deg":       deg,
                "out_deg":   d.get("out_degree", 0),
                "in_deg":    d.get("in_degree", 0),
                "feedback":  node in feedback_nodes,
                "self_loop": node in self_loop_nodes,
            },
            "classes": classes,
        })

    for u, v, data in G.edges(data=True):
        rels  = data.get("relationships", ["no_effect"])
        count = data.get("count", 1)
        pmids = data.get("pmids", [])
        is_fb = (u, v) in feedback_edges
        dom   = Counter(rels).most_common(1)[0][0] if rels else "no_effect"
        classes = dom + (" feedback-edge" if is_fb else "")

        elements.append({
            "data": {
                "id":       u + "__" + v,
                "source":   u,
                "target":   v,
                "rel":      dom,
                "perts":    sorted(set(data.get("perturbations",[]))),
                "effs":     sorted(set(data.get("effects",[]))),
                "stages":   sorted(set(str(s) for s in data.get("stages",[]) if pd.notna(s)))[:6],
                "count":    count,
                "pmids":    pmids[:8],
                "feedback": is_fb,
                "tissue_reg": data.get("tissue_reg",""),
                "tissue_tgt": data.get("tissue_tgt",""),
            },
            "classes": classes,
        })

    return elements