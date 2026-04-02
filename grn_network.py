"""
=============================================================================
Lens GRN Explorer — NetworkX + Pyvis
=============================================================================
Lachke Lab 2016 — Gene Regulatory Network

KEY BIOLOGY NOTE:
    True relationship = Perturbation x Effect:
        Pert(-) x Effect(-) -> ACTIVATING
        Pert(-) x Effect(+) -> INHIBITING
        Pert(+) x Effect(+) -> ACTIVATING
        Pert(+) x Effect(-) -> INHIBITING
        Any    x Effect(o)  -> NO EFFECT

NEW FEATURES:
    - Edge tooltip shows PubMed IDs with clickable links
    - Node tooltip shows MegaTable expression data
    - Legend as collapsible button
    - Compact zoom controls (+/-)
=============================================================================
"""

import os
import sys
import json
import webbrowser
import pandas as pd
import networkx as nx
from pyvis.network import Network
from collections import Counter

# =============================================================================
# 1. CONFIG
# =============================================================================

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

CONFIG = {
    "input_file":    "data/Lens_GRN_June_2016_original FOR HACKATHON - Salil Lachke.xlsx",
    "megatable_file":"data/MegaTable_April_24_2024_for_Microarray_and_RNA_Seq_Sent_to_Murali__1_.xls",
    "output_file":   "grn_output.html",
    "sheet_name":    "Lens_GRN_pert",
    "stage_from":    None,
    "stage_to":      None,
    "stage_single":  None,
    "filter_regulator": None,
    "filter_target":    None,
    "relationships_include": ["activating", "inhibiting", "no_effect"],
    "max_edges":           300,
    "show_feedback_loops": True,
    "color_regulator": WONG["sky_blue"],
    "color_target":    WONG["orange"],
    "color_both":      WONG["pink"],
    "color_selfloop":  WONG["yellow"],
    "color_activating": WONG["green"],
    "color_inhibiting": WONG["vermillion"],
    "color_noeffect":   "#94a3b8",
    "node_size_min": 12,
    "node_size_max": 55,
    "layout": "barnes_hut",
    "theme":  "dark",
}

THEMES = {
    "dark": {
        "bgcolor":        "#0a0e1a",
        "font_color":     "#ffffff",
        "tooltip_bg":     "#0f1525",
        "tooltip_border": "#1e2d4a",
        "tooltip_text":   "#e2e8f0",
    },
    "light": {
        "bgcolor":        "#f8fafc",
        "font_color":     "#0f172a",
        "tooltip_bg":     "#ffffff",
        "tooltip_border": "#cbd5e1",
        "tooltip_text":   "#0f172a",
    },
}

# =============================================================================
# 2. DATA LOADING
# =============================================================================

def load_megatable(config):
    """Load MegaTable and return a symbol -> data dict."""
    path = config.get("megatable_file", "")
    if not path or not os.path.exists(path):
        print("[WARN] MegaTable not found — expression data will be unavailable")
        return {}

    print("      Loading MegaTable from: " + path)
    df = pd.read_excel(path)

    microarray_exp_cols = ['Beebe_E12_exp_Fiber', 'Beebe_E12_exp_Epi',
                           'Naka_P13_fiber_exp',  'Naka_P13_epi_exp']
    microarray_enr_cols = ['Beebe_E12_Fiber_enr', 'Beebe_E12_Epi_enr',
                           'Naka_P13_fiber_enr',  'Naka_P13_epi_enr']
    rnaseq_cols = ['enr_FC_E14_Cv', 'enr_LEC_E14_Cv', 'enr_FC_E16_Cv', 'enr_LEC_E16_Cv',
                   'enr_FC_E18_Cv', 'enr_LEC_E18_Cv', 'enr_FC_P0_Cv',  'enr_LEC_P0_Cv',
                   'enr_LEC_P0_Rob','enr_FC_P0_Rob',  'enr_FC_3Mo',    'enr_LEC_3Mo',
                   'enr_FC_6Mo',    'enr_LEC_6Mo',    'enr_FC_2Y',     'enr_LEC_2Y']

    lookup = {}
    for _, row in df.iterrows():
        sym = str(row.get('Symbol', '')).strip()
        if not sym or sym == 'nan':
            continue

        def safe_float(v):
            try:
                if pd.notna(v):
                    return round(float(v), 3)
            except Exception:
                pass
            return None

        lookup[sym] = {
            'entrez':      str(row.get('Entrez',          '')),
            'uniprot':     str(row.get('UNIPROT',         '')),
            'description': str(row.get('Gene_description','')),
            'microarray_exp': {c: safe_float(row.get(c)) for c in microarray_exp_cols},
            'microarray_enr': {c: safe_float(row.get(c)) for c in microarray_enr_cols},
            'rnaseq':         {c: safe_float(row.get(c)) for c in rnaseq_cols},
        }
    print("      MegaTable loaded: " + str(len(lookup)) + " genes")
    return lookup


def load_data(config):
    path = config["input_file"]
    if not os.path.exists(path):
        print("\n[ERROR] File not found: " + path)
        sys.exit(1)

    print("[1/5] Loading GRN data from: " + path)
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
        df.columns[20]: "reference",
        df.columns[21]: "pmid",
    }
    df = df.rename(columns=col_map)
    keep = ["sno","regulator","target","perturbation","effect","stage","context","reference","pmid"]
    df = df[keep]
    df = df.dropna(subset=["regulator", "target"])

    for col in ["regulator","target","perturbation","effect","stage","context"]:
        df[col] = df[col].astype(str).str.strip()

    df["effect"]       = df["effect"].replace({"nan":"o","none":"o","None":"o","0":"o"})
    df["perturbation"] = df["perturbation"].replace({"nan":"-","none":"-","None":"-"})

    # Clean PMID
    def clean_pmid(v):
        try:
            if pd.notna(v):
                return str(int(float(v)))
        except Exception:
            pass
        return ""
    df["pmid"] = df["pmid"].apply(clean_pmid)

    df["true_relationship"] = df.apply(_compute_relationship, axis=1)
    print("       Loaded " + str(len(df)) + " edges")
    for rel, cnt in df["true_relationship"].value_counts().items():
        print("       " + rel + ": " + str(cnt))
    return df


def _compute_relationship(row):
    pert   = str(row["perturbation"]).strip()
    effect = str(row["effect"]).strip()
    if effect == "o":
        return "no_effect"
    if pert == effect:
        return "activating"
    return "inhibiting"


# =============================================================================
# 3. FILTERING
# =============================================================================

def stage_numeric(stage):
    s = str(stage).strip()
    if s == "Adult":
        return 100000.0
    try:
        if s.startswith("E"):
            return float(s[1:])
        if s.startswith("P"):
            return 1000.0 + float(s[1:])
    except ValueError:
        pass
    return 99999.0


def filter_data(df, config):
    print("[2/5] Applying filters...")
    original = len(df)
    df = df[df["true_relationship"].isin(config["relationships_include"])]

    if config["stage_single"]:
        df = df[df["stage"] == config["stage_single"]]
    else:
        if config["stage_from"]:
            df = df[df["stage"].apply(stage_numeric) >= stage_numeric(config["stage_from"])]
        if config["stage_to"]:
            df = df[df["stage"].apply(stage_numeric) <= stage_numeric(config["stage_to"])]

    if config["filter_regulator"]:
        df = df[df["regulator"] == config["filter_regulator"]]
    if config["filter_target"]:
        df = df[df["target"] == config["filter_target"]]
    if config["max_edges"] and len(df) > config["max_edges"]:
        df = df.head(config["max_edges"])

    print("       " + str(original) + " -> " + str(len(df)) + " edges after filtering")
    return df.reset_index(drop=True)


# =============================================================================
# 4. GRAPH BUILDING
# =============================================================================

def build_graph(df):
    print("[3/5] Building NetworkX DiGraph...")
    G = nx.DiGraph()
    regulators = set(df["regulator"].unique())
    targets    = set(df["target"].unique())

    for node in regulators | targets:
        G.add_node(node, is_reg=(node in regulators), is_tgt=(node in targets))

    for _, row in df.iterrows():
        reg, tgt  = row["regulator"], row["target"]
        pert, eff = row["perturbation"], row["effect"]
        stg, ctx  = row["stage"], row["context"]
        rel       = row["true_relationship"]
        pmid      = str(row.get("pmid", "")).strip()

        if G.has_edge(reg, tgt):
            G[reg][tgt]["perturbations"].append(pert)
            G[reg][tgt]["effects"].append(eff)
            G[reg][tgt]["relationships"].append(rel)
            G[reg][tgt]["stages"].append(stg)
            G[reg][tgt]["contexts"].append(ctx)
            G[reg][tgt]["count"] += 1
            if pmid and pmid not in G[reg][tgt]["pmids"]:
                G[reg][tgt]["pmids"].append(pmid)
        else:
            G.add_edge(reg, tgt,
                       perturbations=[pert], effects=[eff],
                       relationships=[rel], stages=[stg],
                       contexts=[ctx], count=1,
                       pmids=[pmid] if pmid else [])

    print("       " + str(G.number_of_nodes()) + " nodes, " + str(G.number_of_edges()) + " edges")
    return G


# =============================================================================
# 5. ANALYSIS
# =============================================================================

def analyze_graph(G, config):
    print("[4/5] Running graph analysis...")
    results = {}
    in_deg  = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    deg_c   = nx.degree_centrality(G)

    for node in G.nodes():
        G.nodes[node]["in_degree"]      = in_deg[node]
        G.nodes[node]["out_degree"]     = out_deg[node]
        G.nodes[node]["total_degree"]   = in_deg[node] + out_deg[node]
        G.nodes[node]["deg_centrality"] = round(deg_c[node], 4)

    results["hub_genes"] = sorted(out_deg.items(), key=lambda x: x[1], reverse=True)[:10]

    results["feedback_nodes"] = set()
    results["feedback_edges"] = set()
    results["feedback_loops"] = []
    results["self_loops"]     = []

    if config["show_feedback_loops"]:
        try:
            for i, cycle in enumerate(nx.simple_cycles(G)):
                if i > 500:
                    break
                results["feedback_loops"].append(cycle)
                for node in cycle:
                    results["feedback_nodes"].add(node)
                for j in range(len(cycle)):
                    results["feedback_edges"].add((cycle[j], cycle[(j+1) % len(cycle)]))
        except Exception:
            pass
        results["self_loops"] = list(nx.selfloop_edges(G))

    results["components"] = list(nx.weakly_connected_components(G))
    print("       Feedback loops: " + str(len(results["feedback_loops"])) +
          "  |  Self-loops: " + str(len(results["self_loops"])))
    return results


# =============================================================================
# 6. VISUALIZATION
# =============================================================================

def _node_size(G, node, config):
    deg     = G.nodes[node].get("total_degree", 1)
    max_deg = max((d for _, d in G.degree()), default=1)
    ratio   = deg / max_deg
    return int(config["node_size_min"] + ratio * (config["node_size_max"] - config["node_size_min"]))


def _node_color(G, node, analysis, config):
    self_loop_nodes = set(e[0] for e in analysis.get("self_loops", []))
    if node in self_loop_nodes:
        return config["color_selfloop"]
    is_reg = G.nodes[node].get("is_reg", False)
    is_tgt = G.nodes[node].get("is_tgt", False)
    if is_reg and is_tgt:
        return config["color_both"]
    if is_reg:
        return config["color_regulator"]
    return config["color_target"]


def _edge_color(relationships, config):
    if not relationships:
        return config["color_noeffect"]
    dominant = Counter(relationships).most_common(1)[0][0]
    if dominant == "activating":
        return config["color_activating"]
    if dominant == "inhibiting":
        return config["color_inhibiting"]
    return config["color_noeffect"]


def _mega_table_rows(sym, mega_lookup, theme):
    """Build mini expression table HTML for a gene."""
    if not sym or sym not in mega_lookup:
        return ""
    data  = mega_lookup[sym]
    t_bg  = "#1a2540" if theme == "dark" else "#f1f5f9"
    t_hdr = "#334155" if theme == "dark" else "#e2e8f0"
    t_txt = "#e2e8f0" if theme == "dark" else "#0f172a"
    t_mut = "#64748b"

    def val_cell(v):
        if v is None:
            return "<td style='padding:2px 6px;color:" + t_mut + "'>—</td>"
        color = WONG["green"] if v > 0 else (WONG["vermillion"] if v < 0 else t_txt)
        return "<td style='padding:2px 6px;color:" + color + ";text-align:right'>" + str(v) + "</td>"

    html = (
        "<div style='margin-top:8px;font-size:11px'>"
        "<div style='font-weight:bold;color:" + WONG["sky_blue"] + ";margin-bottom:4px'>Expression Data (MegaTable)</div>"
    )

    # Description
    desc = data.get("description", "")
    if desc and desc != "nan":
        html += "<div style='color:" + t_mut + ";font-size:10px;margin-bottom:6px;font-style:italic'>" + desc[:80] + ("..." if len(desc) > 80 else "") + "</div>"

    # Microarray expression
    exp = data.get("microarray_exp", {})
    non_null_exp = {k: v for k, v in exp.items() if v is not None}
    if non_null_exp:
        html += "<div style='font-size:10px;color:" + t_mut + ";margin-bottom:2px'>Microarray Expression</div>"
        html += "<table style='font-size:10px;border-collapse:collapse;width:100%;background:" + t_bg + ";border-radius:4px'>"
        for k, v in non_null_exp.items():
            short = k.replace("_exp","").replace("Beebe_","B_").replace("Naka_","N_")
            html += "<tr><td style='padding:2px 6px;color:" + t_mut + "'>" + short + "</td>" + val_cell(v) + "</tr>"
        html += "</table>"

    # Microarray enrichment
    enr = data.get("microarray_enr", {})
    non_null_enr = {k: v for k, v in enr.items() if v is not None}
    if non_null_enr:
        html += "<div style='font-size:10px;color:" + t_mut + ";margin-top:4px;margin-bottom:2px'>Microarray Enrichment</div>"
        html += "<table style='font-size:10px;border-collapse:collapse;width:100%;background:" + t_bg + ";border-radius:4px'>"
        for k, v in non_null_enr.items():
            short = k.replace("_enr","").replace("Beebe_","B_").replace("Naka_","N_")
            html += "<tr><td style='padding:2px 6px;color:" + t_mut + "'>" + short + "</td>" + val_cell(v) + "</tr>"
        html += "</table>"

    # RNA-seq (show first 8 to keep tooltip compact)
    rna = data.get("rnaseq", {})
    non_null_rna = {k: v for k, v in rna.items() if v is not None}
    if non_null_rna:
        html += "<div style='font-size:10px;color:" + t_mut + ";margin-top:4px;margin-bottom:2px'>RNA-seq Enrichment</div>"
        html += "<table style='font-size:10px;border-collapse:collapse;width:100%;background:" + t_bg + ";border-radius:4px'>"
        for k, v in list(non_null_rna.items())[:8]:
            short = k.replace("enr_","").replace("_Cv","").replace("_Rob","_R")
            html += "<tr><td style='padding:2px 6px;color:" + t_mut + "'>" + short + "</td>" + val_cell(v) + "</tr>"
        if len(non_null_rna) > 8:
            html += "<tr><td colspan='2' style='padding:2px 6px;color:" + t_mut + ";font-style:italic'>+" + str(len(non_null_rna)-8) + " more...</td></tr>"
        html += "</table>"

    html += "</div>"
    return html


def _node_tooltip(node, G, analysis, theme, mega_lookup):
    t       = THEMES[theme]
    d       = G.nodes[node]
    is_reg  = d.get("is_reg", False)
    is_tgt  = d.get("is_tgt", False)
    in_deg  = d.get("in_degree", 0)
    out_deg = d.get("out_degree", 0)
    cent    = d.get("deg_centrality", 0)

    if is_reg and is_tgt:
        role = "Regulator &amp; Target"
    elif is_reg:
        role = "Regulator"
    else:
        role = "Target"

    self_loop   = node in set(e[0] for e in analysis.get("self_loops", []))
    in_feedback = node in analysis.get("feedback_nodes", set())

    # PubMed link for node (from Entrez if available)
    entrez = ""
    if node in mega_lookup:
        entrez = mega_lookup[node].get("entrez", "")

    tip = (
        "<div style='font-family:monospace;font-size:12px;padding:10px;min-width:240px;max-width:320px;"
        "background:" + t["tooltip_bg"] + ";color:" + t["tooltip_text"] + ";border-radius:8px'>"
        "<b style='font-size:14px;color:" + WONG["sky_blue"] + "'>" + node + "</b>"
    )

    # PubMed gene search link
    if entrez and entrez != "nan":
        tip += ("&nbsp;<a href='https://www.ncbi.nlm.nih.gov/gene/" + entrez +
                "' target='_blank' style='color:" + WONG["sky_blue"] +
                ";font-size:10px;text-decoration:none'>[NCBI&#8599;]</a>")

    tip += (
        "<hr style='border-color:" + t["tooltip_border"] + ";margin:6px 0'/>"
        "<b>Role:</b> " + role + "<br/>"
        "<b>Regulates (out):</b> " + str(out_deg) + " genes<br/>"
        "<b>Regulated by (in):</b> " + str(in_deg) + " genes<br/>"
        "<b>Degree centrality:</b> " + str(round(cent, 3)) + "<br/>"
    )
    if self_loop:
        tip += "<br/><span style='color:" + WONG["yellow"] + "'>&#128260; Self-regulatory loop</span><br/>"
    if in_feedback:
        tip += "<span style='color:" + WONG["orange"] + "'>&#9889; Part of feedback loop</span><br/>"

    # MegaTable expression data
    tip += _mega_table_rows(node, mega_lookup, theme)
    tip += "</div>"
    return tip


def _edge_tooltip(u, v, data, analysis, theme):
    t             = THEMES[theme]
    relationships = data.get("relationships", ["no_effect"])
    perturbations = data.get("perturbations", [])
    effects       = data.get("effects", ["o"])
    stages        = sorted(set(str(s) for s in data.get("stages", []) if pd.notna(s)))
    count         = data.get("count", 1)
    pmids         = data.get("pmids", [])
    is_fb         = (u, v) in analysis.get("feedback_edges", set())

    dom_rel = Counter(relationships).most_common(1)[0][0] if relationships else "no_effect"
    if dom_rel == "activating":
        rel_str = "<span style='color:" + WONG["green"] + "'>&#9650; Activating</span>"
    elif dom_rel == "inhibiting":
        rel_str = "<span style='color:" + WONG["vermillion"] + "'>&#9660; Inhibiting</span>"
    else:
        rel_str = "<span style='color:#94a3b8'>&#9675; No effect</span>"

    stg_str     = ", ".join(stages[:5]) + ("..." if len(stages) > 5 else "")
    pert_unique = ", ".join(sorted(set(perturbations)))
    eff_unique  = ", ".join(sorted(set(effects)))

    tip = (
        "<div style='font-family:monospace;font-size:12px;padding:10px;min-width:260px;max-width:340px;"
        "background:" + t["tooltip_bg"] + ";color:" + t["tooltip_text"] + ";border-radius:8px'>"
        "<b style='color:" + WONG["sky_blue"] + "'>" + u + "</b>"
        "<span style='color:#94a3b8'> &#8594; </span>"
        "<b style='color:" + WONG["orange"] + "'>" + v + "</b><br/>"
        "<hr style='border-color:" + t["tooltip_border"] + ";margin:6px 0'/>"
        "<b>True relationship:</b> " + rel_str + "<br/>"
        "<b>Perturbation(s):</b> " + pert_unique + "<br/>"
        "<b>Raw effect(s):</b> " + eff_unique + "<br/>"
        "<b>Stage(s):</b> " + stg_str + "<br/>"
        "<b>Evidence count:</b> " + str(count) + "<br/>"
    )

    # PubMed links
    if pmids:
        tip += "<hr style='border-color:" + t["tooltip_border"] + ";margin:6px 0'/>"
        tip += "<b>PubMed References:</b><br/>"
        for pmid in pmids[:5]:
            tip += ("&nbsp;&#128196; <a href='https://pubmed.ncbi.nlm.nih.gov/" + pmid +
                    "/' target='_blank' style='color:" + WONG["sky_blue"] +
                    ";text-decoration:none'>PMID: " + pmid + " &#8599;</a><br/>")
        if len(pmids) > 5:
            tip += "<span style='color:#64748b'>+" + str(len(pmids)-5) + " more PMIDs</span><br/>"

    if is_fb:
        tip += "<br/><span style='color:" + WONG["orange"] + "'>&#9889; Feedback loop edge</span><br/>"
    tip += "</div>"
    return tip


def _build_legend_button(config, theme):
    """Returns a collapsible legend button injected into the HTML."""
    t = THEMES[theme]
    bg  = "#0f1525" if theme == "dark" else "#ffffff"
    bdr = "#1e2d4a" if theme == "dark" else "#e2e8f0"
    txt = "#e2e8f0" if theme == "dark" else "#0f172a"
    mut = "#475569" if theme == "dark" else "#64748b"

    return """
<div id="legend-container" style="position:fixed;bottom:20px;left:20px;z-index:9999">
  <!-- Toggle button -->
  <button onclick="toggleLegend()" style="
      background:""" + bg + """;border:1px solid """ + bdr + """;border-radius:8px;
      padding:8px 14px;font-family:monospace;font-size:12px;color:""" + WONG["sky_blue"] + """;
      cursor:pointer;box-shadow:0 4px 16px rgba(0,0,0,0.3);display:flex;
      align-items:center;gap:6px">
    &#128300; Legend
    <span id="legend-arrow" style="font-size:10px">&#9650;</span>
  </button>

  <!-- Legend panel -->
  <div id="legend-panel" style="
      display:none;margin-bottom:6px;
      background:""" + bg + """;border:1px solid """ + bdr + """;border-radius:10px;
      padding:14px 18px;font-family:monospace;font-size:12px;color:""" + txt + """;
      box-shadow:0 4px 20px rgba(0,0,0,0.3);min-width:210px">
    <div style="font-weight:bold;font-size:13px;color:""" + WONG["sky_blue"] + """;margin-bottom:10px">
      Lens GRN Legend
    </div>
    <div style="margin-bottom:5px;font-weight:bold;color:""" + mut + """;font-size:10px;text-transform:uppercase">Node - Role</div>
    <div style="margin-bottom:3px"><span style="color:""" + config["color_regulator"] + """">&#9679;</span> &nbsp;Regulator only</div>
    <div style="margin-bottom:3px"><span style="color:""" + config["color_target"] + """">&#9679;</span> &nbsp;Target only</div>
    <div style="margin-bottom:3px"><span style="color:""" + config["color_both"] + """">&#9679;</span> &nbsp;Regulator &amp; Target</div>
    <div style="margin-bottom:10px"><span style="color:""" + config["color_selfloop"] + """">&#9679;</span> &nbsp;Self-regulatory loop</div>
    <div style="margin-bottom:5px;font-weight:bold;color:""" + mut + """;font-size:10px;text-transform:uppercase">Edge - True Relationship</div>
    <div style="margin-bottom:3px"><span style="color:""" + config["color_activating"] + """">&#9654;</span> Activating (Pert x Effect)</div>
    <div style="margin-bottom:3px"><span style="color:""" + config["color_inhibiting"] + """">&#9654;</span> Inhibiting (Pert x Effect)</div>
    <div style="margin-bottom:10px"><span style="color:""" + config["color_noeffect"] + """">&#9654;</span> No effect</div>
    <div style="font-size:10px;color:""" + mut + """">
      Node size = connections<br/>
      Edge width = evidence count<br/>
      Hover edge = PubMed links<br/>
      Hover node = expression data<br/>
      Wong (2011) color-blind safe
    </div>
  </div>
</div>
<script>
function toggleLegend() {
  var panel = document.getElementById('legend-panel');
  var arrow = document.getElementById('legend-arrow');
  if (panel.style.display === 'none') {
    panel.style.display = 'block';
    arrow.innerHTML = '&#9660;';
  } else {
    panel.style.display = 'none';
    arrow.innerHTML = '&#9650;';
  }
}
</script>"""


def _build_zoom_controls(theme):
    """Compact +/- zoom box replacing default vis navigation buttons."""
    bg  = "#0f1525" if theme == "dark" else "#ffffff"
    bdr = "#1e2d4a" if theme == "dark" else "#e2e8f0"
    txt = "#e2e8f0" if theme == "dark" else "#0f172a"

    return """
<div style="position:fixed;bottom:20px;right:20px;z-index:9999;
            display:flex;align-items:center;
            background:""" + bg + """;border:1px solid """ + bdr + """;
            border-radius:8px;overflow:hidden;
            box-shadow:0 4px 16px rgba(0,0,0,0.3)">
  <button onclick="zoomNet(-0.3)" style="
      background:transparent;border:none;border-right:1px solid """ + bdr + """;
      color:""" + txt + """;font-size:18px;padding:6px 14px;cursor:pointer;
      font-family:monospace;line-height:1" title="Zoom out">&#8722;</button>
  <button onclick="zoomNet(0)" style="
      background:transparent;border:none;border-right:1px solid """ + bdr + """;
      color:#64748b;font-size:10px;padding:6px 10px;cursor:pointer;
      font-family:monospace" title="Reset zoom">&#8635;</button>
  <button onclick="zoomNet(0.3)" style="
      background:transparent;border:none;
      color:""" + txt + """;font-size:18px;padding:6px 14px;cursor:pointer;
      font-family:monospace;line-height:1" title="Zoom in">&#43;</button>
</div>
<script>
function zoomNet(delta) {
  var container = document.getElementById('mynetwork');
  if (!container || !window.network) return;
  if (delta === 0) {
    window.network.fit();
  } else {
    var scale = window.network.getScale();
    window.network.moveTo({scale: Math.max(0.05, Math.min(5, scale + delta))});
  }
}
// Store network reference after init
document.addEventListener('DOMContentLoaded', function() {
  setTimeout(function() {
    var keys = Object.keys(window);
    for (var i = 0; i < keys.length; i++) {
      if (window[keys[i]] && window[keys[i]].fit && window[keys[i]].getScale) {
        window.network = window[keys[i]]; break;
      }
    }
  }, 1500);
});
</script>"""


def visualize(G, analysis, config, mega_lookup=None):
    print("[5/5] Building Pyvis visualization...")
    if mega_lookup is None:
        mega_lookup = {}
    theme = config.get("theme", "dark")
    t     = THEMES[theme]

    net = Network(
        height=config.get("height", "100vh"),
        width=config.get("width", "100%"),
        bgcolor=t["bgcolor"],
        font_color=t["font_color"],
        directed=True,
        notebook=False,
        cdn_resources="remote",
    )

    if config["layout"] == "barnes_hut":
        net.barnes_hut(gravity=-8000, central_gravity=0.3,
                       spring_length=120, spring_strength=0.05, damping=0.09)
    elif config["layout"] == "force_atlas_2based":
        net.force_atlas_2based(gravity=-50, central_gravity=0.01,
                               spring_length=100, spring_strength=0.08)
    elif config["layout"] == "repulsion":
        net.repulsion(node_distance=150, central_gravity=0.2,
                      spring_length=200, spring_strength=0.05)

    for node in G.nodes():
        color = _node_color(G, node, analysis, config)
        in_fb = node in analysis.get("feedback_nodes", set())
        net.add_node(
            node,
            label=node,
            color={
                "background": color,
                "border":     WONG["yellow"] if in_fb else color,
                "highlight":  {"background": "#ffffff", "border": "#000000"},
                "hover":      {"background": "#ffffff", "border": "#000000"},
            },
            size=_node_size(G, node, config),
            title=_node_tooltip(node, G, analysis, theme, mega_lookup),
            borderWidth=3 if in_fb else 1,
            font={"color": t["font_color"], "size": 11, "face": "monospace"},
        )

    for u, v, data in G.edges(data=True):
        rels  = data.get("relationships", ["no_effect"])
        count = data.get("count", 1)
        is_fb = (u, v) in analysis.get("feedback_edges", set())
        net.add_edge(
            u, v,
            color={"color": _edge_color(rels, config), "highlight": "#ffffff", "hover": "#ffffff"},
            title=_edge_tooltip(u, v, data, analysis, theme),
            width=1.2 + (count * 0.25),
            arrows={"to": {"enabled": True, "scaleFactor": 0.5}},
            dashes=is_fb,
            smooth={"type": "curvedCW", "roundness": 0.15},
        )

    # Disable default navigation buttons — we use our own zoom controls
    net.set_options("""
    {
      "interaction": {
        "hover": true,
        "tooltipDelay": 80,
        "navigationButtons": false,
        "keyboard": {"enabled": true},
        "multiselect": true,
        "zoomView": true
      },
      "nodes": {
        "shadow": {"enabled": true, "color": "rgba(0,0,0,0.3)", "size": 8, "x": 2, "y": 2}
      },
      "physics": {
        "stabilization": {"iterations": 250, "updateInterval": 25}
      }
    }
    """)

    output_path = config.get("output_file", "grn_output.html")
    net.save_graph(output_path)

    with open(output_path, "r", encoding="utf-8") as f:
        html = f.read()

    # Store network reference in window.network for zoom controls
    html = html.replace(
        "var network = new vis.Network(",
        "window.network = new vis.Network("
    )

    css = (
        "<style>"
        "div.vis-tooltip {"
        "background:" + t["tooltip_bg"] + " !important;"
        "border:1px solid " + t["tooltip_border"] + " !important;"
        "color:" + t["tooltip_text"] + " !important;"
        "border-radius:8px !important;"
        "font-family:monospace !important;"
        "padding:2px !important;"
        "box-shadow:0 4px 20px rgba(0,0,0,0.3) !important;"
        "max-width:360px !important;"
        "}"
        "a { pointer-events: auto !important; }"
        "</style>"
    )

    html = html.replace("</head>", css + "\n</head>")
    html = html.replace("</body>",
                        _build_legend_button(config, theme) +
                        _build_zoom_controls(theme) +
                        "\n</body>")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print("       Saved -> " + output_path)
    return output_path


# =============================================================================
# 7. MAIN
# =============================================================================

def print_summary(G, analysis):
    print()
    print("=" * 60)
    print("  GRN ANALYSIS SUMMARY")
    print("=" * 60)
    regs = sum(1 for n in G.nodes() if G.nodes[n].get("is_reg") and not G.nodes[n].get("is_tgt"))
    tgts = sum(1 for n in G.nodes() if G.nodes[n].get("is_tgt") and not G.nodes[n].get("is_reg"))
    both = sum(1 for n in G.nodes() if G.nodes[n].get("is_reg") and G.nodes[n].get("is_tgt"))
    print("  Nodes        : " + str(G.number_of_nodes()))
    print("  Edges        : " + str(G.number_of_edges()))
    print("  Regulators   : " + str(regs))
    print("  Targets      : " + str(tgts))
    print("  Both         : " + str(both))
    print()
    print("  TOP HUB REGULATORS:")
    for gene, deg in analysis.get("hub_genes", []):
        print("    " + gene.ljust(25) + str(deg).rjust(4) + "  " + "█" * min(deg, 35))
    print()
    print("  FEEDBACK LOOPS : " + str(len(analysis.get("feedback_loops", []))))
    print("  SELF-LOOPS     : " + str(len(analysis.get("self_loops", []))))
    print("  COMPONENTS     : " + str(len(analysis.get("components", []))))
    print("=" * 60)
    print()


def run_pipeline(config):
    """Full pipeline — importable from app.py."""
    mega_lookup = load_megatable(config)
    df          = load_data(config)
    df          = filter_data(df, config)
    if len(df) == 0:
        print("[ERROR] No edges after filtering.")
        return None, None, None, {}
    G        = build_graph(df)
    analysis = analyze_graph(G, config)
    return df, G, analysis, mega_lookup


def main():
    print()
    print("=" * 60)
    print("  Lens GRN Explorer - NetworkX + Pyvis")
    print("=" * 60)
    print()
    df, G, analysis, mega_lookup = run_pipeline(CONFIG)
    if G is None:
        sys.exit(1)
    print_summary(G, analysis)
    output   = visualize(G, analysis, CONFIG, mega_lookup)
    abs_path = os.path.abspath(output)
    print("  Opening -> " + abs_path)
    webbrowser.open("file://" + abs_path)
    print("\n  Done!\n")


if __name__ == "__main__":
    main()