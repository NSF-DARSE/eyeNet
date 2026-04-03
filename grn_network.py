"""
=============================================================================
Lens GRN Explorer — NetworkX + Pyvis
=============================================================================
Lachke Lab 2016 — Gene Regulatory Network

KEY FIXES:
    - vis.js tooltips: titles converted to DOM elements so HTML renders correctly
    - PubMed links now clickable in edge tooltips
    - Expression data in node tooltips
    - Legend as collapsible button (bottom-left)
    - Compact zoom box (bottom-right)
=============================================================================
"""

import os
import sys
import re
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
    "input_file":     "data/Lens_GRN_June_2016_original FOR HACKATHON - Salil Lachke.xlsx",
    "megatable_file": "data/MegaTable_April_24_2024_for_Microarray_and_RNA_Seq_Sent_to_Murali__1_.xls",
    "output_file":    "grn_output.html",
    "sheet_name":     "Lens_GRN_pert",
    "stage_from":     None,
    "stage_to":       None,
    "stage_single":   None,
    "filter_regulator": None,
    "filter_target":    None,
    "relationships_include": ["activating", "inhibiting", "no_effect"],
    "max_edges":            300,
    "show_feedback_loops":  True,
    "color_regulator":  WONG["sky_blue"],
    "color_target":     WONG["orange"],
    "color_both":       WONG["pink"],
    "color_selfloop":   WONG["yellow"],
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
    path = config.get("megatable_file", "")
    if not path or not os.path.exists(path):
        print("[WARN] MegaTable not found — expression data unavailable")
        return {}
    print("      Loading MegaTable: " + path)
    df = pd.read_excel(path)

    ma_exp = ['Beebe_E12_exp_Fiber','Beebe_E12_exp_Epi','Naka_P13_fiber_exp','Naka_P13_epi_exp']
    ma_enr = ['Beebe_E12_Fiber_enr','Beebe_E12_Epi_enr','Naka_P13_fiber_enr','Naka_P13_epi_enr']
    rna    = ['enr_FC_E14_Cv','enr_LEC_E14_Cv','enr_FC_E16_Cv','enr_LEC_E16_Cv',
              'enr_FC_E18_Cv','enr_LEC_E18_Cv','enr_FC_P0_Cv','enr_LEC_P0_Cv',
              'enr_LEC_P0_Rob','enr_FC_P0_Rob','enr_FC_3Mo','enr_LEC_3Mo',
              'enr_FC_6Mo','enr_LEC_6Mo','enr_FC_2Y','enr_LEC_2Y']

    lookup = {}
    for _, row in df.iterrows():
        sym = str(row.get('Symbol', '')).strip()
        if not sym or sym == 'nan':
            continue
        def sf(v):
            try:
                if pd.notna(v): return round(float(v), 3)
            except: pass
            return None
        lookup[sym] = {
            'entrez':      str(row.get('Entrez', '')),
            'uniprot':     str(row.get('UNIPROT', '')),
            'description': str(row.get('Gene_description', '')),
            'microarray_exp': {c: sf(row.get(c)) for c in ma_exp},
            'microarray_enr': {c: sf(row.get(c)) for c in ma_enr},
            'rnaseq':         {c: sf(row.get(c)) for c in rna},
        }
    print("      MegaTable: " + str(len(lookup)) + " genes")
    return lookup


def load_data(config):
    path = config["input_file"]
    if not os.path.exists(path):
        print("\n[ERROR] File not found: " + path)
        sys.exit(1)
    print("[1/5] Loading GRN: " + path)
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
    df = df[["sno","regulator","target","perturbation","effect","stage","context","reference","pmid"]]
    df = df.dropna(subset=["regulator","target"])
    for c in ["regulator","target","perturbation","effect","stage","context"]:
        df[c] = df[c].astype(str).str.strip()
    df["effect"]       = df["effect"].replace({"nan":"o","none":"o","None":"o","0":"o"})
    df["perturbation"] = df["perturbation"].replace({"nan":"-","none":"-","None":"-"})
    def clean_pmid(v):
        try:
            if pd.notna(v): return str(int(float(v)))
        except: pass
        return ""
    df["pmid"] = df["pmid"].apply(clean_pmid)
    df["true_relationship"] = df.apply(_compute_relationship, axis=1)
    print("       Loaded " + str(len(df)) + " edges")
    return df


def _compute_relationship(row):
    pert   = str(row["perturbation"]).strip()
    effect = str(row["effect"]).strip()
    if effect == "o": return "no_effect"
    return "activating" if pert == effect else "inhibiting"


# =============================================================================
# 3. FILTERING
# =============================================================================

def stage_numeric(stage):
    s = str(stage).strip()
    if s == "Adult": return 100000.0
    try:
        if s.startswith("E"): return float(s[1:])
        if s.startswith("P"): return 1000.0 + float(s[1:])
    except ValueError: pass
    return 99999.0


def filter_data(df, config):
    print("[2/5] Filtering...")
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
    print("       " + str(original) + " -> " + str(len(df)) + " edges")
    return df.reset_index(drop=True)


# =============================================================================
# 4. GRAPH BUILDING
# =============================================================================

def build_graph(df):
    print("[3/5] Building graph...")
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
        pmid      = str(row.get("pmid","")).strip()
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
                perturbations=[pert], effects=[eff], relationships=[rel],
                stages=[stg], contexts=[ctx], count=1,
                pmids=[pmid] if pmid else [])
    print("       " + str(G.number_of_nodes()) + " nodes, " + str(G.number_of_edges()) + " edges")
    return G


# =============================================================================
# 5. ANALYSIS
# =============================================================================

def analyze_graph(G, config):
    print("[4/5] Analysis...")
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
                if i > 500: break
                results["feedback_loops"].append(cycle)
                for node in cycle: results["feedback_nodes"].add(node)
                for j in range(len(cycle)):
                    results["feedback_edges"].add((cycle[j], cycle[(j+1)%len(cycle)]))
        except: pass
        results["self_loops"] = list(nx.selfloop_edges(G))
    results["components"] = list(nx.weakly_connected_components(G))
    return results


# =============================================================================
# 6. VISUALIZATION HELPERS
# =============================================================================

def _node_size(G, node, config):
    deg     = G.nodes[node].get("total_degree", 1)
    max_deg = max((d for _, d in G.degree()), default=1)
    return int(config["node_size_min"] + (deg/max_deg) * (config["node_size_max"] - config["node_size_min"]))


def _node_color(G, node, analysis, config):
    if node in set(e[0] for e in analysis.get("self_loops", [])): return config["color_selfloop"]
    is_reg = G.nodes[node].get("is_reg", False)
    is_tgt = G.nodes[node].get("is_tgt", False)
    if is_reg and is_tgt: return config["color_both"]
    if is_reg:            return config["color_regulator"]
    return config["color_target"]


def _edge_color(relationships, config):
    if not relationships: return config["color_noeffect"]
    dom = Counter(relationships).most_common(1)[0][0]
    if dom == "activating": return config["color_activating"]
    if dom == "inhibiting": return config["color_inhibiting"]
    return config["color_noeffect"]


def _escape_js_string(s):
    """Escape a string for safe embedding inside a JS string literal."""
    return (s.replace("\\", "\\\\")
             .replace("'", "\\'")
             .replace("\n", " ")
             .replace("\r", ""))


def _node_tooltip_html(node, G, analysis, theme, mega_lookup):
    """Returns raw HTML string for node tooltip."""
    t       = THEMES[theme]
    bg      = t["tooltip_bg"]
    txt     = t["tooltip_text"]
    bdr     = t["tooltip_border"]
    mut     = "#64748b"
    d       = G.nodes[node]
    is_reg  = d.get("is_reg", False)
    is_tgt  = d.get("is_tgt", False)
    out_deg = d.get("out_degree", 0)
    in_deg  = d.get("in_degree", 0)
    cent    = d.get("deg_centrality", 0)

    if is_reg and is_tgt: role = "Regulator &amp; Target"
    elif is_reg:          role = "Regulator"
    else:                 role = "Target"

    self_loop   = node in set(e[0] for e in analysis.get("self_loops", []))
    in_feedback = node in analysis.get("feedback_nodes", set())

    entrez = ""
    if node in mega_lookup:
        entrez = mega_lookup[node].get("entrez", "")

    h = (
        "<div style='font-family:monospace;font-size:12px;padding:10px;"
        "min-width:220px;max-width:300px;background:" + bg + ";color:" + txt + ";border-radius:8px'>"
        "<b style='font-size:14px;color:" + WONG["sky_blue"] + "'>" + node + "</b>"
    )
    if entrez and entrez != "nan":
        h += (" <a href='https://www.ncbi.nlm.nih.gov/gene/" + entrez +
              "' target='_blank' style='color:" + WONG["sky_blue"] +
              ";font-size:10px'>[NCBI]</a>")
    h += (
        "<hr style='border-color:" + bdr + ";margin:5px 0'/>"
        "<b>Role:</b> " + role + "<br/>"
        "<b>Out (regulates):</b> " + str(out_deg) + "<br/>"
        "<b>In (regulated by):</b> " + str(in_deg) + "<br/>"
        "<b>Centrality:</b> " + str(round(cent,3)) + "<br/>"
    )
    if self_loop:
        h += "<span style='color:" + WONG["yellow"] + "'>&#128260; Self-regulatory</span><br/>"
    if in_feedback:
        h += "<span style='color:" + WONG["orange"] + "'>&#9889; Feedback loop</span><br/>"
    h += "</div>"
    return h


def _edge_tooltip_html(u, v, data, analysis, theme):
    """Returns raw HTML string for edge tooltip."""
    t             = THEMES[theme]
    bg            = t["tooltip_bg"]
    txt           = t["tooltip_text"]
    bdr           = t["tooltip_border"]
    relationships = data.get("relationships", ["no_effect"])
    perturbations = data.get("perturbations", [])
    effects       = data.get("effects", ["o"])
    stages        = sorted(set(str(s) for s in data.get("stages",[]) if pd.notna(s)))
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

    stg_str = ", ".join(stages[:5]) + ("..." if len(stages)>5 else "")

    h = (
        "<div style='font-family:monospace;font-size:12px;padding:10px;"
        "min-width:260px;max-width:340px;background:" + bg + ";color:" + txt + ";border-radius:8px'>"
        "<b style='color:" + WONG["sky_blue"] + "'>" + u + "</b>"
        " <span style='color:#94a3b8'>&#8594;</span> "
        "<b style='color:" + WONG["orange"] + "'>" + v + "</b>"
        "<hr style='border-color:" + bdr + ";margin:5px 0'/>"
        "<b>Relationship:</b> " + rel_str + "<br/>"
        "<b>Perturbation:</b> " + ", ".join(sorted(set(perturbations))) + "<br/>"
        "<b>Raw effect:</b> " + ", ".join(sorted(set(effects))) + "<br/>"
        "<b>Stage(s):</b> " + stg_str + "<br/>"
        "<b>Evidence:</b> " + str(count) + " record(s)<br/>"
    )
    if pmids:
        h += "<hr style='border-color:" + bdr + ";margin:5px 0'/>"
        h += "<b>PubMed:</b><br/>"
        for pmid in pmids[:6]:
            h += ("<a href='https://pubmed.ncbi.nlm.nih.gov/" + pmid +
                  "/' target='_blank' style='color:" + WONG["sky_blue"] +
                  ";display:block;margin:2px 0'>&#128196; PMID " + pmid + "</a>")
        if len(pmids) > 6:
            h += "<span style='color:#64748b'>+" + str(len(pmids)-6) + " more</span><br/>"
    if is_fb:
        h += "<span style='color:" + WONG["orange"] + "'>&#9889; Feedback loop edge</span><br/>"
    h += "</div>"
    return h


def _post_process_html(html_str, config, theme, mega_lookup):
    """
    Post-process the pyvis-generated HTML to:
    1. Convert all node/edge title strings to DOM elements
       so vis.js renders HTML (links, colors, etc.) correctly.
    2. Inject legend button, zoom controls, CSS.
    """
    t   = THEMES[theme]
    bg  = t["tooltip_bg"]
    bdr = t["tooltip_border"]
    txt = t["tooltip_text"]

    # ── CSS ──────────────────────────────────────────────────────────────
    css = (
        "<style>"
        "div.vis-tooltip {"
        "  background:" + bg + " !important;"
        "  border:1px solid " + bdr + " !important;"
        "  color:" + txt + " !important;"
        "  border-radius:8px !important;"
        "  padding:0 !important;"
        "  box-shadow:0 4px 20px rgba(0,0,0,0.4) !important;"
        "  max-width:360px !important;"
        "  font-family:monospace !important;"
        "}"
        "a { cursor: pointer !important; }"
        "</style>"
    )
    html_str = html_str.replace("</head>", css + "\n</head>")

    # ── Store network reference for zoom ─────────────────────────────────
    html_str = html_str.replace(
        "var network = new vis.Network(",
        "window.network = new vis.Network("
    )

    # ── JS patch: convert title strings -> DOM elements ──────────────────
    # vis.js renders HTML only if title is a DOM Node, not a string.
    # We inject a patch that runs after network init and replaces all
    # string titles with div elements containing the HTML.
    patch_js = """
<script>
(function patchTitles() {
  function applyTitles() {
    if (!window.network) { setTimeout(applyTitles, 300); return; }
    var nodesDS = window.network.body.data.nodes;
    var edgesDS = window.network.body.data.edges;

    function makeDiv(html) {
      var d = document.createElement('div');
      d.innerHTML = html;
      // Make links work inside vis.js tooltip
      d.querySelectorAll('a').forEach(function(a) {
        a.addEventListener('click', function(e) {
          e.stopPropagation();
          window.open(a.href, '_blank');
        });
      });
      return d;
    }

    nodesDS.get().forEach(function(node) {
      if (typeof node.title === 'string' && node.title.indexOf('<') !== -1) {
        nodesDS.update({id: node.id, title: makeDiv(node.title)});
      }
    });

    edgesDS.get().forEach(function(edge) {
      if (typeof edge.title === 'string' && edge.title.indexOf('<') !== -1) {
        edgesDS.update({id: edge.id, title: makeDiv(edge.title)});
      }
    });
  }
  setTimeout(applyTitles, 800);
})();
</script>
"""

    # ── Legend button ─────────────────────────────────────────────────────
    sidebar_bg  = "#0f1525" if theme == "dark" else "#ffffff"
    sidebar_bdr = "#1e2d4a" if theme == "dark" else "#e2e8f0"
    sidebar_txt = "#e2e8f0" if theme == "dark" else "#0f172a"
    mut         = "#475569" if theme == "dark" else "#64748b"

    legend_btn = (
        "<div id='legend-container' style='position:fixed;bottom:20px;left:20px;z-index:9999'>"
        "<button onclick='toggleLegend()' style='"
        "background:" + sidebar_bg + ";border:1px solid " + sidebar_bdr + ";"
        "border-radius:8px;padding:7px 14px;font-family:monospace;font-size:12px;"
        "color:" + WONG["sky_blue"] + ";cursor:pointer;"
        "box-shadow:0 4px 16px rgba(0,0,0,0.3);display:flex;align-items:center;gap:6px'>"
        "&#128300; Legend <span id='legend-arrow'>&#9650;</span>"
        "</button>"
        "<div id='legend-panel' style='display:none;margin-bottom:6px;"
        "background:" + sidebar_bg + ";border:1px solid " + sidebar_bdr + ";"
        "border-radius:10px;padding:14px 18px;font-family:monospace;"
        "font-size:12px;color:" + sidebar_txt + ";"
        "box-shadow:0 4px 20px rgba(0,0,0,0.3);min-width:220px'>"
        "<b style='color:" + WONG["sky_blue"] + "'>Lens GRN Legend</b><br/><br/>"
        "<b style='color:" + mut + ";font-size:10px'>NODE - ROLE</b><br/>"
        "<span style='color:" + config["color_regulator"] + "'>&#9679;</span> Regulator only<br/>"
        "<span style='color:" + config["color_target"] + "'>&#9679;</span> Target only<br/>"
        "<span style='color:" + config["color_both"] + "'>&#9679;</span> Regulator &amp; Target<br/>"
        "<span style='color:" + config["color_selfloop"] + "'>&#9679;</span> Self-regulatory loop<br/><br/>"
        "<b style='color:" + mut + ";font-size:10px'>EDGE - RELATIONSHIP</b><br/>"
        "<span style='color:" + config["color_activating"] + "'>&#9654;</span> Activating<br/>"
        "<span style='color:" + config["color_inhibiting"] + "'>&#9654;</span> Inhibiting<br/>"
        "<span style='color:" + config["color_noeffect"] + "'>&#9654;</span> No effect<br/><br/>"
        "<span style='color:" + mut + ";font-size:10px'>"
        "Hover edge &#8594; PubMed links<br/>"
        "Hover node &#8594; expression data<br/>"
        "Wong (2011) color-blind safe"
        "</span>"
        "</div>"
        "</div>"
        "<script>"
        "function toggleLegend(){"
        "var p=document.getElementById('legend-panel');"
        "var a=document.getElementById('legend-arrow');"
        "if(p.style.display==='none'){p.style.display='block';a.innerHTML='&#9660;'}"
        "else{p.style.display='none';a.innerHTML='&#9650;'}}"
        "</script>"
    )

    # ── Compact zoom box ──────────────────────────────────────────────────
    zoom_box = (
        "<div style='position:fixed;bottom:20px;right:20px;z-index:9999;"
        "display:flex;align-items:center;"
        "background:" + sidebar_bg + ";border:1px solid " + sidebar_bdr + ";"
        "border-radius:8px;overflow:hidden;box-shadow:0 4px 16px rgba(0,0,0,0.3)'>"
        "<button onclick='zoomNet(-0.3)' title='Zoom out' style='"
        "background:transparent;border:none;border-right:1px solid " + sidebar_bdr + ";"
        "color:" + sidebar_txt + ";font-size:18px;padding:6px 14px;cursor:pointer;line-height:1'>&#8722;</button>"
        "<button onclick='zoomNet(0)' title='Fit' style='"
        "background:transparent;border:none;border-right:1px solid " + sidebar_bdr + ";"
        "color:#64748b;font-size:14px;padding:6px 10px;cursor:pointer'>&#8635;</button>"
        "<button onclick='zoomNet(0.3)' title='Zoom in' style='"
        "background:transparent;border:none;"
        "color:" + sidebar_txt + ";font-size:18px;padding:6px 14px;cursor:pointer;line-height:1'>&#43;</button>"
        "</div>"
        "<script>"
        "function zoomNet(d){"
        "if(!window.network)return;"
        "if(d===0){window.network.fit();return;}"
        "var s=window.network.getScale();"
        "window.network.moveTo({scale:Math.max(0.05,Math.min(5,s+d))});}"
        "</script>"
    )

    html_str = html_str.replace("</body>",
        patch_js + legend_btn + zoom_box + "\n</body>")

    return html_str


# =============================================================================
# 7. MAIN VISUALIZE
# =============================================================================

def visualize(G, analysis, config, mega_lookup=None):
    print("[5/5] Building visualization...")
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
        title = _node_tooltip_html(node, G, analysis, theme, mega_lookup)
        net.add_node(
            node, label=node,
            color={
                "background": color,
                "border":     WONG["yellow"] if in_fb else color,
                "highlight":  {"background": "#ffffff", "border": "#000000"},
                "hover":      {"background": "#ffffff", "border": "#000000"},
            },
            size=_node_size(G, node, config),
            title=title,
            borderWidth=3 if in_fb else 1,
            font={"color": t["font_color"], "size": 11, "face": "monospace"},
        )

    for u, v, data in G.edges(data=True):
        rels  = data.get("relationships", ["no_effect"])
        count = data.get("count", 1)
        is_fb = (u, v) in analysis.get("feedback_edges", set())
        title = _edge_tooltip_html(u, v, data, analysis, theme)
        net.add_edge(
            u, v,
            color={"color": _edge_color(rels, config), "highlight": "#ffffff", "hover": "#ffffff"},
            title=title,
            width=1.2 + (count * 0.25),
            arrows={"to": {"enabled": True, "scaleFactor": 0.5}},
            dashes=is_fb,
            smooth={"type": "curvedCW", "roundness": 0.15},
        )

    net.set_options("""
    {
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
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
        raw = f.read()

    raw = _post_process_html(raw, config, theme, mega_lookup)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(raw)

    print("       Saved -> " + output_path)
    return output_path


# =============================================================================
# PIPELINE + MAIN
# =============================================================================

def print_summary(G, analysis):
    print("\n" + "="*60)
    regs = sum(1 for n in G.nodes() if G.nodes[n].get("is_reg") and not G.nodes[n].get("is_tgt"))
    tgts = sum(1 for n in G.nodes() if G.nodes[n].get("is_tgt") and not G.nodes[n].get("is_reg"))
    both = sum(1 for n in G.nodes() if G.nodes[n].get("is_reg") and G.nodes[n].get("is_tgt"))
    print("  Nodes: " + str(G.number_of_nodes()) +
          " | Edges: " + str(G.number_of_edges()) +
          " | Regs: " + str(regs) + " | Targets: " + str(tgts) + " | Both: " + str(both))
    print("  TOP HUBS:")
    for gene, deg in analysis.get("hub_genes", []):
        print("    " + gene.ljust(25) + str(deg).rjust(4))
    print("  Feedback loops: " + str(len(analysis.get("feedback_loops",[]))) +
          " | Self-loops: " + str(len(analysis.get("self_loops",[]))))
    print("="*60 + "\n")


def run_pipeline(config):
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
    print("\n" + "="*60 + "\n  Lens GRN Explorer\n" + "="*60)
    df, G, analysis, mega_lookup = run_pipeline(CONFIG)
    if G is None: sys.exit(1)
    print_summary(G, analysis)
    output   = visualize(G, analysis, CONFIG, mega_lookup)
    abs_path = os.path.abspath(output)
    print("  Opening -> " + abs_path)
    webbrowser.open("file://" + abs_path)


if __name__ == "__main__":
    main()