"""
=============================================================================
Lens GRN Explorer — NetworkX + Pyvis
=============================================================================
KEY FIXES:
  - Custom sticky tooltip panel (stays visible so links are clickable)
  - PubMed links open in new tab reliably
  - Legend as collapsible button
  - Compact zoom box
  - MegaTable data tagged on nodes (ring border = has expression data)
=============================================================================
"""

import os, sys, webbrowser
import pandas as pd
import networkx as nx
from pyvis.network import Network
from collections import Counter

# =============================================================================
# WONG palette + CONFIG
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
    "megatable_file": "data/MegaTable April 24 2024 for Microarray and RNA Seq Sent to Murali (1).xls",
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
# DATA LOADING
# =============================================================================

def _find_megatable(config):
    """Try multiple possible filenames for the MegaTable."""
    candidates = [
        config.get("megatable_file", ""),
        "data/MegaTable April 24 2024 for Microarray and RNA Seq Sent to Murali (1).xls",
        "data/MegaTable_April_24_2024_for_Microarray_and_RNA_Seq_Sent_to_Murali__1_.xls",
        "data/MegaTable.xls",
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def load_megatable(config):
    path = _find_megatable(config)
    if not path:
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

    def sf(v):
        try:
            if pd.notna(v): return round(float(v), 3)
        except: pass
        return None

    lookup = {}
    for _, row in df.iterrows():
        sym = str(row.get('Symbol', '')).strip()
        if not sym or sym == 'nan':
            continue
        lookup[sym] = {
            'entrez':      str(row.get('Entrez', '')),
            'uniprot':     str(row.get('UNIPROT', '')),
            'description': str(row.get('Gene_description', '')),
            'microarray_exp': {c: sf(row.get(c)) for c in ma_exp},
            'microarray_enr': {c: sf(row.get(c)) for c in ma_enr},
            'rnaseq':         {c: sf(row.get(c)) for c in rna},
        }
    print("      MegaTable: " + str(len(lookup)) + " genes loaded")
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
        df.columns[0]: "sno", df.columns[1]: "regulator", df.columns[2]: "target",
        df.columns[4]: "perturbation", df.columns[5]: "effect",
        df.columns[6]: "stage", df.columns[7]: "context",
        df.columns[20]: "reference", df.columns[21]: "pmid",
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
# FILTERING
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
# GRAPH BUILDING
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
# ANALYSIS
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
# VISUALIZATION HELPERS
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


def _build_node_data_js(G, analysis, config, mega_lookup):
    """
    Build a JS object mapping node_id -> tooltip HTML and edge data.
    This is embedded in the page so our custom tooltip can access it.
    """
    import json

    node_data = {}
    for node in G.nodes():
        d       = G.nodes[node]
        is_reg  = d.get("is_reg", False)
        is_tgt  = d.get("is_tgt", False)
        out_deg = d.get("out_degree", 0)
        in_deg  = d.get("in_degree", 0)
        cent    = d.get("deg_centrality", 0)
        self_loop   = node in set(e[0] for e in analysis.get("self_loops", []))
        in_feedback = node in analysis.get("feedback_nodes", set())

        if is_reg and is_tgt: role = "Regulator & Target"
        elif is_reg:          role = "Regulator"
        else:                 role = "Target"

        mega = mega_lookup.get(node, {})
        entrez  = mega.get("entrez", "")
        uniprot = mega.get("uniprot", "")
        desc    = mega.get("description", "")
        has_expr = bool(mega)

        node_data[node] = {
            "role": role,
            "out": out_deg,
            "in": in_deg,
            "cent": round(cent, 3),
            "self_loop": self_loop,
            "feedback": in_feedback,
            "entrez": entrez if entrez and entrez != "nan" else "",
            "uniprot": uniprot if uniprot and uniprot != "nan" else "",
            "desc": desc[:80] if desc and desc != "nan" else "",
            "has_expr": has_expr,
        }

    edge_data = {}
    for u, v, data in G.edges(data=True):
        rels  = data.get("relationships", ["no_effect"])
        perts = data.get("perturbations", [])
        effs  = data.get("effects", ["o"])
        stages= sorted(set(str(s) for s in data.get("stages",[]) if pd.notna(s)))
        count = data.get("count", 1)
        pmids = data.get("pmids", [])
        is_fb = (u, v) in analysis.get("feedback_edges", set())
        dom_rel = Counter(rels).most_common(1)[0][0] if rels else "no_effect"

        key = u + "|||" + v
        edge_data[key] = {
            "from": u,
            "to": v,
            "rel": dom_rel,
            "perts": sorted(set(perts)),
            "effs": sorted(set(effs)),
            "stages": stages[:5],
            "count": count,
            "pmids": pmids[:8],
            "feedback": is_fb,
        }

    return json.dumps(node_data), json.dumps(edge_data)


def _build_custom_tooltip_js(config, theme):
    """
    Build JS that creates a custom sticky tooltip panel.
    This replaces vis.js native tooltips with a custom div that
    stays visible when you hover over it, so links are clickable.
    """
    t           = THEMES[theme]
    bg          = t["tooltip_bg"]
    bdr         = t["tooltip_border"]
    txt         = t["tooltip_text"]
    sky         = WONG["sky_blue"]
    orange      = WONG["orange"]
    green       = WONG["green"]
    verm        = WONG["vermillion"]
    yellow      = WONG["yellow"]
    muted       = "#64748b"

    return """
<div id="custom-tooltip" style="
    position:fixed; z-index:99999; display:none;
    background:""" + bg + """; border:1px solid """ + bdr + """;
    color:""" + txt + """; border-radius:10px;
    padding:12px 14px; font-family:monospace; font-size:12px;
    box-shadow:0 8px 32px rgba(0,0,0,0.5); max-width:340px;
    min-width:240px; pointer-events:auto;
    transition: opacity 0.1s;
" id="custom-tooltip"></div>

<script>
var GRN_NODE_DATA = """ + "NODE_DATA_PLACEHOLDER" + """;
var GRN_EDGE_DATA = """ + "EDGE_DATA_PLACEHOLDER" + """;

var tooltip = document.getElementById('custom-tooltip');
var tooltipHideTimer = null;
var tooltipVisible = false;

function showTooltip(html, x, y) {
    clearTimeout(tooltipHideTimer);
    tooltip.innerHTML = html;
    tooltip.style.display = 'block';
    tooltipVisible = true;

    // Position tooltip
    var tw = 350, th = 300;
    var wx = window.innerWidth, wy = window.innerHeight;
    var left = x + 15;
    var top  = y + 15;
    if (left + tw > wx) left = x - tw - 10;
    if (top  + th > wy) top  = y - th - 10;
    tooltip.style.left = left + 'px';
    tooltip.style.top  = top  + 'px';
}

function hideTooltipDelayed() {
    tooltipHideTimer = setTimeout(function() {
        tooltip.style.display = 'none';
        tooltipVisible = false;
    }, 300);
}

// Keep tooltip visible when hovering over it
tooltip.addEventListener('mouseenter', function() {
    clearTimeout(tooltipHideTimer);
});
tooltip.addEventListener('mouseleave', function() {
    hideTooltipDelayed();
});

function buildNodeTooltip(nodeId) {
    var d = GRN_NODE_DATA[nodeId];
    if (!d) return '<b>' + nodeId + '</b>';

    var roleColor = d.role === 'Regulator' ? '""" + sky + """' :
                    d.role === 'Target'     ? '""" + orange + """' : '""" + WONG["pink"] + """';

    var html = '<div style="margin-bottom:6px">'
             + '<b style="font-size:14px;color:""" + sky + """">' + nodeId + '</b>';

    if (d.entrez) {
        html += ' <a href="https://www.ncbi.nlm.nih.gov/gene/' + d.entrez
              + '" target="_blank" style="color:""" + sky + """;font-size:10px;text-decoration:underline">[NCBI]</a>';
    }
    if (d.uniprot) {
        html += ' <a href="https://www.uniprot.org/uniprot/' + d.uniprot
              + '" target="_blank" style="color:""" + sky + """;font-size:10px;text-decoration:underline">[UniProt]</a>';
    }
    html += '</div>';
    html += '<hr style="border-color:""" + bdr + """;margin:4px 0"/>';
    html += '<b>Role:</b> <span style="color:'+roleColor+'">' + d.role + '</span><br/>';
    html += '<b>Regulates:</b> ' + d.out + ' genes &nbsp; <b>Regulated by:</b> ' + d.in + '<br/>';
    html += '<b>Centrality:</b> ' + d.cent + '<br/>';

    if (d.self_loop) html += '<span style="color:""" + yellow + """">&#128260; Self-regulatory loop</span><br/>';
    if (d.feedback)  html += '<span style="color:""" + orange + """">&#9889; Part of feedback loop</span><br/>';

    if (d.desc) {
        html += '<div style="color:""" + muted + """;font-size:10px;font-style:italic;margin-top:4px">' + d.desc + '</div>';
    }

    if (d.has_expr) {
        html += '<div style="margin-top:6px;padding:4px 6px;background:rgba(86,180,233,0.1);'
             +  'border-radius:4px;font-size:10px;color:""" + sky + """">'
             +  '&#128202; Expression data available — see sidebar panel</div>';
    }

    return html;
}

function buildEdgeTooltip(fromId, toId) {
    var key = fromId + '|||' + toId;
    var d = GRN_EDGE_DATA[key];
    if (!d) return '<b>' + fromId + ' → ' + toId + '</b>';

    var relColor = d.rel === 'activating' ? '""" + green + """' :
                   d.rel === 'inhibiting' ? '""" + verm + """' : '""" + muted + """';
    var relIcon  = d.rel === 'activating' ? '▲' :
                   d.rel === 'inhibiting' ? '▼' : '○';
    var relLabel = d.rel.charAt(0).toUpperCase() + d.rel.slice(1);

    var html = '<b style="color:""" + sky + """">' + fromId + '</b>'
             + '<span style="color:""" + muted + """"> → </span>'
             + '<b style="color:""" + orange + """">' + toId + '</b>';
    html += '<hr style="border-color:""" + bdr + """;margin:4px 0"/>';
    html += '<b>Relationship:</b> <span style="color:'+relColor+'">' + relIcon + ' ' + relLabel + '</span><br/>';
    html += '<b>Perturbation:</b> ' + d.perts.join(', ') + '<br/>';
    html += '<b>Raw effect:</b> '   + d.effs.join(', ')  + '<br/>';
    html += '<b>Stage(s):</b> '     + d.stages.join(', ') + '<br/>';
    html += '<b>Evidence:</b> '     + d.count + ' record(s)<br/>';

    if (d.pmids && d.pmids.length > 0) {
        html += '<hr style="border-color:""" + bdr + """;margin:4px 0"/>';
        html += '<b>PubMed References:</b><br/>';
        d.pmids.forEach(function(pmid) {
            html += '<a href="https://pubmed.ncbi.nlm.nih.gov/' + pmid + '/" target="_blank" '
                  + 'style="color:""" + sky + """;display:block;margin:3px 0;text-decoration:underline">'
                  + '&#128196; PMID ' + pmid + '</a>';
        });
    }

    if (d.feedback) {
        html += '<span style="color:""" + orange + """">&#9889; Feedback loop edge</span><br/>';
    }

    return html;
}

// Hook into vis.js events after network is ready
function attachNetworkEvents() {
    if (!window.network) { setTimeout(attachNetworkEvents, 500); return; }

    // Disable native vis.js tooltip
    window.network.on('hoverNode', function(params) {
        var pos = params.event.center || {x: params.event.clientX, y: params.event.clientY};
        var html = buildNodeTooltip(params.node);
        showTooltip(html, pos.x, pos.y);
    });

    window.network.on('blurNode', function(params) {
        hideTooltipDelayed();
    });

    window.network.on('hoverEdge', function(params) {
        var pos = params.event.center || {x: params.event.clientX, y: params.event.clientY};
        var edge = window.network.body.data.edges.get(params.edge);
        if (edge) {
            var html = buildEdgeTooltip(edge.from, edge.to);
            showTooltip(html, pos.x, pos.y);
        }
    });

    window.network.on('blurEdge', function(params) {
        hideTooltipDelayed();
    });

    window.network.on('click', function(params) {
        if (params.nodes.length === 0 && params.edges.length === 0) {
            tooltip.style.display = 'none';
        }
    });
}

setTimeout(attachNetworkEvents, 800);
</script>
"""


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
        color    = _node_color(G, node, analysis, config)
        in_fb    = node in analysis.get("feedback_nodes", set())
        has_expr = node in mega_lookup

        # Nodes with expression data get a white dashed border ring
        border_color = "#ffffff" if has_expr else (WONG["yellow"] if in_fb else color)
        border_width = 3 if has_expr or in_fb else 1

        net.add_node(
            node, label=node,
            color={
                "background": color,
                "border":     border_color,
                "highlight":  {"background": "#ffffff", "border": "#000000"},
                "hover":      {"background": "#ffffff", "border": "#000000"},
            },
            size=_node_size(G, node, config),
            title="",  # We use custom tooltip, not vis.js title
            borderWidth=border_width,
            borderWidthSelected=4,
            font={"color": t["font_color"], "size": 11, "face": "monospace"},
        )

    for u, v, data in G.edges(data=True):
        rels  = data.get("relationships", ["no_effect"])
        count = data.get("count", 1)
        is_fb = (u, v) in analysis.get("feedback_edges", set())
        net.add_edge(
            u, v,
            color={"color": _edge_color(rels, config), "highlight": "#ffffff", "hover": "#ffffff"},
            title="",  # custom tooltip handles this
            width=1.2 + (count * 0.25),
            arrows={"to": {"enabled": True, "scaleFactor": 0.5}},
            dashes=is_fb,
            smooth={"type": "curvedCW", "roundness": 0.15},
        )

    net.set_options("""
    {
      "interaction": {
        "hover": true,
        "tooltipDelay": 99999,
        "navigationButtons": false,
        "keyboard": {"enabled": true},
        "multiselect": true,
        "zoomView": true,
        "hoverConnectedEdges": false
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

    # Store network reference
    raw = raw.replace("var network = new vis.Network(", "window.network = new vis.Network(")

    # Build data JS
    node_data_js, edge_data_js = _build_node_data_js(G, analysis, config, mega_lookup)

    # Build custom tooltip JS and inject data
    tooltip_js = _build_custom_tooltip_js(config, theme)
    tooltip_js = tooltip_js.replace('"NODE_DATA_PLACEHOLDER"', node_data_js)
    tooltip_js = tooltip_js.replace('"EDGE_DATA_PLACEHOLDER"', edge_data_js)

    # Legend button
    sb_bg  = "#0f1525" if theme == "dark" else "#ffffff"
    sb_bdr = "#1e2d4a" if theme == "dark" else "#e2e8f0"
    sb_txt = "#e2e8f0" if theme == "dark" else "#0f172a"
    mut    = "#475569" if theme == "dark" else "#64748b"

    legend = (
        "<div id='legend-container' style='position:fixed;bottom:20px;left:20px;z-index:9998'>"
        "<div id='legend-panel' style='display:none;margin-bottom:6px;"
        "background:" + sb_bg + ";border:1px solid " + sb_bdr + ";"
        "border-radius:10px;padding:14px 18px;font-family:monospace;"
        "font-size:12px;color:" + sb_txt + ";box-shadow:0 4px 20px rgba(0,0,0,0.3);min-width:230px'>"
        "<b style='color:" + WONG["sky_blue"] + "'>Lens GRN Legend</b><br/><br/>"
        "<b style='color:" + mut + ";font-size:10px'>NODE - ROLE</b><br/>"
        "<span style='color:" + config["color_regulator"] + "'>&#9679;</span> Regulator only<br/>"
        "<span style='color:" + config["color_target"] + "'>&#9679;</span> Target only<br/>"
        "<span style='color:" + config["color_both"] + "'>&#9679;</span> Regulator &amp; Target<br/>"
        "<span style='color:" + config["color_selfloop"] + "'>&#9679;</span> Self-regulatory loop<br/>"
        "<span style='color:#ffffff'>&#9711;</span> White border = has expression data<br/><br/>"
        "<b style='color:" + mut + ";font-size:10px'>EDGE - RELATIONSHIP</b><br/>"
        "<span style='color:" + config["color_activating"] + "'>&#9654;</span> Activating (Pert x Effect)<br/>"
        "<span style='color:" + config["color_inhibiting"] + "'>&#9654;</span> Inhibiting (Pert x Effect)<br/>"
        "<span style='color:" + config["color_noeffect"] + "'>&#9654;</span> No effect<br/>"
        "<span style='border:1px dashed #888;display:inline-block;width:20px'>&nbsp;</span> Dashed = feedback loop<br/><br/>"
        "<span style='color:" + mut + ";font-size:10px'>"
        "Hover edge/node for details &amp; links<br/>"
        "Wong (2011) color-blind safe palette"
        "</span>"
        "</div>"
        "<button onclick='toggleLegend()' style='"
        "background:" + sb_bg + ";border:1px solid " + sb_bdr + ";"
        "border-radius:8px;padding:7px 14px;font-family:monospace;font-size:12px;"
        "color:" + WONG["sky_blue"] + ";cursor:pointer;"
        "box-shadow:0 4px 16px rgba(0,0,0,0.3);display:flex;align-items:center;gap:6px'>"
        "&#128300; Legend <span id='legend-arrow'>&#9650;</span>"
        "</button>"
        "</div>"
        "<script>"
        "function toggleLegend(){"
        "var p=document.getElementById('legend-panel');"
        "var a=document.getElementById('legend-arrow');"
        "if(p.style.display==='none'){p.style.display='block';a.innerHTML='&#9660;'}"
        "else{p.style.display='none';a.innerHTML='&#9650;'}}"
        "</script>"
    )

    # Compact zoom
    zoom = (
        "<div style='position:fixed;bottom:20px;right:20px;z-index:9998;"
        "display:flex;align-items:center;"
        "background:" + sb_bg + ";border:1px solid " + sb_bdr + ";"
        "border-radius:8px;overflow:hidden;box-shadow:0 4px 16px rgba(0,0,0,0.3)'>"
        "<button onclick='zoomNet(-0.3)' style='background:transparent;border:none;"
        "border-right:1px solid " + sb_bdr + ";color:" + sb_txt + ";"
        "font-size:18px;padding:6px 14px;cursor:pointer;line-height:1'>&#8722;</button>"
        "<button onclick='zoomNet(0)' style='background:transparent;border:none;"
        "border-right:1px solid " + sb_bdr + ";color:#64748b;"
        "font-size:14px;padding:6px 10px;cursor:pointer'>&#8635;</button>"
        "<button onclick='zoomNet(0.3)' style='background:transparent;border:none;"
        "color:" + sb_txt + ";font-size:18px;padding:6px 14px;cursor:pointer;line-height:1'>&#43;</button>"
        "</div>"
        "<script>"
        "function zoomNet(d){"
        "if(!window.network)return;"
        "if(d===0){window.network.fit();return;}"
        "window.network.moveTo({scale:Math.max(0.05,Math.min(5,window.network.getScale()+d))});}"
        "</script>"
    )

    # CSS — hide vis.js native tooltip completely
    css = (
        "<style>"
        "div.vis-tooltip { display: none !important; }"
        "body { margin:0; overflow:hidden; }"
        "</style>"
    )

    raw = raw.replace("</head>", css + "\n</head>")
    raw = raw.replace("</body>", tooltip_js + legend + zoom + "\n</body>")

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
    print("  Nodes: " + str(G.number_of_nodes()) + " | Edges: " + str(G.number_of_edges()))
    print("  Regs: " + str(regs) + " | Targets: " + str(tgts) + " | Both: " + str(both))
    print("  Feedback: " + str(len(analysis.get("feedback_loops",[]))) +
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
    output = visualize(G, analysis, CONFIG, mega_lookup)
    webbrowser.open("file://" + os.path.abspath(output))


if __name__ == "__main__":
    main()