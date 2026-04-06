"""
=============================================================================
Lens GRN Explorer — Dash + Cytoscape
=============================================================================
Run:  python app.py
Open: http://127.0.0.1:8050
=============================================================================
"""

import os, copy, json
import pandas as pd
import networkx as nx
import dash
from dash import dcc, html, Input, Output, State, callback_context, ALL
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto

from grn_network import (
    CONFIG, THEMES, WONG,
    load_data, load_external_data, filter_data,
    build_graph, analyze_graph, build_cytoscape_elements, stage_numeric,
)

cyto.load_extra_layouts()

# =============================================================================
# DATA SOURCES
# =============================================================================

DATA_SOURCES = {
    "expression": {
        "label":      "Expression & Enrichment",
        "icon":       "📊",
        "path":       "data/MegaTable April 24 2024 for Microarray and RNA Seq Sent to Murali (1).xls",
        "symbol_col": "Symbol",
        "meta_cols": {
            "entrez":      "Entrez",
            "uniprot":     "UNIPROT",
            "description": "Gene_description",
        },
        "sections": [
            {
                "title":        "Microarray Expression",
                "description":  "Raw expression counts — fiber cells vs epithelium.",
                "color_values": False,
                "columns": {
                    "Fiber E12 (Beebe)": "Beebe_E12_exp_Fiber",
                    "Epi E12 (Beebe)":   "Beebe_E12_exp_Epi",
                    "Fiber P13 (Naka)":  "Naka_P13_fiber_exp",
                    "Epi P13 (Naka)":    "Naka_P13_epi_exp",
                },
            },
            {
                "title":        "Microarray Enrichment",
                "description":  "log2 fold change fiber vs epithelium. Positive = fiber-enriched.",
                "color_values": True,
                "columns": {
                    "Fiber E12": "Beebe_E12_Fiber_enr",
                    "Epi E12":   "Beebe_E12_Epi_enr",
                    "Fiber P13": "Naka_P13_fiber_enr",
                    "Epi P13":   "Naka_P13_epi_enr",
                },
            },
            {
                "title":        "RNA-seq Enrichment",
                "description":  "Fiber cell (FC) vs Lens epithelial cell (LEC) enrichment across developmental stages.",
                "color_values": True,
                "columns": {
                    "FC E14":   "enr_FC_E14_Cv",  "LEC E14":  "enr_LEC_E14_Cv",
                    "FC E16":   "enr_FC_E16_Cv",  "LEC E16":  "enr_LEC_E16_Cv",
                    "FC E18":   "enr_FC_E18_Cv",  "LEC E18":  "enr_LEC_E18_Cv",
                    "FC P0":    "enr_FC_P0_Cv",   "LEC P0":   "enr_LEC_P0_Cv",
                    "FC P0 R":  "enr_FC_P0_Rob",  "LEC P0 R": "enr_LEC_P0_Rob",
                    "FC 3Mo":   "enr_FC_3Mo",     "LEC 3Mo":  "enr_LEC_3Mo",
                    "FC 6Mo":   "enr_FC_6Mo",     "LEC 6Mo":  "enr_LEC_6Mo",
                    "FC 2Y":    "enr_FC_2Y",      "LEC 2Y":   "enr_LEC_2Y",
                },
            },
        ],
    },
}

# =============================================================================
# Startup
# =============================================================================

print("\n[APP] Starting...")
BASE_DF  = load_data(CONFIG)
EXT_DATA = load_external_data(DATA_SOURCES)

ALL_STAGES     = sorted(BASE_DF["stage"].dropna().unique(), key=stage_numeric)
ALL_REGULATORS = sorted(BASE_DF["regulator"].dropna().unique())
ALL_TARGETS    = sorted(BASE_DF["target"].dropna().unique())
print("[APP] Ready\n")

def stage_opts(s): return [{"label": x, "value": x} for x in s]
def gene_opts(g):  return [{"label": x, "value": x} for x in g]

# =============================================================================
# Stylesheet — labels BELOW nodes, clean organic look
# =============================================================================

def build_stylesheet(theme="light"):
    t = THEMES[theme]
    return [
        {"selector": "node", "style": {
            "label":              "data(label)",
            "font-size":          "9px",
            "font-family":        "monospace",
            "font-weight":        "normal",
            # ── Label BELOW the node, no heavy outline ──
            "text-valign":        "bottom",
            "text-halign":        "center",
            "text-margin-y":      "5px",
            "text-wrap":          "none",
            "color":              t["text"],
            "text-outline-width": "0px",
            # ── Node size ──
            "width":  "mapData(deg, 1, 50, 20, 60)",
            "height": "mapData(deg, 1, 50, 20, 60)",
            "border-width": "2px",
            "cursor": "pointer",
        }},
        {"selector": "node.regulator",
         "style": {"background-color": WONG["sky_blue"], "border-color": WONG["sky_blue"]}},
        {"selector": "node.target",
         "style": {"background-color": WONG["orange"],   "border-color": WONG["orange"]}},
        {"selector": "node.both",
         "style": {"background-color": WONG["pink"],     "border-color": WONG["pink"]}},
        {"selector": "node.selfloop",
         "style": {"background-color": WONG["yellow"],   "border-color": WONG["yellow"]}},
        {"selector": "node.feedback",
         "style": {"border-width": "4px", "border-color": WONG["yellow"]}},
        {"selector": "node:selected",
         "style": {"border-width": "4px", "border-color": "#1d4ed8",
                   "overlay-color": "#1d4ed8", "overlay-opacity": 0.1}},
        {"selector": "node:grabbed",
         "style": {"border-width": "3px", "border-color": "#1d4ed8"}},
        {"selector": "edge", "style": {
            "curve-style":        "bezier",
            "target-arrow-shape": "triangle",
            "arrow-scale":        0.8,
            "width":              "mapData(count, 1, 10, 1.5, 5)",
            "opacity":            0.75,
            "cursor":             "pointer",
        }},
        {"selector": "edge.activating",
         "style": {"line-color": WONG["green"],      "target-arrow-color": WONG["green"]}},
        {"selector": "edge.inhibiting",
         "style": {"line-color": WONG["vermillion"], "target-arrow-color": WONG["vermillion"]}},
        {"selector": "edge.no_effect",
         "style": {"line-color": "#94a3b8",          "target-arrow-color": "#94a3b8"}},
        {"selector": "edge.feedback-edge",
         "style": {"line-style": "dashed", "line-dash-pattern": [6, 3]}},
        {"selector": "edge:selected",
         "style": {"opacity": 1, "width": 4,
                   "overlay-color": "#1d4ed8", "overlay-opacity": 0.12}},
    ]

# =============================================================================
# App
# =============================================================================

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="Lens GRN Explorer",
    suppress_callback_exceptions=True,
)

GLOBAL_CSS = """
    body { margin:0; font-family:sans-serif; overflow:hidden; }

    /* Dropdown text fix */
    .Select-control { background:#fff !important; border-color:#cbd5e1 !important; }
    .Select-value-label,
    .Select--single .Select-value .Select-value-label { color:#0f172a !important; }
    .Select-placeholder  { color:#94a3b8 !important; }
    .Select-input>input  { color:#0f172a !important; }
    .Select-option       { color:#0f172a !important; background:#fff !important; }
    .Select-option.is-focused  { background:#f1f5f9 !important; }
    .Select-option.is-selected { background:#dbeafe !important; }
    .Select-menu-outer   { z-index:9999 !important; }
    [class*="singleValue"]{ color:#0f172a !important; }
    [class*="placeholder"]{ color:#94a3b8 !important; }
    [class*="option"]     { color:#0f172a !important; background:#fff !important; }
    [class*="menu"]       { z-index:9999 !important; background:#fff !important; }
    [class*="Input"] input{ color:#0f172a !important; }

    /* Hover tooltip */
    #hover-tooltip {
        position: fixed;
        z-index: 99999;
        pointer-events: none;
        border-radius: 10px;
        padding: 10px 14px;
        font-family: monospace;
        font-size: 12px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.13);
        max-width: 260px;
        min-width: 160px;
        display: none;
        top: 70px;
        right: 320px;
    }

    /* Legend overlay — bottom left of graph */
    #legend-overlay {
        position: absolute;
        bottom: 24px;
        left: 16px;
        z-index: 1000;
        font-family: monospace;
    }
    #legend-panel {
        display: none;
        margin-bottom: 6px;
        padding: 14px 16px;
        border-radius: 10px;
        font-size: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.13);
        min-width: 220px;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        color: #0f172a;
    }
    /* The legend button is a plain div with onclick */
    #legend-btn {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 7px 14px;
        border-radius: 8px;
        font-family: monospace;
        font-size: 12px;
        cursor: pointer;
        box-shadow: 0 3px 14px rgba(0,0,0,0.15);
        background: #ffffff;
        border: 1px solid #e2e8f0;
        color: #0369a1;
        user-select: none;
    }
    #legend-btn:hover { background: #f0f9ff; }

    /* Zoom controls — bottom right of graph */
    #zoom-controls {
        position: absolute;
        bottom: 24px;
        right: 16px;
        z-index: 1000;
        display: flex;
        align-items: center;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 3px 14px rgba(0,0,0,0.13);
        background: #ffffff;
        border: 1px solid #e2e8f0;
    }
    /* Each zoom button is a plain span with onclick */
    .zoom-btn {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 38px;
        height: 36px;
        font-size: 18px;
        cursor: pointer;
        color: #334155;
        user-select: none;
        transition: background 0.12s;
    }
    .zoom-btn:hover { background: #f1f5f9; }
    .zoom-sep { width:1px; height:28px; background:#e2e8f0; }

    /* Collapsible sections */
    .sec-header {
        display:flex; justify-content:space-between; align-items:center;
        padding:8px 10px; border-radius:6px; cursor:pointer;
        user-select:none; margin-bottom:2px; transition:background 0.15s;
    }
    .sec-light { background:#f1f5f9; border:1px solid #e2e8f0; }
    .sec-light:hover { background:#e2e8f0; }
    .sec-dark  { background:#1a2540; border:1px solid #1e2d4a; }
    .sec-dark:hover  { background:#1e3055; }

    ::-webkit-scrollbar { width:5px; height:5px; }
    ::-webkit-scrollbar-thumb { background:#cbd5e1; border-radius:3px; }
"""

OVERLAY_JS = """
<script>
// ── 1. Lock dragged nodes in place ───────────────────────────────
(function lockNodes() {
  function tryAttach() {
    var el = document.getElementById('cytoscape-graph');
    if (!el) { setTimeout(tryAttach, 600); return; }
    var poll = setInterval(function() {
      try {
        var cy = el._cyreg && el._cyreg.cy;
        if (!cy) return;
        clearInterval(poll);
        cy.on('dragfree', 'node', function(e) { e.target.lock(); });
      } catch(e) {}
    }, 400);
  }
  document.addEventListener('DOMContentLoaded', tryAttach);
})();

// ── 2. Legend toggle ─────────────────────────────────────────────
function grn_toggleLegend() {
  var p = document.getElementById('legend-panel');
  var a = document.getElementById('legend-arrow');
  if (!p) return;
  if (p.style.display === 'none' || p.style.display === '') {
    p.style.display = 'block';
    if (a) a.textContent = '▼';
  } else {
    p.style.display = 'none';
    if (a) a.textContent = '▲';
  }
}

// ── 3. Zoom helpers ──────────────────────────────────────────────
function _grn_cy() {
  var el = document.getElementById('cytoscape-graph');
  if (!el) return null;
  try { return el._cyreg && el._cyreg.cy; } catch(e) { return null; }
}
function grn_zoomOut() {
  var cy = _grn_cy(); if (!cy) return;
  cy.zoom({ level: Math.max(0.05, cy.zoom() - 0.3), renderedPosition: { x: cy.width()/2, y: cy.height()/2 } });
}
function grn_zoomFit() {
  var cy = _grn_cy(); if (!cy) return;
  cy.fit(undefined, 30);
}
function grn_zoomIn() {
  var cy = _grn_cy(); if (!cy) return;
  cy.zoom({ level: Math.min(3, cy.zoom() + 0.3), renderedPosition: { x: cy.width()/2, y: cy.height()/2 } });
}

// ── 4. Attach DOM click listeners ────────────────────────────────
(function bindOverlayButtons() {
  function attach() {
    var legendBtn = document.getElementById('legend-btn');
    var zoomOutBtn = document.getElementById('zoom-out-btn');
    var zoomFitBtn = document.getElementById('zoom-fit-btn');
    var zoomInBtn = document.getElementById('zoom-in-btn');

    if (legendBtn && !legendBtn.dataset.bound) {
      legendBtn.addEventListener('click', grn_toggleLegend);
      legendBtn.dataset.bound = '1';
    }
    if (zoomOutBtn && !zoomOutBtn.dataset.bound) {
      zoomOutBtn.addEventListener('click', grn_zoomOut);
      zoomOutBtn.dataset.bound = '1';
    }
    if (zoomFitBtn && !zoomFitBtn.dataset.bound) {
      zoomFitBtn.addEventListener('click', grn_zoomFit);
      zoomFitBtn.dataset.bound = '1';
    }
    if (zoomInBtn && !zoomInBtn.dataset.bound) {
      zoomInBtn.addEventListener('click', grn_zoomIn);
      zoomInBtn.dataset.bound = '1';
    }
  }

  document.addEventListener('DOMContentLoaded', function() {
    attach();
    setInterval(attach, 1000);
  });
})();
</script>
"""


app.index_string = (
    "<!DOCTYPE html><html><head>"
    "{%metas%}<title>{%title%}</title>{%favicon%}{%css%}"
    "<style>" + GLOBAL_CSS + "</style>"
    "</head><body>"
    "{%app_entry%}"
    "<footer>{%config%}{%scripts%}{%renderer%}</footer>"
    + OVERLAY_JS +
    "</body></html>"
)

# =============================================================================
# Graph overlays — legend + zoom
# Key fix: use html.Div with id + onclick attribute (NOT html.Button)
# html.Button swallows onClick; html.Div passes it straight to the DOM
# =============================================================================

def build_graph_overlays(theme="light"):
    t     = THEMES[theme]
    muted = t["muted"]
    txt   = t["text"]

    legend_panel = html.Div(id="legend-panel", children=[
        html.Div("Lens GRN Legend",
                 style={"fontWeight":"bold","fontSize":"13px","color":WONG["sky_blue"],"marginBottom":"10px"}),
        html.Div("NODE — ROLE",
                 style={"fontSize":"9px","fontWeight":"bold","color":muted,
                        "textTransform":"uppercase","letterSpacing":"0.08em","marginBottom":"5px"}),
        *[html.Div([html.Span("●",style={"color":c,"marginRight":"6px"}),l],
                   style={"fontSize":"11px","marginBottom":"3px","color":txt})
          for c,l in [(WONG["sky_blue"],"Regulator only"),(WONG["orange"],"Target only"),
                      (WONG["pink"],"Regulator & Target"),(WONG["yellow"],"Self-regulatory loop")]],
        html.Br(),
        html.Div("EDGE — RELATIONSHIP",
                 style={"fontSize":"9px","fontWeight":"bold","color":muted,
                        "textTransform":"uppercase","letterSpacing":"0.08em","marginBottom":"5px"}),
        *[html.Div([html.Span("━▶",style={"color":c,"marginRight":"6px"}),l],
                   style={"fontSize":"11px","marginBottom":"3px","color":txt})
          for c,l in [(WONG["green"],"Activating"),(WONG["vermillion"],"Inhibiting"),("#94a3b8","No effect")]],
        html.Br(),
        html.Div("Yellow border = feedback loop",style={"fontSize":"10px","color":muted}),
        html.Div("Hover = quick info  |  Click = full details",style={"fontSize":"10px","color":muted,"marginTop":"2px"}),
        html.Div("Drag node → locks in place",style={"fontSize":"10px","color":muted,"marginTop":"2px"}),
        html.Div("Wong (2011) color-blind safe",style={"fontSize":"10px","color":muted,"marginTop":"4px","fontStyle":"italic"}),
    ])

    legend_btn = html.Div(
        id="legend-btn",
        children=[html.Span("🔬 Legend "), html.Span("▲", id="legend-arrow")],
        n_clicks=0,
    )

    zoom_box = html.Div(
        id="zoom-controls",
        children=[
            html.Span("−", className="zoom-btn", id="zoom-out-btn", n_clicks=0),
            html.Div(className="zoom-sep"),
            html.Span("⟳", className="zoom-btn", id="zoom-fit-btn",
                      style={"fontSize":"16px"}, n_clicks=0),
            html.Div(className="zoom-sep"),
            html.Span("+", className="zoom-btn", id="zoom-in-btn", n_clicks=0),
        ]
    )

    legend_overlay = html.Div(id="legend-overlay", children=[legend_panel, legend_btn])
    return legend_overlay, zoom_box
# =============================================================================
# Right panel builders
# =============================================================================

def empty_panel(theme="light"):
    t = THEMES[theme]
    return html.Div([
        html.Div("👆 Click any node or edge",
                 style={"color":t["muted"],"fontSize":"13px","fontFamily":"monospace",
                        "textAlign":"center","marginTop":"50px"}),
        html.Div("to see full details, expression data,",
                 style={"color":t["muted"],"fontSize":"12px","textAlign":"center","marginTop":"6px"}),
        html.Div("enrichment values and PubMed links.",
                 style={"color":t["muted"],"fontSize":"12px","textAlign":"center"}),
        html.Div("─"*26,
                 style={"color":t["muted"],"fontSize":"10px","textAlign":"center",
                        "marginTop":"16px","opacity":"0.3"}),
        html.Div("Hover for quick summary",
                 style={"color":t["muted"],"fontSize":"11px","textAlign":"center",
                        "marginTop":"6px","fontStyle":"italic"}),
    ])


def collapsible(sec_id, title, icon, body, open_default, theme):
    t       = THEMES[theme]
    arrow   = "▼" if open_default else "▶"
    display = "block" if open_default else "none"
    cls     = "sec-header sec-dark" if theme=="dark" else "sec-header sec-light"
    return html.Div([
        html.Div(id={"type":"sec-hdr","index":sec_id}, className=cls,
            children=[
                html.Span([html.Span(icon+" "),
                           html.Span(title,style={"fontWeight":"600","fontSize":"12px","color":t["text"]})]),
                html.Span(arrow,id={"type":"sec-arr","index":sec_id},
                          style={"color":t["muted"],"fontSize":"10px","marginLeft":"6px"}),
            ],
        ),
        html.Div(id={"type":"sec-body","index":sec_id},
                 style={"display":display}, children=body),
    ], style={"marginBottom":"6px"})


def data_table(rows, color_values, theme):
    t = THEMES[theme]
    trs = []
    for lbl, val in rows:
        if val is None: continue
        if color_values and isinstance(val,(int,float)):
            pct = min(abs(val)/10*100,100)
            vc  = WONG["green"] if val>0 else (WONG["vermillion"] if val<0 else t["muted"])
            trs.append(html.Tr([
                html.Td(lbl,style={"padding":"3px 6px","color":t["muted"],"fontSize":"11px",
                                   "fontFamily":"monospace","width":"50%"}),
                html.Td([
                    html.Span(str(val),style={"color":vc,"fontSize":"11px","fontFamily":"monospace",
                                              "marginRight":"5px","minWidth":"40px","display":"inline-block"}),
                    html.Div(style={"display":"inline-block","height":"6px","width":str(pct)+"%",
                                    "maxWidth":"50px","background":vc,"borderRadius":"3px","verticalAlign":"middle"}),
                ],style={"padding":"3px 6px"}),
            ]))
        else:
            trs.append(html.Tr([
                html.Td(lbl,style={"padding":"3px 6px","color":t["muted"],"fontSize":"11px","fontFamily":"monospace"}),
                html.Td(str(val),style={"padding":"3px 6px","color":t["text"],"fontSize":"11px",
                                        "fontFamily":"monospace","textAlign":"right"}),
            ]))
    if not trs:
        return html.Div("No values recorded.",style={"color":t["muted"],"fontSize":"11px","fontStyle":"italic","padding":"4px"})
    return html.Table(trs,style={"width":"100%","borderCollapse":"collapse","background":t["row_bg"],"borderRadius":"6px"})


def node_panel(node_id, G, analysis, ext_data, theme="light"):
    t      = THEMES[theme]
    d      = G.nodes.get(node_id,{})
    is_reg = d.get("is_reg",False); is_tgt = d.get("is_tgt",False)
    out_d  = d.get("out_degree",0); in_d   = d.get("in_degree",0)
    cent   = d.get("deg_centrality",0)
    in_fb  = node_id in analysis.get("feedback_nodes",set())
    is_sl  = node_id in set(e[0] for e in analysis.get("self_loops",[]))

    if is_reg and is_tgt: role,rc = "Regulator & Target",WONG["pink"]
    elif is_reg:          role,rc = "Regulator",WONG["sky_blue"]
    else:                 role,rc = "Target",WONG["orange"]

    mut = {"color":t["muted"],"fontSize":"12px"}
    def row(l,v,c=None):
        return html.Div(style={"display":"flex","justifyContent":"space-between","marginBottom":"4px"},
            children=[html.Span(l,style=mut),
                      html.Span(str(v),style={"fontFamily":"monospace","fontSize":"12px","color":c or WONG["sky_blue"]})])

    net_body = html.Div([
        html.Span(role,style={"background":rc,"color":"#fff","padding":"2px 10px","borderRadius":"12px",
                              "fontSize":"11px","fontFamily":"monospace","display":"inline-block","marginBottom":"10px"}),
        row("Regulates (out-edges)",out_d), row("Regulated by (in-edges)",in_d),
        row("Degree centrality",round(cent,4)),
    ]+([html.Div("⚡ Part of feedback loop",style={"color":WONG["orange"],"fontSize":"12px","marginTop":"4px"})] if in_fb else [])
     +([html.Div("🔄 Self-regulatory loop",style={"color":WONG["yellow"],"fontSize":"12px","marginTop":"4px"})] if is_sl else []),
        style={"padding":"2px 4px"})

    sections = [collapsible("net_"+node_id,"Network Info","🔗",net_body,True,theme)]

    for src_key,src_cfg in DATA_SOURCES.items():
        lookup    = ext_data.get(src_key,{})
        gene_data = lookup.get(node_id)
        icon      = src_cfg.get("icon","📋")
        lbl       = src_cfg["label"]
        if gene_data is None:
            body = html.Div("No "+lbl+" data for "+node_id+".",
                            style={"color":t["muted"],"fontSize":"11px","fontStyle":"italic","padding":"6px 4px"})
        else:
            entrez  = gene_data.get("_meta_entrez","")
            uniprot = gene_data.get("_meta_uniprot","")
            desc    = gene_data.get("_meta_description","")
            children = []
            links = []
            if entrez and entrez not in ("nan",""):
                links.append(html.A("NCBI Gene ↗",href="https://www.ncbi.nlm.nih.gov/gene/"+str(entrez),
                    target="_blank",style={"color":WONG["sky_blue"],"fontSize":"11px","marginRight":"10px","textDecoration":"underline"}))
            if uniprot and uniprot not in ("nan",""):
                links.append(html.A("UniProt ↗",href="https://www.uniprot.org/uniprot/"+str(uniprot),
                    target="_blank",style={"color":WONG["sky_blue"],"fontSize":"11px","textDecoration":"underline"}))
            if links: children.append(html.Div(links,style={"marginBottom":"6px","marginTop":"4px"}))
            if desc and desc not in ("nan",""):
                children.append(html.Div(str(desc)[:100]+("..." if len(str(desc))>100 else ""),
                    style={"color":t["muted"],"fontSize":"10px","fontStyle":"italic","marginBottom":"8px"}))
            for sec in src_cfg.get("sections",[]):
                rows_data  = [(dn,gene_data.get(dn)) for dn in sec["columns"]]
                has_any    = any(v is not None for _,v in rows_data)
                color_vals = sec.get("color_values",True)
                children.append(html.Div([
                    html.Div(sec["title"],style={"fontSize":"10px","color":t["muted"],"fontWeight":"bold",
                                                 "textTransform":"uppercase","letterSpacing":"0.06em",
                                                 "marginTop":"10px","marginBottom":"2px"}),
                    html.Div(sec.get("description",""),style={"fontSize":"10px","color":t["muted"],
                                                              "marginBottom":"4px","fontStyle":"italic"})
                    if sec.get("description") else html.Div(),
                    data_table(rows_data,color_vals,theme) if has_any else
                    html.Div("No values.",style={"color":t["muted"],"fontSize":"11px","fontStyle":"italic","padding":"2px 4px"}),
                ]))
            if any(s.get("color_values") for s in src_cfg.get("sections",[])):
                children.append(html.Div([
                    html.Span("■ ",style={"color":WONG["green"]}),
                    html.Span("Positive  ",style={"fontSize":"10px","color":t["muted"]}),
                    html.Span("■ ",style={"color":WONG["vermillion"]}),
                    html.Span("Negative",style={"fontSize":"10px","color":t["muted"]}),
                ],style={"marginTop":"8px"}))
            body = html.Div(children,style={"padding":"0 2px"})
        sections.append(collapsible(src_key+"_"+node_id,lbl,icon,body,False,theme))

    return html.Div([
        html.Div(node_id,style={"fontWeight":"bold","fontSize":"16px","color":WONG["sky_blue"],
                                 "fontFamily":"monospace","marginBottom":"12px"}),
        *sections,
    ])


def edge_panel(edge_data, theme="light"):
    t     = THEMES[theme]
    src   = edge_data.get("source",""); tgt = edge_data.get("target","")
    rel   = edge_data.get("rel","no_effect")
    perts = edge_data.get("perts",[]); effs = edge_data.get("effs",[])
    stgs  = edge_data.get("stages",[]); count = edge_data.get("count",1)
    pmids = edge_data.get("pmids",[]); is_fb = edge_data.get("feedback",False)
    rc   = WONG["green"] if rel=="activating" else (WONG["vermillion"] if rel=="inhibiting" else "#94a3b8")
    rlbl = "▲ Activating" if rel=="activating" else ("▼ Inhibiting" if rel=="inhibiting" else "○ No effect")
    mut  = {"color":t["muted"],"fontSize":"12px"}
    def row(l,v):
        return html.Div(style={"display":"flex","justifyContent":"space-between","marginBottom":"5px"},
            children=[html.Span(l,style=mut),
                      html.Span(v,style={"fontFamily":"monospace","color":t["text"],"fontSize":"12px"})])
    net_body = html.Div([
        html.Span(rlbl,style={"background":rc,"color":"#fff","padding":"2px 10px","borderRadius":"12px",
                              "fontSize":"11px","fontFamily":"monospace","display":"inline-block","marginBottom":"10px"}),
        row("Perturbation",", ".join(perts) if perts else "—"),
        row("Raw effect",", ".join(effs) if effs else "—"),
        row("Stage(s)",", ".join(stgs) if stgs else "—"),
        row("Evidence count",str(count)),
    ]+([html.Div("⚡ Part of feedback loop",style={"color":WONG["orange"],"fontSize":"12px","marginTop":"4px"})] if is_fb else []),
        style={"padding":"2px 4px"})
    pmid_links = [
        html.A("📄 PMID "+pmid+" ↗",href="https://pubmed.ncbi.nlm.nih.gov/"+pmid+"/",target="_blank",
               style={"color":WONG["sky_blue"],"display":"block","fontSize":"12px",
                      "marginBottom":"5px","textDecoration":"underline"})
        for pmid in pmids
    ]
    pmid_body = html.Div(
        pmid_links or [html.Div("No PubMed IDs for this edge.",
                                style={"color":t["muted"],"fontSize":"11px","fontStyle":"italic","padding":"4px"})],
        style={"padding":"2px 4px"})
    return html.Div([
        html.Div([
            html.Span(src,style={"fontWeight":"bold","color":WONG["sky_blue"],"fontFamily":"monospace","fontSize":"14px"}),
            html.Span(" → ",style={"color":t["muted"],"fontSize":"14px"}),
            html.Span(tgt,style={"fontWeight":"bold","color":WONG["orange"],"fontFamily":"monospace","fontSize":"14px"}),
        ],style={"marginBottom":"12px"}),
        collapsible("edge_net_"+src+"_"+tgt,"Network Info",     "🔗",net_body, True, theme),
        collapsible("edge_pm_"+src+"_"+tgt, "PubMed References","📄",pmid_body,False,theme),
    ])

# =============================================================================
# Sidebar + Layout
# =============================================================================

def build_sidebar(theme="light"):
    t     = THEMES[theme]
    label = {"color":t["muted"],"fontSize":"10px","fontWeight":"bold",
              "letterSpacing":"0.1em","fontFamily":"monospace","textTransform":"uppercase"}
    sub   = {"color":t["muted"],"fontSize":"12px","marginTop":"6px"}
    hr    = {"borderColor":t["sidebar_bdr"],"margin":"4px 0"}
    return html.Div(
        style={"width":"260px","minWidth":"260px","background":t["sidebar_bg"],
               "borderRight":"1px solid "+t["sidebar_bdr"],"padding":"12px 14px",
               "overflowY":"auto","display":"flex","flexDirection":"column",
               "gap":"10px","height":"100vh","boxSizing":"border-box"},
        children=[
            html.Div(style={"display":"flex","justifyContent":"space-between","alignItems":"center"},
            children=[
                html.Div([
                    html.Div("Lens GRN Explorer",
                             style={"color":WONG["sky_blue"],"fontFamily":"monospace","fontWeight":"bold","fontSize":"13px"}),
                    html.Div("Lachke Lab 2016",style={"color":t["muted"],"fontSize":"11px"}),
                ]),
                html.Div(id="theme-toggle",n_clicks=0,
                    style={"cursor":"pointer","border":"1px solid "+t["sidebar_bdr"],
                           "borderRadius":"6px","padding":"4px 8px","textAlign":"center","minWidth":"50px"},
                    children=[
                        html.Div("🌙" if theme=="light" else "☀️",style={"fontSize":"14px"}),
                        html.Div("Dark" if theme=="light" else "Light",
                                 style={"fontSize":"9px","color":t["muted"],"fontFamily":"monospace"}),
                    ]),
            ]),
            html.Hr(style=hr),
            html.Div([
                html.Label("Stage Filter",style=label),
                html.Label("Single stage",style=sub),
                dcc.Dropdown(id="stage-single",options=stage_opts(ALL_STAGES),
                             placeholder="All stages",clearable=True,style={"fontSize":"12px"}),
                html.Label("Stage range",style={**sub,"marginTop":"6px"}),
                html.Div([
                    dcc.Dropdown(id="stage-from",options=stage_opts(ALL_STAGES),
                                 placeholder="From",clearable=True,style={"flex":"1","fontSize":"12px"}),
                    html.Span("→",style={"color":t["muted"],"padding":"0 4px","alignSelf":"center"}),
                    dcc.Dropdown(id="stage-to",options=stage_opts(ALL_STAGES),
                                 placeholder="To",clearable=True,style={"flex":"1","fontSize":"12px"}),
                ],style={"display":"flex","alignItems":"center","gap":"3px"}),
            ]),
            html.Hr(style=hr),
            html.Div([
                html.Label("Gene Filter",style=label),
                html.Label("Regulator",style=sub),
                dcc.Dropdown(id="filter-regulator",options=gene_opts(ALL_REGULATORS),
                             placeholder="All regulators",clearable=True,style={"fontSize":"12px"}),
                html.Label("Target",style={**sub,"marginTop":"6px"}),
                dcc.Dropdown(id="filter-target",options=gene_opts(ALL_TARGETS),
                             placeholder="All targets",clearable=True,style={"fontSize":"12px"}),
            ]),
            html.Hr(style=hr),
            html.Div([
                html.Label("Relationship Filter",style=label),
                dcc.Checklist(id="relationship-filter",
                    options=[{"label":"  Activating","value":"activating"},
                             {"label":"  Inhibiting","value":"inhibiting"},
                             {"label":"  No effect","value":"no_effect"}],
                    value=["activating","inhibiting","no_effect"],
                    style={"marginTop":"6px"},
                    labelStyle={"display":"block","color":t["text"],"fontSize":"12px","marginBottom":"3px"}),
                html.Div("True relationship = Perturbation × Effect",
                         style={"fontSize":"10px","color":t["muted"],"marginTop":"3px","fontFamily":"monospace"}),
            ]),
            html.Hr(style=hr),
            html.Div([
                html.Label("Display Options",style=label),
                html.Label("Max edges",style=sub),
                dcc.Dropdown(id="max-edges",
                    options=[{"label":"100","value":100},{"label":"300","value":300},
                             {"label":"600","value":600},{"label":"All","value":9999}],
                    value=300,clearable=False,style={"fontSize":"12px"}),
                html.Label("Layout",style={**sub,"marginTop":"6px"}),
                dcc.Dropdown(id="layout-select",
                    options=[
                        {"label":"Barnes Hut",       "value":"barnes_hut"},
                        {"label":"Force Atlas 2",    "value":"force_atlas_2based"},
                        {"label":"Repulsion",        "value":"repulsion"},
                        {"label":"Circle",           "value":"circle"},
                        {"label":"Grid",             "value":"grid"},
                        {"label":"Dagre (hierarchy)","value":"dagre"},
                    ],
                    value="barnes_hut",clearable=False,style={"fontSize":"12px"}),
            ]),
            html.Hr(style=hr),
            html.Div([
                html.Button("▶  Apply Filters",id="apply-btn",
                    style={"width":"100%","padding":"8px","marginBottom":"5px",
                           "background":"rgba(86,180,233,0.12)",
                           "border":"1px solid "+WONG["sky_blue"],"borderRadius":"7px",
                           "color":WONG["sky_blue"],"fontFamily":"monospace","fontSize":"12px","cursor":"pointer"}),
                html.Button("↺  Reset",id="reset-btn",
                    style={"width":"100%","padding":"8px","background":"transparent",
                           "border":"1px solid "+t["sidebar_bdr"],"borderRadius":"7px",
                           "color":t["muted"],"fontFamily":"monospace","fontSize":"12px","cursor":"pointer"}),
            ]),
            html.Hr(style=hr),
            html.Div(id="stats-panel",children=[
                html.Label("Network Stats",style=label),
                html.Div("Apply filters to see stats.",style={"color":t["muted"],"fontSize":"12px","marginTop":"4px"}),
            ]),
        ],
    )


def build_layout(theme="light"):
    t = THEMES[theme]
    legend_overlay, zoom_box = build_graph_overlays(theme)
    return html.Div(
        id="app-container",
        style={"display":"flex","flexDirection":"column","height":"100vh",
               "background":t["bgcolor"],"overflow":"hidden"},
        children=[
            html.Div(
                style={"background":t["topbar_bg"],"borderBottom":"1px solid "+t["topbar_bdr"],
                       "padding":"8px 20px","display":"flex","alignItems":"center","gap":"12px","flexShrink":"0"},
                children=[
                    html.Span("Lens GRN Explorer",
                              style={"color":WONG["sky_blue"],"fontFamily":"monospace","fontWeight":"bold","fontSize":"14px"}),
                    html.Span("Gene Regulatory Network — Lachke Lab 2016",
                              style={"color":t["muted"],"fontSize":"12px"}),
                    html.Div(id="topbar-stats",
                             style={"marginLeft":"auto","fontFamily":"monospace","fontSize":"11px","color":t["muted"]}),
                ],
            ),
            html.Div(style={"display":"flex","flex":"1","overflow":"hidden"},
            children=[
                build_sidebar(theme),
                html.Div(style={"flex":"1","position":"relative","overflow":"hidden"},
                children=[
                    dcc.Loading(id="loading-graph",type="circle",color=WONG["sky_blue"],
                        children=[
                            cyto.Cytoscape(
                                id="cytoscape-graph",
                                elements=[],
                                stylesheet=build_stylesheet(theme),
                                layout={
                                    "name":          "cose",
                                    "animate":       True,
                                    "randomize":     True,
                                    "idealEdgeLength": 100,
                                    "nodeRepulsion": 450000,
                                    "gravity":       0.25,
                                    "numIter":       1000,
                                    "fit":           True,
                                    "padding":       40,
                                },
                                style={"width":"100%","height":"calc(100vh - 46px)","background":t["bgcolor"]},
                                minZoom=0.05, maxZoom=3,
                                userZoomingEnabled=True,
                                userPanningEnabled=True,
                                boxSelectionEnabled=False,
                                autoungrabify=False,
                                autolock=False,
                            )
                        ],
                    ),
                    html.Div(id="hover-tooltip",
                             style={"background":"#ffffff" if theme=="light" else "#0f1525",
                                    "border":"1px solid #e2e8f0" if theme=="light" else "1px solid #1e2d4a",
                                    "color":"#0f172a" if theme=="light" else "#e2e8f0"}),
                    legend_overlay,
                    zoom_box,
                ]),
                html.Div(
                    style={"width":"300px","minWidth":"300px","background":t["panel_bg"],
                           "borderLeft":"1px solid "+t["panel_bdr"],"padding":"14px",
                           "overflowY":"auto","height":"calc(100vh - 46px)","boxSizing":"border-box"},
                    children=[
                        html.Div("Details",style={"fontSize":"10px","fontWeight":"bold","letterSpacing":"0.1em",
                                                   "color":t["muted"],"fontFamily":"monospace",
                                                   "textTransform":"uppercase","marginBottom":"10px"}),
                        html.Div(id="info-panel",children=[empty_panel(theme)]),
                    ],
                ),
            ]),
            dcc.Store(id="theme-store",data=theme),
            dcc.Store(id="graph-store",data={}),
        ],
    )


app.layout = build_layout("light")

# =============================================================================
# Callbacks
# =============================================================================

@app.callback(Output("theme-store","data"),Input("theme-toggle","n_clicks"),State("theme-store","data"),prevent_initial_call=True)
def toggle_theme(n,cur): return "dark" if cur=="light" else "light"

@app.callback(Output("app-container","children"),Input("theme-store","data"),prevent_initial_call=True)
def update_theme(theme): return build_layout(theme).children

@app.callback(
    Output("stage-from","value"),Output("stage-to","value"),Output("stage-single","value"),
    Output("filter-regulator","value"),Output("filter-target","value"),
    Output("relationship-filter","value"),Output("max-edges","value"),Output("layout-select","value"),
    Input("reset-btn","n_clicks"),prevent_initial_call=True,
)
def reset(_): return None,None,None,None,None,["activating","inhibiting","no_effect"],300,"barnes_hut"

@app.callback(
    Output("cytoscape-graph","elements"),Output("cytoscape-graph","stylesheet"),
    Output("cytoscape-graph","layout"),Output("graph-store","data"),
    Output("stats-panel","children"),Output("topbar-stats","children"),
    Input("apply-btn","n_clicks"),
    State("stage-single","value"),State("stage-from","value"),State("stage-to","value"),
    State("filter-regulator","value"),State("filter-target","value"),
    State("relationship-filter","value"),State("max-edges","value"),
    State("layout-select","value"),State("theme-store","data"),
    prevent_initial_call=True,
)
def apply_filters(n,stage_single,stage_from,stage_to,filter_reg,filter_tgt,
                  relationships,max_edges,layout,theme):
    theme = theme or "light"
    cfg   = copy.deepcopy(CONFIG)
    cfg["stage_single"]          = stage_single
    cfg["stage_from"]            = stage_from
    cfg["stage_to"]              = stage_to
    cfg["filter_regulator"]      = filter_reg
    cfg["filter_target"]         = filter_tgt
    cfg["relationships_include"] = relationships or ["activating","inhibiting","no_effect"]
    cfg["max_edges"]             = int(max_edges) if max_edges != 9999 else None

    df = filter_data(BASE_DF.copy(), cfg)
    if len(df)==0:
        return [],[],{"name":"cose","animate":True},{},_stats_empty(theme),""

    G        = build_graph(df)
    analysis = analyze_graph(G, cfg)
    elements = build_cytoscape_elements(G, analysis, cfg, EXT_DATA)

    g_store = {
        "nodes": {n: dict(G.nodes[n]) for n in G.nodes()},
        "edges": {u+"|||"+v:{**dict(data),"stages":[str(s) for s in data.get("stages",[])]}
                  for u,v,data in G.edges(data=True)},
        "feedback_nodes": list(analysis.get("feedback_nodes",set())),
        "feedback_edges": [list(e) for e in analysis.get("feedback_edges",set())],
        "self_loops":     [list(e) for e in analysis.get("self_loops",[])],
        "hub_genes":      analysis.get("hub_genes",[]),
    }

    layout_map = {
        "barnes_hut":         {"name":"cose","animate":True,"randomize":True,"idealEdgeLength":100,
                               "nodeRepulsion":450000,"gravity":0.25,"numIter":1000,"fit":True,"padding":40},
        "force_atlas_2based": {"name":"cose","animate":True,"randomize":True,"idealEdgeLength":70,
                               "nodeRepulsion":650000,"gravity":0.1,"numIter":1500,"fit":True,"padding":40},
        "repulsion":          {"name":"cose","animate":True,"randomize":True,"idealEdgeLength":160,
                               "nodeRepulsion":900000,"gravity":0.05,"numIter":1000,"fit":True,"padding":50},
        "circle":             {"name":"circle","animate":True,"fit":True,"padding":30},
        "grid":               {"name":"grid",  "animate":True,"fit":True,"padding":30},
        "dagre":              {"name":"dagre", "animate":True,"rankDir":"TB","fit":True,"padding":30},
    }
    layout_cfg = layout_map.get(layout or "barnes_hut",
                                {"name":"cose","animate":True,"randomize":True,"fit":True,"padding":40})

    return (elements,build_stylesheet(theme),layout_cfg,
            g_store,_stats_panel(G,analysis,theme),_topbar_stats(G,analysis))


@app.callback(
    Output("hover-tooltip","children"),Output("hover-tooltip","style"),
    Input("cytoscape-graph","mouseoverNodeData"),Input("cytoscape-graph","mouseoverEdgeData"),
    Input("cytoscape-graph","tapNodeData"),Input("cytoscape-graph","tapEdgeData"),
    State("theme-store","data"),prevent_initial_call=True,
)
def update_hover(node_hover,edge_hover,node_tap,edge_tap,theme):
    theme=theme or "light"; ctx=callback_context
    trigger=ctx.triggered[0]["prop_id"] if ctx.triggered else ""
    if "tapNodeData" in trigger or "tapEdgeData" in trigger:
        return "",{"display":"none"}
    base={"position":"fixed","zIndex":"99999","pointerEvents":"none","borderRadius":"10px",
          "padding":"10px 14px","fontFamily":"monospace","fontSize":"12px",
          "boxShadow":"0 4px 24px rgba(0,0,0,0.13)","maxWidth":"260px","minWidth":"160px",
          "top":"70px","right":"320px",
          "background":"#ffffff" if theme=="light" else "#0f1525",
          "border":"1px solid #e2e8f0" if theme=="light" else "1px solid #1e2d4a",
          "color":"#0f172a" if theme=="light" else "#e2e8f0"}
    muted="#64748b"; sky=WONG["sky_blue"]; org=WONG["orange"]
    if "mouseoverNodeData" in trigger and node_hover:
        role_map={"regulator":"Regulator","target":"Target","both":"Regulator & Target","selfloop":"Self-regulatory"}
        role=role_map.get(node_hover.get("role","target"),"—")
        content=[
            html.Div(node_hover.get("id",""),style={"fontWeight":"bold","color":sky,"fontSize":"13px","marginBottom":"5px"}),
            html.Div("Role: "+role,style={"color":muted,"fontSize":"11px"}),
            html.Div("Out-edges: "+str(node_hover.get("out_deg",0)),style={"color":muted,"fontSize":"11px"}),
            html.Div("In-edges: "+str(node_hover.get("in_deg",0)),style={"color":muted,"fontSize":"11px"}),
            html.Div("Click for full details →",style={"color":sky,"fontSize":"10px","marginTop":"5px","fontStyle":"italic"}),
        ]
        return content,{**base,"display":"block"}
    if "mouseoverEdgeData" in trigger and edge_hover:
        rel=edge_hover.get("rel","no_effect")
        rc=WONG["green"] if rel=="activating" else (WONG["vermillion"] if rel=="inhibiting" else muted)
        content=[
            html.Div([html.Span(edge_hover.get("source",""),style={"color":sky,"fontWeight":"bold"}),
                      html.Span(" → ",style={"color":muted}),
                      html.Span(edge_hover.get("target",""),style={"color":org,"fontWeight":"bold"})],
                     style={"marginBottom":"5px","fontSize":"12px"}),
            html.Div([html.Span("Relationship: ",style={"color":muted,"fontSize":"11px"}),
                      html.Span(rel.capitalize(),style={"color":rc,"fontSize":"11px","fontWeight":"bold"})]),
            html.Div("Stage(s): "+", ".join(edge_hover.get("stages",[])[:3]),style={"color":muted,"fontSize":"11px","marginTop":"2px"}),
            html.Div("Evidence: "+str(edge_hover.get("count",1))+" record(s)",style={"color":muted,"fontSize":"11px"}),
            html.Div("Click for PubMed links →",style={"color":sky,"fontSize":"10px","marginTop":"5px","fontStyle":"italic"}),
        ]
        return content,{**base,"display":"block"}
    return "",{"display":"none"}


@app.callback(
    Output("info-panel","children"),
    Input("cytoscape-graph","tapNodeData"),Input("cytoscape-graph","tapEdgeData"),
    State("graph-store","data"),State("theme-store","data"),prevent_initial_call=True,
)
def on_click(node_data,edge_data,g_store,theme):
    theme=theme or "light"; ctx=callback_context
    trigger=ctx.triggered[0]["prop_id"] if ctx.triggered else ""
    if not g_store: return empty_panel(theme)
    if "tapNodeData" in trigger and node_data:
        node_id=node_data["id"]
        G=nx.DiGraph()
        for nid,attrs in g_store.get("nodes",{}).items(): G.add_node(nid,**attrs)
        for ek,attrs in g_store.get("edges",{}).items():
            parts=ek.split("|||")
            if len(parts)==2: G.add_edge(parts[0],parts[1],**attrs)
        analysis={"feedback_nodes":set(g_store.get("feedback_nodes",[])),
                  "feedback_edges":set(tuple(e) for e in g_store.get("feedback_edges",[])),
                  "self_loops":[tuple(e) for e in g_store.get("self_loops",[])],
                  "hub_genes":g_store.get("hub_genes",[])}
        return node_panel(node_id,G,analysis,EXT_DATA,theme)
    if "tapEdgeData" in trigger and edge_data: return edge_panel(edge_data,theme)
    return empty_panel(theme)


@app.callback(
    Output({"type":"sec-body","index":ALL},"style"),Output({"type":"sec-arr","index":ALL},"children"),
    Input({"type":"sec-hdr","index":ALL},"n_clicks"),State({"type":"sec-body","index":ALL},"style"),
    prevent_initial_call=True,
)
def toggle_section(clicks,styles):
    ctx=callback_context
    if not ctx.triggered: raise dash.exceptions.PreventUpdate
    triggered_index=json.loads(ctx.triggered[0]["prop_id"].split(".")[0])["index"]
    all_ids=[ctx.inputs_list[0][i]["id"]["index"] for i in range(len(styles))]
    new_styles,new_arrows=[],[]
    for i,idx in enumerate(all_ids):
        s=styles[i] or {}; cur=s.get("display","none")
        if idx==triggered_index:
            nd="none" if cur!="none" else "block"
            new_styles.append({**s,"display":nd}); new_arrows.append("▼" if nd=="block" else "▶")
        else:
            new_styles.append(s); new_arrows.append("▼" if cur!="none" else "▶")
    return new_styles,new_arrows

# =============================================================================
# Stats helpers
# =============================================================================

def _stats_panel(G,analysis,theme="light"):
    t=THEMES[theme]; lbl={"fontSize":"10px","fontWeight":"bold","letterSpacing":"0.1em",
                           "color":t["muted"],"fontFamily":"monospace","textTransform":"uppercase"}
    mut=t["muted"]
    regs=sum(1 for n in G.nodes() if G.nodes[n].get("is_reg") and not G.nodes[n].get("is_tgt"))
    tgts=sum(1 for n in G.nodes() if G.nodes[n].get("is_tgt") and not G.nodes[n].get("is_reg"))
    both=sum(1 for n in G.nodes() if G.nodes[n].get("is_reg") and G.nodes[n].get("is_tgt"))
    loops=len(analysis.get("feedback_loops",[])); sl=len(analysis.get("self_loops",[]))
    hubs=analysis.get("hub_genes",[])
    def row(l,v,c):
        return html.Div(style={"display":"flex","justifyContent":"space-between","marginBottom":"3px"},
            children=[html.Span(l,style={"color":mut,"fontSize":"12px"}),
                      html.Span(str(v),style={"color":c,"fontFamily":"monospace","fontSize":"12px"})])
    return html.Div([
        html.Label("Network Stats",style={**lbl,"marginBottom":"6px","display":"block"}),
        row("Nodes",G.number_of_nodes(),WONG["sky_blue"]),row("Edges",G.number_of_edges(),WONG["sky_blue"]),
        row("Regulators",regs,WONG["sky_blue"]),row("Targets",tgts,WONG["orange"]),
        row("Both",both,WONG["pink"]),row("Feedback loops",loops,WONG["orange"]),row("Self-loops",sl,WONG["yellow"]),
        html.Div(style={"marginTop":"8px"},children=[
            html.Label("Top Hubs",style={**lbl,"display":"block","marginBottom":"4px"}),
            *[html.Div(style={"display":"flex","justifyContent":"space-between","marginBottom":"2px"},
                children=[html.Span(g,style={"color":t["text"],"fontFamily":"monospace","fontSize":"11px"}),
                           html.Span(str(d),style={"color":WONG["sky_blue"],"fontFamily":"monospace","fontSize":"11px"})])
              for g,d in hubs[:6]],
        ]),
    ])

def _stats_empty(theme="light"):
    t=THEMES[theme]; lbl={"fontSize":"10px","fontWeight":"bold","letterSpacing":"0.1em",
                           "color":t["muted"],"fontFamily":"monospace","textTransform":"uppercase"}
    return html.Div([html.Label("Network Stats",style=lbl),
                     html.Div("Apply filters to see stats.",style={"color":t["muted"],"fontSize":"12px","marginTop":"4px"})])

def _topbar_stats(G,analysis):
    return ("Nodes: "+str(G.number_of_nodes())+"  |  Edges: "+str(G.number_of_edges())+
            "  |  Feedback loops: "+str(len(analysis.get("feedback_loops",[]))))

# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    print("\n"+"="*60+"\n  Lens GRN — Dash + Cytoscape\n"+"="*60)
    print("  http://127.0.0.1:8050\n")
    server = app.server
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT",8050)))