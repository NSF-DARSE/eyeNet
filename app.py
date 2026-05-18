"""
=============================================================================
Lens GRN Explorer — Dash 4.0 + Cytoscape
=============================================================================
Key fixes in this version:
  1. RNA-seq expression merged into single data panel (not separate dropdown)
  2. Node re-drag fixed (unlock on mousedown, re-lock on dragfree)
  3. Split network view when tissue of regulator != tissue of target
  4. RNA-seq data correctly loaded via symbol_col="Symbol"
=============================================================================
"""

import os, copy, json
import pandas as pd
import networkx as nx
import dash
from dash import dcc, html, Input, Output, State, callback_context, ALL, clientside_callback
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto

from grn_network import (
    CONFIG, THEMES, WONG,
    load_data, load_external_data, filter_data,
    build_graph, analyze_graph, build_cytoscape_elements, stage_numeric,
)

cyto.load_extra_layouts()

# =============================================================================
# SINGLE DATA SOURCE — all expression data in one panel
# =============================================================================

DATA_SOURCES = {
    "all_expression": {
        "label":      "Expression & Enrichment",
        "icon":       "📊",
        # Primary file — microarray raw counts + RNA-seq enrichment
        "path":       "data/MegaTable April 24 2024 for Microarray and RNA Seq Sent to Murali (1).xls",
        "symbol_col": "Symbol",
        "meta_cols":  {"entrez":"Entrez","uniprot":"UNIPROT","description":"Gene_description"},
        # FPKM values from sheet 3 of the FPKM file
        "extra_paths": [
            {
                "path":       "data/LogCPM_FPKM_TPM_FiberEpi120225 USE THIS FEB 13 2025.xls",
                "symbol_col": "Symbol",
                "sheet_name": "FPKM",
            }
        ],
        "sections": [
            {
                "title":        "Microarray Expression (raw counts)",
                "description":  "Raw expression counts. Epi then Fiber per stage.",
                "color_values": False,
                "columns": {
                    "Epi E12 (Beebe)":   "Beebe_E12_exp_Epi",
                    "Fiber E12 (Beebe)": "Beebe_E12_exp_Fiber",
                    "Epi P13 (Naka)":    "Naka_P13_epi_exp",
                    "Fiber P13 (Naka)":  "Naka_P13_fiber_exp",
                },
            },
            {
                "title":        "RNA-seq Expression (FPKM) — Epi then Fiber per stage",
                "description":  "FPKM expression values. Epi then Fiber, stage ascending.",
                "color_values": False,
                "columns": {
                    "Epi P0b":   "P0b_FPKM_Epi",   "Fiber P0b": "P0b_FPKM_Fiber",
                    "Epi E14":   "E14_FPKM_Epi",    "Fiber E14": "E14_FPKM_Fiber",
                    "Epi E16":   "E16_FPKM_Epi",    "Fiber E16": "E16_FPKM_Fiber",
                    "Epi E18":   "E18_FPKM_Epi",    "Fiber E18": "E18_FPKM_Fiber",
                    "Epi P0":    "P0_FPKM_Epi",     "Fiber P0":  "P0_FPKM_Fiber",
                    "Epi 3Mo":   "3Mo_FPKM_Epi",    "Fiber 3Mo": "3Mo_FPKM_Fiber",
                    "Epi 6Mo":   "6Mo_FPKM_Epi",    "Fiber 6Mo": "6Mo_FPKM_Fiber",
                    "Epi 2Y":    "2Y_FPKM_Epi",     "Fiber 2Y":  "2Y_FPKM_Fiber",
                },
            },
            {
                "title":        "RNA-seq Enrichment (LEC then FC by stage)",
                "description":  "LEC = lens epithelial cell (Epi), FC = fiber cell. Positive = enriched.",
                "color_values": True,
                "columns": {
                    "LEC E14": "enr_LEC_E14_Cv", "FC E14": "enr_FC_E14_Cv",
                    "LEC E16": "enr_LEC_E16_Cv", "FC E16": "enr_FC_E16_Cv",
                    "LEC E18": "enr_LEC_E18_Cv", "FC E18": "enr_FC_E18_Cv",
                    "LEC P0":  "enr_LEC_P0_Cv",  "FC P0":  "enr_FC_P0_Cv",
                    "LEC P0b": "enr_LEC_P0_Rob", "FC P0b": "enr_FC_P0_Rob",
                    "LEC 3Mo": "enr_LEC_3Mo",    "FC 3Mo": "enr_FC_3Mo",
                    "LEC 6Mo": "enr_LEC_6Mo",    "FC 6Mo": "enr_FC_6Mo",
                    "LEC 2Y":  "enr_LEC_2Y",     "FC 2Y":  "enr_FC_2Y",
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
ALL_TISSUE_REG = sorted([t for t in BASE_DF["tissue_reg"].dropna().unique() if t and t not in ("nan","Unknown")])
ALL_TISSUE_TGT = sorted([t for t in BASE_DF["tissue_tgt"].dropna().unique() if t and t not in ("nan","Unknown")])

# Log what loaded
for key, lookup in EXT_DATA.items():
    print(f"[APP] {DATA_SOURCES[key]['label']}: {len(lookup)} genes")
print("[APP] Ready\n")

def stage_opts(s): return [{"label":x,"value":x} for x in s]
def gene_opts(g):  return [{"label":x,"value":x} for x in g]
def tissue_opts(t):return [{"label":x,"value":x} for x in t]

# =============================================================================
# Stylesheet
# =============================================================================

def build_stylesheet(theme="light"):
    t = THEMES[theme]
    return [
        {"selector":"node","style":{
            "label":"data(label)","font-size":"9px","font-family":"monospace","font-weight":"normal",
            "text-valign":"bottom","text-halign":"center","text-margin-y":"5px",
            "text-wrap":"none","color":t["text"],"text-outline-width":"0px",
            "width":"mapData(deg, 1, 50, 20, 60)","height":"mapData(deg, 1, 50, 20, 60)",
            "border-width":"2px","cursor":"pointer",
        }},
        {"selector":"node.regulator","style":{"background-color":WONG["sky_blue"],"border-color":WONG["sky_blue"]}},
        {"selector":"node.target",   "style":{"background-color":WONG["orange"],  "border-color":WONG["orange"]}},
        {"selector":"node.both",     "style":{"background-color":WONG["pink"],    "border-color":WONG["pink"]}},
        {"selector":"node.selfloop", "style":{"background-color":WONG["yellow"],  "border-color":WONG["yellow"]}},
        {"selector":"node.feedback", "style":{"border-width":"4px","border-color":WONG["yellow"]}},
        {"selector":"node:selected", "style":{"border-width":"4px","border-color":"#1d4ed8","overlay-color":"#1d4ed8","overlay-opacity":0.1}},
        {"selector":"edge","style":{
            "curve-style":"bezier","target-arrow-shape":"triangle",
            "arrow-scale":0.9,"width":2,"opacity":0.75,"cursor":"pointer",
        }},
        {"selector":"edge.activating","style":{"line-color":WONG["green"],     "target-arrow-color":WONG["green"]}},
        {"selector":"edge.inhibiting","style":{"line-color":WONG["vermillion"],"target-arrow-color":WONG["vermillion"]}},
        {"selector":"edge.no_effect", "style":{"line-color":"#94a3b8",         "target-arrow-color":"#94a3b8"}},
        {"selector":"edge.feedback-edge","style":{"line-style":"dashed","line-dash-pattern":[6,3]}},
        {"selector":"edge:selected","style":{"opacity":1,"width":3,"overlay-color":"#1d4ed8","overlay-opacity":0.12}},
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
    body{margin:0;font-family:sans-serif;overflow:hidden;}
    .Select-control{background:#fff !important;border-color:#cbd5e1 !important;}
    .Select-value-label,.Select--single .Select-value .Select-value-label{color:#0f172a !important;}
    .Select-placeholder{color:#94a3b8 !important;}
    .Select-input>input{color:#0f172a !important;}
    .Select-option{color:#0f172a !important;background:#fff !important;}
    .Select-option.is-focused{background:#f1f5f9 !important;}
    .Select-option.is-selected{background:#dbeafe !important;}
    .Select-menu-outer{z-index:9999 !important;}
    [class*="singleValue"]{color:#0f172a !important;}
    [class*="placeholder"]{color:#94a3b8 !important;}
    [class*="option"]{color:#0f172a !important;background:#fff !important;}
    [class*="menu"]{z-index:9999 !important;background:#fff !important;}
    [class*="Input"] input{color:#0f172a !important;}

    #hover-tooltip{
        position:fixed;z-index:99999;pointer-events:none;border-radius:10px;
        padding:10px 14px;font-family:monospace;font-size:12px;
        box-shadow:0 4px 24px rgba(0,0,0,0.13);max-width:260px;min-width:160px;
        display:none;top:70px;right:320px;
    }
    /* Graph area container for split view */
    #graph-area{
        flex:1;display:flex;flex-direction:row;overflow:hidden;position:relative;
    }
    .graph-pane{
        flex:1;position:relative;overflow:hidden;
    }
    .graph-pane-label{
        position:absolute;top:10px;left:50%;transform:translateX(-50%);
        z-index:100;background:rgba(255,255,255,0.92);border:1px solid #e2e8f0;
        border-radius:8px;padding:4px 12px;font-family:monospace;font-size:11px;
        color:#0369a1;font-weight:bold;pointer-events:none;
        box-shadow:0 2px 8px rgba(0,0,0,0.08);
    }
    .graph-divider{
        width:3px;background:#e2e8f0;flex-shrink:0;
    }
    /* Overlay controls */
    #legend-overlay{position:absolute;bottom:24px;left:16px;z-index:1000;font-family:monospace;}
    #legend-panel{
        display:none;margin-bottom:6px;padding:14px 16px;border-radius:10px;font-size:12px;
        box-shadow:0 4px 20px rgba(0,0,0,0.13);min-width:220px;
        background:#ffffff;border:1px solid #e2e8f0;color:#0f172a;
    }
    #legend-btn{
        display:inline-flex;align-items:center;gap:6px;padding:7px 14px;
        border-radius:8px;font-family:monospace;font-size:12px;cursor:pointer;
        box-shadow:0 3px 14px rgba(0,0,0,0.15);background:#ffffff;
        border:1px solid #e2e8f0;color:#0369a1;user-select:none;
    }
    #legend-btn:hover{background:#f0f9ff;}
    #zoom-controls{
        position:absolute;bottom:24px;right:16px;z-index:1000;
        display:flex;align-items:center;border-radius:8px;overflow:hidden;
        box-shadow:0 3px 14px rgba(0,0,0,0.13);background:#ffffff;border:1px solid #e2e8f0;
    }
    .zoom-btn{
        display:inline-flex;align-items:center;justify-content:center;
        width:38px;height:36px;font-size:18px;cursor:pointer;
        color:#334155;user-select:none;transition:background 0.12s;
    }
    .zoom-btn:hover{background:#f1f5f9;}
    .zoom-sep{width:1px;height:28px;background:#e2e8f0;}
    #save-btn{
        position:absolute;bottom:24px;right:140px;z-index:1000;
        display:inline-flex;align-items:center;gap:5px;padding:7px 13px;
        border-radius:8px;font-family:monospace;font-size:12px;cursor:pointer;
        box-shadow:0 3px 14px rgba(0,0,0,0.13);background:#ffffff;
        border:1px solid #e2e8f0;color:#0369a1;user-select:none;
    }
    #save-btn:hover{background:#f0f9ff;}
    .sec-header{
        display:flex;justify-content:space-between;align-items:center;
        padding:8px 10px;border-radius:6px;cursor:pointer;
        user-select:none;margin-bottom:2px;transition:background 0.15s;
    }
    .sec-light{background:#f1f5f9;border:1px solid #e2e8f0;}
    .sec-light:hover{background:#e2e8f0;}
    .sec-dark{background:#1a2540;border:1px solid #1e2d4a;}
    .sec-dark:hover{background:#1e3055;}
    ::-webkit-scrollbar{width:5px;height:5px;}
    ::-webkit-scrollbar-thumb{background:#cbd5e1;border-radius:3px;}
"""

INIT_JS = """
<script>
// ── Attach drag-lock to a single Cytoscape instance by element id ─────────
function attachDragLock(elId) {
  var el = document.getElementById(elId);
  if (!el) return;
  var poll = setInterval(function() {
    try {
      var cy = el._cyreg && el._cyreg.cy;
      if (!cy) return;
      clearInterval(poll);
      // After layout: unlock all so initial layout runs free, then re-arm
      cy.on('layoutstop', function() {
        cy.nodes().unlock();
        armDragLock(cy);
      });
      armDragLock(cy);
    } catch(e) {}
  }, 400);
}

function armDragLock(cy) {
  cy.off('mousedown', 'node');
  cy.off('dragfree', 'node');
  // Unlock on mousedown so node can be dragged again
  cy.on('mousedown', 'node', function(e) {
    e.target.unlock();
  });
  // Re-lock on release
  cy.on('dragfree', 'node', function(e) {
    e.target.lock();
  });
}

(function() {
  function init() {
    attachDragLock('cytoscape-graph');
    attachDragLock('cytoscape-graph-2');
  }
  document.addEventListener('DOMContentLoaded', init);
  // Re-attach after Dash re-renders (theme toggle, filter apply)
  var observer = new MutationObserver(function() {
    attachDragLock('cytoscape-graph');
    attachDragLock('cytoscape-graph-2');
  });
  observer.observe(document.body, { childList: true, subtree: true });
})();

// ── Legend ─────────────────────────────────────────────────────────────────
function grn_toggleLegend() {
  var p = document.getElementById('legend-panel');
  var a = document.getElementById('legend-arrow');
  if (!p) return;
  if (p.style.display==='none'||p.style.display==='') {
    p.style.display='block'; if(a) a.textContent='▼';
  } else {
    p.style.display='none'; if(a) a.textContent='▲';
  }
}

// ── Zoom (operates on first visible Cytoscape) ─────────────────────────────
function _getActiveCy() {
  for (var id of ['cytoscape-graph','cytoscape-graph-2']) {
    var el = document.getElementById(id);
    if (el && el.style.display!=='none') {
      try { var cy=el._cyreg&&el._cyreg.cy; if(cy) return cy; } catch(e){}
    }
  }
  var el2 = document.getElementById('cytoscape-graph');
  if(el2) { try { return el2._cyreg&&el2._cyreg.cy; } catch(e){} }
  return null;
}
function grn_zoomOut(){ var cy=_getActiveCy();if(!cy)return; cy.zoom({level:Math.max(0.05,cy.zoom()-0.3),renderedPosition:{x:cy.width()/2,y:cy.height()/2}}); }
function grn_zoomFit(){ var cy=_getActiveCy();if(!cy)return; cy.fit(undefined,30); }
function grn_zoomIn(){  var cy=_getActiveCy();if(!cy)return; cy.zoom({level:Math.min(3,cy.zoom()+0.3),renderedPosition:{x:cy.width()/2,y:cy.height()/2}}); }

// ── Save PNG (saves both graphs if split view) ─────────────────────────────
function grn_saveImage() {
  for (var id of ['cytoscape-graph','cytoscape-graph-2']) {
    var el = document.getElementById(id);
    if (!el || el.closest('[style*="display: none"]')) continue;
    try {
      var cy=el._cyreg&&el._cyreg.cy; if(!cy) continue;
      var png=cy.png({output:'blob',bg:'white',full:true,scale:2});
      var url=URL.createObjectURL(png);
      var a=document.createElement('a'); a.href=url;
      a.download='lens_grn_'+id+'.png'; a.click();
      URL.revokeObjectURL(url);
    } catch(e){ console.error(e); }
  }
}
</script>
"""

app.index_string = (
    "<!DOCTYPE html><html><head>"
    "{%metas%}<title>{%title%}</title>{%favicon%}{%css%}"
    "<style>" + GLOBAL_CSS + "</style>"
    "</head><body>"
    "{%app_entry%}"
    "<footer>{%config%}{%scripts%}{%renderer%}</footer>"
    + INIT_JS +
    "</body></html>"
)

# =============================================================================
# Clientside callbacks — Dash 4.0 way for button events
# =============================================================================

clientside_callback("function(n){if(n)grn_toggleLegend();return n;}",
    Output("legend-btn","n_clicks"), Input("legend-btn","n_clicks"), prevent_initial_call=True)
clientside_callback("function(n){if(n)grn_zoomOut();return n;}",
    Output("zoom-out","n_clicks"), Input("zoom-out","n_clicks"), prevent_initial_call=True)
clientside_callback("function(n){if(n)grn_zoomFit();return n;}",
    Output("zoom-fit","n_clicks"), Input("zoom-fit","n_clicks"), prevent_initial_call=True)
clientside_callback("function(n){if(n)grn_zoomIn();return n;}",
    Output("zoom-in","n_clicks"), Input("zoom-in","n_clicks"), prevent_initial_call=True)
clientside_callback("function(n){if(n)grn_saveImage();return n;}",
    Output("save-btn","n_clicks"), Input("save-btn","n_clicks"), prevent_initial_call=True)


# =============================================================================
# Overlays
# =============================================================================

def build_graph_overlays(theme="light"):
    t=THEMES[theme]; muted=t["muted"]; txt=t["text"]
    legend_panel = html.Div(id="legend-panel", children=[
        html.Div("Lens GRN Legend",style={"fontWeight":"bold","fontSize":"13px","color":WONG["sky_blue"],"marginBottom":"10px"}),
        html.Div("NODE — ROLE",style={"fontSize":"9px","fontWeight":"bold","color":muted,"textTransform":"uppercase","letterSpacing":"0.08em","marginBottom":"5px"}),
        *[html.Div([html.Span("●",style={"color":c,"marginRight":"6px"}),l],style={"fontSize":"11px","marginBottom":"3px","color":txt})
          for c,l in [(WONG["sky_blue"],"Regulator only"),(WONG["orange"],"Target only"),
                      (WONG["pink"],"Regulator & Target"),(WONG["yellow"],"Self-regulatory loop")]],
        html.Br(),
        html.Div("EDGE — RELATIONSHIP",style={"fontSize":"9px","fontWeight":"bold","color":muted,"textTransform":"uppercase","letterSpacing":"0.08em","marginBottom":"5px"}),
        *[html.Div([html.Span("━▶",style={"color":c,"marginRight":"6px"}),l],style={"fontSize":"11px","marginBottom":"3px","color":txt})
          for c,l in [(WONG["green"],"Activating"),(WONG["vermillion"],"Inhibiting"),("#94a3b8","No effect")]],
        html.Br(),
        html.Div("Yellow border = feedback loop",style={"fontSize":"10px","color":muted}),
        html.Div("Hover = quick info  |  Click = full details",style={"fontSize":"10px","color":muted,"marginTop":"2px"}),
        html.Div("Drag node → locks; drag again to move",style={"fontSize":"10px","color":muted,"marginTop":"2px"}),
        html.Div("Split view = different tissue networks",style={"fontSize":"10px","color":muted,"marginTop":"2px"}),
        html.Div("Wong (2011) color-blind safe",style={"fontSize":"10px","color":muted,"marginTop":"4px","fontStyle":"italic"}),
    ])
    legend_btn = html.Div(id="legend-btn",n_clicks=0,
        children=[html.Span("🔬 Legend "),html.Span("▲",id="legend-arrow")])
    save_btn   = html.Div(id="save-btn",n_clicks=0,children=[html.Span("💾 Save PNG")])
    zoom_box   = html.Div(id="zoom-controls",children=[
        html.Div("−",id="zoom-out",n_clicks=0,className="zoom-btn"),
        html.Div(className="zoom-sep"),
        html.Div("⟳",id="zoom-fit",n_clicks=0,className="zoom-btn",style={"fontSize":"16px"}),
        html.Div(className="zoom-sep"),
        html.Div("+",id="zoom-in",n_clicks=0,className="zoom-btn"),
    ])
    return html.Div(id="legend-overlay",children=[legend_panel,legend_btn]), save_btn, zoom_box

# =============================================================================
# Right panel builders
# =============================================================================

def empty_panel(theme="light"):
    t=THEMES[theme]
    return html.Div([
        html.Div("👆 Click any node or edge",style={"color":t["muted"],"fontSize":"13px","fontFamily":"monospace","textAlign":"center","marginTop":"50px"}),
        html.Div("to see full details, expression data,",style={"color":t["muted"],"fontSize":"12px","textAlign":"center","marginTop":"6px"}),
        html.Div("enrichment values and PubMed links.",style={"color":t["muted"],"fontSize":"12px","textAlign":"center"}),
        html.Div("─"*26,style={"color":t["muted"],"fontSize":"10px","textAlign":"center","marginTop":"16px","opacity":"0.3"}),
        html.Div("Hover for quick summary",style={"color":t["muted"],"fontSize":"11px","textAlign":"center","marginTop":"6px","fontStyle":"italic"}),
    ])

def collapsible(sec_id,title,icon,body,open_default,theme):
    t=THEMES[theme]
    cls="sec-header sec-dark" if theme=="dark" else "sec-header sec-light"
    return html.Div([
        html.Div(id={"type":"sec-hdr","index":sec_id},className=cls,children=[
            html.Span([html.Span(icon+" "),html.Span(title,style={"fontWeight":"600","fontSize":"12px","color":t["text"]})]),
            html.Span("▼" if open_default else "▶",id={"type":"sec-arr","index":sec_id},
                      style={"color":t["muted"],"fontSize":"10px","marginLeft":"6px"}),
        ]),
        html.Div(id={"type":"sec-body","index":sec_id},
                 style={"display":"block" if open_default else "none"},children=body),
    ],style={"marginBottom":"6px"})

def data_table(rows, color_values, theme):
    """Display rows in original column order (Epi before Fiber as defined in DATA_SOURCES)."""
    t=THEMES[theme]
    # NO sorting — preserve the Epi-first, Fiber-second order from DATA_SOURCES columns dict
    trs=[]
    for lbl,val in rows:
        if val is None: continue
        if color_values and isinstance(val,(int,float)):
            pct=min(abs(val)/10*100,100)
            vc=WONG["green"] if val>0 else (WONG["vermillion"] if val<0 else t["muted"])
            trs.append(html.Tr([
                html.Td(lbl,style={"padding":"3px 6px","color":t["muted"],"fontSize":"11px","fontFamily":"monospace","width":"50%"}),
                html.Td([
                    html.Span(str(round(val,1)),style={"color":vc,"fontSize":"11px","fontFamily":"monospace","marginRight":"5px","minWidth":"35px","display":"inline-block"}),
                    html.Div(style={"display":"inline-block","height":"6px","width":str(pct)+"%","maxWidth":"50px","background":vc,"borderRadius":"3px","verticalAlign":"middle"}),
                ],style={"padding":"3px 6px"}),
            ]))
        else:
            disp=str(round(val,1)) if isinstance(val,(int,float)) else str(val)
            trs.append(html.Tr([
                html.Td(lbl,style={"padding":"3px 6px","color":t["muted"],"fontSize":"11px","fontFamily":"monospace"}),
                html.Td(disp,style={"padding":"3px 6px","color":t["text"],"fontSize":"11px","fontFamily":"monospace","textAlign":"right"}),
            ]))
    if not trs:
        return html.Div("No values recorded.",style={"color":t["muted"],"fontSize":"11px","fontStyle":"italic","padding":"4px"})
    return html.Table(trs,style={"width":"100%","borderCollapse":"collapse","background":t["row_bg"],"borderRadius":"6px"})

def node_panel(node_id, G, analysis, ext_data, theme="light"):
    t=THEMES[theme]; d=G.nodes.get(node_id,{})
    is_reg=d.get("is_reg",False); is_tgt=d.get("is_tgt",False)
    out_d=d.get("out_degree",0); in_d=d.get("in_degree",0)
    in_fb=node_id in analysis.get("feedback_nodes",set())
    is_sl=node_id in set(e[0] for e in analysis.get("self_loops",[]))

    if is_reg and is_tgt: role,rc="Regulator & Target",WONG["pink"]
    elif is_reg:          role,rc="Regulator",WONG["sky_blue"]
    else:                 role,rc="Target",WONG["orange"]

    mut={"color":t["muted"],"fontSize":"12px"}
    def row(l,v,c=None):
        return html.Div(style={"display":"flex","justifyContent":"space-between","marginBottom":"4px"},
            children=[html.Span(l,style=mut),html.Span(str(v),style={"fontFamily":"monospace","fontSize":"12px","color":c or WONG["sky_blue"]})])

    net_body=html.Div([
        html.Span(role,style={"background":rc,"color":"#fff","padding":"2px 10px","borderRadius":"12px","fontSize":"11px","fontFamily":"monospace","display":"inline-block","marginBottom":"10px"}),
        row("Regulates (out-edges)",out_d),
        row("Regulated by (in-edges)",in_d),
    ]+([html.Div("⚡ Part of feedback loop",style={"color":WONG["orange"],"fontSize":"12px","marginTop":"4px"})] if in_fb else [])
     +([html.Div("🔄 Self-regulatory loop",style={"color":WONG["yellow"],"fontSize":"12px","marginTop":"4px"})] if is_sl else []),
        style={"padding":"2px 4px"})

    sections=[collapsible("net_"+node_id,"Network Info","🔗",net_body,True,theme)]

    # Build a combined data body — all sources in one collapsible
    combined_children = []
    has_any_data = False

    for src_key, src_cfg in DATA_SOURCES.items():
        lookup=ext_data.get(src_key,{})
        gene_data=lookup.get(node_id)
        if gene_data is None:
            continue
        has_any_data = True

        # Links (only from first source that has them)
        entrez=gene_data.get("_meta_entrez",""); uniprot=gene_data.get("_meta_uniprot","")
        desc=gene_data.get("_meta_description","")
        links=[]
        if entrez and entrez not in ("nan",""):
            links.append(html.A("NCBI Gene ↗",href="https://www.ncbi.nlm.nih.gov/gene/"+str(entrez),target="_blank",
                style={"color":WONG["sky_blue"],"fontSize":"11px","marginRight":"10px","textDecoration":"underline"}))
        if uniprot and uniprot not in ("nan",""):
            links.append(html.A("UniProt ↗",href="https://www.uniprot.org/uniprot/"+str(uniprot),target="_blank",
                style={"color":WONG["sky_blue"],"fontSize":"11px","textDecoration":"underline"}))
        if links and not combined_children:
            combined_children.append(html.Div(links,style={"marginBottom":"6px","marginTop":"4px"}))
        if desc and desc not in ("nan","") and not combined_children:
            combined_children.append(html.Div(str(desc)[:120]+("..." if len(str(desc))>120 else ""),
                style={"color":t["muted"],"fontSize":"10px","fontStyle":"italic","marginBottom":"8px"}))

        # Sections
        for sec in src_cfg.get("sections",[]):
            rows_data=[(dn,gene_data.get(dn)) for dn in sec["columns"]]
            has_sec_data=any(v is not None for _,v in rows_data)
            color_vals=sec.get("color_values",True)
            combined_children.append(html.Div([
                html.Div(sec["title"],style={"fontSize":"10px","color":t["muted"],"fontWeight":"bold",
                                             "textTransform":"uppercase","letterSpacing":"0.06em",
                                             "marginTop":"10px","marginBottom":"2px"}),
                html.Div(sec.get("description",""),style={"fontSize":"10px","color":t["muted"],"marginBottom":"4px","fontStyle":"italic"})
                if sec.get("description") else html.Div(),
                data_table(rows_data,color_vals,theme) if has_sec_data else
                html.Div("No values.",style={"color":t["muted"],"fontSize":"11px","fontStyle":"italic","padding":"2px 4px"}),
            ]))

    if combined_children:
        # Color legend
        combined_children.append(html.Div([
            html.Span("■ ",style={"color":WONG["green"]}),html.Span("Positive  ",style={"fontSize":"10px","color":t["muted"]}),
            html.Span("■ ",style={"color":WONG["vermillion"]}),html.Span("Negative",style={"fontSize":"10px","color":t["muted"]}),
        ],style={"marginTop":"8px"}))
        sections.append(collapsible("expr_"+node_id,"Expression & Enrichment","📊",
                                    html.Div(combined_children,style={"padding":"0 2px"}),False,theme))
    elif not has_any_data:
        sections.append(collapsible("expr_"+node_id,"Expression & Enrichment","📊",
            html.Div("No expression data available for "+node_id+".",
                     style={"color":t["muted"],"fontSize":"11px","fontStyle":"italic","padding":"6px 4px"}),
            False,theme))

    return html.Div([
        html.Div(node_id,style={"fontWeight":"bold","fontSize":"16px","color":WONG["sky_blue"],"fontFamily":"monospace","marginBottom":"12px"}),
        *sections,
    ])

def edge_panel(edge_data,theme="light"):
    t=THEMES[theme]
    # Support both original and tagged node IDs
    src=edge_data.get("source_gene", edge_data.get("source",""))
    tgt=edge_data.get("target_gene", edge_data.get("target",""))
    rel=edge_data.get("rel","no_effect")
    perts=edge_data.get("perts",[]); effs=edge_data.get("effs",[])
    stgs=edge_data.get("stages",[]); count=edge_data.get("count",1)
    pmids=edge_data.get("pmids",[]); is_fb=edge_data.get("feedback",False)
    t_reg=edge_data.get("tissue_reg",""); t_tgt=edge_data.get("tissue_tgt","")
    rc=WONG["green"] if rel=="activating" else (WONG["vermillion"] if rel=="inhibiting" else "#94a3b8")
    rlbl="▲ Activating" if rel=="activating" else ("▼ Inhibiting" if rel=="inhibiting" else "○ No effect")
    mut={"color":t["muted"],"fontSize":"12px"}
    def row(l,v):
        return html.Div(style={"display":"flex","justifyContent":"space-between","marginBottom":"5px"},
            children=[html.Span(l,style=mut),html.Span(v,style={"fontFamily":"monospace","color":t["text"],"fontSize":"12px"})])
    net_body=html.Div([
        html.Span(rlbl,style={"background":rc,"color":"#fff","padding":"2px 10px","borderRadius":"12px","fontSize":"11px","fontFamily":"monospace","display":"inline-block","marginBottom":"10px"}),
        row("Perturbation",", ".join(perts) if perts else "—"),
        row("Raw effect",", ".join(effs) if effs else "—"),
        row("Stage(s)",", ".join(stgs) if stgs else "—"),
        row("Evidence count",str(count)),
        row("Tissue of regulator",t_reg or "—"),
        row("Tissue of target",t_tgt or "—"),
    ]+([html.Div("⚡ Feedback loop",style={"color":WONG["orange"],"fontSize":"12px","marginTop":"4px"})] if is_fb else []),
        style={"padding":"2px 4px"})
    pmid_links=[html.A("📄 PMID "+pmid+" ↗",href="https://pubmed.ncbi.nlm.nih.gov/"+pmid+"/",target="_blank",
               style={"color":WONG["sky_blue"],"display":"block","fontSize":"12px","marginBottom":"5px","textDecoration":"underline"})
               for pmid in pmids]
    pmid_body=html.Div(pmid_links or [html.Div("No PubMed IDs.",style={"color":t["muted"],"fontSize":"11px","fontStyle":"italic","padding":"4px"})],style={"padding":"2px 4px"})
    return html.Div([
        html.Div([
            html.Span(src,style={"fontWeight":"bold","color":WONG["sky_blue"],"fontFamily":"monospace","fontSize":"14px"}),
            html.Span(" → ",style={"color":t["muted"],"fontSize":"14px"}),
            html.Span(tgt,style={"fontWeight":"bold","color":WONG["orange"],"fontFamily":"monospace","fontSize":"14px"}),
        ],style={"marginBottom":"12px"}),
        collapsible("edge_net_"+src+"_"+tgt,"Network Info","🔗",net_body,True,theme),
        collapsible("edge_pm_"+src+"_"+tgt,"PubMed References","📄",pmid_body,False,theme),
    ])

# =============================================================================
# Sidebar
# =============================================================================

def build_sidebar(theme="light"):
    t=THEMES[theme]
    label={"color":t["muted"],"fontSize":"10px","fontWeight":"bold","letterSpacing":"0.1em","fontFamily":"monospace","textTransform":"uppercase"}
    sub={"color":t["muted"],"fontSize":"12px","marginTop":"6px"}
    hr={"borderColor":t["sidebar_bdr"],"margin":"4px 0"}
    return html.Div(
        style={"width":"260px","minWidth":"260px","background":t["sidebar_bg"],
               "borderRight":"1px solid "+t["sidebar_bdr"],"padding":"12px 14px",
               "overflowY":"auto","display":"flex","flexDirection":"column","gap":"10px",
               "height":"100vh","boxSizing":"border-box"},
        children=[
            html.Div(style={"display":"flex","justifyContent":"space-between","alignItems":"center"},children=[
                html.Div([html.Div("Lens GRN Explorer",style={"color":WONG["sky_blue"],"fontFamily":"monospace","fontWeight":"bold","fontSize":"13px"}),
                          html.Div("Lachke Lab 2016",style={"color":t["muted"],"fontSize":"11px"})]),
                html.Div(id="theme-toggle",n_clicks=0,
                    style={"cursor":"pointer","border":"1px solid "+t["sidebar_bdr"],"borderRadius":"6px","padding":"4px 8px","textAlign":"center","minWidth":"50px"},
                    children=[html.Div("🌙" if theme=="light" else "☀️",style={"fontSize":"14px"}),
                              html.Div("Dark" if theme=="light" else "Light",style={"fontSize":"9px","color":t["muted"],"fontFamily":"monospace"})]),
            ]),
            html.Hr(style=hr),
            html.Div([
                html.Label("Stage Filter",style=label),
                html.Label("Single stage",style=sub),
                dcc.Dropdown(id="stage-single",options=stage_opts(ALL_STAGES),placeholder="All stages",clearable=True,style={"fontSize":"12px"}),
                html.Label("Stage range",style={**sub,"marginTop":"6px"}),
                html.Div([
                    dcc.Dropdown(id="stage-from",options=stage_opts(ALL_STAGES),placeholder="From",clearable=True,style={"flex":"1","fontSize":"12px"}),
                    html.Span("→",style={"color":t["muted"],"padding":"0 4px","alignSelf":"center"}),
                    dcc.Dropdown(id="stage-to",options=stage_opts(ALL_STAGES),placeholder="To",clearable=True,style={"flex":"1","fontSize":"12px"}),
                ],style={"display":"flex","alignItems":"center","gap":"3px"}),
            ]),
            html.Hr(style=hr),
            html.Div([
                html.Label("Gene Filter",style=label),
                html.Label("Regulator",style=sub),
                dcc.Dropdown(id="filter-regulator",options=gene_opts(ALL_REGULATORS),placeholder="All regulators",clearable=True,style={"fontSize":"12px"}),
                html.Label("Target",style={**sub,"marginTop":"6px"}),
                dcc.Dropdown(id="filter-target",options=gene_opts(ALL_TARGETS),placeholder="All targets",clearable=True,style={"fontSize":"12px"}),
            ]),
            html.Hr(style=hr),
            html.Div([
                html.Label("Tissue Filter",style=label),
                html.Div("Auto-splits into two networks if different tissues exist in data",
                         style={"fontSize":"10px","color":WONG["sky_blue"],"fontFamily":"monospace",
                                "marginTop":"3px","marginBottom":"6px"}),
                html.Label("Tissue of regulator",style=sub),
                html.Div([
                    dcc.Dropdown(id="filter-tissue-reg",options=tissue_opts(ALL_TISSUE_REG),
                                 placeholder="All tissues",clearable=True,style={"flex":"1","fontSize":"12px"}),
                    html.Button("All",id="tissue-reg-all",n_clicks=0,
                        style={"padding":"2px 7px","fontSize":"10px","cursor":"pointer","marginLeft":"4px",
                               "border":"1px solid #cbd5e1","borderRadius":"4px","background":"#f1f5f9",
                               "color":"#64748b","whiteSpace":"nowrap","height":"36px"}),
                    html.Button("✕",id="tissue-reg-clear",n_clicks=0,
                        style={"padding":"2px 7px","fontSize":"10px","cursor":"pointer","marginLeft":"2px",
                               "border":"1px solid #cbd5e1","borderRadius":"4px","background":"#f1f5f9",
                               "color":"#64748b","height":"36px"}),
                ],style={"display":"flex","alignItems":"center"}),
                html.Label("Tissue of target",style={**sub,"marginTop":"6px"}),
                html.Div([
                    dcc.Dropdown(id="filter-tissue-tgt",options=tissue_opts(ALL_TISSUE_TGT),
                                 placeholder="All tissues",clearable=True,style={"flex":"1","fontSize":"12px"}),
                    html.Button("All",id="tissue-tgt-all",n_clicks=0,
                        style={"padding":"2px 7px","fontSize":"10px","cursor":"pointer","marginLeft":"4px",
                               "border":"1px solid #cbd5e1","borderRadius":"4px","background":"#f1f5f9",
                               "color":"#64748b","whiteSpace":"nowrap","height":"36px"}),
                    html.Button("✕",id="tissue-tgt-clear",n_clicks=0,
                        style={"padding":"2px 7px","fontSize":"10px","cursor":"pointer","marginLeft":"2px",
                               "border":"1px solid #cbd5e1","borderRadius":"4px","background":"#f1f5f9",
                               "color":"#64748b","height":"36px"}),
                ],style={"display":"flex","alignItems":"center"}),
            ]),
            html.Hr(style=hr),
            html.Div([
                html.Label("Relationship Filter",style=label),
                dcc.Checklist(id="relationship-filter",
                    options=[{"label":"  Activating","value":"activating"},{"label":"  Inhibiting","value":"inhibiting"},{"label":"  No effect","value":"no_effect"}],
                    value=["activating","inhibiting","no_effect"],style={"marginTop":"6px"},
                    labelStyle={"display":"block","color":t["text"],"fontSize":"12px","marginBottom":"3px"}),
                html.Div("True relationship = Perturbation × Effect",style={"fontSize":"10px","color":t["muted"],"marginTop":"3px","fontFamily":"monospace"}),
            ]),
            html.Hr(style=hr),
            html.Div([
                html.Label("Display Options",style=label),
                html.Label("Max edges",style=sub),
                dcc.Dropdown(id="max-edges",options=[{"label":"100","value":100},{"label":"300","value":300},{"label":"600","value":600},{"label":"All","value":9999}],
                    value=300,clearable=False,style={"fontSize":"12px"}),
                html.Label("Layout",style={**sub,"marginTop":"6px"}),
                dcc.Dropdown(id="layout-select",options=[
                    {"label":"Barnes Hut","value":"barnes_hut"},{"label":"Force Atlas 2","value":"force_atlas_2based"},
                    {"label":"Repulsion","value":"repulsion"},{"label":"Circle","value":"circle"},
                    {"label":"Grid","value":"grid"},{"label":"Dagre (hierarchy)","value":"dagre"},
                ],value="barnes_hut",clearable=False,style={"fontSize":"12px"}),
            ]),
            html.Hr(style=hr),
            html.Div([
                html.Button("▶  Apply Filters",id="apply-btn",
                    style={"width":"100%","padding":"8px","marginBottom":"5px","background":"rgba(86,180,233,0.12)",
                           "border":"1px solid "+WONG["sky_blue"],"borderRadius":"7px","color":WONG["sky_blue"],
                           "fontFamily":"monospace","fontSize":"12px","cursor":"pointer"}),
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

# =============================================================================
# Build Layout — supports single or split graph view
# =============================================================================

def make_cytoscape(cy_id, theme, layout_cfg):
    t=THEMES[theme]
    return cyto.Cytoscape(
        id=cy_id, elements=[], stylesheet=build_stylesheet(theme),
        layout=layout_cfg,
        style={"width":"100%","height":"calc(100vh - 46px)","background":t["bgcolor"]},
        minZoom=0.05, maxZoom=3,
        userZoomingEnabled=True, userPanningEnabled=True,
        boxSelectionEnabled=False, autoungrabify=False, autolock=False,
    )

DEFAULT_LAYOUT = {"name":"cose","animate":True,"randomize":True,
                  "idealEdgeLength":100,"nodeRepulsion":450000,
                  "gravity":0.25,"numIter":1000,"fit":True,"padding":40}

def build_layout(theme="light"):
    t=THEMES[theme]
    legend_overlay,save_btn,zoom_box=build_graph_overlays(theme)
    return html.Div(id="app-container",
        style={"display":"flex","flexDirection":"column","height":"100vh","background":t["bgcolor"],"overflow":"hidden"},
        children=[
            # Topbar
            html.Div(style={"background":t["topbar_bg"],"borderBottom":"1px solid "+t["topbar_bdr"],
                            "padding":"8px 20px","display":"flex","alignItems":"center","gap":"12px","flexShrink":"0"},
                children=[
                    html.Span("Lens GRN Explorer",style={"color":WONG["sky_blue"],"fontFamily":"monospace","fontWeight":"bold","fontSize":"14px"}),
                    html.Span("Gene Regulatory Network — Lachke Lab 2016",style={"color":t["muted"],"fontSize":"12px"}),
                    html.Div(id="topbar-stats",style={"marginLeft":"auto","fontFamily":"monospace","fontSize":"11px","color":t["muted"]}),
                ]),
            # Body
            html.Div(style={"display":"flex","flex":"1","overflow":"hidden"},children=[
                build_sidebar(theme),
                # Graph area — single or split (controlled by graph-mode store)
                html.Div(id="graph-area",children=[
                    # Pane 1 — always visible
                    html.Div(id="graph-pane-1",className="graph-pane",children=[
                        html.Div(id="pane1-label",className="graph-pane-label",style={"display":"none"}),
                        dcc.Loading(id="loading-graph",type="circle",color=WONG["sky_blue"],children=[
                            make_cytoscape("cytoscape-graph",theme,DEFAULT_LAYOUT)
                        ]),
                    ]),
                    # Divider — hidden in single mode
                    html.Div(id="graph-divider",className="graph-divider",style={"display":"none"}),
                    # Pane 2 — hidden in single mode
                    html.Div(id="graph-pane-2",className="graph-pane",style={"display":"none"},children=[
                        html.Div(id="pane2-label",className="graph-pane-label",style={"display":"none"}),
                        dcc.Loading(id="loading-graph-2",type="circle",color=WONG["sky_blue"],children=[
                            make_cytoscape("cytoscape-graph-2",theme,DEFAULT_LAYOUT)
                        ]),
                    ]),
                    # Shared overlays
                    html.Div(id="hover-tooltip",style={
                        "background":"#ffffff" if theme=="light" else "#0f1525",
                        "border":"1px solid #e2e8f0" if theme=="light" else "1px solid #1e2d4a",
                        "color":"#0f172a" if theme=="light" else "#e2e8f0"}),
                    legend_overlay, save_btn, zoom_box,
                ]),
                # Right panel
                html.Div(style={"width":"300px","minWidth":"300px","background":t["panel_bg"],
                                "borderLeft":"1px solid "+t["panel_bdr"],"padding":"14px",
                                "overflowY":"auto","height":"calc(100vh - 46px)","boxSizing":"border-box"},
                    children=[
                        html.Div("Details",style={"fontSize":"10px","fontWeight":"bold","letterSpacing":"0.1em",
                                                   "color":t["muted"],"fontFamily":"monospace","textTransform":"uppercase","marginBottom":"10px"}),
                        html.Div(id="info-panel",children=[empty_panel(theme)]),
                    ]),
            ]),
            dcc.Store(id="theme-store",data=theme),
            dcc.Store(id="graph-store",data={}),
            dcc.Store(id="graph-store-2",data={}),
        ])

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
    Output("filter-tissue-reg","value"),Output("filter-tissue-tgt","value"),
    Output("relationship-filter","value"),Output("max-edges","value"),Output("layout-select","value"),
    Input("reset-btn","n_clicks"),
    Input("tissue-reg-all","n_clicks"),Input("tissue-reg-clear","n_clicks"),
    Input("tissue-tgt-all","n_clicks"),Input("tissue-tgt-clear","n_clicks"),
    prevent_initial_call=True,
)
def reset(n_reset, n_reg_all, n_reg_clear, n_tgt_all, n_tgt_clear):
    from dash import no_update
    ctx = callback_context
    trigger = ctx.triggered[0]["prop_id"] if ctx.triggered else ""
    # Tissue regulator buttons — only clear tissue reg dropdown
    if "tissue-reg-all" in trigger or "tissue-reg-clear" in trigger:
        return (no_update,no_update,no_update,no_update,no_update,
                None,no_update,no_update,no_update,no_update)
    # Tissue target buttons — only clear tissue tgt dropdown
    if "tissue-tgt-all" in trigger or "tissue-tgt-clear" in trigger:
        return (no_update,no_update,no_update,no_update,no_update,
                no_update,None,no_update,no_update,no_update)
    # Full reset
    return None,None,None,None,None,None,None,["activating","inhibiting","no_effect"],300,"barnes_hut"


def get_layout_cfg(layout):
    layout_map={
        "barnes_hut":        {"name":"cose","animate":True,"randomize":True,"idealEdgeLength":100,"nodeRepulsion":450000,"gravity":0.25,"numIter":1000,"fit":True,"padding":40},
        "force_atlas_2based":{"name":"cose","animate":True,"randomize":True,"idealEdgeLength":70, "nodeRepulsion":650000,"gravity":0.1, "numIter":1500,"fit":True,"padding":40},
        "repulsion":         {"name":"cose","animate":True,"randomize":True,"idealEdgeLength":160,"nodeRepulsion":900000,"gravity":0.05,"numIter":1000,"fit":True,"padding":50},
        "circle":            {"name":"circle","animate":True,"fit":True,"padding":30},
        "grid":              {"name":"grid",  "animate":True,"fit":True,"padding":30},
        "dagre":             {"name":"dagre", "animate":True,"rankDir":"TB","fit":True,"padding":30},
    }
    return layout_map.get(layout or "barnes_hut", {"name":"cose","animate":True,"randomize":True,"fit":True,"padding":40})

@app.callback(
    # Graph 1
    Output("cytoscape-graph","elements"),
    Output("cytoscape-graph","stylesheet"),
    Output("cytoscape-graph","layout"),
    Output("graph-store","data"),
    # Graph 2 (split view)
    Output("cytoscape-graph-2","elements"),
    Output("cytoscape-graph-2","layout"),
    Output("graph-store-2","data"),
    # Split view controls
    Output("graph-pane-2","style"),
    Output("graph-divider","style"),
    Output("pane1-label","style"),
    Output("pane1-label","children"),
    Output("pane2-label","style"),
    Output("pane2-label","children"),
    # Stats
    Output("stats-panel","children"),
    Output("topbar-stats","children"),
    Input("apply-btn","n_clicks"),
    State("stage-single","value"),State("stage-from","value"),State("stage-to","value"),
    State("filter-regulator","value"),State("filter-target","value"),
    State("filter-tissue-reg","value"),State("filter-tissue-tgt","value"),
    State("relationship-filter","value"),State("max-edges","value"),
    State("layout-select","value"),State("theme-store","data"),
    prevent_initial_call=True,
)
def apply_filters(n,stage_single,stage_from,stage_to,filter_reg,filter_tgt,
                  filter_t_reg,filter_t_tgt,relationships,max_edges,layout,theme):
    theme=theme or "light"
    cfg=copy.deepcopy(CONFIG)
    cfg.update({"stage_single":stage_single,"stage_from":stage_from,"stage_to":stage_to,
                "filter_regulator":filter_reg,"filter_target":filter_tgt,
                "filter_tissue_reg":filter_t_reg,"filter_tissue_tgt":filter_t_tgt,
                "relationships_include":relationships or ["activating","inhibiting","no_effect"],
                "max_edges":int(max_edges) if max_edges!=9999 else None})

    df=filter_data(BASE_DF.copy(),cfg)
    layout_cfg=get_layout_cfg(layout)
    stylesheet=build_stylesheet(theme)
    hide={"display":"none"}
    show_flex={"display":"flex"}
    show_block={"display":"block"}

    if len(df)==0:
        empty_store={}
        return [],[],layout_cfg,empty_store,[],layout_cfg,empty_store,\
               hide,hide,hide,"",hide,"",_stats_empty(theme),""

    # ── Auto-split: check if filtered data has multiple distinct tissue groups ──
    # Get unique regulator tissues and target tissues in this filtered dataset
    unique_t_reg = set(df["tissue_reg"].dropna().unique()) - {"nan","Unknown",""}
    unique_t_tgt = set(df["tissue_tgt"].dropna().unique()) - {"nan","Unknown",""}

    # Split if there are different tissues between regulators and targets
    # AND the union of tissues has more than one unique value
    all_tissues = unique_t_reg | unique_t_tgt
    do_split = len(all_tissues) > 1 and unique_t_reg != unique_t_tgt

    # Also split if user explicitly set different tissue filters
    if filter_t_reg and filter_t_tgt and filter_t_reg != filter_t_tgt:
        do_split = True

    if do_split:
        # Determine the two tissue groups to show
        # Pane 1: edges grouped by regulator tissue
        # Pane 2: edges grouped by target tissue
        # Use user-selected tissues if set, otherwise use dominant tissues from data
        if filter_t_reg:
            pane1_tissue = filter_t_reg
            pane1_label  = filter_t_reg + " (Regulator tissue)"
        else:
            # Pick most common regulator tissue
            pane1_tissue = df["tissue_reg"].value_counts().index[0]
            pane1_label  = pane1_tissue + " (Regulator tissue)"

        if filter_t_tgt:
            pane2_tissue = filter_t_tgt
            pane2_label  = filter_t_tgt + " (Target tissue)"
        else:
            # Pick most common target tissue that differs from pane1
            tgt_counts = df["tissue_tgt"].value_counts()
            pane2_tissue = None
            for t in tgt_counts.index:
                if t != pane1_tissue:
                    pane2_tissue = t
                    break
            if not pane2_tissue:
                pane2_tissue = tgt_counts.index[0]
            pane2_label = pane2_tissue + " (Target tissue)"

        # Build pane 1: filter by regulator tissue
        cfg1 = copy.deepcopy(cfg)
        cfg1["filter_tissue_reg"] = pane1_tissue
        cfg1["filter_tissue_tgt"] = None
        df1 = filter_data(BASE_DF.copy(), cfg1)

        # Build pane 2: filter by target tissue
        cfg2 = copy.deepcopy(cfg)
        cfg2["filter_tissue_reg"] = None
        cfg2["filter_tissue_tgt"] = pane2_tissue
        df2 = filter_data(BASE_DF.copy(), cfg2)

        G1=build_graph(df1); analysis1=analyze_graph(G1,cfg1)
        G2=build_graph(df2); analysis2=analyze_graph(G2,cfg2)
        el1=build_cytoscape_elements(G1,analysis1,cfg1,EXT_DATA,tissue_tag="")
        el2=build_cytoscape_elements(G2,analysis2,cfg2,EXT_DATA,tissue_tag="__t2")

        def make_store(G,analysis):
            return {
                "nodes":{n:dict(G.nodes[n]) for n in G.nodes()},
                "edges":{u+"|||"+v:{**dict(data),"stages":[str(s) for s in data.get("stages",[])]}
                         for u,v,data in G.edges(data=True)},
                "feedback_nodes":list(analysis.get("feedback_nodes",set())),
                "feedback_edges":[list(e) for e in analysis.get("feedback_edges",set())],
                "self_loops":[list(e) for e in analysis.get("self_loops",[])],
                "hub_genes":analysis.get("hub_genes",[]),
            }

        store1=make_store(G1,analysis1); store2=make_store(G2,analysis2)
        lbl_style={"display":"block"}

        return (el1,stylesheet,layout_cfg,store1,
                el2,layout_cfg,store2,
                show_block,{"width":"3px","background":"#e2e8f0","flexShrink":"0"},
                lbl_style,pane1_label,lbl_style,pane2_label,
                _stats_panel(G1,analysis1,theme),
                "Pane 1: "+_topbar_stats(G1,analysis1)+" | Pane 2: "+_topbar_stats(G2,analysis2))
    else:
        # Single view
        G=build_graph(df); analysis=analyze_graph(G,cfg)
        elements=build_cytoscape_elements(G,analysis,cfg,EXT_DATA)
        g_store={
            "nodes":{n:dict(G.nodes[n]) for n in G.nodes()},
            "edges":{u+"|||"+v:{**dict(data),"stages":[str(s) for s in data.get("stages",[])]}
                     for u,v,data in G.edges(data=True)},
            "feedback_nodes":list(analysis.get("feedback_nodes",set())),
            "feedback_edges":[list(e) for e in analysis.get("feedback_edges",set())],
            "self_loops":[list(e) for e in analysis.get("self_loops",[])],
            "hub_genes":analysis.get("hub_genes",[]),
        }
        return (elements,stylesheet,layout_cfg,g_store,
                [],[],{},
                hide,hide,hide,"",hide,"",
                _stats_panel(G,analysis,theme),_topbar_stats(G,analysis))


@app.callback(
    Output("hover-tooltip","children"),Output("hover-tooltip","style"),
    Input("cytoscape-graph","mouseoverNodeData"),Input("cytoscape-graph","mouseoverEdgeData"),
    Input("cytoscape-graph-2","mouseoverNodeData"),Input("cytoscape-graph-2","mouseoverEdgeData"),
    Input("cytoscape-graph","tapNodeData"),Input("cytoscape-graph","tapEdgeData"),
    Input("cytoscape-graph-2","tapNodeData"),Input("cytoscape-graph-2","tapEdgeData"),
    State("theme-store","data"),prevent_initial_call=True,
)
def update_hover(nh1,eh1,nh2,eh2,nt1,et1,nt2,et2,theme):
    theme=theme or "light"; ctx=callback_context
    trigger=ctx.triggered[0]["prop_id"] if ctx.triggered else ""
    if any(x in trigger for x in ["tapNodeData","tapEdgeData"]): return "",{"display":"none"}
    base={"position":"fixed","zIndex":"99999","pointerEvents":"none","borderRadius":"10px",
          "padding":"10px 14px","fontFamily":"monospace","fontSize":"12px",
          "boxShadow":"0 4px 24px rgba(0,0,0,0.13)","maxWidth":"260px","minWidth":"160px",
          "top":"70px","right":"320px",
          "background":"#ffffff" if theme=="light" else "#0f1525",
          "border":"1px solid #e2e8f0" if theme=="light" else "1px solid #1e2d4a",
          "color":"#0f172a" if theme=="light" else "#e2e8f0"}
    muted="#64748b"; sky=WONG["sky_blue"]; org=WONG["orange"]
    node_hover = nh1 or nh2
    edge_hover = eh1 or eh2
    if "mouseoverNodeData" in trigger and node_hover:
        role_map={"regulator":"Regulator","target":"Target","both":"Regulator & Target","selfloop":"Self-regulatory"}
        role=role_map.get(node_hover.get("role","target"),"—")
        return [
            html.Div(node_hover.get("label",node_hover.get("id","")),style={"fontWeight":"bold","color":sky,"fontSize":"13px","marginBottom":"5px"}),
            html.Div("Role: "+role,style={"color":muted,"fontSize":"11px"}),
            html.Div("Out-edges: "+str(node_hover.get("out_deg",0)),style={"color":muted,"fontSize":"11px"}),
            html.Div("In-edges: "+str(node_hover.get("in_deg",0)),style={"color":muted,"fontSize":"11px"}),
            html.Div("Click for full details →",style={"color":sky,"fontSize":"10px","marginTop":"5px","fontStyle":"italic"}),
        ],{**base,"display":"block"}
    if "mouseoverEdgeData" in trigger and edge_hover:
        rel=edge_hover.get("rel","no_effect")
        rc=WONG["green"] if rel=="activating" else (WONG["vermillion"] if rel=="inhibiting" else muted)
        src=edge_hover.get("source_gene",edge_hover.get("source",""))
        tgt=edge_hover.get("target_gene",edge_hover.get("target",""))
        return [
            html.Div([html.Span(src,style={"color":sky,"fontWeight":"bold"}),html.Span(" → ",style={"color":muted}),html.Span(tgt,style={"color":org,"fontWeight":"bold"})],style={"marginBottom":"5px","fontSize":"12px"}),
            html.Div([html.Span("Relationship: ",style={"color":muted,"fontSize":"11px"}),html.Span(rel.capitalize(),style={"color":rc,"fontSize":"11px","fontWeight":"bold"})]),
            html.Div("Stage(s): "+", ".join(edge_hover.get("stages",[])[:3]),style={"color":muted,"fontSize":"11px","marginTop":"2px"}),
            html.Div("Evidence: "+str(edge_hover.get("count",1))+" record(s)",style={"color":muted,"fontSize":"11px"}),
            html.Div("Click for PubMed links →",style={"color":sky,"fontSize":"10px","marginTop":"5px","fontStyle":"italic"}),
        ],{**base,"display":"block"}
    return "",{"display":"none"}


@app.callback(
    Output("info-panel","children"),
    Input("cytoscape-graph","tapNodeData"),Input("cytoscape-graph","tapEdgeData"),
    Input("cytoscape-graph-2","tapNodeData"),Input("cytoscape-graph-2","tapEdgeData"),
    State("graph-store","data"),State("graph-store-2","data"),
    State("theme-store","data"),prevent_initial_call=True,
)
def on_click(nd1,ed1,nd2,ed2,g_store,g_store2,theme):
    theme=theme or "light"; ctx=callback_context
    trigger=ctx.triggered[0]["prop_id"] if ctx.triggered else ""
    node_data = nd1 if "cytoscape-graph." in trigger else (nd2 if nd2 else None)
    edge_data = ed1 if "cytoscape-graph." in trigger else (ed2 if ed2 else None)
    store = g_store if "cytoscape-graph." in trigger else (g_store2 if g_store2 else g_store)
    if not store: return empty_panel(theme)

    if "tapNodeData" in trigger and node_data:
        # Use label (original gene name) not id (which may have tissue tag)
        node_id = node_data.get("label", node_data.get("id",""))
        G=nx.DiGraph()
        for nid,attrs in store.get("nodes",{}).items(): G.add_node(nid,**attrs)
        for ek,attrs in store.get("edges",{}).items():
            parts=ek.split("|||")
            if len(parts)==2: G.add_edge(parts[0],parts[1],**attrs)
        analysis={"feedback_nodes":set(store.get("feedback_nodes",[])),
                  "feedback_edges":set(tuple(e) for e in store.get("feedback_edges",[])),
                  "self_loops":[tuple(e) for e in store.get("self_loops",[])],
                  "hub_genes":store.get("hub_genes",[])}
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
# Stats
# =============================================================================

def _stats_panel(G,analysis,theme="light"):
    t=THEMES[theme]; lbl={"fontSize":"10px","fontWeight":"bold","letterSpacing":"0.1em","color":t["muted"],"fontFamily":"monospace","textTransform":"uppercase"}
    mut=t["muted"]
    regs=sum(1 for n in G.nodes() if G.nodes[n].get("is_reg") and not G.nodes[n].get("is_tgt"))
    tgts=sum(1 for n in G.nodes() if G.nodes[n].get("is_tgt") and not G.nodes[n].get("is_reg"))
    both=sum(1 for n in G.nodes() if G.nodes[n].get("is_reg") and G.nodes[n].get("is_tgt"))
    loops=len(analysis.get("feedback_loops",[])); sl=len(analysis.get("self_loops",[]))
    hubs=analysis.get("hub_genes",[])
    def row(l,v,c):
        return html.Div(style={"display":"flex","justifyContent":"space-between","marginBottom":"3px"},
            children=[html.Span(l,style={"color":mut,"fontSize":"12px"}),html.Span(str(v),style={"color":c,"fontFamily":"monospace","fontSize":"12px"})])
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
    t=THEMES[theme]; lbl={"fontSize":"10px","fontWeight":"bold","letterSpacing":"0.1em","color":t["muted"],"fontFamily":"monospace","textTransform":"uppercase"}
    return html.Div([html.Label("Network Stats",style=lbl),html.Div("Apply filters to see stats.",style={"color":t["muted"],"fontSize":"12px","marginTop":"4px"})])

def _topbar_stats(G,analysis):
    return "Nodes: "+str(G.number_of_nodes())+"  |  Edges: "+str(G.number_of_edges())+"  |  Feedback loops: "+str(len(analysis.get("feedback_loops",[])))

if __name__ == "__main__":
    print("\n"+"="*60+"\n  Lens GRN — Dash 4.0\n"+"="*60)
    print("  http://127.0.0.1:8050\n")
    server=app.server
    app.run(debug=False,host="0.0.0.0",port=int(os.environ.get("PORT",8050)))