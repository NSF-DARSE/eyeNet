"""
=============================================================================
Lens GRN Explorer — Dash App
=============================================================================
Run:  python app.py
Open: http://127.0.0.1:8050
=============================================================================
"""

import os, copy, tempfile
import pandas as pd
import networkx as nx
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

from grn_network import (
    CONFIG, THEMES, WONG,
    load_data, load_megatable, filter_data,
    build_graph, analyze_graph, visualize, stage_numeric,
)

# =============================================================================
# Startup — load data once
# =============================================================================

print("\n[APP] Loading data...")
BASE_DF     = load_data(CONFIG)
MEGA_LOOKUP = load_megatable(CONFIG)

ALL_STAGES     = sorted(BASE_DF["stage"].dropna().unique(), key=stage_numeric)
ALL_REGULATORS = sorted(BASE_DF["regulator"].dropna().unique())
ALL_TARGETS    = sorted(BASE_DF["target"].dropna().unique())

# Expression gene list — GRN genes that exist in MegaTable
EXPR_GENES = sorted([g for g in MEGA_LOOKUP
                     if g in set(BASE_DF["regulator"].unique()) | set(BASE_DF["target"].unique())])

print("[APP] MegaTable genes available for expression panel: " + str(len(EXPR_GENES)))
print("[APP] Ready\n")

def stage_options(s): return [{"label": x, "value": x} for x in s]
def gene_options(g):  return [{"label": x, "value": x} for x in g]

# =============================================================================
# Dash app
# =============================================================================

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="Lens GRN Explorer",
    suppress_callback_exceptions=True,
)

# ── CSS: fix dropdown selected text visibility ────────────────────────────────
# Dash 2.x uses react-select. We target both old (Select-*) and new (css module) classes.
DROPDOWN_CSS = """
    /* react-select v1/v2 (Dash older) */
    .Select-control                                     { background:#fff !important; border:1px solid #cbd5e1 !important; }
    .Select--single>.Select-control .Select-value,
    .Select--single>.Select-control .Select-value .Select-value-label,
    .Select-value-label                                 { color:#0f172a !important; }
    .Select-placeholder                                 { color:#94a3b8 !important; }
    .Select-input>input, .Select-input, .dash-dropdown input { color:#0f172a !important; background:transparent !important; }
    .Select-menu-outer                                  { background:#fff !important; border:1px solid #cbd5e1 !important; z-index:9999 !important; }
    .Select-option                                      { color:#0f172a !important; background:#fff !important; }
    .Select-option.is-focused                           { background:#f1f5f9 !important; color:#0f172a !important; }
    .Select-option.is-selected                          { background:#dbeafe !important; color:#0f172a !important; }
    .VirtualizedSelectOption                            { color:#0f172a !important; background:#fff !important; }
    .VirtualizedSelectFocusedOption                     { background:#f1f5f9 !important; }

    /* react-select v3+ (Dash 2.x) — css modules use data attributes */
    [class*="control"]   { background:#fff !important; }
    [class*="singleValue"]  { color:#0f172a !important; }
    [class*="placeholder"]  { color:#94a3b8 !important; }
    [class*="Input"] input  { color:#0f172a !important; }
    [class*="option"]       { color:#0f172a !important; background:#fff !important; }
    [class*="option"]:hover { background:#f1f5f9 !important; }
    [class*="menu"]         { background:#fff !important; border:1px solid #cbd5e1 !important; z-index:9999 !important; }
    [class*="MenuList"]     { background:#fff !important; }

    /* Scrollbar */
    ::-webkit-scrollbar { width:4px; }
    ::-webkit-scrollbar-thumb { background:#334155; border-radius:2px; }
"""

app.index_string = (
    "<!DOCTYPE html><html><head>"
    "{%metas%}<title>{%title%}</title>{%favicon%}{%css%}"
    "<style>" + DROPDOWN_CSS + "</style>"
    "</head><body>"
    "{%app_entry%}"
    "<footer>{%config%}{%scripts%}{%renderer%}</footer>"
    "</body></html>"
)


# =============================================================================
# Expression data panel
# =============================================================================

def build_expression_panel(gene_name, mega_lookup, theme="dark"):
    t         = THEMES[theme]
    lbl_color = "#64748b" if theme == "dark" else "#475569"
    muted     = "#94a3b8" if theme == "dark" else "#64748b"
    txt       = t["tooltip_text"]
    row_bg    = "#1a2540" if theme == "dark" else "#f1f5f9"
    label_style = {"color": lbl_color, "fontSize": "10px", "fontWeight": "bold",
                   "letterSpacing": "0.1em", "fontFamily": "monospace", "display": "block"}

    if not gene_name or gene_name not in mega_lookup:
        return html.Div(
            "No expression data for this gene." if gene_name else "Select a gene above.",
            style={"color": muted, "fontSize": "12px", "fontStyle": "italic", "marginTop": "6px"}
        )

    data    = mega_lookup[gene_name]
    entrez  = data.get("entrez", "")
    uniprot = data.get("uniprot", "")
    desc    = data.get("description", "")

    # Short column name formatter
    def short(k):
        return (k.replace("_exp","").replace("_enr","").replace("enr_","")
                 .replace("Beebe_","B_").replace("Naka_","N_")
                 .replace("_Cv","").replace("_Rob","_R"))

    def make_table(title, data_dict):
        non_null = {k: v for k, v in data_dict.items() if v is not None}
        if not non_null:
            return html.Div()
        rows = []
        for k, v in non_null.items():
            val_color = WONG["green"] if v > 0 else (WONG["vermillion"] if v < 0 else txt)
            rows.append(html.Tr([
                html.Td(short(k), style={"padding":"2px 6px","color":muted,"fontSize":"11px","fontFamily":"monospace"}),
                html.Td(str(v),   style={"padding":"2px 6px","color":val_color,"fontSize":"11px","fontFamily":"monospace","textAlign":"right"}),
            ]))
        return html.Div([
            html.Div(title, style={"fontSize":"10px","color":muted,"fontWeight":"bold","marginBottom":"2px","marginTop":"8px"}),
            html.Table(rows, style={"width":"100%","borderCollapse":"collapse","background":row_bg,"borderRadius":"4px"}),
        ])

    children = [
        # Gene name + links
        html.Div([
            html.Span(gene_name, style={"color": WONG["sky_blue"], "fontFamily":"monospace",
                                         "fontWeight":"bold", "fontSize":"14px"}),
        ], style={"marginBottom":"4px"}),
        html.Div([
            html.A("NCBI Gene ↗", href="https://www.ncbi.nlm.nih.gov/gene/"+entrez,
                   target="_blank", style={"color":WONG["sky_blue"],"fontSize":"11px","marginRight":"10px"})
            if entrez and entrez != "nan" else html.Span(),
            html.A("UniProt ↗", href="https://www.uniprot.org/uniprot/"+uniprot,
                   target="_blank", style={"color":WONG["sky_blue"],"fontSize":"11px"})
            if uniprot and uniprot != "nan" else html.Span(),
        ], style={"marginBottom":"6px"}),
    ]

    if desc and desc != "nan":
        children.append(
            html.Div(desc[:80]+("..." if len(desc)>80 else ""),
                     style={"color":muted,"fontSize":"10px","fontStyle":"italic","marginBottom":"6px"})
        )

    children += [
        make_table("Microarray Expression (raw counts)", data.get("microarray_exp",{})),
        make_table("Microarray Enrichment (fiber/epi ratio)", data.get("microarray_enr",{})),
        make_table("RNA-seq Enrichment (FC vs LEC)", data.get("rnaseq",{})),
        html.Div([
            html.Span("■ ", style={"color":WONG["green"]}),
            html.Span("Positive  ", style={"fontSize":"10px","color":muted}),
            html.Span("■ ", style={"color":WONG["vermillion"]}),
            html.Span("Negative", style={"fontSize":"10px","color":muted}),
        ], style={"marginTop":"8px"}),
    ]

    return html.Div(children)


# =============================================================================
# Sidebar + Layout
# =============================================================================

def build_sidebar(theme="dark"):
    sb_bg  = "#0f1525" if theme == "dark" else "#f1f5f9"
    sb_br  = "#1e2d4a" if theme == "dark" else "#e2e8f0"
    txt    = "#e2e8f0" if theme == "dark" else "#0f172a"
    muted  = "#94a3b8" if theme == "dark" else "#64748b"
    lbl    = "#64748b" if theme == "dark" else "#475569"
    hr     = {"borderColor": "#1e2d4a" if theme=="dark" else "#e2e8f0", "margin":"0"}
    label  = {"color":lbl,"fontSize":"10px","fontWeight":"bold","letterSpacing":"0.1em","fontFamily":"monospace"}
    sub    = {"color":muted,"fontSize":"12px","marginTop":"8px"}

    return html.Div(
        style={"width":"300px","minWidth":"300px","background":sb_bg,
               "borderRight":"1px solid "+sb_br,"padding":"16px",
               "overflowY":"auto","display":"flex","flexDirection":"column","gap":"14px"},
        children=[

            # Title + theme toggle
            html.Div(style={"display":"flex","justifyContent":"space-between","alignItems":"flex-start"}, children=[
                html.Div([
                    html.H5("Lens GRN Explorer",
                            style={"color":WONG["sky_blue"],"fontFamily":"monospace","marginBottom":"2px","fontSize":"14px"}),
                    html.Small("Lachke Lab 2016", style={"color":muted,"fontSize":"11px"}),
                ]),
                html.Div(id="theme-toggle", n_clicks=0,
                    style={"cursor":"pointer","display":"flex","flexDirection":"column",
                           "alignItems":"center","gap":"2px","padding":"6px 8px",
                           "border":"1px solid "+sb_br,"borderRadius":"6px"},
                    children=[
                        html.Div("☀️" if theme=="light" else "🌙", style={"fontSize":"16px"}),
                        html.Div("Light" if theme=="light" else "Dark",
                                 style={"fontSize":"9px","color":muted,"fontFamily":"monospace"}),
                    ]),
            ]),

            html.Hr(style=hr),

            # Stage Filter
            html.Div([
                html.Label("STAGE FILTER", style=label),
                html.Label("Single Stage", style=sub),
                dcc.Dropdown(id="stage-single", options=stage_options(ALL_STAGES),
                             placeholder="All stages", clearable=True, style={"fontSize":"12px"}),
                html.Label("Stage Range", style={**sub,"marginTop":"10px"}),
                html.Div([
                    dcc.Dropdown(id="stage-from", options=stage_options(ALL_STAGES),
                                 placeholder="From", clearable=True, style={"flex":"1","fontSize":"12px"}),
                    html.Span("to", style={"color":muted,"padding":"0 6px","alignSelf":"center","fontSize":"11px"}),
                    dcc.Dropdown(id="stage-to", options=stage_options(ALL_STAGES),
                                 placeholder="To", clearable=True, style={"flex":"1","fontSize":"12px"}),
                ], style={"display":"flex","alignItems":"center","gap":"4px"}),
            ]),

            html.Hr(style=hr),

            # Gene Filter
            html.Div([
                html.Label("GENE FILTER", style=label),
                html.Label("Regulator", style=sub),
                dcc.Dropdown(id="filter-regulator", options=gene_options(ALL_REGULATORS),
                             placeholder="All regulators", clearable=True, style={"fontSize":"12px"}),
                html.Label("Target", style={**sub,"marginTop":"10px"}),
                dcc.Dropdown(id="filter-target", options=gene_options(ALL_TARGETS),
                             placeholder="All targets", clearable=True, style={"fontSize":"12px"}),
            ]),

            html.Hr(style=hr),

            # Relationship Filter
            html.Div([
                html.Label("RELATIONSHIP FILTER", style=label),
                html.Div(style={"marginTop":"8px"}, children=[
                    dcc.Checklist(
                        id="relationship-filter",
                        options=[
                            {"label":"  Activating","value":"activating"},
                            {"label":"  Inhibiting","value":"inhibiting"},
                            {"label":"  No effect", "value":"no_effect"},
                        ],
                        value=["activating","inhibiting","no_effect"],
                        labelStyle={"display":"block","color":txt,"fontSize":"13px","marginBottom":"4px"},
                    )
                ]),
                html.Div("True relationship = Perturbation x Effect",
                         style={"fontSize":"10px","color":muted,"marginTop":"4px","fontFamily":"monospace"}),
            ]),

            html.Hr(style=hr),

            # Display Options
            html.Div([
                html.Label("DISPLAY OPTIONS", style=label),
                html.Label("Max edges", style=sub),
                dcc.Dropdown(id="max-edges",
                    options=[{"label":"100 edges","value":100},{"label":"300 edges","value":300},
                             {"label":"600 edges","value":600},{"label":"All edges","value":9999}],
                    value=300, clearable=False, style={"fontSize":"12px"}),
                html.Label("Layout", style={**sub,"marginTop":"10px"}),
                dcc.Dropdown(id="layout-select",
                    options=[{"label":"Barnes Hut","value":"barnes_hut"},
                             {"label":"Force Atlas 2","value":"force_atlas_2based"},
                             {"label":"Repulsion","value":"repulsion"}],
                    value="barnes_hut", clearable=False, style={"fontSize":"12px"}),
            ]),

            html.Hr(style=hr),

            # Buttons
            html.Div([
                html.Button("Apply Filters", id="apply-btn", style={
                    "width":"100%","padding":"9px",
                    "background":"rgba(86,180,233,0.15)",
                    "border":"1px solid "+WONG["sky_blue"],"borderRadius":"7px",
                    "color":WONG["sky_blue"],"fontFamily":"monospace",
                    "fontSize":"12px","cursor":"pointer","marginBottom":"6px"}),
                html.Button("Reset All", id="reset-btn", style={
                    "width":"100%","padding":"9px","background":"transparent",
                    "border":"1px solid "+sb_br,"borderRadius":"7px",
                    "color":muted,"fontFamily":"monospace","fontSize":"12px","cursor":"pointer"}),
            ]),

            html.Hr(style=hr),

            # Network stats
            html.Div(id="stats-panel", children=[
                html.Label("NETWORK STATS", style=label),
                html.Div("Apply filters to see stats.",
                         style={"color":muted,"fontSize":"12px","marginTop":"6px"}),
            ]),

            html.Hr(style=hr),

            # Expression data panel
            html.Div([
                html.Label("EXPRESSION DATA", style=label),
                html.Div([
                    html.Span("&#128202; ", style={"fontSize":"11px"}),
                    html.Span("Microarray expression + enrichment + RNA-seq enrichment",
                              style={"color":muted,"fontSize":"10px","fontFamily":"monospace"}),
                ], style={"marginBottom":"6px","marginTop":"4px"}),
                html.Label("Select gene", style=sub),
                dcc.Dropdown(
                    id="expr-gene-select",
                    options=gene_options(EXPR_GENES),
                    placeholder="Choose a gene (" + str(len(EXPR_GENES)) + " available)...",
                    clearable=True,
                    style={"fontSize":"12px"},
                ),
                html.Div(id="expression-panel",
                         style={"marginTop":"8px","fontSize":"12px","color":txt}),
            ]),

            html.Hr(style=hr),

            # Legend
            html.Div([
                html.Label("LEGEND", style={**label,"marginBottom":"8px"}),
                html.Div([html.Span("●",style={"color":WONG["sky_blue"]}), " Regulator only"],
                         style={"fontSize":"12px","marginBottom":"3px","color":txt}),
                html.Div([html.Span("●",style={"color":WONG["orange"]}), " Target only"],
                         style={"fontSize":"12px","marginBottom":"3px","color":txt}),
                html.Div([html.Span("●",style={"color":WONG["pink"]}), " Regulator & Target"],
                         style={"fontSize":"12px","marginBottom":"3px","color":txt}),
                html.Div([html.Span("●",style={"color":WONG["yellow"]}), " Self-regulatory loop"],
                         style={"fontSize":"12px","marginBottom":"3px","color":txt}),
                html.Div([html.Span("○",style={"color":"#ffffff","fontWeight":"bold"}), " White border = has expression data"],
                         style={"fontSize":"12px","marginBottom":"10px","color":txt}),
                html.Div([html.Span("▶",style={"color":WONG["green"]}), " Activating"],
                         style={"fontSize":"12px","marginBottom":"3px","color":txt}),
                html.Div([html.Span("▶",style={"color":WONG["vermillion"]}), " Inhibiting"],
                         style={"fontSize":"12px","marginBottom":"3px","color":txt}),
                html.Div([html.Span("▶",style={"color":"#94a3b8"}), " No effect"],
                         style={"fontSize":"12px","marginBottom":"8px","color":txt}),
                html.Div("Hover any node/edge for details",
                         style={"fontSize":"10px","color":muted,"fontFamily":"monospace"}),
                html.Div("Links stay open (sticky tooltip)",
                         style={"fontSize":"10px","color":muted,"fontFamily":"monospace"}),
                html.Div("Wong (2011) color-blind safe",
                         style={"fontSize":"10px","color":muted,"fontFamily":"monospace","marginTop":"3px"}),
            ]),
        ],
    )


def build_layout(theme="dark"):
    t         = THEMES[theme]
    bg        = t["bgcolor"]
    topbar_bg = "#0f1525" if theme=="dark" else "#ffffff"
    topbar_br = "#1e2d4a" if theme=="dark" else "#e2e8f0"
    muted     = "#64748b"

    placeholder = (
        "<div style='color:#64748b;font-family:monospace;padding:40px;font-size:14px;"
        "background:" + bg + ";height:100%'>"
        "Click <b style='color:" + WONG["sky_blue"] + "'>Apply Filters</b> to render the network.<br/><br/>"
        "<span style='font-size:12px;color:#475569'>"
        "Hover over edges for sticky PubMed links &nbsp;|&nbsp; "
        "Hover over nodes for expression data info"
        "</span></div>"
    )

    return html.Div(
        id="app-container",
        style={"display":"flex","flexDirection":"column","height":"100vh",
               "background":bg,"color":t["font_color"],"fontFamily":"sans-serif","overflow":"hidden"},
        children=[
            html.Div(
                style={"background":topbar_bg,"borderBottom":"1px solid "+topbar_br,
                       "padding":"10px 20px","display":"flex","alignItems":"center","gap":"12px","flexShrink":"0"},
                children=[
                    html.H4("Lens GRN Explorer",
                            style={"color":WONG["sky_blue"],"fontFamily":"monospace",
                                   "fontSize":"14px","margin":"0","fontWeight":"700"}),
                    html.Span("Gene Regulatory Network — Lachke Lab 2016",
                              style={"color":muted,"fontSize":"12px"}),
                    html.Div(id="topbar-stats",
                             style={"marginLeft":"auto","fontFamily":"monospace","fontSize":"11px","color":muted}),
                ],
            ),
            html.Div(
                style={"display":"flex","flex":"1","overflow":"hidden"},
                children=[
                    build_sidebar(theme),
                    html.Div(
                        style={"flex":"1","display":"flex","flexDirection":"column","overflow":"hidden"},
                        children=[
                            dcc.Loading(
                                id="loading-graph", type="circle", color=WONG["sky_blue"],
                                children=[
                                    html.Iframe(
                                        id="graph-frame",
                                        style={"width":"100%","height":"calc(100vh - 48px)",
                                               "border":"none","flex":"1","background":bg,"display":"block"},
                                        srcDoc=placeholder,
                                    )
                                ],
                                style={"height":"calc(100vh - 48px)","display":"block"},
                            ),
                        ],
                    ),
                ],
            ),
            dcc.Store(id="theme-store", data=theme),
        ],
    )


app.layout = build_layout("dark")


# =============================================================================
# Callbacks
# =============================================================================

@app.callback(
    Output("theme-store","data"),
    Input("theme-toggle","n_clicks"),
    State("theme-store","data"),
    prevent_initial_call=True,
)
def toggle_theme(n, cur):
    return "light" if cur=="dark" else "dark"


@app.callback(
    Output("app-container","children"),
    Input("theme-store","data"),
    prevent_initial_call=True,
)
def update_theme(theme):
    return build_layout(theme).children


@app.callback(
    Output("stage-from","value"), Output("stage-to","value"),
    Output("stage-single","value"), Output("filter-regulator","value"),
    Output("filter-target","value"), Output("relationship-filter","value"),
    Output("max-edges","value"), Output("layout-select","value"),
    Input("reset-btn","n_clicks"),
    prevent_initial_call=True,
)
def reset_filters(_):
    return None,None,None,None,None,["activating","inhibiting","no_effect"],300,"barnes_hut"


@app.callback(
    Output("graph-frame","srcDoc"),
    Output("stats-panel","children"),
    Output("topbar-stats","children"),
    Input("apply-btn","n_clicks"),
    State("stage-single","value"), State("stage-from","value"), State("stage-to","value"),
    State("filter-regulator","value"), State("filter-target","value"),
    State("relationship-filter","value"), State("max-edges","value"),
    State("layout-select","value"), State("theme-store","data"),
    prevent_initial_call=True,
)
def update_graph(n, stage_single, stage_from, stage_to,
                 filter_reg, filter_tgt, relationships, max_edges, layout, theme):

    cfg = copy.deepcopy(CONFIG)
    cfg["stage_single"]          = stage_single
    cfg["stage_from"]            = stage_from
    cfg["stage_to"]              = stage_to
    cfg["filter_regulator"]      = filter_reg
    cfg["filter_target"]         = filter_tgt
    cfg["relationships_include"] = relationships or ["activating","inhibiting","no_effect"]
    cfg["max_edges"]             = int(max_edges) if max_edges != 9999 else None
    cfg["layout"]                = layout
    cfg["theme"]                 = theme or "dark"
    cfg["height"]                = "100vh"
    cfg["output_file"]           = os.path.join(tempfile.gettempdir(), "grn_dash_output.html")

    df = filter_data(BASE_DF.copy(), cfg)
    if len(df) == 0:
        t     = THEMES[cfg["theme"]]
        empty = ("<div style='color:#ef4444;font-family:monospace;padding:40px;"
                 "font-size:14px;background:" + t["bgcolor"] + ";height:100%'>"
                 "No edges match the current filters.</div>")
        return empty, _stats_panel_empty(cfg["theme"]), ""

    G        = build_graph(df)
    analysis = analyze_graph(G, cfg)
    out_path = visualize(G, analysis, cfg, MEGA_LOOKUP)

    with open(out_path,"r",encoding="utf-8") as f:
        html_content = f.read()

    return html_content, _stats_panel(G, analysis, cfg["theme"]), _topbar_stats(G, analysis)


@app.callback(
    Output("expression-panel","children"),
    Input("expr-gene-select","value"),
    State("theme-store","data"),
    prevent_initial_call=True,
)
def update_expression(gene, theme):
    return build_expression_panel(gene, MEGA_LOOKUP, theme or "dark")


# =============================================================================
# Stats helpers
# =============================================================================

def _stats_panel(G, analysis, theme="dark"):
    lbl   = "#64748b" if theme=="dark" else "#475569"
    muted = "#94a3b8" if theme=="dark" else "#64748b"
    t     = THEMES[theme]
    regs  = sum(1 for n in G.nodes() if G.nodes[n].get("is_reg") and not G.nodes[n].get("is_tgt"))
    tgts  = sum(1 for n in G.nodes() if G.nodes[n].get("is_tgt") and not G.nodes[n].get("is_reg"))
    both  = sum(1 for n in G.nodes() if G.nodes[n].get("is_reg") and G.nodes[n].get("is_tgt"))
    loops = len(analysis.get("feedback_loops",[]))
    sl    = len(analysis.get("self_loops",[]))
    hubs  = analysis.get("hub_genes",[])
    ls    = {"color":lbl,"fontSize":"10px","fontWeight":"bold","letterSpacing":"0.1em","fontFamily":"monospace"}

    def row(l, v, c):
        return html.Div(style={"display":"flex","justifyContent":"space-between","marginBottom":"4px","fontSize":"12px"},
            children=[html.Span(l,style={"color":muted}), html.Span(str(v),style={"color":c,"fontFamily":"monospace"})])

    return html.Div([
        html.Label("NETWORK STATS", style={**ls,"marginBottom":"8px","display":"block"}),
        row("Nodes",          G.number_of_nodes(), WONG["sky_blue"]),
        row("Edges",          G.number_of_edges(), WONG["sky_blue"]),
        row("Regulators",     regs,  WONG["sky_blue"]),
        row("Targets",        tgts,  WONG["orange"]),
        row("Both",           both,  WONG["pink"]),
        row("Feedback loops", loops, WONG["orange"]),
        row("Self-loops",     sl,    WONG["yellow"]),
        html.Div(style={"marginTop":"10px"}, children=[
            html.Label("TOP HUBS", style={**ls,"display":"block","marginBottom":"6px"}),
            *[html.Div(style={"display":"flex","justifyContent":"space-between","marginBottom":"2px","fontSize":"11px"},
                children=[html.Span(g,style={"color":t["tooltip_text"],"fontFamily":"monospace"}),
                           html.Span(str(d),style={"color":WONG["sky_blue"],"fontFamily":"monospace"})])
              for g, d in hubs[:6]],
        ]),
    ])


def _stats_panel_empty(theme="dark"):
    lbl = "#64748b" if theme=="dark" else "#475569"
    return html.Div([
        html.Label("NETWORK STATS", style={"color":lbl,"fontSize":"10px","fontWeight":"bold",
                                           "letterSpacing":"0.1em","fontFamily":"monospace"}),
        html.Div("No edges match filters.", style={"color":WONG["vermillion"],"fontSize":"12px","marginTop":"6px"}),
    ])


def _topbar_stats(G, analysis):
    return ("Nodes: " + str(G.number_of_nodes()) +
            "  |  Edges: " + str(G.number_of_edges()) +
            "  |  Feedback loops: " + str(len(analysis.get("feedback_loops",[]))))


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60 + "\n  Lens GRN Dash App\n" + "="*60)
    print("  http://127.0.0.1:8050\n")
    server = app.server
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8050)))