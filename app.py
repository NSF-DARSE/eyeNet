"""
=============================================================================
Lens GRN Explorer — Dash App
=============================================================================
Run:
    python app.py

Then open:  http://127.0.0.1:8050
=============================================================================
"""

import os
import copy
import tempfile
import pandas as pd
import networkx as nx

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

from grn_network import (
    CONFIG,
    THEMES,
    WONG,
    load_data,
    filter_data,
    build_graph,
    analyze_graph,
    visualize,
    stage_numeric,
)


# =============================================================================
# Load data once at startup
# =============================================================================

print("\n[APP] Loading GRN data...")
BASE_DF = load_data(CONFIG)

ALL_STAGES     = sorted(BASE_DF["stage"].dropna().unique(), key=stage_numeric)
ALL_REGULATORS = sorted(BASE_DF["regulator"].dropna().unique())
ALL_TARGETS    = sorted(BASE_DF["target"].dropna().unique())

print(f"[APP] {len(BASE_DF):,} edges | {len(ALL_STAGES)} stages | "
      f"{len(ALL_REGULATORS)} regulators | {len(ALL_TARGETS)} targets\n")


# =============================================================================
# Helpers
# =============================================================================

def stage_options(stages):
    return [{"label": s, "value": s} for s in stages]

def gene_options(genes):
    return [{"label": g, "value": g} for g in genes]


# =============================================================================
# App init
# =============================================================================

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    title="Lens GRN Explorer",
    suppress_callback_exceptions=True,
)

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* Dropdown text colors */
            .Select-value-label { color: #0f172a !important; }
            .Select-placeholder { color: #64748b !important; }
            .Select-input input  { color: #0f172a !important; }
            .Select-menu-outer   { background: #1e293b !important; }
            .Select-option       { color: #e2e8f0 !important; }
            .Select-option.is-focused { background: #334155 !important; }
            .dash-dropdown .Select-control    { background: #f8fafc !important; }
            .dash-dropdown .Select-value-label { color: #0f172a !important; }

            /* Theme transition */
            body, .sidebar-div { transition: background 0.3s, color 0.3s; }

            /* Light mode sidebar override */
            body.light-mode .sidebar-div {
                background: #f1f5f9 !important;
                border-right: 1px solid #e2e8f0 !important;
                color: #0f172a !important;
            }
            body.light-mode { background: #f8fafc !important; }

            /* Toggle switch style */
            .theme-toggle {
                display: flex;
                align-items: center;
                gap: 8px;
                cursor: pointer;
                font-size: 12px;
                font-family: monospace;
            }
            .toggle-track {
                width: 36px; height: 18px;
                border-radius: 9px;
                background: #334155;
                position: relative;
                transition: background 0.2s;
            }
            .toggle-track.light { background: #56B4E9; }
            .toggle-thumb {
                width: 14px; height: 14px;
                border-radius: 50%;
                background: white;
                position: absolute;
                top: 2px; left: 2px;
                transition: left 0.2s;
            }
            .toggle-track.light .toggle-thumb { left: 20px; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''


# =============================================================================
# Layout builder — theme-aware
# =============================================================================

def build_sidebar(theme="dark"):
    t = THEMES[theme]
    label_style  = {"color": "#64748b" if theme == "dark" else "#475569",
                    "fontSize": "10px", "fontWeight": "bold",
                    "letterSpacing": "0.1em", "fontFamily": "monospace"}
    sub_style    = {"color": "#94a3b8" if theme == "dark" else "#64748b",
                    "fontSize": "12px", "marginTop": "8px"}
    hr_style     = {"borderColor": "#1e2d4a" if theme == "dark" else "#e2e8f0", "margin": "0"}
    sidebar_bg   = "#0f1525" if theme == "dark" else "#f1f5f9"
    sidebar_bdr  = "#1e2d4a" if theme == "dark" else "#e2e8f0"
    text_color   = "#e2e8f0" if theme == "dark" else "#0f172a"
    muted_color  = "#475569" if theme == "dark" else "#64748b"

    return html.Div(
        className="sidebar-div",
        style={
            "width": "300px", "minWidth": "300px",
            "background": sidebar_bg,
            "borderRight": f"1px solid {sidebar_bdr}",
            "padding": "16px", "overflowY": "auto",
            "display": "flex", "flexDirection": "column", "gap": "14px",
        },
        children=[

            # Title + theme toggle
            html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "flex-start"}, children=[
                html.Div([
                    html.H5("🔬 Lens GRN Explorer",
                            style={"color": WONG["sky_blue"], "fontFamily": "monospace",
                                   "marginBottom": "2px", "fontSize": "14px"}),
                    html.Small("Lachke Lab 2016 — Gene Regulatory Network",
                               style={"color": muted_color, "fontSize": "11px"}),
                ]),
                # Theme toggle
                html.Div(
                    id="theme-toggle",
                    style={"cursor": "pointer", "display": "flex", "flexDirection": "column",
                           "alignItems": "center", "gap": "3px"},
                    children=[
                        html.Div("☀️" if theme == "light" else "🌙",
                                 style={"fontSize": "16px"}),
                        html.Div("Light" if theme == "light" else "Dark",
                                 style={"fontSize": "9px", "color": muted_color,
                                        "fontFamily": "monospace"}),
                    ]
                ),
            ]),

            html.Hr(style=hr_style),

            # Stage Filter
            html.Div([
                html.Label("STAGE FILTER", style=label_style),
                html.Label("Single Stage", style=sub_style),
                dcc.Dropdown(id="stage-single", options=stage_options(ALL_STAGES),
                             placeholder="All stages", clearable=True,
                             style={"fontSize": "12px"}),
                html.Label("Stage Range", style={**sub_style, "marginTop": "10px"}),
                html.Div([
                    dcc.Dropdown(id="stage-from", options=stage_options(ALL_STAGES),
                                 placeholder="From", clearable=True,
                                 style={"flex": "1", "fontSize": "12px"}),
                    html.Span("→", style={"color": muted_color, "padding": "0 6px", "alignSelf": "center"}),
                    dcc.Dropdown(id="stage-to", options=stage_options(ALL_STAGES),
                                 placeholder="To", clearable=True,
                                 style={"flex": "1", "fontSize": "12px"}),
                ], style={"display": "flex", "alignItems": "center", "gap": "4px"}),
            ]),

            html.Hr(style=hr_style),

            # Gene Filter
            html.Div([
                html.Label("GENE FILTER", style=label_style),
                html.Label("Regulator", style=sub_style),
                dcc.Dropdown(id="filter-regulator", options=gene_options(ALL_REGULATORS),
                             placeholder="All regulators", clearable=True,
                             style={"fontSize": "12px"}),
                html.Label("Target", style={**sub_style, "marginTop": "10px"}),
                dcc.Dropdown(id="filter-target", options=gene_options(ALL_TARGETS),
                             placeholder="All targets", clearable=True,
                             style={"fontSize": "12px"}),
            ]),

            html.Hr(style=hr_style),

            # Relationship Filter
            html.Div([
                html.Label("RELATIONSHIP FILTER", style=label_style),
                html.Div(style={"marginTop": "8px"}, children=[
                    dcc.Checklist(
                        id="relationship-filter",
                        options=[
                            {"label": f"  ▲ Activating", "value": "activating"},
                            {"label": f"  ▼ Inhibiting", "value": "inhibiting"},
                            {"label": f"  ○ No effect",  "value": "no_effect"},
                        ],
                        value=["activating", "inhibiting", "no_effect"],
                        labelStyle={"display": "block", "color": text_color,
                                    "fontSize": "13px", "marginBottom": "4px"},
                    )
                ]),
                html.Div("True relationship = Perturbation × Effect",
                         style={"fontSize": "10px", "color": muted_color, "marginTop": "6px",
                                "fontFamily": "monospace"}),
            ]),

            html.Hr(style=hr_style),

            # Display Options
            html.Div([
                html.Label("DISPLAY OPTIONS", style=label_style),
                html.Label("Max edges", style=sub_style),
                dcc.Dropdown(id="max-edges",
                             options=[{"label": "100 edges", "value": 100},
                                      {"label": "300 edges", "value": 300},
                                      {"label": "600 edges", "value": 600},
                                      {"label": "All edges", "value": 9999}],
                             value=300, clearable=False, style={"fontSize": "12px"}),
                html.Label("Layout", style={**sub_style, "marginTop": "10px"}),
                dcc.Dropdown(id="layout-select",
                             options=[{"label": "Barnes Hut",   "value": "barnes_hut"},
                                      {"label": "Force Atlas 2","value": "force_atlas_2based"},
                                      {"label": "Repulsion",    "value": "repulsion"}],
                             value="barnes_hut", clearable=False, style={"fontSize": "12px"}),
            ]),

            html.Hr(style=hr_style),

            # Buttons
            html.Div([
                html.Button("▶  Apply Filters", id="apply-btn", style={
                    "width": "100%", "padding": "9px",
                    "background": f"rgba(86,180,233,0.15)",
                    "border": f"1px solid {WONG['sky_blue']}", "borderRadius": "7px",
                    "color": WONG["sky_blue"], "fontFamily": "monospace",
                    "fontSize": "12px", "cursor": "pointer", "marginBottom": "6px",
                }),
                html.Button("↺  Reset All", id="reset-btn", style={
                    "width": "100%", "padding": "9px",
                    "background": "transparent",
                    "border": f"1px solid {sidebar_bdr}", "borderRadius": "7px",
                    "color": muted_color, "fontFamily": "monospace",
                    "fontSize": "12px", "cursor": "pointer",
                }),
            ]),

            html.Hr(style=hr_style),

            # Stats panel
            html.Div(id="stats-panel", children=[
                html.Label("NETWORK STATS", style=label_style),
                html.Div("Apply filters to see stats.",
                         style={"color": muted_color, "fontSize": "12px", "marginTop": "6px"}),
            ]),

            html.Hr(style=hr_style),

            # Legend
            html.Div([
                html.Label("LEGEND", style={**label_style, "marginBottom": "8px"}),
                html.Div([html.Span("●", style={"color": WONG["sky_blue"]}), " Regulator only"],
                         style={"fontSize": "12px", "marginBottom": "3px", "color": text_color}),
                html.Div([html.Span("●", style={"color": WONG["orange"]}), " Target only"],
                         style={"fontSize": "12px", "marginBottom": "3px", "color": text_color}),
                html.Div([html.Span("●", style={"color": WONG["pink"]}), " Regulator & Target"],
                         style={"fontSize": "12px", "marginBottom": "3px", "color": text_color}),
                html.Div([html.Span("●", style={"color": WONG["yellow"]}), " Self-regulatory loop"],
                         style={"fontSize": "12px", "marginBottom": "10px", "color": text_color}),
                html.Div([html.Span("━━▶", style={"color": WONG["green"]}), " Activating"],
                         style={"fontSize": "12px", "marginBottom": "3px", "color": text_color}),
                html.Div([html.Span("━━▶", style={"color": WONG["vermillion"]}), " Inhibiting"],
                         style={"fontSize": "12px", "marginBottom": "3px", "color": text_color}),
                html.Div([html.Span("━━▶", style={"color": "#94a3b8"}), " No effect"],
                         style={"fontSize": "12px", "color": text_color}),
                html.Div("Wong (2011) color-blind safe palette",
                         style={"fontSize": "10px", "color": muted_color,
                                "marginTop": "8px", "fontFamily": "monospace"}),
            ]),
        ],
    )


def build_layout(theme="dark"):
    t        = THEMES[theme]
    bg       = t["bgcolor"]
    topbar_bg = "#0f1525" if theme == "dark" else "#ffffff"
    topbar_bdr = "#1e2d4a" if theme == "dark" else "#e2e8f0"
    muted    = "#64748b"

    return html.Div(
        style={"display": "flex", "flexDirection": "column", "height": "100vh",
               "background": bg, "color": t["font_color"],
               "fontFamily": "DM Sans, sans-serif", "overflow": "hidden"},
        children=[
            # Top bar
            html.Div(
                style={"background": topbar_bg, "borderBottom": f"1px solid {topbar_bdr}",
                       "padding": "10px 20px", "display": "flex",
                       "alignItems": "center", "gap": "12px", "flexShrink": "0"},
                children=[
                    html.H4("🔬 Lens GRN Explorer",
                            style={"color": WONG["sky_blue"], "fontFamily": "monospace",
                                   "fontSize": "14px", "margin": "0", "fontWeight": "700"}),
                    html.Span("Gene Regulatory Network — Lachke Lab 2016",
                              style={"color": muted, "fontSize": "12px"}),
                    html.Div(id="topbar-stats",
                             style={"marginLeft": "auto", "fontFamily": "monospace",
                                    "fontSize": "11px", "color": muted}),
                ],
            ),
            # Body
            html.Div(
                style={"display": "flex", "flex": "1", "overflow": "hidden"},
                children=[
                    build_sidebar(theme),
                    # Graph panel
                    html.Div(
                        style={"flex": "1", "display": "flex", "flexDirection": "column", "overflow": "hidden"},
                        children=[
                            dcc.Loading(
                                id="loading-graph", type="circle", color=WONG["sky_blue"],
                                children=[
                                    html.Iframe(
                                        id="graph-frame",
                                        style={"width": "100%", "height": "calc(100vh - 48px)",
                                               "border": "none", "flex": "1",
                                               "background": bg, "display": "block"},
                                        srcDoc=f"<div style='color:#64748b;font-family:monospace;"
                                               f"padding:40px;font-size:14px;background:#0a0e1a;height:100%'>"
                                               f"Click <b style='color:#56B4E9'>&#9654; Apply Filters</b> "
                                               f"to render the network.</div>",
                                    )
                                ],
                                style={"height": "calc(100vh - 48px)", "display": "block"},
                            ),
                        ],
                    ),
                ],
            ),
            # Hidden store for theme
            dcc.Store(id="theme-store", data=theme),
        ],
    )


app.layout = build_layout("dark")


# =============================================================================
# Callbacks
# =============================================================================

# Toggle theme
@app.callback(
    Output("theme-store", "data"),
    Input("theme-toggle", "n_clicks"),
    State("theme-store", "data"),
    prevent_initial_call=True,
)
def toggle_theme(n, current_theme):
    return "light" if current_theme == "dark" else "dark"


# Re-render layout on theme change
@app.callback(
    Output("app-container", "children"),
    Input("theme-store", "data"),
    prevent_initial_call=True,
)
def update_theme_layout(theme):
    return build_layout(theme).children


# Reset filters
@app.callback(
    Output("stage-from",           "value"),
    Output("stage-to",             "value"),
    Output("stage-single",         "value"),
    Output("filter-regulator",     "value"),
    Output("filter-target",        "value"),
    Output("relationship-filter",  "value"),
    Output("max-edges",            "value"),
    Output("layout-select",        "value"),
    Input("reset-btn", "n_clicks"),
    prevent_initial_call=True,
)
def reset_filters(_):
    return None, None, None, None, None, ["activating", "inhibiting", "no_effect"], 300, "barnes_hut"


# Main graph update
@app.callback(
    Output("graph-frame",  "srcDoc"),
    Output("stats-panel",  "children"),
    Output("topbar-stats", "children"),
    Input("apply-btn", "n_clicks"),
    State("stage-single",          "value"),
    State("stage-from",            "value"),
    State("stage-to",              "value"),
    State("filter-regulator",      "value"),
    State("filter-target",         "value"),
    State("relationship-filter",   "value"),
    State("max-edges",             "value"),
    State("layout-select",         "value"),
    State("theme-store",           "data"),
    prevent_initial_call=True,
)
def update_graph(n_clicks, stage_single, stage_from, stage_to,
                 filter_reg, filter_tgt, relationships, max_edges, layout, theme):

    cfg = copy.deepcopy(CONFIG)
    cfg["stage_single"]            = stage_single
    cfg["stage_from"]              = stage_from
    cfg["stage_to"]                = stage_to
    cfg["filter_regulator"]        = filter_reg
    cfg["filter_target"]           = filter_tgt
    cfg["relationships_include"]   = relationships or ["activating", "inhibiting", "no_effect"]
    cfg["max_edges"]               = int(max_edges) if max_edges != 9999 else None
    cfg["layout"]                  = layout
    cfg["theme"]                   = theme or "dark"
    cfg["height"]                  = "100vh"
    cfg["output_file"]             = os.path.join(tempfile.gettempdir(), "grn_dash_output.html")

    df = filter_data(BASE_DF.copy(), cfg)

    if len(df) == 0:
        t     = THEMES[cfg["theme"]]
        empty = (f"<div style='color:#ef4444;font-family:monospace;padding:40px;"
                 f"font-size:14px;background:{t[\"bgcolor\"]};height:100%'>"
                 f"No edges match the current filters. Try adjusting your selection.</div>")
        return empty, _stats_panel_empty(cfg["theme"]), ""

    G        = build_graph(df)
    analysis = analyze_graph(G, cfg)
    out_path = visualize(G, analysis, cfg)

    with open(out_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    return html_content, _stats_panel(G, analysis, cfg["theme"]), _topbar_stats(G, analysis)


def _stats_panel(G, analysis, theme="dark"):
    t     = THEMES[theme]
    regs  = sum(1 for n in G.nodes() if G.nodes[n].get("is_reg") and not G.nodes[n].get("is_tgt"))
    tgts  = sum(1 for n in G.nodes() if G.nodes[n].get("is_tgt") and not G.nodes[n].get("is_reg"))
    both  = sum(1 for n in G.nodes() if G.nodes[n].get("is_reg") and G.nodes[n].get("is_tgt"))
    loops = len(analysis.get("feedback_loops", []))
    sl    = len(analysis.get("self_loops", []))
    hubs  = analysis.get("hub_genes", [])
    label_style = {"color": "#64748b" if theme == "dark" else "#475569",
                   "fontSize": "10px", "fontWeight": "bold",
                   "letterSpacing": "0.1em", "fontFamily": "monospace"}

    def row(label, value, color=None):
        color = color or WONG["sky_blue"]
        return html.Div(
            style={"display": "flex", "justifyContent": "space-between",
                   "marginBottom": "4px", "fontSize": "12px"},
            children=[
                html.Span(label, style={"color": "#94a3b8" if theme == "dark" else "#64748b"}),
                html.Span(str(value), style={"color": color, "fontFamily": "monospace"}),
            ]
        )

    hub_items = [
        html.Div(style={"display": "flex", "justifyContent": "space-between",
                        "marginBottom": "2px", "fontSize": "11px"},
                 children=[
                     html.Span(gene, style={"color": t["tooltip_text"], "fontFamily": "monospace"}),
                     html.Span(str(deg), style={"color": WONG["sky_blue"], "fontFamily": "monospace"}),
                 ])
        for gene, deg in hubs[:6]
    ]

    return html.Div([
        html.Label("NETWORK STATS", style={**label_style, "marginBottom": "8px", "display": "block"}),
        row("Nodes",          G.number_of_nodes()),
        row("Edges",          G.number_of_edges()),
        row("Regulators",     regs,  WONG["sky_blue"]),
        row("Targets",        tgts,  WONG["orange"]),
        row("Both",           both,  WONG["pink"]),
        row("Feedback loops", loops, WONG["orange"]),
        row("Self-loops",     sl,    WONG["yellow"]),
        html.Div(style={"marginTop": "10px"}, children=[
            html.Label("TOP HUBS", style={**label_style, "display": "block", "marginBottom": "6px"}),
            *hub_items,
        ]),
    ])


def _stats_panel_empty(theme="dark"):
    label_style = {"color": "#64748b", "fontSize": "10px", "fontWeight": "bold",
                   "letterSpacing": "0.1em", "fontFamily": "monospace"}
    return html.Div([
        html.Label("NETWORK STATS", style=label_style),
        html.Div("No edges match filters.",
                 style={"color": WONG["vermillion"], "fontSize": "12px", "marginTop": "6px"}),
    ])


def _topbar_stats(G, analysis):
    return (f"Nodes: {G.number_of_nodes()}  |  Edges: {G.number_of_edges()}  |  "
            f"Feedback loops: {len(analysis.get('feedback_loops', []))}")


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Lens GRN Dash App")
    print("=" * 60)
    print("  Open in browser:  http://127.0.0.1:8050")
    print("  Stop server:      Ctrl + C")
    print("=" * 60 + "\n")
    server = app.server
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8050)))