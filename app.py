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
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc

from grn_network import (
    CONFIG,
    load_data,
    filter_data,
    build_graph,
    analyze_graph,
    visualize,
    stage_numeric,
    print_summary,
)


# =============================================================================
# Load data once at startup
# =============================================================================

print("\n[APP] Loading GRN data...")
BASE_DF = load_data(CONFIG)

ALL_STAGES = sorted(
    BASE_DF["stage"].dropna().unique(),
    key=stage_numeric
)
ALL_REGULATORS = sorted(BASE_DF["regulator"].dropna().unique())
ALL_TARGETS    = sorted(BASE_DF["target"].dropna().unique())

print(f"[APP] {len(BASE_DF):,} edges | {len(ALL_STAGES)} stages | "
      f"{len(ALL_REGULATORS)} regulators | {len(ALL_TARGETS)} targets\n")


# =============================================================================
# Helper — stage dropdown options
# =============================================================================

def stage_options(stages):
    return [{"label": s, "value": s} for s in stages]


def gene_options(genes):
    return [{"label": g, "value": g} for g in genes]


# =============================================================================
# App layout
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
            .Select-value-label { color: #0f172a !important; }
            .Select-placeholder { color: #64748b !important; }
            .Select-input input { color: #0f172a !important; }
            .Select-menu-outer { background: #1e293b !important; }
            .Select-option { color: #e2e8f0 !important; }
            .Select-option.is-focused { background: #334155 !important; }
            .dash-dropdown .Select-control { background: #f8fafc !important; }
            .dash-dropdown .Select-value-label { color: #0f172a !important; }
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


# ---- Sidebar ----
sidebar = html.Div(
    style={
        "width": "300px",
        "minWidth": "300px",
        "background": "#0f1525",
        "borderRight": "1px solid #1e2d4a",
        "padding": "16px",
        "overflowY": "auto",
        "display": "flex",
        "flexDirection": "column",
        "gap": "16px",
    },
    children=[

        # Title
        html.Div([
            html.H5("🔬 Lens GRN Explorer",
                    style={"color": "#00d4ff", "fontFamily": "monospace",
                           "marginBottom": "2px", "fontSize": "14px"}),
            html.Small("Lachke Lab 2016 — Gene Regulatory Network",
                       style={"color": "#64748b", "fontSize": "11px"}),
        ]),

        html.Hr(style={"borderColor": "#1e2d4a", "margin": "0"}),

        # ── Stage Filter ──
        html.Div([
            html.Label("STAGE FILTER",
                       style={"color": "#64748b", "fontSize": "10px",
                              "fontWeight": "bold", "letterSpacing": "0.1em",
                              "fontFamily": "monospace"}),

            html.Label("Single Stage", style={"color": "#94a3b8", "fontSize": "12px", "marginTop": "8px"}),
            dcc.Dropdown(
                id="stage-single",
                options=stage_options(ALL_STAGES),
                placeholder="All stages",
                clearable=True,
                style={"background": "#141b2d", "color": "#e2e8f0", "fontSize": "12px"},
            ),

            html.Label("Stage Range", style={"color": "#94a3b8", "fontSize": "12px", "marginTop": "10px"}),
            html.Div([
                dcc.Dropdown(
                    id="stage-from",
                    options=stage_options(ALL_STAGES),
                    placeholder="From",
                    clearable=True,
                    style={"flex": "1", "fontSize": "12px"},
                ),
                html.Span("→", style={"color": "#64748b", "padding": "0 6px", "alignSelf": "center"}),
                dcc.Dropdown(
                    id="stage-to",
                    options=stage_options(ALL_STAGES),
                    placeholder="To",
                    clearable=True,
                    style={"flex": "1", "fontSize": "12px"},
                ),
            ], style={"display": "flex", "alignItems": "center", "gap": "4px"}),
        ]),

        html.Hr(style={"borderColor": "#1e2d4a", "margin": "0"}),

        # ── Gene Filter ──
        html.Div([
            html.Label("GENE FILTER",
                       style={"color": "#64748b", "fontSize": "10px",
                              "fontWeight": "bold", "letterSpacing": "0.1em",
                              "fontFamily": "monospace"}),

            html.Label("Regulator", style={"color": "#94a3b8", "fontSize": "12px", "marginTop": "8px"}),
            dcc.Dropdown(
                id="filter-regulator",
                options=gene_options(ALL_REGULATORS),
                placeholder="All regulators",
                clearable=True,
                style={"fontSize": "12px"},
            ),

            html.Label("Target", style={"color": "#94a3b8", "fontSize": "12px", "marginTop": "10px"}),
            dcc.Dropdown(
                id="filter-target",
                options=gene_options(ALL_TARGETS),
                placeholder="All targets",
                clearable=True,
                style={"fontSize": "12px"},
            ),
        ]),

        html.Hr(style={"borderColor": "#1e2d4a", "margin": "0"}),

        # ── Effect Filter ──
        html.Div([
            html.Label("EFFECT FILTER",
                       style={"color": "#64748b", "fontSize": "10px",
                              "fontWeight": "bold", "letterSpacing": "0.1em",
                              "fontFamily": "monospace"}),
            html.Div(style={"marginTop": "8px"}, children=[
                dcc.Checklist(
                    id="effect-filter",
                    options=[
                        {"label": "  ✅ Activating (+)", "value": "+"},
                        {"label": "  ❌ Inhibiting (−)", "value": "-"},
                        {"label": "  ⚪ No effect (○)", "value": "o"},
                    ],
                    value=["+", "-", "o"],
                    labelStyle={"display": "block", "color": "#e2e8f0",
                                "fontSize": "13px", "marginBottom": "4px"},
                )
            ]),
        ]),

        html.Hr(style={"borderColor": "#1e2d4a", "margin": "0"}),

        # ── Display Options ──
        html.Div([
            html.Label("DISPLAY OPTIONS",
                       style={"color": "#64748b", "fontSize": "10px",
                              "fontWeight": "bold", "letterSpacing": "0.1em",
                              "fontFamily": "monospace"}),

            html.Label("Max edges", style={"color": "#94a3b8", "fontSize": "12px", "marginTop": "8px"}),
            dcc.Dropdown(
                id="max-edges",
                options=[
                    {"label": "100 edges",  "value": 100},
                    {"label": "300 edges",  "value": 300},
                    {"label": "600 edges",  "value": 600},
                    {"label": "All edges",  "value": 9999},
                ],
                value=300,
                clearable=False,
                style={"fontSize": "12px"},
            ),

            html.Label("Layout", style={"color": "#94a3b8", "fontSize": "12px", "marginTop": "10px"}),
            dcc.Dropdown(
                id="layout-select",
                options=[
                    {"label": "Barnes Hut",        "value": "barnes_hut"},
                    {"label": "Force Atlas 2",      "value": "force_atlas_2based"},
                    {"label": "Repulsion",          "value": "repulsion"},
                ],
                value="barnes_hut",
                clearable=False,
                style={"fontSize": "12px"},
            ),
        ]),

        html.Hr(style={"borderColor": "#1e2d4a", "margin": "0"}),

        # ── Buttons ──
        html.Div([
            html.Button("▶  Apply Filters", id="apply-btn",
                style={
                    "width": "100%", "padding": "9px",
                    "background": "rgba(0,212,255,0.12)",
                    "border": "1px solid #00d4ff", "borderRadius": "7px",
                    "color": "#00d4ff", "fontFamily": "monospace",
                    "fontSize": "12px", "cursor": "pointer", "marginBottom": "6px",
                }),
            html.Button("↺  Reset All", id="reset-btn",
                style={
                    "width": "100%", "padding": "9px",
                    "background": "transparent",
                    "border": "1px solid #1e2d4a", "borderRadius": "7px",
                    "color": "#64748b", "fontFamily": "monospace",
                    "fontSize": "12px", "cursor": "pointer",
                }),
        ]),

        html.Hr(style={"borderColor": "#1e2d4a", "margin": "0"}),

        # ── Stats panel (updated after render) ──
        html.Div(id="stats-panel", children=[
            html.Label("NETWORK STATS",
                       style={"color": "#64748b", "fontSize": "10px",
                              "fontWeight": "bold", "letterSpacing": "0.1em",
                              "fontFamily": "monospace"}),
            html.Div("Apply filters to see stats.",
                     style={"color": "#475569", "fontSize": "12px", "marginTop": "6px"}),
        ]),

        html.Hr(style={"borderColor": "#1e2d4a", "margin": "0"}),

        # ── Legend ──
        html.Div([
            html.Label("LEGEND",
                       style={"color": "#64748b", "fontSize": "10px",
                              "fontWeight": "bold", "letterSpacing": "0.1em",
                              "fontFamily": "monospace", "marginBottom": "8px"}),
            html.Div([
                html.Span("●", style={"color": "#00d4ff"}), " Regulator only",
            ], style={"fontSize": "12px", "marginBottom": "3px"}),
            html.Div([
                html.Span("●", style={"color": "#ff6b35"}), " Target only",
            ], style={"fontSize": "12px", "marginBottom": "3px"}),
            html.Div([
                html.Span("●", style={"color": "#a855f7"}), " Regulator & Target",
            ], style={"fontSize": "12px", "marginBottom": "3px"}),
            html.Div([
                html.Span("●", style={"color": "#facc15"}), " Self-regulatory loop",
            ], style={"fontSize": "12px", "marginBottom": "10px"}),
            html.Div([
                html.Span("━━▶", style={"color": "#22c55e"}), " Activating (+)",
            ], style={"fontSize": "12px", "marginBottom": "3px"}),
            html.Div([
                html.Span("━━▶", style={"color": "#ef4444"}), " Inhibiting (−)",
            ], style={"fontSize": "12px", "marginBottom": "3px"}),
            html.Div([
                html.Span("━━▶", style={"color": "#94a3b8"}), " No effect (○)",
            ], style={"fontSize": "12px"}),
        ]),
    ],
)

# ---- Main graph panel ----
graph_panel = html.Div(
    style={"flex": "1", "display": "flex", "flexDirection": "column", "overflow": "hidden"},
    children=[
        # Loading indicator
        dcc.Loading(
            id="loading-graph",
            type="circle",
            color="#00d4ff",
            children=[
                html.Iframe(
                    id="graph-frame",
                    style={
                        "width": "100%",
                        "height": "calc(100vh - 48px)",
                        "border": "none",
                        "flex": "1",
                        "background": "#0a0e1a",
                        "display": "block",
                    },
                    srcDoc="<div style='color:#64748b;font-family:monospace;padding:40px;font-size:14px'>"
                           "Click <b style='color:#00d4ff'>▶ Apply Filters</b> to render the network.</div>",
                )
            ],
            style={"height": "calc(100vh - 48px)", "display": "block"},
        ),
    ],
)

# ---- Full layout ----
app.layout = html.Div(
    style={
        "display": "flex",
        "flexDirection": "column",
        "height": "100vh",
        "background": "#0a0e1a",
        "color": "#e2e8f0",
        "fontFamily": "DM Sans, sans-serif",
        "overflow": "hidden",
    },
    children=[

        # Top bar
        html.Div(
            style={
                "background": "#0f1525",
                "borderBottom": "1px solid #1e2d4a",
                "padding": "10px 20px",
                "display": "flex",
                "alignItems": "center",
                "gap": "12px",
                "flexShrink": "0",
            },
            children=[
                html.H4("🔬 Lens GRN Explorer",
                        style={"color": "#00d4ff", "fontFamily": "monospace",
                               "fontSize": "14px", "margin": "0", "fontWeight": "700"}),
                html.Span("Gene Regulatory Network — Lachke Lab 2016",
                          style={"color": "#64748b", "fontSize": "12px"}),
                html.Div(id="topbar-stats",
                         style={"marginLeft": "auto", "fontFamily": "monospace",
                                "fontSize": "11px", "color": "#64748b"}),
            ],
        ),

        # Body: sidebar + graph
        html.Div(
            style={"display": "flex", "flex": "1", "overflow": "hidden"},
            children=[sidebar, graph_panel],
        ),
    ],
)


# =============================================================================
# Callbacks
# =============================================================================

@app.callback(
    Output("stage-from",  "value"),
    Output("stage-to",    "value"),
    Output("stage-single","value"),
    Output("filter-regulator", "value"),
    Output("filter-target",    "value"),
    Output("effect-filter",    "value"),
    Output("max-edges",        "value"),
    Output("layout-select",    "value"),
    Input("reset-btn", "n_clicks"),
    prevent_initial_call=True,
)
def reset_filters(_):
    return None, None, None, None, None, ["+", "-", "o"], 300, "barnes_hut"


@app.callback(
    Output("graph-frame",  "srcDoc"),
    Output("stats-panel",  "children"),
    Output("topbar-stats", "children"),
    Input("apply-btn", "n_clicks"),
    State("stage-single",       "value"),
    State("stage-from",         "value"),
    State("stage-to",           "value"),
    State("filter-regulator",   "value"),
    State("filter-target",      "value"),
    State("effect-filter",      "value"),
    State("max-edges",          "value"),
    State("layout-select",      "value"),
    prevent_initial_call=True,
)
def update_graph(n_clicks, stage_single, stage_from, stage_to,
                 filter_reg, filter_tgt, effects, max_edges, layout):

    # Build config for this render
    cfg = copy.deepcopy(CONFIG)
    cfg["stage_single"]       = stage_single
    cfg["stage_from"]         = stage_from
    cfg["stage_to"]           = stage_to
    cfg["filter_regulator"]   = filter_reg
    cfg["filter_target"]      = filter_tgt
    cfg["effects_include"]    = effects or ["+", "-", "o"]
    cfg["max_edges"]          = int(max_edges) if max_edges != 9999 else None
    cfg["layout"]             = layout
    cfg["height"]             = "100vh"
    cfg["output_file"]        = os.path.join(tempfile.gettempdir(), "grn_dash_output.html")

    # Filter + build + analyze
    df = filter_data(BASE_DF.copy(), cfg)

    if len(df) == 0:
        empty = "<div style='color:#ef4444;font-family:monospace;padding:40px;font-size:14px'>No edges match the current filters. Try adjusting your selection.</div>"
        return empty, _stats_panel_empty(), ""

    G        = build_graph(df)
    analysis = analyze_graph(G, cfg)
    out_path = visualize(G, analysis, cfg)

    with open(out_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    # Build stats panel
    stats = _stats_panel(G, analysis)
    topbar = _topbar_stats(G, analysis)

    return html_content, stats, topbar


def _stats_panel(G: nx.DiGraph, analysis: dict):
    regs  = sum(1 for n in G.nodes() if G.nodes[n].get("is_reg") and not G.nodes[n].get("is_tgt"))
    tgts  = sum(1 for n in G.nodes() if G.nodes[n].get("is_tgt") and not G.nodes[n].get("is_reg"))
    both  = sum(1 for n in G.nodes() if G.nodes[n].get("is_reg") and G.nodes[n].get("is_tgt"))
    loops = len(analysis.get("feedback_loops", []))
    sl    = len(analysis.get("self_loops", []))
    hubs  = analysis.get("hub_genes", [])

    def row(label, value, color="#00d4ff"):
        return html.Div(
            style={"display": "flex", "justifyContent": "space-between",
                   "marginBottom": "4px", "fontSize": "12px"},
            children=[
                html.Span(label, style={"color": "#94a3b8"}),
                html.Span(str(value), style={"color": color, "fontFamily": "monospace"}),
            ]
        )

    hub_items = []
    for gene, deg in hubs[:6]:
        hub_items.append(
            html.Div(
                style={"display": "flex", "justifyContent": "space-between",
                       "marginBottom": "2px", "fontSize": "11px"},
                children=[
                    html.Span(gene, style={"color": "#e2e8f0", "fontFamily": "monospace"}),
                    html.Span(str(deg), style={"color": "#00d4ff", "fontFamily": "monospace"}),
                ]
            )
        )

    return html.Div([
        html.Label("NETWORK STATS",
                   style={"color": "#64748b", "fontSize": "10px",
                          "fontWeight": "bold", "letterSpacing": "0.1em",
                          "fontFamily": "monospace", "marginBottom": "8px",
                          "display": "block"}),
        row("Nodes",       G.number_of_nodes()),
        row("Edges",       G.number_of_edges()),
        row("Regulators",  regs),
        row("Targets",     tgts),
        row("Both",        both, "#a855f7"),
        row("Feedback loops", loops, "#fb923c"),
        row("Self-loops",  sl, "#facc15"),
        html.Div(style={"marginTop": "10px"}, children=[
            html.Label("TOP HUBS",
                       style={"color": "#64748b", "fontSize": "10px",
                              "fontWeight": "bold", "letterSpacing": "0.1em",
                              "fontFamily": "monospace", "display": "block",
                              "marginBottom": "6px"}),
            *hub_items,
        ]),
    ])


def _stats_panel_empty():
    return html.Div([
        html.Label("NETWORK STATS",
                   style={"color": "#64748b", "fontSize": "10px",
                          "fontWeight": "bold", "letterSpacing": "0.1em",
                          "fontFamily": "monospace"}),
        html.Div("No edges match filters.",
                 style={"color": "#ef4444", "fontSize": "12px", "marginTop": "6px"}),
    ])


def _topbar_stats(G: nx.DiGraph, analysis: dict):
    return (
        f"Nodes: {G.number_of_nodes()}  |  "
        f"Edges: {G.number_of_edges()}  |  "
        f"Feedback loops: {len(analysis.get('feedback_loops', []))}"
    )


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
    app.run(debug=True, port=8050)