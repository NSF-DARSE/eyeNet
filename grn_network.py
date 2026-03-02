"""
=============================================================================
Lens GRN Explorer — NetworkX + Pyvis
=============================================================================
Lachke Lab 2016 — Gene Regulatory Network

SETUP (run once):
    pip install networkx pyvis pandas openpyxl plotly dash

USAGE:
    python grn_network.py        ← opens grn_output.html in browser
    python app.py                ← runs full Dash app with live filters

STRUCTURE:
    1. CONFIG         — all tunable settings
    2. DATA LOADING   — reads Excel
    3. FILTERING      — stage, gene, effect filters
    4. GRAPH BUILDING — NetworkX DiGraph
    5. ANALYSIS       — hubs, feedback loops, centrality
    6. VISUALIZATION  — Pyvis HTML output
    7. MAIN           — runs pipeline
=============================================================================
"""

import os
import sys
import webbrowser
import pandas as pd
import networkx as nx
from pyvis.network import Network


# =============================================================================
# 1. CONFIG
# =============================================================================

CONFIG = {
    # --- File paths ---
    "input_file":  "data/Lens_GRN_June_2016_original FOR HACKATHON - Salil Lachke.xlsx",
    "output_file": "grn_output.html",
    "sheet_name":  "Lens_GRN_pert",

    # --- Stage filters ---
    # Stages: 'E8.0' ... 'E19.5', 'P0' ... 'P665', 'Adult'
    "stage_from":   None,       # e.g. "E9.5"
    "stage_to":     None,       # e.g. "E14.5"
    "stage_single": None,       # e.g. "E12.5" — overrides range if set

    # --- Gene filters ---
    "filter_regulator": None,   # e.g. "Pax6"
    "filter_target":    None,   # e.g. "Prox1"

    # --- Effect filters ---
    # "+" = activating, "-" = inhibiting, "o" = no effect
    "effects_include": ["+", "-", "o"],

    # --- Graph display ---
    "max_edges":           300,
    "show_feedback_loops": True,

    # --- Node colors ---
    "color_regulator": "#00d4ff",   # cyan    — regulates others
    "color_target":    "#ff6b35",   # orange  — regulated by others
    "color_both":      "#a855f7",   # purple  — regulates AND is regulated
    "color_selfloop":  "#facc15",   # yellow  — self-regulatory gene

    # --- Edge colors ---
    "color_activating": "#22c55e",  # green  — activates target
    "color_inhibiting": "#ef4444",  # red    — inhibits target
    "color_noeffect":   "#94a3b8",  # grey   — no effect observed

    # --- Node sizing ---
    "node_size_min": 12,
    "node_size_max": 55,

    # --- Layout ---
    # Options: "barnes_hut", "force_atlas_2based", "repulsion", "hierarchical"
    "layout": "barnes_hut",

    # --- Window ---
    "width":      "100%",
    "height":     "100vh",
    "bgcolor":    "#0a0e1a",
    "font_color": "#ffffff",
}


# =============================================================================
# 2. DATA LOADING
# =============================================================================

def load_data(config: dict) -> pd.DataFrame:
    path = config["input_file"]
    if not os.path.exists(path):
        print(f"\n[ERROR] File not found: {path}")
        print("        Make sure the Excel file is at the correct path.\n")
        sys.exit(1)

    print(f"[1/5] Loading data from: {path}")
    df = pd.read_excel(path, sheet_name=config["sheet_name"])
    df.columns = [str(c).strip() for c in df.columns]

    col_map = {
        df.columns[0]: "sno",
        df.columns[1]: "regulator",
        df.columns[2]: "target",
        df.columns[4]: "perturbation",
        df.columns[5]: "effect",
        df.columns[6]: "stage",
        df.columns[7]: "context",
    }
    df = df.rename(columns=col_map)
    df = df[["sno", "regulator", "target", "perturbation", "effect", "stage", "context"]]
    df = df.dropna(subset=["regulator", "target"])

    df["regulator"]    = df["regulator"].astype(str).str.strip()
    df["target"]       = df["target"].astype(str).str.strip()
    df["effect"]       = df["effect"].astype(str).str.strip()
    df["stage"]        = df["stage"].astype(str).str.strip()
    df["context"]      = df["context"].astype(str).str.strip()
    df["perturbation"] = df["perturbation"].astype(str).str.strip()
    df["effect"]       = df["effect"].replace({"nan": "o", "none": "o", "None": "o", "0": "o"})

    print(f"       Loaded {len(df):,} edges")
    return df


# =============================================================================
# 3. FILTERING
# =============================================================================

def stage_numeric(stage: str) -> float:
    s = str(stage).strip()
    if s == "Adult": return 100000.0
    try:
        if s.startswith("E"): return float(s[1:])
        if s.startswith("P"): return 1000.0 + float(s[1:])
    except ValueError:
        pass
    return 99999.0


def filter_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    print(f"[2/5] Applying filters...")
    original = len(df)

    df = df[df["effect"].isin(config["effects_include"])]

    if config["stage_single"]:
        df = df[df["stage"] == config["stage_single"]]
        print(f"       Stage = {config['stage_single']}")
    else:
        if config["stage_from"]:
            df = df[df["stage"].apply(stage_numeric) >= stage_numeric(config["stage_from"])]
        if config["stage_to"]:
            df = df[df["stage"].apply(stage_numeric) <= stage_numeric(config["stage_to"])]
        if config["stage_from"] or config["stage_to"]:
            print(f"       Stage range: {config['stage_from'] or 'start'} → {config['stage_to'] or 'end'}")

    if config["filter_regulator"]:
        df = df[df["regulator"] == config["filter_regulator"]]
        print(f"       Regulator = {config['filter_regulator']}")

    if config["filter_target"]:
        df = df[df["target"] == config["filter_target"]]
        print(f"       Target = {config['filter_target']}")

    if config["max_edges"] and len(df) > config["max_edges"]:
        df = df.head(config["max_edges"])
        print(f"       Capped at {config['max_edges']} edges")

    print(f"       {original:,} → {len(df):,} edges after filtering")
    return df.reset_index(drop=True)


# =============================================================================
# 4. GRAPH BUILDING
# =============================================================================

def build_graph(df: pd.DataFrame) -> nx.DiGraph:
    print(f"[3/5] Building NetworkX DiGraph...")
    G = nx.DiGraph()

    regulators = set(df["regulator"].unique())
    targets    = set(df["target"].unique())

    for node in regulators | targets:
        G.add_node(node, is_reg=(node in regulators), is_tgt=(node in targets))

    for _, row in df.iterrows():
        reg, tgt = row["regulator"], row["target"]
        eff, stg = row["effect"], row["stage"]
        ctx      = row["context"]

        if G.has_edge(reg, tgt):
            G[reg][tgt]["effects"].append(eff)
            G[reg][tgt]["stages"].append(stg)
            G[reg][tgt]["contexts"].append(ctx)
            G[reg][tgt]["count"] += 1
        else:
            G.add_edge(reg, tgt, effects=[eff], stages=[stg], contexts=[ctx], count=1)

    print(f"       {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G


# =============================================================================
# 5. ANALYSIS
# =============================================================================

def analyze_graph(G: nx.DiGraph, config: dict) -> dict:
    print(f"[4/5] Running graph analysis...")
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
    print(f"       Top hubs: {', '.join(g[0] for g in results['hub_genes'][:5])}")

    results["feedback_nodes"] = set()
    results["feedback_edges"] = set()
    results["feedback_loops"] = []
    results["self_loops"]     = []

    if config["show_feedback_loops"]:
        try:
            for i, cycle in enumerate(nx.simple_cycles(G)):
                if i > 500: break
                results["feedback_loops"].append(cycle)
                for node in cycle:
                    results["feedback_nodes"].add(node)
                for j in range(len(cycle)):
                    results["feedback_edges"].add((cycle[j], cycle[(j+1) % len(cycle)]))
        except Exception:
            pass
        results["self_loops"] = list(nx.selfloop_edges(G))
        print(f"       Feedback loops: {len(results['feedback_loops'])}  |  Self-loops: {len(results['self_loops'])}")

    results["components"] = list(nx.weakly_connected_components(G))
    print(f"       Components: {len(results['components'])}")
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
    if node in self_loop_nodes:       return config["color_selfloop"]
    is_reg = G.nodes[node].get("is_reg", False)
    is_tgt = G.nodes[node].get("is_tgt", False)
    if is_reg and is_tgt:             return config["color_both"]
    if is_reg:                        return config["color_regulator"]
    return config["color_target"]


def _edge_color(effects, config):
    from collections import Counter
    if not effects: return config["color_noeffect"]
    dominant = Counter(effects).most_common(1)[0][0]
    if dominant == "+": return config["color_activating"]
    if dominant == "-": return config["color_inhibiting"]
    return config["color_noeffect"]


def _node_tooltip(node, G, analysis):
    d       = G.nodes[node]
    is_reg  = d.get("is_reg", False)
    is_tgt  = d.get("is_tgt", False)
    in_deg  = d.get("in_degree", 0)
    out_deg = d.get("out_degree", 0)
    cent    = d.get("deg_centrality", 0)

    if is_reg and is_tgt: role = "Regulator &amp; Target"
    elif is_reg:          role = "Regulator"
    else:                 role = "Target"

    self_loop   = node in set(e[0] for e in analysis.get("self_loops", []))
    in_feedback = node in analysis.get("feedback_nodes", set())

    tip = (
        f"<div style='font-family:monospace;font-size:13px;padding:8px;min-width:200px'>"
        f"<b style='font-size:15px;color:#00d4ff'>{node}</b><br/>"
        f"<hr style='border-color:#334155;margin:6px 0'/>"
        f"<b>Role:</b> {role}<br/>"
        f"<b>Regulates (out):</b> {out_deg} genes<br/>"
        f"<b>Regulated by (in):</b> {in_deg} genes<br/>"
        f"<b>Degree centrality:</b> {cent:.3f}<br/>"
    )
    if self_loop:
        tip += "<br/><span style='color:#facc15'>🔄 Self-regulatory loop</span><br/>"
    if in_feedback:
        tip += "<span style='color:#fb923c'>⚡ Part of feedback loop</span><br/>"
    tip += "</div>"
    return tip


def _edge_tooltip(u, v, data, analysis):
    effects  = data.get("effects", ["o"])
    stages   = sorted(set(str(s) for s in data.get("stages", []) if pd.notna(s)))
    count    = data.get("count", 1)
    is_fb    = (u, v) in analysis.get("feedback_edges", set())
    labels   = {"+": "✅ Activating", "-": "❌ Inhibiting", "o": "⚪ No effect"}
    eff_str  = ", ".join(set(labels.get(e, e) for e in effects))
    stg_str  = ", ".join(stages[:5]) + ("…" if len(stages) > 5 else "")

    tip = (
        f"<div style='font-family:monospace;font-size:13px;padding:8px;min-width:220px'>"
        f"<b style='color:#00d4ff'>{u}</b>"
        f"<span style='color:#94a3b8'> → </span>"
        f"<b style='color:#ff6b35'>{v}</b><br/>"
        f"<hr style='border-color:#334155;margin:6px 0'/>"
        f"<b>Effect:</b> {eff_str}<br/>"
        f"<b>Stage(s):</b> {stg_str}<br/>"
        f"<b>Evidence count:</b> {count}<br/>"
    )
    if is_fb:
        tip += "<br/><span style='color:#fb923c'>⚡ Feedback loop edge</span><br/>"
    tip += "</div>"
    return tip


def _legend_html(config):
    return f"""
    <div style='position:fixed;bottom:20px;left:20px;z-index:9999;
                background:#0f1525;border:1px solid #1e2d4a;border-radius:10px;
                padding:14px 18px;font-family:monospace;font-size:12px;color:#e2e8f0;
                box-shadow:0 4px 20px rgba(0,0,0,0.5);min-width:200px'>
      <div style='font-weight:bold;font-size:13px;color:#00d4ff;margin-bottom:10px'>
        🔬 Lens GRN Legend
      </div>
      <div style='margin-bottom:6px;font-weight:bold;color:#94a3b8;font-size:10px;text-transform:uppercase'>
        Node — Role
      </div>
      <div style='margin-bottom:3px'><span style='color:{config["color_regulator"]}'>●</span> &nbsp;Regulator only</div>
      <div style='margin-bottom:3px'><span style='color:{config["color_target"]}'>●</span> &nbsp;Target only</div>
      <div style='margin-bottom:3px'><span style='color:{config["color_both"]}'>●</span> &nbsp;Regulator &amp; Target</div>
      <div style='margin-bottom:10px'><span style='color:{config["color_selfloop"]}'>●</span> &nbsp;Self-regulatory loop</div>
      <div style='margin-bottom:6px;font-weight:bold;color:#94a3b8;font-size:10px;text-transform:uppercase'>
        Edge — Effect
      </div>
      <div style='margin-bottom:3px'><span style='color:{config["color_activating"]}'>━━▶</span> &nbsp;Activating (+)</div>
      <div style='margin-bottom:3px'><span style='color:{config["color_inhibiting"]}'>━━▶</span> &nbsp;Inhibiting (−)</div>
      <div style='margin-bottom:3px'><span style='color:{config["color_noeffect"]}'>━━▶</span> &nbsp;No effect (○)</div>
      <div style='margin-top:10px;font-size:10px;color:#475569'>
        Node size = connections<br/>Edge width = evidence count
      </div>
    </div>"""


def visualize(G: nx.DiGraph, analysis: dict, config: dict) -> str:
    print(f"[5/5] Building Pyvis visualization...")

    net = Network(
        height=config["height"],
        width=config["width"],
        bgcolor=config["bgcolor"],
        font_color=config["font_color"],
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
                "border":     "#facc15" if in_fb else color,
                "highlight":  {"background": "#ffffff", "border": "#ffffff"},
                "hover":      {"background": "#ffffff", "border": "#ffffff"},
            },
            size=_node_size(G, node, config),
            title=_node_tooltip(node, G, analysis),
            borderWidth=3 if in_fb else 1,
            font={"color": "#ffffff", "size": 11, "face": "monospace"},
        )

    for u, v, data in G.edges(data=True):
        effects = data.get("effects", ["o"])
        count   = data.get("count", 1)
        is_fb   = (u, v) in analysis.get("feedback_edges", set())
        net.add_edge(
            u, v,
            color={"color": _edge_color(effects, config), "highlight": "#ffffff", "hover": "#ffffff"},
            title=_edge_tooltip(u, v, data, analysis),
            width=1.2 + (count * 0.25),
            arrows={"to": {"enabled": True, "scaleFactor": 0.5}},
            dashes=is_fb,
            smooth={"type": "curvedCW", "roundness": 0.15},
        )

    net.set_options("""
    {
      "interaction": {
        "hover": true,
        "tooltipDelay": 80,
        "navigationButtons": true,
        "keyboard": {"enabled": true},
        "multiselect": true,
        "zoomView": true
      },
      "nodes": {
        "shadow": {"enabled": true, "color": "rgba(0,0,0,0.4)", "size": 8, "x": 2, "y": 2}
      },
      "physics": {
        "stabilization": {"iterations": 250, "updateInterval": 25}
      }
    }
    """)

    output_path = config["output_file"]
    net.save_graph(output_path)

    # Post-process: inject legend + fix white UI controls + style tooltips
    with open(output_path, "r", encoding="utf-8") as f:
        html = f.read()

    css = """<style>
      .vis-navigation .vis-button { filter: invert(1) brightness(2) !important; }
      div.vis-tooltip {
        background: #0f1525 !important;
        border: 1px solid #1e2d4a !important;
        color: #e2e8f0 !important;
        border-radius: 8px !important;
        font-family: monospace !important;
        padding: 2px !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5) !important;
        max-width: 300px !important;
      }
    </style>"""

    html = html.replace("</head>", css + "\n</head>")
    html = html.replace("</body>", _legend_html(config) + "\n</body>")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"       Saved → {output_path}")
    return output_path


# =============================================================================
# 7. MAIN
# =============================================================================

def print_summary(G: nx.DiGraph, analysis: dict):
    print()
    print("=" * 60)
    print("  GRN ANALYSIS SUMMARY")
    print("=" * 60)
    regs = sum(1 for n in G.nodes() if G.nodes[n].get("is_reg") and not G.nodes[n].get("is_tgt"))
    tgts = sum(1 for n in G.nodes() if G.nodes[n].get("is_tgt") and not G.nodes[n].get("is_reg"))
    both = sum(1 for n in G.nodes() if G.nodes[n].get("is_reg") and G.nodes[n].get("is_tgt"))
    print(f"  Nodes        : {G.number_of_nodes():,}")
    print(f"  Edges        : {G.number_of_edges():,}")
    print(f"  Regulators   : {regs}")
    print(f"  Targets      : {tgts}")
    print(f"  Both         : {both}")
    print()
    print("  TOP HUB REGULATORS (out-degree):")
    for gene, deg in analysis.get("hub_genes", []):
        print(f"    {gene:<25} {deg:>4}  {'█' * min(deg, 35)}")
    print()
    print(f"  FEEDBACK LOOPS : {len(analysis.get('feedback_loops', []))}")
    print(f"  SELF-LOOPS     : {len(analysis.get('self_loops', []))}")
    if analysis.get("self_loops"):
        print(f"  Self-loop genes: {', '.join(set(e[0] for e in analysis['self_loops']))}")
    print(f"  COMPONENTS     : {len(analysis.get('components', []))}")
    print("=" * 60)
    print()


def run_pipeline(config: dict):
    """Full pipeline — importable from app.py."""
    df       = load_data(config)
    df       = filter_data(df, config)
    if len(df) == 0:
        print("[ERROR] No edges after filtering.")
        return None, None, None
    G        = build_graph(df)
    analysis = analyze_graph(G, config)
    return df, G, analysis


def main():
    print()
    print("=" * 60)
    print("  Lens GRN Explorer — NetworkX + Pyvis")
    print("=" * 60)
    print()
    df, G, analysis = run_pipeline(CONFIG)
    if G is None:
        sys.exit(1)
    print_summary(G, analysis)
    output   = visualize(G, analysis, CONFIG)
    abs_path = os.path.abspath(output)
    print(f"  Opening → {abs_path}")
    webbrowser.open(f"file://{abs_path}")
    print("\n  Done! Edit CONFIG at the top to change filters.\n")


if __name__ == "__main__":
    main()


# =============================================================================
# QUICK REFERENCE — useful NetworkX one-liners
# =============================================================================
#
#   df, G, analysis = run_pipeline(CONFIG)
#
#   nx.shortest_path(G, 'Pax6', 'Prox1')      # path between two genes
#   nx.descendants(G, 'Pax6')                  # everything downstream of Pax6
#   nx.ancestors(G, 'Foxe3')                   # everything upstream of Foxe3
#   nx.pagerank(G)                             # influence score per gene
#   list(nx.simple_cycles(G))                  # all feedback loops
#   nx.in_degree_centrality(G)                 # most regulated genes
#   nx.out_degree_centrality(G)                # most active regulators
#   nx.write_graphml(G, 'grn.graphml')         # export → Cytoscape
#   nx.write_gexf(G, 'grn.gexf')              # export → Gephi
# =============================================================================