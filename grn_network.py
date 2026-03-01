"""
=============================================================================
Lens GRN Explorer — NetworkX + Pyvis
=============================================================================
Lachke Lab 2016 Gene Regulatory Network

SETUP (run once in your terminal):
    pip install networkx pyvis pandas openpyxl

USAGE:
    python grn_network.py

    Opens an interactive HTML file in your browser.
    Edit the CONFIG section below to change filters, colors, layout, etc.

STRUCTURE:
    1. CONFIG          — all your tunable settings in one place
    2. DATA LOADING    — reads the Excel file
    3. FILTERING       — stage, regulator, target, effect filters
    4. GRAPH BUILDING  — NetworkX directed graph (DiGraph)
    5. ANALYSIS        — degree, hub genes, feedback loops, centrality
    6. VISUALIZATION   — Pyvis interactive network → HTML output
    7. MAIN            — runs everything, prints summary
=============================================================================
"""

import os
import sys
import math
import webbrowser
import pandas as pd
import networkx as nx
from pyvis.network import Network


# =============================================================================
# 1. CONFIG — Edit everything here
# =============================================================================

CONFIG = {
    # --- File paths ---
    "input_file": "Lens_GRN_June_2016_original_FOR_HACKATHON_-_Salil_Lachke.xlsx",
    "output_file": "grn_output.html",
    "sheet_name": "Lens_GRN_pert",

    # --- Stage filters ---
    # Set to None to disable. Stages: 'E8.0' ... 'E19.5', 'P0' ... 'P665', 'Adult'
    "stage_from": None,         # e.g. "E9.5"  — include stages >= this
    "stage_to":   None,         # e.g. "E14.5" — include stages <= this
    "stage_single": None,       # e.g. "E12.5" — overrides stage_from/to if set

    # --- Gene filters ---
    "filter_regulator": None,   # e.g. "Pax6" — show only this regulator's network
    "filter_target":    None,   # e.g. "Prox1" — show only edges targeting this gene

    # --- Effect filters ---
    # Include any combination of: "+" (activating), "-" (inhibiting), "o" (no effect)
    "effects_include": ["+", "-", "o"],

    # --- Graph display ---
    "max_edges": 300,           # cap for performance; set to None for all edges
    "show_feedback_loops": True,  # highlight detected feedback/auto-regulatory loops

    # --- Node colors ---
    "color_regulator": "#00d4ff",   # cyan  — regulator only
    "color_target":    "#ff6b35",   # orange — target only
    "color_both":      "#a855f7",   # purple — acts as both
    "color_loop":      "#facc15",   # yellow — self-loop / feedback node

    # --- Edge colors ---
    "color_activating":  "#22c55e", # green
    "color_inhibiting":  "#ef4444", # red
    "color_noeffect":    "#f59e0b", # amber

    # --- Node sizing ---
    "node_size_min": 10,
    "node_size_max": 50,

    # --- Layout ---
    # Options: "barnes_hut", "force_atlas_2based", "hierarchical", "repulsion"
    "layout": "barnes_hut",

    # --- Window ---
    "width":  "100%",
    "height": "92vh",
    "bgcolor": "#0a0e1a",
    "font_color": "#e2e8f0",
}


# =============================================================================
# 2. DATA LOADING
# =============================================================================

def load_data(config: dict) -> pd.DataFrame:
    """Load GRN data from Excel file."""
    path = config["input_file"]
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        print("        Make sure the Excel file is in the same folder as this script.")
        sys.exit(1)

    print(f"[1/5] Loading data from: {path}")
    df = pd.read_excel(path, sheet_name=config["sheet_name"])

    # Rename columns for clarity
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

    # Clean
    df = df.dropna(subset=["regulator", "target"])
    df["regulator"] = df["regulator"].astype(str).str.strip()
    df["target"]    = df["target"].astype(str).str.strip()
    df["effect"]    = df["effect"].astype(str).str.strip().str.lower()
    df["stage"]     = df["stage"].astype(str).str.strip()

    # Normalize effect values
    df["effect"] = df["effect"].replace({"nan": "o", "none": "o", "0": "o"})

    print(f"       Loaded {len(df):,} edges")
    return df


# =============================================================================
# 3. FILTERING
# =============================================================================

def stage_numeric(stage: str) -> float:
    """Convert stage string to a sortable numeric value."""
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


def filter_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Apply all filters from config."""
    print(f"[2/5] Applying filters...")

    original_count = len(df)

    # Effect filter
    df = df[df["effect"].isin(config["effects_include"])]

    # Stage filters
    if config["stage_single"]:
        df = df[df["stage"] == config["stage_single"]]
        print(f"       Stage = {config['stage_single']}")
    else:
        if config["stage_from"]:
            min_val = stage_numeric(config["stage_from"])
            df = df[df["stage"].apply(stage_numeric) >= min_val]
        if config["stage_to"]:
            max_val = stage_numeric(config["stage_to"])
            df = df[df["stage"].apply(stage_numeric) <= max_val]
        if config["stage_from"] or config["stage_to"]:
            print(f"       Stage range: {config['stage_from'] or 'start'} → {config['stage_to'] or 'end'}")

    # Gene filters
    if config["filter_regulator"]:
        df = df[df["regulator"] == config["filter_regulator"]]
        print(f"       Regulator = {config['filter_regulator']}")

    if config["filter_target"]:
        df = df[df["target"] == config["filter_target"]]
        print(f"       Target = {config['filter_target']}")

    # Cap edges
    if config["max_edges"] and len(df) > config["max_edges"]:
        df = df.head(config["max_edges"])
        print(f"       Capped at {config['max_edges']} edges")

    print(f"       {original_count:,} → {len(df):,} edges after filtering")
    return df.reset_index(drop=True)


# =============================================================================
# 4. GRAPH BUILDING
# =============================================================================

def build_graph(df: pd.DataFrame) -> nx.DiGraph:
    """Build a directed NetworkX graph from the filtered dataframe."""
    print(f"[3/5] Building NetworkX DiGraph...")

    G = nx.DiGraph()

    for _, row in df.iterrows():
        reg = row["regulator"]
        tgt = row["target"]
        eff = row["effect"]

        # Add nodes (will not duplicate)
        if not G.has_node(reg):
            G.add_node(reg)
        if not G.has_node(tgt):
            G.add_node(tgt)

        # Add edge (or accumulate multiple edges as list)
        if G.has_edge(reg, tgt):
            G[reg][tgt]["effects"].append(eff)
            G[reg][tgt]["stages"].append(str(row["stage"]))
            G[reg][tgt]["count"] += 1
        else:
            G.add_edge(reg, tgt,
                       effects=[eff],
                       stages=[str(row["stage"])],
                       contexts=[str(row["context"])],
                       count=1)

    # Tag each node: is it a regulator, target, or both?
    regulators = set(df["regulator"].unique())
    targets    = set(df["target"].unique())

    for node in G.nodes():
        G.nodes[node]["is_reg"] = node in regulators
        G.nodes[node]["is_tgt"] = node in targets

    print(f"       {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G


# =============================================================================
# 5. ANALYSIS
# =============================================================================

def analyze_graph(G: nx.DiGraph, config: dict) -> dict:
    """Run graph analytics. Results are embedded into node attributes."""
    print(f"[4/5] Running graph analysis...")

    results = {}

    # --- Degree centrality ---
    in_deg  = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    deg_centrality = nx.degree_centrality(G)

    for node in G.nodes():
        G.nodes[node]["in_degree"]       = in_deg[node]
        G.nodes[node]["out_degree"]      = out_deg[node]
        G.nodes[node]["total_degree"]    = in_deg[node] + out_deg[node]
        G.nodes[node]["deg_centrality"]  = round(deg_centrality[node], 4)

    # --- Hub genes (top 10 by out-degree = most active regulators) ---
    hub_genes = sorted(out_deg.items(), key=lambda x: x[1], reverse=True)[:10]
    results["hub_genes"] = hub_genes
    print(f"       Top hub regulators: {', '.join(g[0] for g in hub_genes[:5])}")

    # --- Feedback loops (cycles) ---
    feedback_nodes = set()
    feedback_edges = set()
    cycles = []

    if config["show_feedback_loops"]:
        try:
            # simple_cycles can be slow on large graphs — limit search
            cycle_gen = nx.simple_cycles(G)
            for i, cycle in enumerate(cycle_gen):
                if i > 500:  # safety cap
                    break
                cycles.append(cycle)
                for node in cycle:
                    feedback_nodes.add(node)
                for j in range(len(cycle)):
                    feedback_edges.add((cycle[j], cycle[(j+1) % len(cycle)]))
        except Exception:
            pass

        results["feedback_loops"]  = cycles
        results["feedback_nodes"]  = feedback_nodes
        results["feedback_edges"]  = feedback_edges

        # Self-loops (auto-regulatory)
        self_loops = list(nx.selfloop_edges(G))
        results["self_loops"] = self_loops

        print(f"       Feedback loops detected: {len(cycles)}")
        print(f"       Self-regulatory loops: {len(self_loops)}")
        if self_loops:
            print(f"       Self-loop genes: {', '.join(set(e[0] for e in self_loops))}")

    # --- Weakly connected components ---
    components = list(nx.weakly_connected_components(G))
    results["components"] = components
    print(f"       Connected components: {len(components)}")

    return results


# =============================================================================
# 6. VISUALIZATION — Pyvis
# =============================================================================

def node_size(G: nx.DiGraph, node: str, config: dict) -> int:
    """Scale node size by total degree."""
    deg = G.nodes[node].get("total_degree", 1)
    max_deg = max(d for _, d in G.degree()) or 1
    ratio = deg / max_deg
    size_range = config["node_size_max"] - config["node_size_min"]
    return int(config["node_size_min"] + ratio * size_range)


def node_color(G: nx.DiGraph, node: str, analysis: dict, config: dict) -> str:
    """Determine node color based on role and feedback status."""
    # Feedback loop node takes priority
    if config["show_feedback_loops"]:
        if node in analysis.get("feedback_nodes", set()):
            # Only override with loop color if it's a self-loop
            self_loop_nodes = set(e[0] for e in analysis.get("self_loops", []))
            if node in self_loop_nodes:
                return config["color_loop"]

    is_reg = G.nodes[node].get("is_reg", False)
    is_tgt = G.nodes[node].get("is_tgt", False)

    if is_reg and is_tgt:
        return config["color_both"]
    elif is_reg:
        return config["color_regulator"]
    else:
        return config["color_target"]


def edge_color(effects: list, config: dict) -> str:
    """Pick edge color based on dominant effect."""
    if not effects:
        return config["color_noeffect"]
    # Most common effect wins
    from collections import Counter
    dominant = Counter(effects).most_common(1)[0][0]
    if dominant == "+":
        return config["color_activating"]
    elif dominant == "-":
        return config["color_inhibiting"]
    return config["color_noeffect"]


def edge_label(effects: list, stages: list) -> str:
    """Build a short edge label."""
    dominant = max(set(effects), key=effects.count) if effects else "o"
    symbol = {"+" : "▲", "-": "▼", "o": "○"}.get(dominant, "?")
    stage_str = stages[0] if stages else ""
    return f"{symbol} {stage_str}"


def build_tooltip(node: str, G: nx.DiGraph, analysis: dict) -> str:
    """Build rich HTML tooltip for a node."""
    data = G.nodes[node]
    is_reg = data.get("is_reg", False)
    is_tgt = data.get("is_tgt", False)
    role   = "Regulator & Target" if (is_reg and is_tgt) else ("Regulator" if is_reg else "Target")

    in_deg  = data.get("in_degree", 0)
    out_deg = data.get("out_degree", 0)
    cent    = data.get("deg_centrality", 0)

    in_fb  = node in analysis.get("feedback_nodes", set())
    self_l = node in set(e[0] for e in analysis.get("self_loops", []))

    tip = f"""<b style='color:#00d4ff'>{node}</b><br/>
Role: {role}<br/>
Out-edges (targets): {out_deg}<br/>
In-edges (regulators): {in_deg}<br/>
Degree centrality: {cent:.3f}<br/>"""

    if in_fb:
        tip += "⚡ Part of feedback loop<br/>"
    if self_l:
        tip += "🔄 Self-regulatory loop<br/>"

    return tip


def visualize(G: nx.DiGraph, analysis: dict, config: dict):
    """Build and save the Pyvis interactive HTML."""
    print(f"[5/5] Building Pyvis visualization...")

    net = Network(
        height=config["height"],
        width=config["width"],
        bgcolor=config["bgcolor"],
        font_color=config["font_color"],
        directed=True,
        notebook=False,
    )

    # Physics layout
    layout = config["layout"]
    if layout == "barnes_hut":
        net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=120, spring_strength=0.05)
    elif layout == "force_atlas_2based":
        net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=100, spring_strength=0.08)
    elif layout == "repulsion":
        net.repulsion(node_distance=150, central_gravity=0.2, spring_length=200, spring_strength=0.05)
    elif layout == "hierarchical":
        net.set_options("""{"layout": {"hierarchical": {"enabled": true, "direction": "UD", "sortMethod": "directed"}}}""")

    # Add nodes
    feedback_nodes = analysis.get("feedback_nodes", set())
    feedback_edges = analysis.get("feedback_edges", set())

    for node in G.nodes():
        color  = node_color(G, node, analysis, config)
        size   = node_size(G, node, config)
        title  = build_tooltip(node, G, analysis)

        # Glow effect for feedback/hub nodes
        border_width = 3 if node in feedback_nodes else 1
        border_color = "#facc15" if node in feedback_nodes else color

        net.add_node(
            node,
            label=node,
            color={"background": color, "border": border_color,
                   "highlight": {"background": "#ffffff", "border": "#ffffff"}},
            size=size,
            title=title,
            borderWidth=border_width,
            font={"color": "#e2e8f0", "size": 11, "face": "monospace"},
        )

    # Add edges
    for u, v, data in G.edges(data=True):
        effects  = data.get("effects", ["o"])
        stages   = data.get("stages", [])
        count    = data.get("count", 1)
        color    = edge_color(effects, config)
        label    = edge_label(effects, stages)
        is_fb    = (u, v) in feedback_edges

        # Build edge tooltip
        stage_list = ", ".join(sorted(set(stages)))
        tip = f"<b>{u} → {v}</b><br/>Effect(s): {', '.join(set(effects))}<br/>Stages: {stage_list}<br/>Evidence count: {count}"
        if is_fb:
            tip += "<br/>⚡ Part of feedback loop"

        net.add_edge(
            u, v,
            color={"color": color, "highlight": "#ffffff"},
            label=label,
            title=tip,
            width=1.5 + (count * 0.3),       # thicker = more evidence
            arrows={"to": {"enabled": True, "scaleFactor": 0.6}},
            dashes=is_fb,                      # dashed = feedback loop edge
            font={"color": "#94a3b8", "size": 9, "align": "middle"},
            smooth={"type": "curvedCW", "roundness": 0.1},
        )

    # Enable interaction options
    net.set_options("""
    {
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "navigationButtons": true,
        "keyboard": true,
        "multiselect": true
      },
      "nodes": {
        "shadow": {"enabled": true, "size": 8, "x": 2, "y": 2}
      },
      "edges": {
        "shadow": false,
        "smooth": {"type": "curvedCW", "roundness": 0.1}
      },
      "physics": {
        "stabilization": {"iterations": 200}
      }
    }
    """)

    output_path = config["output_file"]
    net.save_graph(output_path)
    print(f"       Saved → {output_path}")
    return output_path


# =============================================================================
# 7. MAIN
# =============================================================================

def print_summary(G: nx.DiGraph, analysis: dict):
    """Print a clean summary to terminal."""
    print()
    print("=" * 60)
    print("  GRN ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"  Nodes        : {G.number_of_nodes():,}")
    print(f"  Edges        : {G.number_of_edges():,}")
    regs = sum(1 for n in G.nodes() if G.nodes[n].get("is_reg") and not G.nodes[n].get("is_tgt"))
    tgts = sum(1 for n in G.nodes() if G.nodes[n].get("is_tgt") and not G.nodes[n].get("is_reg"))
    both = sum(1 for n in G.nodes() if G.nodes[n].get("is_reg") and G.nodes[n].get("is_tgt"))
    print(f"  Regulators   : {regs}")
    print(f"  Targets      : {tgts}")
    print(f"  Both         : {both}")
    print()

    print("  TOP HUB REGULATORS (by out-degree):")
    for gene, deg in analysis.get("hub_genes", []):
        bar = "█" * min(deg, 40)
        print(f"    {gene:<25} {deg:>4}  {bar}")
    print()

    loops = analysis.get("feedback_loops", [])
    self_loops = analysis.get("self_loops", [])
    print(f"  FEEDBACK LOOPS : {len(loops)}")
    print(f"  SELF-LOOPS     : {len(self_loops)}")
    if self_loops:
        print(f"  Self-loop genes: {', '.join(set(e[0] for e in self_loops))}")

    components = analysis.get("components", [])
    print(f"  COMPONENTS     : {len(components)}")
    print("=" * 60)
    print()


def main():
    print()
    print("=" * 60)
    print("  Lens GRN Explorer — NetworkX + Pyvis")
    print("=" * 60)
    print()

    # Run pipeline
    df       = load_data(CONFIG)
    df       = filter_data(df, CONFIG)

    if len(df) == 0:
        print("[ERROR] No edges remain after filtering. Adjust your CONFIG filters.")
        sys.exit(1)

    G        = build_graph(df)
    analysis = analyze_graph(G, CONFIG)

    print_summary(G, analysis)

    output   = visualize(G, analysis, CONFIG)

    # Open in browser
    abs_path = os.path.abspath(output)
    print(f"  Opening in browser: {abs_path}")
    webbrowser.open(f"file://{abs_path}")
    print()
    print("  Done! Edit CONFIG at the top of this file to change filters.")
    print()


if __name__ == "__main__":
    main()


# =============================================================================
# QUICK REFERENCE — useful one-liners to run in a Python REPL or Jupyter
# =============================================================================
#
# Load graph manually:
#   df = load_data(CONFIG)
#   df = filter_data(df, CONFIG)
#   G  = build_graph(df)
#
# Explore:
#   list(G.nodes())[:20]                        # all nodes
#   list(G.edges(data=True))[:5]               # edges with attributes
#   nx.in_degree_centrality(G)                 # which genes are most regulated
#   nx.out_degree_centrality(G)                # which genes regulate the most
#   nx.shortest_path(G, 'Pax6', 'Prox1')      # path between two genes
#   list(nx.simple_cycles(G))                  # all feedback loops
#   nx.descendants(G, 'Pax6')                  # everything downstream of Pax6
#   nx.ancestors(G, 'Foxe3')                   # everything upstream of Foxe3
#   nx.pagerank(G)                             # PageRank influence score
#
# Export:
#   nx.write_graphml(G, 'grn.graphml')         # open in Cytoscape
#   nx.write_gexf(G, 'grn.gexf')              # open in Gephi
# =============================================================================