"""
=============================================================================
Lens GRN Explorer — NetworkX + Pyvis
=============================================================================
Lachke Lab 2016 — Gene Regulatory Network

SETUP (run once):
    pip install networkx pyvis pandas openpyxl plotly dash dash-bootstrap-components

USAGE:
    python grn_network.py        ← opens grn_output.html in browser
    python app.py                ← runs full Dash app with live filters

KEY BIOLOGY NOTE:
    The TRUE regulatory relationship requires combining TWO columns:
        Perturbation: what was done to the regulator (+/- = overexpressed/knocked out)
        Effect:       what happened to the target   (+/- = increased/decreased, o = no change)

    True relationship = Perturbation × Effect:
        Pert(-) × Effect(-) → ACTIVATING  (knockout reduces target → reg normally activates)
        Pert(-) × Effect(+) → INHIBITING  (knockout increases target → reg normally inhibits)
        Pert(+) × Effect(+) → ACTIVATING  (overexpress increases target → reg activates)
        Pert(+) × Effect(-) → INHIBITING  (overexpress decreases target → reg inhibits)
        Any    × Effect(o)  → NO EFFECT

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

# --- Wong (2011) Color-Blind Friendly Palette ---
# Nature Methods standard, safe for all types of color blindness
WONG = {
    "black":        "#000000",
    "orange":       "#E69F00",
    "sky_blue":     "#56B4E9",
    "green":        "#009E73",
    "yellow":       "#F0E442",
    "blue":         "#0072B2",
    "vermillion":   "#D55E00",
    "pink":         "#CC79A7",
}

CONFIG = {
    # --- File paths ---
    "input_file":  "data/Lens_GRN_June_2016_original FOR HACKATHON - Salil Lachke.xlsx",
    "output_file": "grn_output.html",
    "sheet_name":  "Lens_GRN_pert",

    # --- Stage filters ---
    "stage_from":   None,
    "stage_to":     None,
    "stage_single": None,

    # --- Gene filters ---
    "filter_regulator": None,
    "filter_target":    None,

    # --- True relationship filter ---
    # Options: "activating", "inhibiting", "no_effect" — or all three
    "relationships_include": ["activating", "inhibiting", "no_effect"],

    # --- Graph display ---
    "max_edges":           300,
    "show_feedback_loops": True,

    # --- Node colors (Wong palette) ---
    "color_regulator": WONG["sky_blue"],    # sky blue  — regulates others
    "color_target":    WONG["orange"],      # orange    — regulated by others
    "color_both":      WONG["pink"],        # pink      — regulates AND is regulated
    "color_selfloop":  WONG["yellow"],      # yellow    — self-regulatory gene

    # --- Edge colors (Wong palette) ---
    "color_activating": WONG["green"],      # green     — activates target
    "color_inhibiting": WONG["vermillion"], # vermillion — inhibits target
    "color_noeffect":   "#94a3b8",          # grey      — no effect observed

    # --- Node sizing ---
    "node_size_min": 12,
    "node_size_max": 55,

    # --- Layout ---
    "layout": "barnes_hut",

    # --- Theme: "dark" or "light" ---
    "theme": "dark",
}

# Theme palettes
THEMES = {
    "dark": {
        "bgcolor":      "#0a0e1a",
        "font_color":   "#ffffff",
        "tooltip_bg":   "#0f1525",
        "tooltip_border": "#1e2d4a",
        "tooltip_text": "#e2e8f0",
        "legend_bg":    "#0f1525",
        "legend_border": "#1e2d4a",
        "legend_text":  "#e2e8f0",
        "legend_muted": "#475569",
    },
    "light": {
        "bgcolor":      "#f8fafc",
        "font_color":   "#0f172a",
        "tooltip_bg":   "#ffffff",
        "tooltip_border": "#cbd5e1",
        "tooltip_text": "#0f172a",
        "legend_bg":    "#ffffff",
        "legend_border": "#e2e8f0",
        "legend_text":  "#0f172a",
        "legend_muted": "#64748b",
    },
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
    df["perturbation"] = df["perturbation"].astype(str).str.strip()
    df["effect"]       = df["effect"].astype(str).str.strip()
    df["stage"]        = df["stage"].astype(str).str.strip()
    df["context"]      = df["context"].astype(str).str.strip()

    df["effect"]       = df["effect"].replace({"nan": "o", "none": "o", "None": "o", "0": "o"})
    df["perturbation"] = df["perturbation"].replace({"nan": "-", "none": "-", "None": "-"})

    # --- Compute TRUE regulatory relationship ---
    # Perturbation × Effect → true_relationship
    df["true_relationship"] = df.apply(_compute_relationship, axis=1)

    print(f"       Loaded {len(df):,} edges")
    rel_counts = df["true_relationship"].value_counts()
    for rel, cnt in rel_counts.items():
        print(f"       {rel}: {cnt:,}")
    return df


def _compute_relationship(row) -> str:
    """
    Compute the true regulatory relationship from perturbation × effect.
    
    Logic:
      Effect = 'o'           → no_effect  (regardless of perturbation)
      Pert('-') × Effect('-') → activating (knockout reduces target → reg activates)
      Pert('-') × Effect('+') → inhibiting (knockout increases target → reg inhibits)
      Pert('+') × Effect('+') → activating (overexpress increases target → reg activates)
      Pert('+') × Effect('-') → inhibiting (overexpress decreases target → reg inhibits)
    """
    pert   = str(row["perturbation"]).strip()
    effect = str(row["effect"]).strip()

    if effect == "o":
        return "no_effect"

    # Same sign → activating, opposite sign → inhibiting
    if pert == effect:
        return "activating"
    else:
        return "inhibiting"


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

    # Filter by true relationship
    df = df[df["true_relationship"].isin(config["relationships_include"])]

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
        reg, tgt  = row["regulator"], row["target"]
        pert, eff = row["perturbation"], row["effect"]
        stg, ctx  = row["stage"], row["context"]
        rel       = row["true_relationship"]

        if G.has_edge(reg, tgt):
            G[reg][tgt]["perturbations"].append(pert)
            G[reg][tgt]["effects"].append(eff)
            G[reg][tgt]["relationships"].append(rel)
            G[reg][tgt]["stages"].append(stg)
            G[reg][tgt]["contexts"].append(ctx)
            G[reg][tgt]["count"] += 1
        else:
            G.add_edge(reg, tgt,
                       perturbations=[pert],
                       effects=[eff],
                       relationships=[rel],
                       stages=[stg],
                       contexts=[ctx],
                       count=1)

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
    if node in self_loop_nodes:   return config["color_selfloop"]
    is_reg = G.nodes[node].get("is_reg", False)
    is_tgt = G.nodes[node].get("is_tgt", False)
    if is_reg and is_tgt:         return config["color_both"]
    if is_reg:                    return config["color_regulator"]
    return config["color_target"]


def _edge_color(relationships, config):
    """Color based on dominant TRUE relationship."""
    from collections import Counter
    if not relationships: return config["color_noeffect"]
    dominant = Counter(relationships).most_common(1)[0][0]
    if dominant == "activating":  return config["color_activating"]
    if dominant == "inhibiting":  return config["color_inhibiting"]
    return config["color_noeffect"]


def _node_tooltip(node, G, analysis, theme):
    t       = THEMES[theme]
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
        f"<div style='font-family:monospace;font-size:13px;padding:8px;min-width:200px;"
        f"background:{t['tooltip_bg']};color:{t['tooltip_text']};border-radius:6px'>"
        f"<b style='font-size:15px;color:{WONG['sky_blue']}'>{node}</b><br/>"
        f"<hr style='border-color:{t['tooltip_border']};margin:6px 0'/>"
        f"<b>Role:</b> {role}<br/>"
        f"<b>Regulates (out):</b> {out_deg} genes<br/>"
        f"<b>Regulated by (in):</b> {in_deg} genes<br/>"
        f"<b>Degree centrality:</b> {cent:.3f}<br/>"
    )
    if self_loop:
        tip += f"<br/><span style='color:{WONG['yellow']}'>🔄 Self-regulatory loop</span><br/>"
    if in_feedback:
        tip += f"<span style='color:{WONG['orange']}'>⚡ Part of feedback loop</span><br/>"
    tip += "</div>"
    return tip


def _edge_tooltip(u, v, data, analysis, theme):
    t             = THEMES[theme]
    relationships = data.get("relationships", ["no_effect"])
    perturbations = data.get("perturbations", [])
    effects       = data.get("effects", ["o"])
    stages        = sorted(set(str(s) for s in data.get("stages", []) if pd.notna(s)))
    count         = data.get("count", 1)
    is_fb         = (u, v) in analysis.get("feedback_edges", set())

    from collections import Counter
    dom_rel  = Counter(relationships).most_common(1)[0][0] if relationships else "no_effect"
    rel_labels = {
        "activating": f"<span style='color:{WONG["green"]}'>▲ Activating</span>",
        "inhibiting": f"<span style='color:{WONG["vermillion"]}'>▼ Inhibiting</span>",
        "no_effect":  "<span style='color:#94a3b8'>○ No effect</span>",
    }
    rel_str  = rel_labels.get(dom_rel, dom_rel)
    stg_str  = ", ".join(stages[:5]) + ("…" if len(stages) > 5 else "")

    # Show raw data for transparency
    pert_unique = ", ".join(sorted(set(perturbations)))
    eff_unique  = ", ".join(sorted(set(effects)))

    tip = (
        f"<div style='font-family:monospace;font-size:13px;padding:8px;min-width:240px;"
        f"background:{t['tooltip_bg']};color:{t['tooltip_text']};border-radius:6px'>"
        f"<b style='color:{WONG['sky_blue']}'>{u}</b>"
        f"<span style='color:#94a3b8'> → </span>"
        f"<b style='color:{WONG['orange']}'>{v}</b><br/>"
        f"<hr style='border-color:{t['tooltip_border']};margin:6px 0'/>"
        f"<b>True relationship:</b> {rel_str}<br/>"
        f"<b>Perturbation(s):</b> {pert_unique}<br/>"
        f"<b>Raw effect(s):</b> {eff_unique}<br/>"
        f"<b>Stage(s):</b> {stg_str}<br/>"
        f"<b>Evidence count:</b> {count}<br/>"
    )
    if is_fb:
        tip += f"<br/><span style='color:{WONG['orange']}'>⚡ Feedback loop edge</span><br/>"
    tip += "</div>"
    return tip


def _legend_html(config, theme):
    t = THEMES[theme]
    return f"""
    <div style='position:fixed;bottom:20px;left:20px;z-index:9999;
                background:{t['legend_bg']};border:1px solid {t['legend_border']};
                border-radius:10px;padding:14px 18px;font-family:monospace;
                font-size:12px;color:{t['legend_text']};
                box-shadow:0 4px 20px rgba(0,0,0,0.3);min-width:210px'>
      <div style='font-weight:bold;font-size:13px;color:{WONG["sky_blue"]};margin-bottom:10px'>
        🔬 Lens GRN Legend
      </div>
      <div style='margin-bottom:6px;font-weight:bold;color:{t["legend_muted"]};font-size:10px;text-transform:uppercase'>
        Node — Role
      </div>
      <div style='margin-bottom:3px'><span style='color:{config["color_regulator"]}'>●</span> &nbsp;Regulator only</div>
      <div style='margin-bottom:3px'><span style='color:{config["color_target"]}'>●</span> &nbsp;Target only</div>
      <div style='margin-bottom:3px'><span style='color:{config["color_both"]}'>●</span> &nbsp;Regulator &amp; Target</div>
      <div style='margin-bottom:10px'><span style='color:{config["color_selfloop"]}'>●</span> &nbsp;Self-regulatory loop</div>
      <div style='margin-bottom:6px;font-weight:bold;color:{t["legend_muted"]};font-size:10px;text-transform:uppercase'>
        Edge — True Relationship
      </div>
      <div style='margin-bottom:3px'><span style='color:{config["color_activating"]}'>━━▶</span> &nbsp;Activating</div>
      <div style='margin-bottom:3px'><span style='color:{config["color_inhibiting"]}'>━━▶</span> &nbsp;Inhibiting</div>
      <div style='margin-bottom:10px'><span style='color:{config["color_noeffect"]}'>━━▶</span> &nbsp;No effect</div>
      <div style='font-size:10px;color:{t["legend_muted"]}'>
        True relationship = Perturbation × Effect<br/>
        Node size = connections<br/>
        Edge width = evidence count<br/>
        Colors: Wong (2011) color-blind safe
      </div>
    </div>"""


def visualize(G: nx.DiGraph, analysis: dict, config: dict) -> str:
    print(f"[5/5] Building Pyvis visualization...")
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
            title=_node_tooltip(node, G, analysis, theme),
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
        "shadow": {"enabled": true, "color": "rgba(0,0,0,0.3)", "size": 8, "x": 2, "y": 2}
      },
      "physics": {
        "stabilization": {"iterations": 250, "updateInterval": 25}
      }
    }
    """)

    output_path = config.get("output_file", "grn_output.html")
    net.save_graph(output_path)

    # Post-process: inject legend + style tooltip + white nav buttons
    with open(output_path, "r", encoding="utf-8") as f:
        html = f.read()

    nav_filter = "invert(1) brightness(2)" if theme == "dark" else "invert(0) brightness(0)"
    css = f"""<style>
      .vis-navigation .vis-button {{ filter: {nav_filter} !important; }}
      div.vis-tooltip {{
        background: {t['tooltip_bg']} !important;
        border: 1px solid {t['tooltip_border']} !important;
        color: {t['tooltip_text']} !important;
        border-radius: 8px !important;
        font-family: monospace !important;
        padding: 2px !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3) !important;
        max-width: 300px !important;
      }}
    </style>"""

    html = html.replace("</head>", css + "\n</head>")
    html = html.replace("</body>", _legend_html(config, theme) + "\n</body>")

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
# QUICK REFERENCE
# =============================================================================
#
#   df, G, analysis = run_pipeline(CONFIG)
#
#   nx.shortest_path(G, 'Pax6', 'Prox1')
#   nx.descendants(G, 'Pax6')
#   nx.ancestors(G, 'Foxe3')
#   nx.pagerank(G)
#   list(nx.simple_cycles(G))
#   nx.write_graphml(G, 'grn.graphml')   # → Cytoscape
#   nx.write_gexf(G, 'grn.gexf')        # → Gephi
# =============================================================================