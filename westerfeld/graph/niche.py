import pandas as pd
import numpy as np    
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
from typing import Literal, Optional, Tuple

from _utils import edge_kingdom_type

# Niche-breadth thresholds (https://doi.org/10.1093/femsec/fiw174).
# A taxon below the lower mean-relative-abundance cutoff is ignored; otherwise
# Bj below the specialist threshold marks a specialist and Bj above the
# generalist threshold marks a generalist.
HABITAT_THRESHOLDS = {
    "Field_Soil": {
        "mean_rel_abundance": 2e-5,
        "specialist": 1.5,
        "generalist": 27.5,  
    },
    "Rhizosphere": {
        "mean_rel_abundance": 2e-5,
        "specialist": 1.5,
        "generalist": 25.0, 
    }
}

def identify_generalists_or_specialists(Pj, habitat_type):
    """
        Niche-breadth approach as described in https://doi.org/10.1093/femsec/fiw174.

        Returns the classification (or ``None``), the mean relative abundance, and
        the niche-breadth value Bj.
    """
    if habitat_type not in HABITAT_THRESHOLDS:
        raise ValueError(f"Unknown habitat type: {habitat_type}. Only Field_Soil or Rhizosphere possible.")

    thresholds = HABITAT_THRESHOLDS[habitat_type]
    gen_thresh = thresholds["generalist"]
    spec_thresh = thresholds["specialist"]
    mean_thresh = thresholds["mean_rel_abundance"]

    Pj = Pj / Pj.sum()
    mean_relative_abundance = Pj.mean()
    if mean_relative_abundance < mean_thresh:
        return None, np.array([]), np.array([])
    Bj = 1 / (Pj**2).sum()
    if Bj > gen_thresh:
        return "Generalist", mean_relative_abundance, Bj
    elif Bj < spec_thresh:
        return "Specialist", mean_relative_abundance, Bj
    return None, mean_relative_abundance, Bj

def _annotate_niche(G, df_lookup, df_relative):
    """Attach the lookup attributes plus a niche classification to each node."""
    nodes_attr = dict(G.nodes)
    
    for node in G.nodes:
        attributes = df_lookup.loc[node]
        habitat_type = attributes["habitat"]

        spec_or_gen, _, Bj = identify_generalists_or_specialists(
            df_relative[node].to_numpy(),
            habitat_type=habitat_type
        )
        
        attributes.loc["generalist_or_specialists"] = (
            spec_or_gen if spec_or_gen is not None else "None"
        )
        
        attributes.loc["niche_breadth"] = Bj if Bj > 0 else np.nan
        attributes.loc["mean_relative_abundance"] = df_relative[node].mean()
        nodes_attr[node] = attributes
        
    nx.set_node_attributes(G, nodes_attr)

def extract_niche_data(graphs, labels):
    """
    Extracts niche data from graphs and returns it as a structured DataFrame.
    """
    data = []
    
    for G, label in zip(graphs, labels):
        for node, attrs in G.nodes(data=True):
            bj = attrs.get("niche_breadth", np.nan)
            mean_ab = attrs.get("mean_relative_abundance", np.nan)
            niche_class = attrs.get("generalist_or_specialists", "None")
            kingdom = attrs.get("kingdom", "Unknown")

            if not np.isnan(bj) and not np.isnan(mean_ab) and mean_ab > 0:
                data.append({
                    "Habitat": label,
                    "Taxon": node,
                    "Kingdom": kingdom,
                    "Niche_Class": niche_class,
                    "Niche_Breadth": bj,
                    "Log_Mean_Abundance": np.log10(mean_ab),
                })
    
    return pd.DataFrame(data)

def plot_niche_breadth_boxplot(
    graphs,
    labels,
    path="FigS2_niche_breadth_boxplot.png",
    figsize=(8, 6),
):   
    df = extract_niche_data(graphs, labels)

    fig, ax = plt.subplots(figsize=figsize)
    
    box_data = [df[df["Habitat"] == label]["Niche_Breadth"].values for label in labels]
    bp = ax.boxplot(box_data, tick_labels=labels, patch_artist=True, showmeans=True)
    
    colors = ['#1f77b4', '#ff7f0e'] 
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("Niche Breadth ($B_j$)", fontsize=12)
    ax.set_title("Niche Breadth Distribution per Habitat", fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    for label in labels:
        if "Field" in label:
            ax.axhline(y=27.5, color='blue', linestyle=':', alpha=0.5, label='FS Threshold Generalist (27.5)')
            ax.axhline(y=1.5, color='blue', linestyle='-', alpha=0.5, label='FS Threshold Specialist (1.5)')

        elif "Rhizo" in label:
            ax.axhline(y=25, color='orange', linestyle=':', alpha=0.5, label='RH Threshold Generalist (25)')
            ax.axhline(y=1.5, color='orange', linestyle='-', alpha=0.5, label='RH Threshold Specialist (1.5)')
    
    handles, labels_legend = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_legend, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2, frameon=False)

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Boxplot saved to {path}")
    return path

def plot_niche_breadth_vs_abundance_grid(
    graphs,
    labels,
    path="FigS3_niche_breadth_vs_abundance.png",
    figsize=(14, 7),
):
    df = extract_niche_data(graphs, labels)

    fig, axes = plt.subplots(1, len(graphs), figsize=figsize)
    if len(graphs) == 1:
        axes = [axes]

    color_map = {
        "Generalist": "#2ca02c",
        "Specialist": "#d62728",
        "None": "#7f7f7f",
    }
    
    marker_map = {
        "Fungi": "o",
        "Bacteria": "^"
    }

    unique_kingdoms = df["Kingdom"].unique()
    unique_classes = df["Niche_Class"].unique()

    for ax, label in zip(axes, labels):
        df_habitat = df[df["Habitat"] == label]
        
        for kingdom in unique_kingdoms:
            for niche_class in unique_classes:
                subset = df_habitat[
                    (df_habitat["Kingdom"] == kingdom) & 
                    (df_habitat["Niche_Class"] == niche_class)
                ]
                
                if not subset.empty:
                    ax.scatter(
                        subset["Log_Mean_Abundance"],
                        subset["Niche_Breadth"],
                        c=color_map[niche_class],
                        marker=marker_map[kingdom],
                        label=f"{kingdom} {niche_class}" if ax == axes[0] else "",
                        alpha=0.7,
                        edgecolors='black',
                        linewidth=0.5,
                        s=50
                    )

        ax.set_xlabel("Log10(Mean Relative Abundance)")
        ax.set_ylabel("Niche Breadth ($B_j$)")
        ax.set_title(label)
        ax.grid(True, linestyle='--', alpha=0.5)

    handles, labels_legend = axes[0].get_legend_handles_labels()
    
    legend_marker = []
    for kingdom in unique_kingdoms:
        mk = marker_map.get(kingdom, "o")
        legend_marker.append(
            Line2D([0], [0], marker=mk, color="w", markerfacecolor="gray", 
                    markersize=10, label=kingdom, markeredgecolor='black')
        )
    
    legend_color = []
    for n_class in unique_classes:
        col = color_map.get(n_class, "gray")
        label_text = "Unclassified" if n_class == "None" else n_class
        legend_color.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor=col, 
                    markersize=10, label=label_text)
        )

    fig.legend(handles=legend_marker, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=2, frameon=False, title="Kingdom")
    fig.legend(handles=legend_color, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False, title="Classification")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    return path

def plot_graphs_side_by_side_by_niche(
    graphs,
    labels,
    path="Fig6_graph_side_by_side_niche.png",
    figsize=(14, 7),
    node_size=50,
    edge_width=1.0,
):

    fig, axes = plt.subplots(1, len(graphs), figsize=figsize)
    if len(graphs) == 1:
        axes = [axes]

    classification_colors = {
        "Generalist": "#2422b8",
        "Specialist": "#d62728",
        "None": "#7f7f7f",
    }

    for ax, G, label in zip(axes, graphs, labels):
        if G.number_of_nodes() == 0:
            ax.set_axis_off()
            continue

        pos = nx.spring_layout(G, seed=42)
        
        # Knoten in zwei Listen aufteilen
        nodes_colored = []
        nodes_gray = []
        colors_colored = []
        
        for n in G.nodes:
            niche = G.nodes[n].get("generalist_or_specialists", "None")
            if niche == "None":
                nodes_gray.append(n)
            else:
                nodes_colored.append(n)
                colors_colored.append(classification_colors.get(niche, "#7f7f7f"))

        # Kanten zeichnen (wie gehabt)
        for edge_type in sorted({edge_kingdom_type(G, u, v) for u, v in G.edges()}):
            edges = [e for e in G.edges() if edge_kingdom_type(G, e[0], e[1]) == edge_type]
            if not edges:
                continue
            nx.draw_networkx_edges(
                G, pos, edgelist=edges, edge_color="#999999",
                width=edge_width, alpha=0.6, ax=ax,
            )

        # 1. Graue Knoten zeichnen (Transparent)
        if nodes_gray:
            nx.draw_networkx_nodes(
                G, pos, nodelist=nodes_gray, node_color="#7f7f7f",
                node_size=node_size, alpha=0.2, ax=ax  # <--- Alpha 0.2
            )

        # 2. Bunte Knoten zeichnen (Deckend)
        if nodes_colored:
            nx.draw_networkx_nodes(
                G, pos, nodelist=nodes_colored, node_color=colors_colored,
                node_size=node_size, alpha=1.0, 
                edgecolors="#454545", linewidths=0.5,
                ax=ax
            )

        ax.set_title(label)
        ax.set_axis_on()
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=10, label=label)
        for label, color in (
            ("Generalist", classification_colors["Generalist"]),
            ("Specialist", classification_colors["Specialist"]),
            ("Unclassified", classification_colors["None"]),
        )
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path

def analyze_niche(graphs, labels):
    """
    1. Analyzes the overlap of generalists and specialists between two habitats.
    2. Calculates the change in degree (number of connections) for taxa present in both habitats.
    """
    if len(graphs) != 2:
        print("'Two graphs are required.'")
        return

    G1, G2 = graphs
    label1, label2 = labels

    common_nodes = set(G1.nodes()).intersection(set(G2.nodes()))

    overlap_data = []
    change_data = []

    for node in common_nodes:
        attrs1 = G1.nodes[node]
        attrs2 = G2.nodes[node]

        niche1 = attrs1.get("generalist_or_specialists", "None")
        niche2 = attrs2.get("generalist_or_specialists", "None")
        taxon_name = node.split(":")[-1]
        kingdom = attrs1.get("kingdom", "Unknown")

        # 1. Overlap Data
        overlap_data.append({
            "Taxon": taxon_name,
            "Kingdom": kingdom,
            f"Niche_{label1}": niche1,
            f"Niche_{label2}": niche2,
            "Status": "Consistent" if niche1 == niche2 else "Changed"
        })

        # 2. Degree Change Data
        deg1 = G1.degree(node)
        deg2 = G2.degree(node)
        delta = deg2 - deg1

        change_data.append({
            "Taxon": taxon_name,
            "Kingdom": kingdom,
            f"Degree_{label1}": deg1,
            f"Degree_{label2}": deg2,
            "Delta_Degree": delta,
            f"Niche_{label1}": niche1,
            f"Niche_{label2}": niche2
        })

    df_overlap = pd.DataFrame(overlap_data)
    df_change = pd.DataFrame(change_data)

    with pd.ExcelWriter("niche_analysis.xlsx") as writer:
        df_overlap.to_excel(writer, sheet_name="overlap", index=False)
        df_change.to_excel(writer, sheet_name="change", index=False)

    return df_overlap, df_change

def plot_degree_change(df_change, path="degree_change_analysis.png"):
    """
    Plots the change in degree (number of connections) for taxa present in both habitats.
    Requires a DataFrame with columns 'Taxon', 'Delta_Degree', etc.
    """

    df_change["Abs_Delta"] = df_change["Delta_Degree"].abs()
    df_sorted = df_change.sort_values(by="Abs_Delta", ascending=False)

    top_n = 20
    df_plot = df_sorted.head(top_n).copy()
    
    colors = ["#d62728" if x < 0 else "#1f77b4" for x in df_plot["Delta_Degree"]]

    fig, ax = plt.subplots(figsize=(10, 8))

    y_pos = np.arange(len(df_plot))
    ax.barh(y_pos, df_plot["Delta_Degree"], color=colors, alpha=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot["Taxon"])
    ax.invert_yaxis()
    ax.set_xlabel("Change in Number of Connections (Rhizo - Field)", fontsize=12)
    ax.set_title(f"Top {top_n} Taxa with Largest Network Changes", fontsize=14)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#d62728", lw=4, label='Lost connections'),
        Line2D([0], [0], color="#1f77b4", lw=4, label='Gained connections'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_consistent_degree_change_comparison(
    df_change, 
    df_overlap, 
    labels,
    path="degree_change_consistent_comparison.png",
    figsize=(16, 8)
):
    """
    Creates a plot with 2 subplots comparing degree changes for consistent generalists and specialists.
    """

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=False)
    
    targets = ["Generalist", "Specialist"]
    label1, label2 = labels

    for ax, target_class in zip(axes, targets):
        # 1. Filtere df_overlap nach konsistenten Taxa dieser Klasse
        mask = (
            (df_overlap["Status"] == "Consistent") & 
            (df_overlap[f"Niche_{label1}"] == target_class) & 
            (df_overlap[f"Niche_{label2}"] == target_class)
        )
        consistent_taxa_list = df_overlap[mask]["Taxon"].tolist()

        if not consistent_taxa_list:
            ax.text(0.5, 0.5, f"No consistent {target_class}s found", ha='center', va='center')
            ax.set_title(f"Consistent {target_class}s")
            continue

        df_subset = df_change[df_change["Taxon"].isin(consistent_taxa_list)].copy()
        
        df_subset["Abs_Delta"] = df_subset["Delta_Degree"].abs()
        df_sorted = df_subset.sort_values(by="Abs_Delta", ascending=False)

        y_pos = np.arange(len(df_sorted))
        colors = ["#d62728" if x < 0 else "#1f77b4" for x in df_sorted["Delta_Degree"]]
        
        ax.barh(y_pos, df_sorted["Delta_Degree"], color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_sorted["Taxon"])
        ax.invert_yaxis()
        ax.set_title(f"Consistent {target_class}s (n={len(df_sorted)})", fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_xlabel("Change in Connections (Rhizo - Field)")

    legend_elements = [
        Line2D([0], [0], color="#d62728", lw=4, label='Lost connections'),
        Line2D([0], [0], color="#1f77b4", lw=4, label='Gained connections'),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.05))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()