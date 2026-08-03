import pandas as pd
import numpy as np    
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx

from _utils import edge_kingdom_type

import pandas as pd
import numpy as np

def identify_generalists_or_specialists(df_relative):
    """
    Identifiziert Generalisten und Spezialisten basierend auf Occurrence Frequency 
    und lokaler Abundanz (Mean Abundance given Presence).

    Kriterien:
    - Generalist: Occurrence Frequency >= 60% der Samples.
    - Spezialist: Occurrence Frequency <= 5% der Samples UND 
                  lokal abundant (mittlere Abundanz in positiven Samples >= 5%).
    
    Parameters
    ----------
    df_relative : pd.DataFrame
        DataFrame mit relativen Abundanzen. 
        Rows = Samples, Columns = Taxa.

    Returns
    -------
    classifications : pd.Series
        Klassifizierung pro Taxon ("Generalist", "Specialist" oder None).
    mean_rel_abundances : pd.Series
        Globale mittlere relative Abundanz pro Taxon (über alle Samples).
    occurrence_frequencies : pd.Series
        Anteil der Samples, in denen das Taxon vorkommt (> 0).
    """
    if df_relative.empty:
        return pd.Series(dtype=object), pd.Series(dtype=float), pd.Series(dtype=float)

    n_samples, n_taxa = df_relative.shape
    mean_rel_abundances = df_relative.mean(axis=0)
    occurrence_frequencies = (np.count_nonzero(df_relative, axis=0) / n_samples)

    local_abundances = df_relative.replace(0, np.nan).mean(axis=0)

    classifications = pd.Series(None, index=df_relative.columns, dtype=object)

    # --- Generalisten Logik ---
    gen_occ_mask = occurrence_frequencies >= 0.60
    gen_abund_thresh = local_abundances.quantile(0.60)
    gen_abund_mask = local_abundances < gen_abund_thresh
    generalist_mask = gen_occ_mask & gen_abund_mask
    classifications[generalist_mask] = "Generalist"

    # --- Spezialisten Logik ---
    spec_occ_mask = occurrence_frequencies <= 0.40
    spec_abund_thresh = local_abundances.quantile(0.60)
    spec_abund_mask = local_abundances >= spec_abund_thresh
    specialist_mask = spec_occ_mask & spec_abund_mask
    classifications[specialist_mask] = "Specialist"

    # >>> HIER DIE PRINTS EINFÜGEN <<<
    print(f"Generalists: {generalist_mask.sum()} ({generalist_mask.mean()*100:.1f}%)")
    print(f"Specialists: {specialist_mask.sum()} ({specialist_mask.mean()*100:.1f}%)")
    print(f"Unclassified: {n_taxa - generalist_mask.sum() - specialist_mask.sum()} ({(1 - generalist_mask.mean() - specialist_mask.mean())*100:.1f}%)")
    # >>> ENDE <<<

    return classifications, mean_rel_abundances, occurrence_frequencies

def _annotate_niche(G, df_lookup, df_relative):
    """
    Attach the lookup attributes plus a niche classification to each node.
    
    Die Niche-Klassifizierung wird einmal für alle Taxa berechnet und dann
    pro Node zugewiesen.
    """

    classifications, mean_rel_abundances, occurrence_frequencies = (
        identify_generalists_or_specialists(df_relative)
    )

    if isinstance(classifications, np.ndarray):
        classifications = pd.Series(classifications, index=df_relative.columns)
    if isinstance(mean_rel_abundances, np.ndarray):
        mean_rel_abundances = pd.Series(mean_rel_abundances, index=df_relative.columns)
    if isinstance(occurrence_frequencies, np.ndarray):
        occurrence_frequencies = pd.Series(occurrence_frequencies, index=df_relative.columns)
    
    nodes_attr = dict(G.nodes)
    
    for node in G.nodes:
        attributes = df_lookup.loc[node].copy()
        
        spec_or_gen = classifications.get(node, None)
        attributes["generalist_or_specialist"] = (
            spec_or_gen if spec_or_gen is not None else "None"
        )
        
        attributes["mean_relative_abundance"] = mean_rel_abundances.get(node, np.nan)
        attributes["occurrence_frequency"] = occurrence_frequencies.get(node, np.nan)
        
        nodes_attr[node] = attributes
    
    nx.set_node_attributes(G, nodes_attr)

def extract_niche_data(graphs, labels):
    """
    Extracts niche data from graphs and returns it as a structured DataFrame.
    """
    data = []
    
    for G, label in zip(graphs, labels):
        for node, attrs in G.nodes(data=True):
            mean_ab = attrs.get("mean_relative_abundance", np.nan)
            occ_freq = attrs.get("occurrence_frequency", np.nan)
            niche_class = attrs.get("generalist_or_specialist", "None")
            kingdom = attrs.get("kingdom", "Unknown")

            if not np.isnan(mean_ab) and not np.isnan(occ_freq) and mean_ab > 0:
                data.append({
                    "Habitat": label,
                    "Taxon": node,
                    "Kingdom": kingdom,
                    "Niche_Class": niche_class,
                    "Occurrence_Frequency": occ_freq,
                    "Mean_Relative_Abundance": mean_ab,
                    "Log_Mean_Abundance": np.log10(mean_ab),
                })
    
    return pd.DataFrame(data)
from matplotlib.lines import Line2D

def plot_occurrence_vs_abundance_grid(
    graphs,
    labels,
    path="FigS2_occurrence_vs_abundance.png",
    figsize=(14, 7),
):
    """
    Plots Occurrence Frequency vs. Mean Relative Abundance for each habitat.
    """
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
    unique_classes = df["Niche_Class"].fillna("None").unique()

    for ax, label in zip(axes, labels):
        df_habitat = df[df["Habitat"] == label]
        
        for kingdom in unique_kingdoms:
            for niche_class in unique_classes:
                niche_class_filter = "None" if niche_class is None else niche_class
                subset = df_habitat[
                    (df_habitat["Kingdom"] == kingdom) & 
                    (df_habitat["Niche_Class"].fillna("None") == niche_class_filter)
                ]
                
                if not subset.empty:
                    ax.scatter(
                        subset["Log_Mean_Abundance"],
                        subset["Occurrence_Frequency"],  # ← Geändert von Niche_Breadth
                        c=color_map.get(niche_class_filter, "#7f7f7f"),
                        marker=marker_map[kingdom],
                        label=f"{kingdom} {niche_class_filter}" if ax == axes[0] else "",
                        alpha=0.7,
                        edgecolors='black',
                        linewidth=0.5,
                        s=50
                    )

        ax.set_xlabel("Log10(Mean Relative Abundance)")
        ax.set_ylabel("Occurrence Frequency")  # ← Geändert von Niche Breadth
        ax.set_title(label)
        ax.grid(True, linestyle='--', alpha=0.5)

    # Legenden erstellen
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
        # None-Objekt zu String konvertieren für sicheren Vergleich
        n_class_str = str(n_class) if n_class is not None else "None"
        col = color_map.get(n_class_str, "gray")
        label_text = "Unclassified" if n_class_str == "None" else n_class_str
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
    """
    Visualizes networks side-by-side with nodes colored by niche classification
    (Generalist, Specialist, or Unclassified).
    """
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
            # ← Geändert von "generalist_or_specialists" zu "generalist_or_specialist"
            niche = G.nodes[n].get("generalist_or_specialist", "None")
            if niche == "None":
                nodes_gray.append(n)
            else:
                nodes_colored.append(n)
                colors_colored.append(classification_colors.get(niche, "#7f7f7f"))

        # Kanten zeichnen
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
                node_size=node_size, alpha=0.2, ax=ax
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