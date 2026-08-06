import pandas as pd
import numpy as np    
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx

from _utils import edge_kingdom_type
from ncm import ncm, taxa_bounds
from graph.comparison import _edge_color, _node_color

def identify_generalists_or_specialists_ncm():
    """
    Identifiziert Generalisten und Spezialisten basierend auf dem Neutral Community Model (NCM).

    Für jedes combination von 'type_label' und 'habitat' wird ein eigenes NCM-Modell
    und ein eigenes taxa_bounds(result) berechnet. Die Ergebnisse werden gesammelt.

    Klassifizierung:
        - 'above' → Generalist
        - 'below' → Specialist
        - 'neutral' → None (Unclassified)

    Returns
    -------
    classifications : pd.Series
        Klassifizierung pro Taxon: "Generalist", "Specialist" oder None.
    mean_rel_abundances : pd.Series
        Globale mittlere relative Abundanz pro Taxon (aus taxa_bounds).
    occurrence_frequencies : pd.Series
        Anteil der Samples, in denen das Taxon vorkommt (aus taxa_bounds).
    """

    type_labels = ["Fungi", "Bacteria"] 
    habitats = ["Field_Soil", "Rhizosphere"]
    taxonomy = "Genus"
    crops = ["Winter wheat 1", "Winter wheat 2"]
    years = 2019

    all_results_nested = {}
    all_taxa_dfs = []

    for type_label in type_labels:
        results = []
        taxa_dfs = []

        for habitat in habitats:
            result = ncm(
                type_label,
                habitat,
                taxonomy,
                years=years,
                habitats=[habitat],
                crops=crops,
            )
            results.append(result)
            print(f"  → Habitat: {habitat} → result created")

            df_taxa = taxa_bounds(result)
            taxa_dfs.append(df_taxa)
            print(f"  → Habitat: {habitat} → taxa_bounds created")

        all_results_nested[type_label] = results
        all_taxa_dfs.extend(taxa_dfs)  

    bounds_df = pd.concat(all_taxa_dfs, ignore_index=True)

    habitat_mapping = {
        "FS": "Field_Soil",
        "RH": "Rhizosphere",
    }
    bounds_df["Habitat"] = bounds_df["Habitat"].replace(habitat_mapping)

    bounds_df["Taxa"] = bounds_df["Kingdom"].astype(str) + ":" + bounds_df["Taxa"].astype(str)

    mapping = {
        "above": "Generalist",
        "below": "Specialist",
        "neutral": None
    }

    classifications = bounds_df.set_index("Taxa")["Prediction"].map(mapping)
    mean_rel_abundances = bounds_df.set_index("Taxa")["Mean Relative Abundance"]
    occurrence_frequencies = bounds_df.set_index("Taxa")["Occurrence Frequency"]

    total = len(classifications)
    n_gen = (classifications == "Generalist").sum()
    n_spec = (classifications == "Specialist").sum()
    n_unc = total - n_gen - n_spec

    print(f"✅ Generalists: {n_gen} ({n_gen/total*100:.1f}%)")
    print(f"✅ Specialists: {n_spec} ({n_spec/total*100:.1f}%)")
    print(f"✅ Unclassified: {n_unc} ({n_unc/total*100:.1f}%)")

     # --- 9. Statistik pro Habitat ausgeben ---
    print("\n" + "-"*60)
    print("📊 Klassifizierung pro Habitat")
    print("-"*60)

    # ✅ Gruppiere nach Habitat
    habitat_stats = bounds_df.groupby("Habitat").apply(
        lambda x: pd.Series({
            "Total": len(x),
            "Generalist": (x["Prediction"] == "above").sum(),
            "Specialist": (x["Prediction"] == "below").sum(),
            "Unclassified": (x["Prediction"] == "neutral").sum()
        })
    ).round(1)

    # ✅ Prozentwerte berechnen
    habitat_stats["Generalist_%"] = (habitat_stats["Generalist"] / habitat_stats["Total"]) * 100
    habitat_stats["Specialist_%"] = (habitat_stats["Specialist"] / habitat_stats["Total"]) * 100
    habitat_stats["Unclassified_%"] = (habitat_stats["Unclassified"] / habitat_stats["Total"]) * 100

    # ✅ Ausgabe
    for habitat, row in habitat_stats.iterrows():
        print(f"📍 Habitat: {habitat}")
        print(f"  ✅ Gesamt: {row['Total']}")
        print(f"  ✅ Generalisten: {row['Generalist']} ({row['Generalist_%']:.1f}%)")
        print(f"  ✅ Spezialisten: {row['Specialist']} ({row['Specialist_%']:.1f}%)")
        print(f"  ✅ Unklassifiziert: {row['Unclassified']} ({row['Unclassified_%']:.1f}%)")

    print("-"*60)

    return classifications, mean_rel_abundances, occurrence_frequencies

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

    print(f"Generalists: {generalist_mask.sum()} ({generalist_mask.mean()*100:.1f}%)")
    print(f"Specialists: {specialist_mask.sum()} ({specialist_mask.mean()*100:.1f}%)")
    print(f"Unclassified: {n_taxa - generalist_mask.sum() - specialist_mask.sum()} ({(1 - generalist_mask.mean() - specialist_mask.mean())*100:.1f}%)")

    return classifications, mean_rel_abundances, occurrence_frequencies

def _annotate_niche(G, df_lookup, df_relative):
    """
    Attach the lookup attributes plus a niche classification to each node.
    
    Die Knotennamen bleiben unverändert (z. B. 'Fungi:Absidia').
    Die Taxa-Namen in bounds_df werden aus 'Kingdom' und 'Taxa' zusammengesetzt: 'Kingdom:Taxa'
    um sie mit df_relative.columns zu vergleichen.
    """

    classifications, mean_rel_abundances, occurrence_frequencies = (
        identify_generalists_or_specialists_ncm()
    )

    # --- 2. Sicherstellen, dass alle Rückgabewerte als Series vorliegen ---
    if isinstance(classifications, np.ndarray):
        classifications = pd.Series(classifications, index=df_relative.columns)
    if isinstance(mean_rel_abundances, np.ndarray):
        mean_rel_abundances = pd.Series(mean_rel_abundances, index=df_relative.columns)
    if isinstance(occurrence_frequencies, np.ndarray):
        occurrence_frequencies = pd.Series(occurrence_frequencies, index=df_relative.columns)


    taxa_in_df = set(df_relative.columns)
    nodes_attr = dict(G.nodes)

    for node in G.nodes:
        node_str = str(node)  

        if node_str in taxa_in_df:
            spec_or_gen = classifications.get(node_str, None)
            mean_ab = mean_rel_abundances.get(node_str)
            occ_freq = occurrence_frequencies.get(node_str)

            if isinstance(mean_ab, pd.Series):
                mean_ab = mean_ab.iloc[0]
            if isinstance(occ_freq, pd.Series):
                occ_freq = occ_freq.iloc[0]
        else:
            # ❌ Kein Match → setze np.nan
            print(f"❌ Node '{node_str}' ist nicht in df_relative.columns!")
            spec_or_gen = "None"
            mean_ab = np.nan
            occ_freq = np.nan


        attributes = df_lookup.loc[node_str].copy()
        attributes["generalist_or_specialist"] = (
            spec_or_gen if spec_or_gen is not None else "None"
        )
        attributes["mean_relative_abundance"] = mean_ab
        attributes["occurrence_frequency"] = occ_freq

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

    df = pd.DataFrame(data)
    # DEBUG: Prüfe, ob Daten vorhanden sind
    if df.empty:
        print("⚠️ WARNUNG: extract_niche_data gab ein leeres DataFrame zurück!")
        print(f"  Anzahl der Graphen: {len(graphs)}")
        print(f"  Anzahl der Labels: {len(labels)}")
        print(f"  Anzahl der Nodes in allen Graphen: {sum(len(G.nodes) for G in graphs)}")
        print(f"  Anzahl der Nodes mit gültigen Daten: {len(data)}")
        print(f"  Spalten: {list(df.columns) if not df.empty else 'keine'}")
    
    return df

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
        "Generalist": "#2422b8",
        "Specialist": "#d62728",
        "None": "#7f7f7f",
    }
    
    marker_map = {
        "Fungi": "o",
        "Bacteria": "^"
    }

    unique_kingdoms = df["Kingdom"].unique()

    niche_values = df["Niche_Class"].dropna()
    unique_classes = set()
    for val in niche_values:
        if isinstance(val, pd.Series):
            val = val.iloc[0]  # Extrahiere Skalar
        unique_classes.add(str(val) if val is not None else "None")

    for ax, label in zip(axes, labels):
        df_habitat = df[df["Habitat"] == label]
        
        for kingdom in unique_kingdoms:
            for niche_class in unique_classes:
                # ✅ Konvertiere niche_class zu String
                niche_class_str = str(niche_class) if niche_class is not None else "None"
                
                # ✅ Extrahiere Skalar aus jeder Zeile in df_habitat["Niche_Class"]
                subset = df_habitat[
                    (df_habitat["Kingdom"] == kingdom) & 
                    (df_habitat["Niche_Class"].apply(lambda x: str(x.iloc[0]) if isinstance(x, pd.Series) else str(x) if x is not None else "None") == niche_class_str)
                ]
                
                if not subset.empty:
                    ax.scatter(
                        subset["Log_Mean_Abundance"],
                        subset["Occurrence_Frequency"],  # ← Geändert von Niche_Breadth
                        c=color_map.get(niche_class_str, "#7f7f7f"),
                        marker=marker_map[kingdom],
                        label=f"{kingdom} {niche_class_str}" if ax == axes[0] else "",
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

def plot_niche_grid(
    graphs: list[nx.Graph],
    labels: list[str],
    path: str = "Fig6_niche_analysis.png",
    figsize: tuple[float, float] = (16, 10),
    node_size: int = 25, # Etwas größer, damit die Ränder gut sichtbar sind
    edge_width: float = 0.75,
):
    """
    Erstellt ein 2x2 Grid-Vergleichsplot für zwei Netzwerke.
    Zeile 1: Alle Knoten und Kanten.
    Zeile 2: Nur Generalisten, Spezialisten und deren Nachbarn.
    
    Design:
    - Füllfarbe: Taxonomie (Pilz vs Bakterium)
    - Umrandung: Nische (Generalist=Blau/Dick, Specialist=Rot/Dick, Rest=Grau/Dünn)
    """
    if len(graphs) != 2:
        raise ValueError("Exactly 2 graphs are required.")

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    edges_g1 = set(graphs[0].edges())
    edges_g2 = set(graphs[1].edges())
    
    def get_filtered_graph(G_original, edges_subset, focus_mode=False):
        """Filtert den Graphen wie gehabt."""
        important_nodes = set()
        for n in G_original.nodes:
            niche = G_original.nodes[n].get("generalist_or_specialist", "None")
            if isinstance(niche, pd.Series):
                niche = niche.iloc[0]
            niche_str = str(niche) if niche is not None else "None"
            if niche_str in ["Generalist", "Specialist"]:
                important_nodes.add(n)
        
        if focus_mode:
            neighbors = set()
            for node in important_nodes:
                if node in G_original:
                    neighbors.update(G_original.neighbors(node))
            nodes_to_keep = important_nodes.union(neighbors)
        else:
            nodes_in_edges = set()
            for u, v in edges_subset:
                nodes_in_edges.add(u)
                nodes_in_edges.add(v)
            nodes_to_keep = nodes_in_edges

        G_nodes_filtered = G_original.subgraph(nodes_to_keep).copy()
        edges_to_draw = [e for e in G_nodes_filtered.edges() if e in edges_subset]
        G_final = G_nodes_filtered.edge_subgraph(edges_to_draw).copy()
        
        return G_final

    def draw_subplot(ax, G, edges_subset, title, focus_mode=False, use_fixed_pos=None):
        ax.set_title(title)
        
        G_filtered = get_filtered_graph(G, edges_subset, focus_mode=focus_mode)
        
        if G_filtered.number_of_nodes() == 0:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
            return

        current_pos = use_fixed_pos if use_fixed_pos is not None else nx.spring_layout(G_filtered, seed=42, k=0.3)
        
        if use_fixed_pos is not None:
            current_pos = {k: v for k, v in use_fixed_pos.items() if k in G_filtered}

        # Kanten zeichnen
        for edge_type in sorted({edge_kingdom_type(G_filtered, u, v) for u, v in G_filtered.edges()}):
            edges_of_type = [e for e in G_filtered.edges() if edge_kingdom_type(G_filtered, e[0], e[1]) == edge_type]
            nx.draw_networkx_edges(
                G_filtered, current_pos, edgelist=edges_of_type,
                edge_color=_edge_color(edge_type),
                width=edge_width, alpha=0.3, ax=ax
            )

        # Knoten vorbereiten
        # Wir listen alle Knoten auf und sammeln ihre Eigenschaften
        node_list = list(G_filtered.nodes())
        face_colors = []   # Taxonomie (Pilz/Bakterie)
        edge_colors = []   # Nische (Gen/Spec)
        linewidths = []    # Dicke des Randes
        
        for n in node_list:
            # 1. Füllfarbe (Taxonomie)
            face_colors.append(_node_color(G_filtered, n))
            
            # 2. Nische für Rand
            niche = G_filtered.nodes[n].get("generalist_or_specialist", "None")
            if isinstance(niche, pd.Series):
                niche = niche.iloc[0]
            niche_str = str(niche) if niche is not None else "None"
            
            if niche_str == "Generalist":
                edge_colors.append("blue")
                linewidths.append(2.0) # Dick
            elif niche_str == "Specialist":
                edge_colors.append("red")
                linewidths.append(2.0) # Dick
            else:
                edge_colors.append("#454545") # Grau
                linewidths.append(0.5) # Dünn

        # Einmaliges Zeichnen aller Knoten mit Listen von Eigenschaften
        nx.draw_networkx_nodes(
            G_filtered, current_pos, nodelist=node_list,
            node_color=face_colors,
            edgecolors=edge_colors,
            linewidths=linewidths,
            node_size=node_size, 
            ax=ax
        )
        
        ax.set_title(f"{title}\n({G_filtered.number_of_nodes()} Nodes, {G_filtered.number_of_edges()} Edges)")
        ax.set_axis_off()

    # --- Zeile 1: Alle Kanten ---
    draw_subplot(axes[0, 0], graphs[0], edges_g1, f"{labels[0]} - All Nodes & Edges", focus_mode=False)
    draw_subplot(axes[0, 1], graphs[1], edges_g2, f"{labels[1]} - All Nodes & Edges", focus_mode=False)

    # --- Zeile 2: Nische Focus ---
    draw_subplot(axes[1, 0], graphs[0], edges_g1, f"{labels[0]} - Niche Focus (Gen & Spec)", focus_mode=True)
    draw_subplot(axes[1, 1], graphs[1], edges_g2, f"{labels[1]} - Niche Focus (Gen & Spec)", focus_mode=True)

    # Legende anpassen
    legend_handles = [
        # Taxonomie (Füllfarbe)
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#2E8B57", markersize=10, label="Fungi", markeredgecolor="#454545", markeredgewidth=0.5),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#81BADB", markersize=10, label="Bacteria", markeredgecolor="#454545", markeredgewidth=0.5),
        plt.Line2D([], [], color="none", label=""), 
        # Nische (Umrandung) - Dummy-Kreise zur Demonstration der Ränder
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#CCCCCC", markersize=10, label="Generalist", markeredgecolor="blue", markeredgewidth=2),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#CCCCCC", markersize=10, label="Specialist", markeredgecolor="red", markeredgewidth=2),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#CCCCCC", markersize=10, label="Unclassified", markeredgecolor="#454545", markeredgewidth=0.5),
        plt.Line2D([], [], color="none", label=""), 
        # Kanten
        plt.Line2D([0], [0], color="#2E8B57", linewidth=2, label="Fungi-Fungi", alpha=0.3),
        plt.Line2D([0], [0], color="#81BADB", linewidth=2, label="Bacteria-Bacteria", alpha=0.3),
        plt.Line2D([0], [0], color="#882255", linewidth=2, label="Fungi-Bacteria", alpha=0.3),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=5, frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Niche-Grid saved to {path}")
    return path

def plot_niche_consistency(
    graphs: list[nx.Graph],
    labels: list[str],
    path: str = "FigS3_niche_consistency.png",
    figsize: tuple[float, float] = (12, 10),
    node_size: int = 100,
    edge_width: float = 1.0,
):
    """
    Analysiert die gemeinsamen Kanten (Common Edges) und prüft, 
    ob die Nischen-Zuweisung (Generalist/Spezialist) konsistent ist.
    """
    if len(graphs) != 2:
        raise ValueError("Exactly 2 graphs are required.")

    G1, G2 = graphs[0], graphs[1]
    
    # 1. Gemeinsame Kanten finden
    edges_g1 = set(G1.edges())
    edges_g2 = set(G2.edges())
    common_edges = edges_g1.intersection(edges_g2)
    
    if not common_edges:
        print("Keine gemeinsamen Kanten gefunden.")
        return

    # 2. Subgraph der gemeinsamen Struktur erstellen
    # Wir nutzen G1 als Basis für die Topologie
    G_common = G1.edge_subgraph(common_edges).copy()
    
    # Layout berechnen
    pos = nx.spring_layout(G_common, seed=42, k=0.3)

    fig, ax = plt.subplots(figsize=figsize)

    # 3. Knoten analysieren: Konsistenz der Nische prüfen
    node_colors = []
    edge_colors_nodes = [] # Für die Umrandung
    linewidths_nodes = []
    
    # Kategorien für die Legende
    status_counts = {"Consistent Gen": 0, "Consistent Spec": 0, "Switched": 0, "Unclassified": 0}

    for n in G_common.nodes():
        # Nische in G1
        n1 = G1.nodes[n].get("generalist_or_specialist", "None")
        if isinstance(n1, pd.Series): n1 = n1.iloc[0]
        n1_str = str(n1) if n1 is not None else "None"
        
        # Nische in G2
        n2 = G2.nodes[n].get("generalist_or_specialist", "None")
        if isinstance(n2, pd.Series): n2 = n2.iloc[0]
        n2_str = str(n2) if n2 is not None else "None"

        # Logik zur Färbung
        face_color = _node_color(G_common, n) # Taxonomie (Pilz/Bakt)
        
        is_gen_1 = (n1_str == "Generalist")
        is_spec_1 = (n1_str == "Specialist")
        is_gen_2 = (n2_str == "Generalist")
        is_spec_2 = (n2_str == "Specialist")
        
        # Fallunterscheidung
        if is_gen_1 and is_gen_2:
            # Konsistenter Generalist
            edge_color = "blue"
            lw = 2.5
            status_counts["Consistent Gen"] += 1
        elif is_spec_1 and is_spec_2:
            # Konsistenter Spezialist
            edge_color = "red"
            lw = 2.5
            status_counts["Consistent Spec"] += 1
        elif (is_gen_1 and is_spec_2) or (is_spec_1 and is_gen_2):
            # Nischen-Wechsel (Switch)
            edge_color = "orange" # Auffällig
            lw = 3.0
            status_counts["Switched"] += 1
        else:
            # Unclassified oder uneinheitlich (z.B. Gen vs None)
            edge_color = "#454545"
            lw = 0.5
            status_counts["Unclassified"] += 1
            
        node_colors.append(face_color)
        edge_colors_nodes.append(edge_color)
        linewidths_nodes.append(lw)

    # Kanten zeichnen (alle grau oder nach Typ, hier einfach grau für Fokus auf Knoten)
    nx.draw_networkx_edges(
        G_common, pos, edge_color="#DDDDDD", width=edge_width, alpha=0.5, ax=ax
    )

    # Knoten zeichnen
    nx.draw_networkx_nodes(
        G_common, pos, 
        node_color=node_colors,
        edgecolors=edge_colors_nodes,
        linewidths=linewidths_nodes,
        node_size=node_size, 
        ax=ax
    )
    
    ax.set_title(f"Niche Consistency on Common Edges\n"
                 f"({G_common.number_of_nodes()} Nodes, {G_common.number_of_edges()} Edges)")
    ax.axis('off')

    # Legende
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#2E8B57", markersize=8, label="Fungi", markeredgecolor="gray"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#81BADB", markersize=8, label="Bacteria", markeredgecolor="gray"),
        plt.Line2D([], [], color="none", label=""), 
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=10, label=f"Consistent Generalist (n={status_counts['Consistent Gen']})", markeredgecolor="blue", markeredgewidth=2.5),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=10, label=f"Consistent Specialist (n={status_counts['Consistent Spec']})", markeredgecolor="red", markeredgewidth=2.5),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=10, label=f"Niche Switched (n={status_counts['Switched']})", markeredgecolor="orange", markeredgewidth=3.0),
    ]
    
    fig.legend(handles=legend_handles, loc="upper center", ncol=3, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Niche Consistency Plot saved to {path}")
    return path

def plot_niche_stability_and_degree_change(
    graphs: list[nx.Graph],
    labels: list[str],
    path: str = "Fig7_niche_stability_degree_change.png",
    figsize: tuple[float, float] = (12, 10),
    base_node_size: int = 50,
):
    """
    Vergleicht alle Knoten, die in beiden Graphen existieren.
    Zeigt, ob sich der Degree (Vernetzungsgrad) geändert hat, 
    während die Nische stabil blieb.
    """
    if len(graphs) != 2:
        raise ValueError("Exactly 2 graphs are required.")

    G1, G2 = graphs[0], graphs[1]
    
    # 1. Gemeinsame Knoten finden (Schnittmenge)
    nodes_g1 = set(G1.nodes())
    nodes_g2 = set(G2.nodes())
    common_nodes = nodes_g1.intersection(nodes_g2)
    
    if not common_nodes:
        print("Keine gemeinsamen Knoten gefunden.")
        return

    # Wir erstellen einen neuen Graphen nur für die Visualisierung der Topologie
    # Wir nutzen die Topologie von G1 als Basis
    G_vis = G1.subgraph(common_nodes).copy()
    
    # Layout berechnen
    pos = nx.spring_layout(G_vis, seed=42, k=0.3)

    fig, ax = plt.subplots(figsize=figsize)

    # 2. Daten für jeden Knoten sammeln
    node_list = list(G_vis.nodes())
    face_colors = []
    edge_colors = []
    linewidths = []
    node_sizes = []
    
    # Statistiken
    stats = {
        "Gen_Stable": 0, "Spec_Stable": 0, "Unclassified": 0,
        "High_Change": 0, "Low_Change": 0
    }

    for n in node_list:
        # Nische bestimmen (wir nehmen an, sie ist gleich, da 0 Switches, 
        # aber wir prüfen es zur Sicherheit für G1)
        niche = G1.nodes[n].get("generalist_or_specialist", "None")
        if isinstance(niche, pd.Series): niche = niche.iloc[0]
        niche_str = str(niche) if niche is not None else "None"
        
        # Degree berechnen
        deg1 = G1.degree(n)
        deg2 = G2.degree(n)
        degree_diff = abs(deg1 - deg2)
        
        # Füllfarbe (Taxonomie)
        face_color = _node_color(G_vis, n)
        
        # Umrandung (Nische)
        if niche_str == "Generalist":
            edge_color = "blue"
            lw = 2.0
            stats["Gen_Stable"] += 1
        elif niche_str == "Specialist":
            edge_color = "red"
            lw = 2.0
            stats["Spec_Stable"] += 1
        else:
            edge_color = "#454545"
            lw = 0.5
            stats["Unclassified"] += 1
            
        # Knotengröße basierend auf Degree-Änderung
        size = base_node_size + (degree_diff * 30)
        
        if degree_diff > 2:
            stats["High_Change"] += 1
        else:
            stats["Low_Change"] += 1

        face_colors.append(face_color)
        edge_colors.append(edge_color)
        linewidths.append(lw)
        node_sizes.append(size)

    # Kanten zeichnen (aus G1)
    nx.draw_networkx_edges(
        G_vis, pos, edge_color="#DDDDDD", width=0.5, alpha=0.5, ax=ax
    )

    # Knoten zeichnen
    nx.draw_networkx_nodes(
        G_vis, pos, 
        node_color=face_colors,      # Korrigiert
        edgecolors=edge_colors,      # Korrigiert
        linewidths=linewidths,       # Korrigiert
        node_size=node_sizes, 
        ax=ax
    )
    
    ax.set_title(f"Niche Stability & Network Change (Common Nodes)\n"
                 f"Node Size = Degree Difference | {labels[0]} vs {labels[1]}")
    ax.axis('off')

    # Legende
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#2E8B57", markersize=8, label="Fungi", markeredgecolor="gray"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#81BADB", markersize=8, label="Bacteria", markeredgecolor="gray"),
        plt.Line2D([], [], color="none", label=""), 
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=10, label=f"Generalist (n={stats['Gen_Stable']})", markeredgecolor="blue", markeredgewidth=2),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=10, label=f"Specialist (n={stats['Spec_Stable']})", markeredgecolor="red", markeredgewidth=2),
        plt.Line2D([], [], color="none", label=""), 
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=5, label=f"Stable Degree (n={stats['Low_Change']})", markeredgecolor="black", markeredgewidth=1),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=15, label=f"High Degree Change (n={stats['High_Change']})", markeredgecolor="black", markeredgewidth=1),
    ]
    
    fig.legend(handles=legend_handles, loc="upper center", ncol=4, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Niche Stability Plot saved to {path}")
    return path

def plot_degree_shift_scatter(
    graphs: list[nx.Graph],
    labels: list[str],
    path: str = "Fig8_degree_shift_scatter.png",
    figsize: tuple[float, float] = (10, 8),
):
    """
    Scatterplot comparing the Degree (Connectivity) of nodes between two habitats.
    X-Axis: Degree in Habitat 1
    Y-Axis: Degree in Habitat 2
    Color: Niche (Generalist/Specialist)
    Marker: Taxonomy (Fungi/Bacteria)
    """
    if len(graphs) != 2:
        raise ValueError("Exactly 2 graphs are required.")

    G1, G2 = graphs[0], graphs[1]
    
    # Gemeinsame Knoten finden
    common_nodes = set(G1.nodes()).intersection(set(G2.nodes()))
    
    if not common_nodes:
        print("Keine gemeinsamen Knoten gefunden.")
        return

    # Daten sammeln
    data = []
    for n in common_nodes:
        # Degree
        deg1 = G1.degree(n)
        deg2 = G2.degree(n)
        
        # Nische
        niche = G1.nodes[n].get("generalist_or_specialist", "None")
        if isinstance(niche, pd.Series): niche = niche.iloc[0]
        niche_str = str(niche) if niche is not None else "None"
        
        # Taxonomie (für Marker)
        # Wir nutzen _node_color, um zu prüfen, ob es Pilz oder Bakterium ist
        tax_color = _node_color(G1, n)
        marker = '^' if tax_color == "#2E8B57" else 'o' # Dreieck für Pilz, Kreis für Bakterium
        
        # Farbe für Plot
        if niche_str == "Generalist":
            color = "blue"
            label_niche = "Generalist"
        elif niche_str == "Specialist":
            color = "red"
            label_niche = "Specialist"
        else:
            color = "gray"
            label_niche = "Unclassified"
            
        data.append({
            "x": deg1,
            "y": deg2,
            "color": color,
            "marker": marker,
            "niche_label": label_niche,
            "tax_label": "Fungi" if marker == '^' else "Bacteria"
        })

    # Plot erstellen
    fig, ax = plt.subplots(figsize=figsize)
    
    # Diagonale Linie zeichnen (x=y) für Referenz
    max_deg = max([max(d['x'], d['y']) for d in data]) if data else 10
    ax.plot([0, max_deg], [0, max_deg], 'k--', alpha=0.3, label="No Change (x=y)")

    # Punkte plotten
    # Wir gruppieren für die Legende
    groups = {
        ("Generalist", "Fungi"): {"x": [], "y": [], "c": "blue", "m": "^"},
        ("Generalist", "Bacteria"): {"x": [], "y": [], "c": "blue", "m": "o"},
        ("Specialist", "Fungi"): {"x": [], "y": [], "c": "red", "m": "^"},
        ("Specialist", "Bacteria"): {"x": [], "y": [], "c": "red", "m": "o"},
        ("Unclassified", "Fungi"): {"x": [], "y": [], "c": "gray", "m": "^", "alpha": 0.3},
        ("Unclassified", "Bacteria"): {"x": [], "y": [], "c": "gray", "m": "o", "alpha": 0.3},
    }
    
    for d in data:
        key = (d['niche_label'], d['tax_label'])
        if key in groups:
            groups[key]["x"].append(d["x"])
            groups[key]["y"].append(d["y"])

    # Zeichnen
    for key, vals in groups.items():
        if vals["x"]:
            ax.scatter(vals["x"], vals["y"], 
                       c=vals["c"], 
                       marker=vals["m"], 
                       label=f"{key[0]} - {key[1]}", 
                       alpha=vals.get("alpha", 0.7),
                       edgecolors='black', linewidth=0.5, s=50)

    ax.set_xlabel(f"Degree in {labels[0]}")
    ax.set_ylabel(f"Degree in {labels[1]}")
    ax.set_title("Node Degree Shift between Habitats")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Degree Shift Scatterplot saved to {path}")
    return path

def plot_degree_shift_scatter_labeled(
    graphs: list[nx.Graph],
    labels: list[str],
    path: str = "Fig5_degree_shift_labeled.png",
    figsize: tuple[float, float] = (12, 10), # Etwas größer für die Labels
):
    """
    Scatterplot comparing Degree between habitats for Generalists and Specialists only.
    Includes Taxa labels for direct identification.
    """
    if len(graphs) != 2:
        raise ValueError("Exactly 2 graphs are required.")

    G1, G2 = graphs[0], graphs[1]
    
    # Gemeinsame Knoten finden
    common_nodes = set(G1.nodes()).intersection(set(G2.nodes()))
    
    if not common_nodes:
        print("Keine gemeinsamen Knoten gefunden.")
        return

    # Daten sammeln
    data = []
    for n in common_nodes:
        # Nische prüfen
        niche = G1.nodes[n].get("generalist_or_specialist", "None")
        if isinstance(niche, pd.Series): niche = niche.iloc[0]
        niche_str = str(niche) if niche is not None else "None"
        
        # Nur Generalisten und Spezialisten behalten
        if niche_str not in ["Generalist", "Specialist"]:
            continue
            
        # Degree
        deg1 = G1.degree(n)
        deg2 = G2.degree(n)
        
        # Taxonomie (für Marker)
        tax_color = _node_color(G1, n)
        marker = '^' if tax_color == "#2E8B57" else 'o' # Dreieck für Pilz, Kreis für Bakterium
        
        # Farbe für Plot
        color = "blue" if niche_str == "Generalist" else "red"
        
        # Taxon Name (Annahme: Der Knoten-Name 'n' ist der Taxon-Name)
        taxon_name = str(n) 

        data.append({
            "x": deg1,
            "y": deg2,
            "color": color,
            "marker": marker,
            "niche": niche_str,
            "taxon": taxon_name
        })

    if not data:
        print("Keine Generalisten oder Spezialisten in beiden Graphen gefunden.")
        return

    # Plot erstellen
    fig, ax = plt.subplots(figsize=figsize)
    
    # Diagonale Linie zeichnen
    max_deg = max([max(d['x'], d['y']) for d in data])
    ax.plot([0, max_deg + 1], [0, max_deg + 1], 'k--', alpha=0.3, label="No Change (x=y)")

    # Punkte plotten
    for d in data:
        ax.scatter(d["x"], d["y"], 
                   c=d["color"], 
                   marker=d["marker"], 
                   s=100, # Etwas größere Punkte für die Labels
                   edgecolors='black', linewidth=0.8, zorder=2)
        
        # Label hinzufügen
        # Offset berechnen, damit Labels sich nicht überlappen (einfache Heuristik)
        offset_x = 0.2
        offset_y = 0.2
        
        ax.annotate(d["taxon"], 
                    (d["x"], d["y"]), 
                    xytext=(offset_x, offset_y), 
                    textcoords='offset points',
                    fontsize=9, 
                    alpha=0.8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7))

    # Achsen und Titel
    ax.set_xlabel(f"Degree in {labels[0]}")
    ax.set_ylabel(f"Degree in {labels[1]}")
    ax.set_title("Degree Shift of Generalists & Specialists (Common Taxa)")
    
    # Legende manuell erstellen (für saubere Darstellung)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Bacteria', markerfacecolor='gray', markersize=10, markeredgecolor='black'),
        Line2D([0], [0], marker='^', color='w', label='Fungi', markerfacecolor='gray', markersize=10, markeredgecolor='black'),
        Line2D([0], [0], marker='o', color='w', label='Generalist', markerfacecolor='blue', markersize=10, markeredgecolor='black'),
        Line2D([0], [0], marker='o', color='w', label='Specialist', markerfacecolor='red', markersize=10, markeredgecolor='black'),
    ]
    ax.legend(handles=legend_elements, loc="upper left")
    
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Labeled Degree Shift Scatterplot saved to {path}")
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

def plot_degree_change_old(df_change, path="degree_change_analysis.png"):
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

def plot_consistent_degree_change_comparison_old(
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