import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def load_and_build_graph(nodes_path, edges_path):
    # Load nodes data
    nodes_df = pd.read_csv(nodes_path)
    
    G = nx.Graph()
    
    # Add nodes with attributes - using corrected column names from your file
    for _, row in nodes_df.iterrows():
        G.add_node(row['Symbol'], 
                   name=row['Name'],
                   sector=row['Sector'], 
                   market_cap=row['Market Cap']) # Matches your CSV column name
    
    # Load edges from correlation-based edge list
    # Each edge has Source, Target, Weight (Pearson correlation >= 0.6)
    edges_df = pd.read_csv(edges_path)

    for _, row in edges_df.iterrows():
        # Only add edge if both nodes exist in the graph
        if row['Source'] in G.nodes and row['Target'] in G.nodes:
            G.add_edge(row['Source'], row['Target'], weight=row['Weight'])
            
    return G

# --- Analysis functions based on project plan ---

def calculate_basic_metrics(G):
    print("--- 2. Basic Properties ---")

    # 1a. Average degree (UNWEIGHTED): How many partners does an average company have?
    avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
    print(f"1a. Average Degree [unweighted]: {avg_degree:.2f}")

    # 1b. Average weighted degree (STRENGTH): Sum of correlation weights per node
    # Strength = sum of edge weights for each node
    strengths = [s for n, s in G.degree(weight='weight')]
    avg_strength = sum(strengths) / G.number_of_nodes()
    print(f"1b. Average Strength [weighted degree]: {avg_strength:.4f}")

    # 2. Diameter (UNWEIGHTED): Distance in hops between furthest companies
    # Note: Using hop count, not weighted distance
    gcc_nodes = max(nx.connected_components(G), key=len)
    gcc = G.subgraph(gcc_nodes)
    diameter = nx.diameter(gcc)
    print(f"2. Diameter (GCC) [unweighted hops]: {diameter}")

    # 3a. Clustering Coefficient (UNWEIGHTED): Measures "communities"
    avg_clustering = nx.average_clustering(G)
    print(f"3a. Average Clustering Coefficient [unweighted]: {avg_clustering:.4f}")

    # 3b. Clustering Coefficient (WEIGHTED): Considers correlation strength in triangles
    avg_clustering_weighted = nx.average_clustering(G, weight='weight')
    print(f"3b. Average Clustering Coefficient [weighted]: {avg_clustering_weighted:.4f}")

    # 4. Small World Phenomenon (UNWEIGHTED): Average hops between nodes
    # Note: Using hop count, not weighted distance
    avg_path = nx.average_shortest_path_length(gcc)
    print(f"4. Average Path Length [unweighted hops]: {avg_path:.2f}")

def calculate_deep_metrics(G):
    print("\n--- 3. Deep Properties ---")

    # 1. Degree and Strength Distributions
    # Plot both unweighted degree and weighted degree (strength) distributions

    degrees = [d for n, d in G.degree()]
    strengths = [s for n, s in G.degree(weight='weight')]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Unweighted degree distribution
    axes[0].hist(degrees, bins=30, alpha=0.7, color='blue', density=True)
    axes[0].set_title("Degree Distribution P(k) [unweighted]")
    axes[0].set_xlabel("k (Degree)")
    axes[0].set_ylabel("P(k)")

    # Right: Weighted degree (strength) distribution
    axes[1].hist(strengths, bins=30, alpha=0.7, color='green', density=True)
    axes[1].set_title("Strength Distribution P(s) [weighted]")
    axes[1].set_xlabel("s (Strength = sum of correlation weights)")
    axes[1].set_ylabel("P(s)")

    plt.tight_layout()
    plt.savefig('degree_strength_distribution.png')
    print("1. Degree and strength distributions saved as 'degree_strength_distribution.png'")
    
    

    # 3. System Stability - Removing top Hub [cite: 166, 167]
    hubs = sorted(G.degree, key=lambda x: x[1], reverse=True)
    top_hub = hubs[0][0]
    
    G_no_hub = G.copy()
    G_no_hub.remove_node(top_hub)
    new_gcc_size = len(max(nx.connected_components(G_no_hub), key=len))
    print(f"3. Stability: Removing top hub ({top_hub}) reduced GCC size to {new_gcc_size}")
    
    # 4. Configuration Model - Comparing actual vs. random clustering [cite: 168]
    config_G = nx.configuration_model([d for n, d in G.degree()])
    config_G = nx.Graph(config_G) # Clean duplicate edges
    print(f"4. Configuration Model: Random model clustering: {nx.average_clustering(config_G):.4f}")
    
    # 5. Structural Correlations (Assortativity) [cite: 169, 170]
    degree_r = nx.degree_assortativity_coefficient(G)
    sector_r = nx.attribute_assortativity_coefficient(G, 'sector')
    print(f"5. Degree Assortativity (Hubs to Hubs): {degree_r:.4f}")
    print(f"5. Sector Assortativity (Intra-sector links): {sector_r:.4f}")
    
    

# Execution
G = load_and_build_graph('sp500_nodes.csv', 'sp500_edges.csv')

# --- 1. Graph Summary ---
print("--- 1. Graph Summary ---")
print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
connected_components = list(nx.connected_components(G))
print(f"Connected Components: {len(connected_components)}")
gcc_size = len(max(connected_components, key=len))
print(f"Giant Connected Component (GCC) size: {gcc_size}")

# Robustness warnings
if G.number_of_edges() == 0:
    print("WARNING: Graph has no edges. Metrics may be undefined or misleading.")
if gcc_size < 10:
    print("WARNING: GCC is very small. Diameter and path length metrics may not be meaningful.")
print()

calculate_basic_metrics(G)
calculate_deep_metrics(G)

# Export to Gephi
print("\n--- 4. Export to Gephi ---")

# Validate sector attributes before export
nodes_with_sector = sum(1 for n in G.nodes() if 'sector' in G.nodes[n])
sectors = set(G.nodes[n].get('sector') for n in G.nodes() if 'sector' in G.nodes[n])
print(f"Nodes with 'sector' attribute: {nodes_with_sector}/{G.number_of_nodes()}")
print(f"Unique sectors: {len(sectors)}")
print(f"Sectors: {sorted(sectors)}")

nx.write_gexf(G, "sp500_project.gexf")
print("\nFile 'sp500_project.gexf' is ready for Gephi.")
print("Tip: In Gephi, use Appearance > Nodes > Partition > sector to color by sector.")