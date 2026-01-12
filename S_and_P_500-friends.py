import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random

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
    
    # Load or generate edges
    # Note: If your edges file contains Source/Target pairs, use those columns here.
    # Currently creating links between companies in the same sector for demonstration.
    sectors = nodes_df['Sector'].unique()
    for sector in sectors:
        sector_nodes = nodes_df[nodes_df['Sector'] == sector]['Symbol'].tolist()
        for i in range(len(sector_nodes)):
            for j in range(i + 1, len(sector_nodes)):
                # Link creation with probability to avoid an overly dense graph
                if random.random() < 0.1: 
                    G.add_edge(sector_nodes[i], sector_nodes[j])
            
    return G

# --- Analysis functions based on project plan ---

def calculate_basic_metrics(G):
    print("--- 2. Basic Properties ---")
    # 1. Average degree: How many partners does an average company have? [cite: 159]
    avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
    print(f"1. Average Degree: {avg_degree:.2f}")
    
    # 2. Diameter: Distance between the two furthest companies [cite: 160]
    # Calculated on the Giant Connected Component (GCC)
    gcc_nodes = max(nx.connected_components(G), key=len)
    gcc = G.subgraph(gcc_nodes)
    diameter = nx.diameter(gcc)
    print(f"2. Diameter (GCC): {diameter}")
    
    # 3. Clustering Coefficient: Measures "communities" [cite: 161]
    avg_clustering = nx.average_clustering(G)
    print(f"3. Average Clustering Coefficient: {avg_clustering:.4f}")
    
    # 4. Small World Phenomenon [cite: 162]
    avg_path = nx.average_shortest_path_length(gcc)
    print(f"4. Average Path Length (Small World): {avg_path:.2f}")

def calculate_deep_metrics(G):
    print("\n--- 3. Deep Properties ---")
    
    # 1+2. Degree Distribution P(k) and Power Law [cite: 164, 165]
    degrees = [d for n, d in G.degree()]
    plt.figure(figsize=(8, 5))
    plt.hist(degrees, bins=30, alpha=0.7, color='blue', density=True)
    plt.title("Degree Distribution P(k)")
    plt.xlabel("k (Degree)")
    plt.ylabel("P(k)")
    plt.savefig('degree_distribution.png')
    print("1-2. Degree distribution plot saved as 'degree_distribution.png'")
    
    

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
calculate_basic_metrics(G)
calculate_deep_metrics(G)

# Export to Gephi
nx.write_gexf(G, "sp500_project.gexf")
print("\nFile 'sp500_project.gexf' is ready for Gephi.")