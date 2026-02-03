import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# Configuration
CORRELATION_THRESHOLD = 0.6
MIN_EDGES = 5
MAX_EDGES = 10

def download_price_data(symbols, period='1y'):
    """Download historical price data for given symbols."""
    print(f"Downloading price data for {len(symbols)} stocks...")
    data = yf.download(symbols, period=period, progress=False)['Adj Close']
    return data

def calculate_correlation_matrix(price_data):
    """Calculate correlation matrix from price returns."""
    returns = price_data.pct_change().dropna()
    correlation_matrix = returns.corr()
    return correlation_matrix

def load_and_build_graph(nodes_path, edges_path):
    # Load nodes data
    nodes_df = pd.read_csv(nodes_path)
    symbols = nodes_df['Symbol'].tolist()

    # Download price data and calculate correlations
    price_data = download_price_data(symbols)

    # Filter symbols that have valid data
    valid_symbols = [s for s in symbols if s in price_data.columns]
    print(f"Valid symbols with price data: {len(valid_symbols)}")

    correlation_matrix = calculate_correlation_matrix(price_data[valid_symbols])

    G = nx.Graph()

    # Add nodes with attributes - using corrected column names from your file
    for _, row in nodes_df.iterrows():
        if row['Symbol'] in valid_symbols:
            G.add_node(row['Symbol'],
                       name=row['Name'],
                       sector=row['Sector'],
                       market_cap=row['Market Cap'])

    # Build dictionary of all potential edges with their correlations
    # (correlation >= threshold)
    potential_edges = {}  # {(stock1, stock2): correlation}
    for i, sym1 in enumerate(valid_symbols):
        for sym2 in valid_symbols[i+1:]:
            corr = correlation_matrix.loc[sym1, sym2]
            if not np.isnan(corr) and corr >= CORRELATION_THRESHOLD:
                # Store with sorted tuple key to avoid duplicates
                edge_key = tuple(sorted([sym1, sym2]))
                potential_edges[edge_key] = corr

    print(f"Edges with correlation >= {CORRELATION_THRESHOLD}: {len(potential_edges)}")

    # Count edges per node from threshold-based edges
    node_edges = {sym: [] for sym in valid_symbols}  # {node: [(neighbor, corr), ...]}
    for (sym1, sym2), corr in potential_edges.items():
        node_edges[sym1].append((sym2, corr))
        node_edges[sym2].append((sym1, corr))

    # Apply min/max edge constraints
    final_edges = set()

    for node in valid_symbols:
        edges = node_edges[node]
        # Sort by correlation (descending)
        edges_sorted = sorted(edges, key=lambda x: x[1], reverse=True)

        current_count = len(edges_sorted)

        if current_count > MAX_EDGES:
            # Keep only the top MAX_EDGES strongest correlations
            edges_sorted = edges_sorted[:MAX_EDGES]
        elif current_count < MIN_EDGES:
            # Need to add more edges - find additional candidates from full correlation matrix
            existing_neighbors = set(e[0] for e in edges_sorted)
            candidates = []
            for other in valid_symbols:
                if other != node and other not in existing_neighbors:
                    corr = correlation_matrix.loc[node, other]
                    if not np.isnan(corr) and corr > 0:  # Only positive correlations
                        candidates.append((other, corr))
            # Sort candidates by correlation (descending) and add until we reach MIN_EDGES
            candidates_sorted = sorted(candidates, key=lambda x: x[1], reverse=True)
            needed = MIN_EDGES - current_count
            edges_sorted.extend(candidates_sorted[:needed])

        # Add these edges to final set
        for neighbor, corr in edges_sorted:
            edge_key = tuple(sorted([node, neighbor]))
            final_edges.add((edge_key, corr))

    # Create weighted edges in graph
    for (sym1, sym2), corr in final_edges:
        # Use correlation as weight
        G.add_edge(sym1, sym2, weight=corr)

    print(f"Final graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Print degree statistics
    degrees = [d for n, d in G.degree()]
    print(f"Degree stats: min={min(degrees)}, max={max(degrees)}, avg={np.mean(degrees):.2f}")

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
    
    

def export_edges_with_weights(G, output_path):
    """Export edges with their correlation weights to CSV."""
    edges_data = []
    for u, v, data in G.edges(data=True):
        edges_data.append({
            'Source': u,
            'Target': v,
            'Weight': data.get('weight', 0)
        })
    edges_df = pd.DataFrame(edges_data)
    edges_df.to_csv(output_path, index=False)
    print(f"Edges exported to '{output_path}'")

def export_centrality(G, output_path):
    """Export node centrality measures to CSV."""
    # Calculate various centrality measures
    degree_cent = nx.degree_centrality(G)
    betweenness_cent = nx.betweenness_centrality(G, weight='weight')
    closeness_cent = nx.closeness_centrality(G)
    eigenvector_cent = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)

    centrality_data = []
    for node in G.nodes():
        centrality_data.append({
            'Symbol': node,
            'Degree_Centrality': degree_cent[node],
            'Betweenness_Centrality': betweenness_cent[node],
            'Closeness_Centrality': closeness_cent[node],
            'Eigenvector_Centrality': eigenvector_cent[node]
        })

    centrality_df = pd.DataFrame(centrality_data)
    centrality_df = centrality_df.sort_values('Eigenvector_Centrality', ascending=False)
    centrality_df.to_csv(output_path, index=False)
    print(f"Centrality measures exported to '{output_path}'")

# Execution
G = load_and_build_graph('sp500_nodes.csv', 'sp500_edges.csv')
calculate_basic_metrics(G)
calculate_deep_metrics(G)

# Export edges with weights
export_edges_with_weights(G, 'sp500_edges.csv')

# Export centrality measures
export_centrality(G, 'edges_centrality_0.5-0.7/centrality_of_stocks_0.6cor.csv')

# Export to Gephi (includes weights)
nx.write_gexf(G, "sp500_project.gexf")
print("\nFile 'sp500_project.gexf' is ready for Gephi.")