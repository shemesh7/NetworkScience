"""
compute_correlations.py

Downloads S&P 500 stock prices and computes a correlation-based edge list.
Keeps only the top N% strongest correlations to create a sparse backbone network.

Output: sp500_edges.csv with columns (Source, Target, Weight)
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# =============================================================================
# Configuration
# =============================================================================

NODES_FILE = "sp500_nodes.csv"
OUTPUT_FILE = "sp500_edges.csv"
TOP_PERCENT = 6  # Keep only the top 6% strongest correlations
LOOKBACK_DAYS = 365  # 1 year of data

# =============================================================================
# Step 1: Read company symbols from sp500_nodes.csv
# =============================================================================

print("Step 1: Reading symbols from", NODES_FILE)
nodes_df = pd.read_csv(NODES_FILE)
original_symbols = nodes_df['Symbol'].tolist()
print(f"  Found {len(original_symbols)} symbols")

# Convert dots to hyphens for yfinance compatibility (e.g., BRK.B -> BRK-B)
# Keep a mapping to restore original names in the output
yf_symbols = [s.replace('.', '-') for s in original_symbols]
yf_to_original = {yf: orig for yf, orig in zip(yf_symbols, original_symbols)}

# =============================================================================
# Step 2: Download daily closing prices using yfinance
# =============================================================================

print("\nStep 2: Downloading stock prices (this may take a few minutes)...")

# Calculate date range: last 1 year
end_date = datetime.today()
start_date = end_date - timedelta(days=LOOKBACK_DAYS)

print(f"  Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

# Download all symbols at once (yfinance handles batching internally)
# Using 'Adj Close' to account for stock splits and dividends
data = yf.download(
    tickers=yf_symbols,
    start=start_date,
    end=end_date,
    progress=True,
    auto_adjust=True  # Returns adjusted prices directly in 'Close'
)

# Extract closing prices
# yfinance returns MultiIndex columns when downloading multiple tickers
prices_df = data['Close']

print(f"  Downloaded {len(prices_df)} trading days")
print(f"  Symbols with data: {len(prices_df.columns)}")

# =============================================================================
# Step 3: Check data quality
# =============================================================================

print("\nStep 3: Checking data quality...")

# Report symbols that failed to download
downloaded_symbols = set(prices_df.columns)
requested_symbols = set(yf_symbols)
missing_symbols = requested_symbols - downloaded_symbols

if missing_symbols:
    print(f"  Warning: {len(missing_symbols)} symbols have no data:")
    print(f"    {sorted(missing_symbols)[:10]}...")  # Show first 10

# Drop columns (symbols) with too much missing data (>20%)
max_missing_ratio = 0.2
missing_ratio = prices_df.isnull().sum() / len(prices_df)
valid_symbols = missing_ratio[missing_ratio <= max_missing_ratio].index.tolist()
prices_df = prices_df[valid_symbols]

print(f"  Symbols after quality filter: {len(prices_df.columns)}")

# Forward-fill then back-fill small gaps (handles holidays, short halts)
prices_df = prices_df.ffill().bfill()

# =============================================================================
# Step 4: Compute Pearson correlation matrix
# =============================================================================

print("\nStep 4: Computing Pearson correlation matrix...")

correlation_matrix = prices_df.corr(method='pearson')

print(f"  Correlation matrix shape: {correlation_matrix.shape}")

# =============================================================================
# Step 5: Extract all pairwise correlations and keep top N%
# =============================================================================

print(f"\nStep 5: Extracting top {TOP_PERCENT}% strongest correlations...")

# Get list of symbols in the correlation matrix
corr_symbols = correlation_matrix.columns.tolist()

# Collect all pairwise correlations (upper triangle only)
all_pairs = []
for i in range(len(corr_symbols)):
    for j in range(i + 1, len(corr_symbols)):  # j > i ensures no self-loops and no duplicates
        symbol_a = corr_symbols[i]
        symbol_b = corr_symbols[j]
        correlation = correlation_matrix.iloc[i, j]

        # Skip NaN correlations
        if pd.isna(correlation):
            continue

        all_pairs.append({
            'symbol_a': symbol_a,
            'symbol_b': symbol_b,
            'correlation': correlation,
            'abs_correlation': abs(correlation)
        })

total_pairs = len(all_pairs)
print(f"  Total valid pairs: {total_pairs}")

# Sort by absolute correlation (strongest first)
all_pairs.sort(key=lambda x: x['abs_correlation'], reverse=True)

# Keep only top N%
num_edges_to_keep = int(total_pairs * TOP_PERCENT / 100)
top_pairs = all_pairs[:num_edges_to_keep]

print(f"  Keeping top {TOP_PERCENT}%: {num_edges_to_keep} edges")

if top_pairs:
    min_corr = top_pairs[-1]['abs_correlation']
    max_corr = top_pairs[0]['abs_correlation']
    print(f"  Correlation range in top {TOP_PERCENT}%: {min_corr:.4f} to {max_corr:.4f}")

# =============================================================================
# Step 6: Create edge list from top pairs
# =============================================================================

print("\nStep 6: Creating edge list...")

edges = []
for pair in top_pairs:
    # Convert back to original symbol names (e.g., BRK-B -> BRK.B)
    orig_a = yf_to_original.get(pair['symbol_a'], pair['symbol_a'])
    orig_b = yf_to_original.get(pair['symbol_b'], pair['symbol_b'])
    # Alphabetical order for consistency
    source, target = sorted([orig_a, orig_b])
    edges.append({
        'Source': source,
        'Target': target,
        'Weight': round(pair['correlation'], 6)
    })

# =============================================================================
# Step 7: Save edge list to CSV
# =============================================================================

print(f"\nStep 7: Saving edge list to {OUTPUT_FILE}...")

edges_df = pd.DataFrame(edges)
if not edges_df.empty:
    edges_df = edges_df.sort_values(['Source', 'Target'])  # Sort for readability
edges_df.to_csv(OUTPUT_FILE, index=False)

if edges_df.empty:
    print("  Warning: No edges found. Output file will be empty.")
else:
    print(f"  Saved {len(edges_df)} edges to {OUTPUT_FILE}")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print(f"  Input symbols:     {len(original_symbols)}")
print(f"  Symbols with data: {len(prices_df.columns)}")
print(f"  Trading days:      {len(prices_df)}")
print(f"  Total pairs:       {total_pairs}")
print(f"  Top percent kept:  {TOP_PERCENT}%")
print(f"  Edges created:     {len(edges_df)}")
if top_pairs:
    print(f"  Min correlation:   {min_corr:.4f}")
    print(f"  Max correlation:   {max_corr:.4f}")
print(f"  Output file:       {OUTPUT_FILE}")
print("="*50)
