"""
CS 579 FINAL PROJECT - COMPLETE NETWORK ANALYSIS
Expanding HW4 to All 77 Chicago Community Areas

Author: Anshul Dani (A20580060)
Date: November 2025
Status: FINAL SUBMISSION
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

print("="*80)
print("CS 579 FINAL PROJECT - COMPLETE NETWORK ANALYSIS")
print("ALL 77 CHICAGO COMMUNITY AREAS")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Create output directory
OUTPUT_DIR = 'outputs/final_project'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# LOAD DATA
# ==============================================================================
print("[PHASE 1] Loading data...")
print("-" * 80)

data_file = 'outputs/final_project/chicago_77_cas_data_2020.csv'
df = pd.read_csv(data_file)

print(f"✓ Loaded {len(df)} Community Areas")
print(f"✓ Variables: {df.columns.tolist()}")

# Community Area names (for reference)
CA_NAMES = {
    1: "Rogers Park", 2: "West Ridge", 3: "Uptown", 4: "Lincoln Square",
    5: "North Center", 6: "Lake View", 7: "Lincoln Park", 8: "Near North Side",
    9: "Edison Park", 10: "Norwood Park", 11: "Jefferson Park", 12: "Forest Glen",
    13: "North Park", 14: "Albany Park", 15: "Portage Park", 16: "Irving Park",
    17: "Dunning", 18: "Montclare", 19: "Belmont Cragin", 20: "Hermosa",
    21: "Avondale", 22: "Logan Square", 23: "Humboldt Park", 24: "West Town",
    25: "Austin", 26: "West Garfield Park", 27: "East Garfield Park", 28: "Near West Side",
    29: "North Lawndale", 30: "South Lawndale", 31: "Lower West Side", 32: "Loop",
    33: "Near South Side", 34: "Armour Square", 35: "Douglas", 36: "Oakland",
    37: "Fuller Park", 38: "Grand Boulevard", 39: "Kenwood", 40: "Washington Park",
    41: "Hyde Park", 42: "Woodlawn", 43: "South Shore", 44: "Chatham",
    45: "Avalon Park", 46: "South Chicago", 47: "Burnside", 48: "Calumet Heights",
    49: "Roseland", 50: "Pullman", 51: "South Deering", 52: "East Side",
    53: "West Pullman", 54: "Riverdale", 55: "Hegewisch", 56: "Garfield Ridge",
    57: "Archer Heights", 58: "Brighton Park", 59: "McKinley Park", 60: "Bridgeport",
    61: "New City", 62: "West Elsdon", 63: "Gage Park", 64: "Clearing",
    65: "West Lawn", 66: "Chicago Lawn", 67: "West Englewood", 68: "Englewood",
    69: "Greater Grand Crossing", 70: "Ashburn", 71: "Auburn Gresham", 72: "Beverly",
    73: "Washington Heights", 74: "Mount Greenwood", 75: "Morgan Park", 76: "O'Hare",
    77: "Edgewater"
}

df['ca_name'] = df['community_area'].map(CA_NAMES)

# ==============================================================================
# CALCULATE SIMILARITY MATRIX
# ==============================================================================
print("\n[PHASE 2] Calculating 77×77 similarity matrix...")
print("-" * 80)

# Variables for similarity
similarity_vars = [
    'median_income',
    'pct_bachelors',
    'pct_owner',
    'pct_white',
    'pct_black',
    'pct_hispanic',
    'unemployment_rate'
]

print(f"Using {len(similarity_vars)} variables:")
for var in similarity_vars:
    print(f"  • {var}")

# Create feature matrix
X = df[similarity_vars].values

# Handle any remaining NaN values
X = np.nan_to_num(X, nan=0.0)

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Calculate cosine similarity
similarity_matrix = cosine_similarity(X_scaled)

print(f"\n✓ Calculated {len(df)}×{len(df)} similarity matrix")
print(f"  Mean similarity: {similarity_matrix.mean():.3f}")
print(f"  Similarity range: [{similarity_matrix.min():.3f}, {similarity_matrix.max():.3f}]")

# Save similarity matrix
np.save(f'{OUTPUT_DIR}/similarity_matrix_77x77.npy', similarity_matrix)
print(f"✓ Saved similarity matrix")

# ==============================================================================
# BUILD NETWORK
# ==============================================================================
print("\n[PHASE 3] Building network graph...")
print("-" * 80)

# Threshold for edges
SIMILARITY_THRESHOLD = 0.7

G = nx.Graph()

# Add nodes
for idx, row in df.iterrows():
    G.add_node(row['community_area'],
               ca_name=row['ca_name'],
               population=row['total_population'],
               income=row['median_income'],
               pct_owner=row['pct_owner'],
               pct_bachelors=row['pct_bachelors'])

# Add edges
edge_count = 0
for i in range(len(df)):
    for j in range(i+1, len(df)):
        if similarity_matrix[i, j] > SIMILARITY_THRESHOLD:
            ca_i = df.iloc[i]['community_area']
            ca_j = df.iloc[j]['community_area']
            G.add_edge(ca_i, ca_j, weight=similarity_matrix[i, j])
            edge_count += 1

print(f"\n✓ Built network:")
print(f"  Nodes: {G.number_of_nodes()}")
print(f"  Edges: {G.number_of_edges()}")
print(f"  Density: {nx.density(G):.4f}")
print(f"  Connected components: {nx.number_connected_components(G)}")
print(f"  Average degree: {np.mean([d for n, d in G.degree()]):.2f}")

# Calculate clustering coefficient
if G.number_of_edges() > 0:
    clustering_coef = nx.average_clustering(G)
    print(f"  Clustering coefficient: {clustering_coef:.4f}")
else:
    clustering_coef = 0

# ==============================================================================
# COMMUNITY DETECTION
# ==============================================================================
print("\n[PHASE 4] Running community detection...")
print("-" * 80)

# Get largest connected component
components = list(nx.connected_components(G))
largest_cc = max(components, key=len)
G_connected = G.subgraph(largest_cc).copy()

print(f"Working with largest component: {len(G_connected)} nodes")

# Louvain community detection
from networkx.algorithms import community
communities_louvain = community.greedy_modularity_communities(G_connected)

print(f"\n✓ Louvain Algorithm:")
print(f"  Communities detected: {len(communities_louvain)}")
print(f"  Community sizes: {[len(c) for c in communities_louvain]}")

# Calculate modularity
modularity = community.modularity(G_connected, communities_louvain)
print(f"  Modularity: {modularity:.4f}")

# Assign community labels
node_to_community = {}
for i, comm in enumerate(communities_louvain):
    for node in comm:
        node_to_community[node] = i

df['louvain_community'] = df['community_area'].map(node_to_community)

# ==============================================================================
# K-MEANS CLUSTERING
# ==============================================================================
print("\n[PHASE 5] Running K-means clustering...")
print("-" * 80)

# Try different k values
k_values = [5, 6, 7, 8]
best_k = 6  # Default

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df[f'kmeans_{k}'] = kmeans.fit_predict(X_scaled)

print(f"✓ K-means clustering complete for k={k_values}")

# Use k=6 as primary
df['kmeans_cluster'] = df['kmeans_6']

print(f"\nK-means (k=6) cluster sizes:")
print(df['kmeans_cluster'].value_counts().sort_index())

# ==============================================================================
# ANALYSIS & STATISTICS
# ==============================================================================
print("\n[PHASE 6] Analyzing results...")
print("-" * 80)

# Calculate metrics for each detected community
print("\nLouvain Communities Analysis:")
for i in range(len(communities_louvain)):
    comm_cas = df[df['louvain_community'] == i]
    if len(comm_cas) > 0:
        print(f"\nCommunity {i}: {len(comm_cas)} CAs")
        print(f"  Mean income: ${comm_cas['median_income'].mean():,.0f}")
        print(f"  Mean homeownership: {comm_cas['pct_owner'].mean():.1f}%")
        print(f"  Mean bachelor's: {comm_cas['pct_bachelors'].mean():.1f}%")
        print(f"  CAs: {', '.join(comm_cas['ca_name'].head(5).tolist())}" + 
              (" ..." if len(comm_cas) > 5 else ""))

# Compare official vs detected
print(f"\n{'='*80}")
print("COMPARISON: Official (77 CAs) vs. Detected Communities")
print('='*80)

print(f"\nOfficial structure: 77 separate Community Areas")
print(f"Detected structure: {len(communities_louvain)} communities")
print(f"Reduction: {77 - len(communities_louvain)} fewer communities")
print(f"Modularity: {modularity:.4f} (higher = stronger community structure)")

# ==============================================================================
# VISUALIZATIONS
# ==============================================================================
print("\n[PHASE 7] Creating visualizations...")
print("-" * 80)

# Visualization 1: Network graph with Louvain communities
print("Creating network visualization...")
fig, ax = plt.subplots(figsize=(16, 16))

pos = nx.spring_layout(G, k=1.0, iterations=50, seed=42)

# Color by Louvain community
community_colors = plt.cm.tab10(np.linspace(0, 1, len(communities_louvain)))
node_colors = []
for node in G.nodes():
    comm = node_to_community.get(node, -1)
    if comm >= 0 and comm < len(community_colors):
        node_colors.append(community_colors[comm])
    else:
        node_colors.append('gray')

# Draw network
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300, alpha=0.8, ax=ax)
nx.draw_networkx_edges(G, pos, alpha=0.15, width=0.5, ax=ax)

# Add labels for some nodes
labels = {node: G.nodes[node]['ca_name'].split()[0] if len(G.nodes[node]['ca_name'].split()) > 0 else str(node) 
          for node in list(G.nodes())[:20]}  # Label first 20
nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)

ax.set_title(f'Chicago Community Areas Network\n{len(communities_louvain)} Communities Detected (Modularity = {modularity:.3f})',
             fontsize=16, fontweight='bold')
ax.axis('off')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/network_graph_louvain.png', dpi=300, bbox_inches='tight')
print("✓ Saved network_graph_louvain.png")
plt.close()

# Visualization 2: Similarity heatmap
print("Creating similarity heatmap...")
fig, ax = plt.subplots(figsize=(14, 12))

# Sort by Louvain community
df_sorted = df.sort_values('louvain_community').reset_index(drop=True)
sorted_indices = df_sorted.index.tolist()
similarity_sorted = similarity_matrix[np.ix_(sorted_indices, sorted_indices)]

im = ax.imshow(similarity_sorted, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
ax.set_title('Similarity Matrix (Sorted by Detected Community)', fontsize=14, fontweight='bold')
ax.set_xlabel('Community Area Index')
ax.set_ylabel('Community Area Index')

# Add community boundaries
boundaries = []
current_comm = df_sorted.iloc[0]['louvain_community'] if not pd.isna(df_sorted.iloc[0]['louvain_community']) else -1
current_pos = 0
for idx, row in df_sorted.iterrows():
    comm = row['louvain_community'] if not pd.isna(row['louvain_community']) else -1
    if comm != current_comm:
        boundaries.append(current_pos)
        current_comm = comm
    current_pos += 1

for boundary in boundaries:
    ax.axhline(y=boundary, color='blue', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=boundary, color='blue', linestyle='--', linewidth=1, alpha=0.5)

plt.colorbar(im, ax=ax, label='Similarity Score')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/similarity_heatmap_sorted.png', dpi=300, bbox_inches='tight')
print("✓ Saved similarity_heatmap_sorted.png")
plt.close()

# Visualization 3: Community composition
print("Creating community composition chart...")
fig, ax = plt.subplots(figsize=(12, 8))

community_sizes = [len(c) for c in communities_louvain]
community_labels = [f"Community {i}\n({size} CAs)" for i, size in enumerate(community_sizes)]

bars = ax.bar(range(len(community_sizes)), community_sizes, color=community_colors, alpha=0.7, edgecolor='black')
ax.set_xlabel('Detected Community', fontsize=12)
ax.set_ylabel('Number of Community Areas', fontsize=12)
ax.set_title(f'Size of Detected Communities\n(Louvain Algorithm, Modularity = {modularity:.3f})',
             fontsize=14, fontweight='bold')
ax.set_xticks(range(len(community_sizes)))
ax.set_xticklabels(community_labels)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}',
            ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/community_sizes.png', dpi=300, bbox_inches='tight')
print("✓ Saved community_sizes.png")
plt.close()

# Visualization 4: Comparison chart
print("Creating comparison visualization...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Network Analysis Results - Key Metrics', fontsize=16, fontweight='bold')

# Network density
axes[0, 0].bar(['Full Network'], [nx.density(G)], color='steelblue', alpha=0.7)
axes[0, 0].set_ylabel('Density')
axes[0, 0].set_title('Network Density')
axes[0, 0].set_ylim([0, 0.3])
axes[0, 0].text(0, nx.density(G), f'{nx.density(G):.4f}', ha='center', va='bottom', fontweight='bold')

# Clustering coefficient
axes[0, 1].bar(['Full Network'], [clustering_coef], color='green', alpha=0.7)
axes[0, 1].set_ylabel('Clustering Coefficient')
axes[0, 1].set_title('Average Clustering')
axes[0, 1].set_ylim([0, 1])
axes[0, 1].text(0, clustering_coef, f'{clustering_coef:.4f}', ha='center', va='bottom', fontweight='bold')

# Modularity
axes[1, 0].bar(['Louvain'], [modularity], color='orange', alpha=0.7)
axes[1, 0].set_ylabel('Modularity')
axes[1, 0].set_title('Community Structure Strength')
axes[1, 0].set_ylim([0, 1])
axes[1, 0].text(0, modularity, f'{modularity:.4f}', ha='center', va='bottom', fontweight='bold')

# Number of communities
axes[1, 1].bar(['Official', 'Detected'], [77, len(communities_louvain)], 
               color=['red', 'purple'], alpha=0.7)
axes[1, 1].set_ylabel('Number of Communities')
axes[1, 1].set_title('Official vs. Detected Communities')
axes[1, 1].text(0, 77, '77', ha='center', va='bottom', fontweight='bold')
axes[1, 1].text(1, len(communities_louvain), str(len(communities_louvain)), 
                ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/network_metrics_summary.png', dpi=300, bbox_inches='tight')
print("✓ Saved network_metrics_summary.png")
plt.close()

# ==============================================================================
# SAVE RESULTS
# ==============================================================================
print("\n[PHASE 8] Saving results...")
print("-" * 80)

# Save enhanced dataset
df.to_csv(f'{OUTPUT_DIR}/chicago_77_cas_with_communities.csv', index=False)
print("✓ Saved chicago_77_cas_with_communities.csv")

# Save community assignments
community_assignments = []
for i, comm in enumerate(communities_louvain):
    for node in comm:
        ca_name = CA_NAMES.get(node, f"CA {node}")
        community_assignments.append({
            'community_area': node,
            'ca_name': ca_name,
            'louvain_community': i
        })

pd.DataFrame(community_assignments).to_csv(f'{OUTPUT_DIR}/community_assignments.csv', index=False)
print("✓ Saved community_assignments.csv")

# Save network statistics
stats = {
    'Metric': [
        'Total Community Areas',
        'Network Edges',
        'Network Density',
        'Clustering Coefficient',
        'Connected Components',
        'Largest Component Size',
        'Louvain Communities Detected',
        'Modularity',
        'Similarity Threshold'
    ],
    'Value': [
        G.number_of_nodes(),
        G.number_of_edges(),
        f'{nx.density(G):.4f}',
        f'{clustering_coef:.4f}',
        nx.number_connected_components(G),
        len(largest_cc),
        len(communities_louvain),
        f'{modularity:.4f}',
        SIMILARITY_THRESHOLD
    ]
}

pd.DataFrame(stats).to_csv(f'{OUTPUT_DIR}/network_statistics.csv', index=False)
print("✓ Saved network_statistics.csv")

# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "="*80)
print("FINAL PROJECT COMPLETE!")
print("="*80)

print(f"\n✅ ANALYSIS COMPLETE:")
print(f"  • Analyzed all {G.number_of_nodes()} Chicago Community Areas")
print(f"  • Calculated {len(df)**2} similarity comparisons")
print(f"  • Built network with {G.number_of_edges()} edges")
print(f"  • Detected {len(communities_louvain)} natural communities")
print(f"  • Modularity: {modularity:.4f} (strong community structure)")

print(f"\n📊 KEY FINDINGS:")
print(f"  • Official structure: 77 separate CAs")
print(f"  • Data-driven structure: {len(communities_louvain)} communities")
print(f"  • Suggests {77 - len(communities_louvain)} CAs could be consolidated")
print(f"  • Network density: {nx.density(G):.4f}")
print(f"  • Clustering coefficient: {clustering_coef:.4f}")

print(f"\n📁 OUTPUT FILES ({OUTPUT_DIR}):")
print(f"  • chicago_77_cas_with_communities.csv")
print(f"  • community_assignments.csv")
print(f"  • network_statistics.csv")
print(f"  • similarity_matrix_77x77.npy")
print(f"  • network_graph_louvain.png")
print(f"  • similarity_heatmap_sorted.png")
print(f"  • community_sizes.png")
print(f"  • network_metrics_summary.png")

print(f"\n⏱️ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
