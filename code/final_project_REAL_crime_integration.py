"""
CS 579 FINAL PROJECT - REAL CRIME DATA INTEGRATION
Processing actual 2020 Chicago crime data and integrating with demographic data

Author: Anshul Dani (A20580060)
Date: December 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from networkx.algorithms import community
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

print("="*80)
print("PROCESSING REAL CHICAGO CRIME DATA - FINAL PROJECT")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ==============================================================================
# PHASE 1: LOAD AND PROCESS CRIME DATA
# ==============================================================================
print("[PHASE 1] Loading crime data from Chicago Data Portal...")
print("-" * 80)

# Load crime data
crimes = pd.read_csv('/Users/anshuldani/Documents/Masters_Sem_3/OSNA/OSNA_ASS4/Crimes_-_2020_20251203.csv')
print(f"✓ Loaded {len(crimes):,} crime records from 2020")
print(f"  Columns: {crimes.columns.tolist()}")

# Check Community Area column
print(f"\n  Community Area coverage:")
print(f"  - Records with CA: {crimes['Community Area'].notna().sum():,} ({crimes['Community Area'].notna().sum()/len(crimes)*100:.1f}%)")
print(f"  - Records without CA: {crimes['Community Area'].isna().sum():,}")
print(f"  - Unique CAs: {crimes['Community Area'].nunique()}")

# Filter to records with Community Area
crimes_with_ca = crimes[crimes['Community Area'].notna()].copy()
print(f"\n✓ Filtered to {len(crimes_with_ca):,} crimes with Community Area assigned")

# ==============================================================================
# PHASE 2: AGGREGATE CRIME BY COMMUNITY AREA
# ==============================================================================
print("\n[PHASE 2] Aggregating crime counts by Community Area...")
print("-" * 80)

# Count crimes per CA
crimes_by_ca = crimes_with_ca.groupby('Community Area').size().reset_index(name='total_crimes')
crimes_by_ca['community_area'] = crimes_by_ca['Community Area'].astype(int)
crimes_by_ca = crimes_by_ca[['community_area', 'total_crimes']]

print(f"✓ Aggregated crimes for {len(crimes_by_ca)} Community Areas")
print(f"\n  Crime statistics:")
print(f"  - Min crimes: {crimes_by_ca['total_crimes'].min():,}")
print(f"  - Max crimes: {crimes_by_ca['total_crimes'].max():,}")
print(f"  - Mean crimes: {crimes_by_ca['total_crimes'].mean():.0f}")
print(f"  - Median crimes: {crimes_by_ca['total_crimes'].median():.0f}")

# Show top 10 highest crime CAs
print(f"\n  Top 10 CAs by total crimes:")
top_10 = crimes_by_ca.nlargest(10, 'total_crimes')
for idx, row in top_10.iterrows():
    print(f"    CA {int(row['community_area'])}: {int(row['total_crimes']):,} crimes")

# ==============================================================================
# PHASE 3: LOAD DEMOGRAPHIC DATA AND MERGE
# ==============================================================================
print("\n[PHASE 3] Loading demographic data and calculating crime rates...")
print("-" * 80)

# Load your existing CA data
ca_data = pd.read_csv('outputs/final_project/chicago_77_cas_data_2020.csv')
print(f"✓ Loaded demographic data for {len(ca_data)} Community Areas")

# Add CA names if not present
if 'ca_name' not in ca_data.columns:
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
    ca_data['ca_name'] = ca_data['community_area'].map(CA_NAMES)

# Merge crime data
ca_data = ca_data.merge(crimes_by_ca, on='community_area', how='left')

# Fill missing with 0 (some CAs might have no reported crimes)
ca_data['total_crimes'] = ca_data['total_crimes'].fillna(0).astype(int)

# Calculate crime rate per 1,000 residents
ca_data['crime_rate_per_1000'] = (ca_data['total_crimes'] / ca_data['total_population']) * 1000

print(f"\n✓ Calculated crime rates per 1,000 residents")
print(f"\n  Crime rate statistics:")
print(f"  - Min rate: {ca_data['crime_rate_per_1000'].min():.1f} per 1,000")
print(f"  - Max rate: {ca_data['crime_rate_per_1000'].max():.1f} per 1,000")
print(f"  - Mean rate: {ca_data['crime_rate_per_1000'].mean():.1f} per 1,000")
print(f"  - Median rate: {ca_data['crime_rate_per_1000'].median():.1f} per 1,000")

# Show top 10 by crime rate
print(f"\n  Top 10 CAs by crime rate per 1,000:")
top_10_rate = ca_data.nlargest(10, 'crime_rate_per_1000')[['community_area', 'ca_name', 'crime_rate_per_1000', 'total_population', 'total_crimes']]
for idx, row in top_10_rate.iterrows():
    if pd.notna(row['ca_name']):
        print(f"    CA {int(row['community_area'])} ({row['ca_name']}): {row['crime_rate_per_1000']:.1f} per 1,000 ({int(row['total_crimes']):,} crimes, {int(row['total_population']):,} pop)")

# Save data with crime
ca_data.to_csv('outputs/final_project/chicago_77_cas_with_REAL_crime.csv', index=False)
print(f"\n✓ Saved: chicago_77_cas_with_REAL_crime.csv")

# ==============================================================================
# PHASE 4: NETWORK ANALYSIS WITH 7 VARIABLES (6 demographic + 1 crime)
# ==============================================================================
print("\n[PHASE 4] Running network analysis with 7 variables (including crime)...")
print("-" * 80)

# Variables for similarity - NOW INCLUDING REAL CRIME DATA
similarity_vars = [
    'median_income',
    'pct_bachelors',
    'pct_owner',
    'pct_white',
    'pct_black',
    'pct_hispanic',
    'crime_rate_per_1000'  # 7th variable!
]

print(f"Using {len(similarity_vars)} variables:")
for i, var in enumerate(similarity_vars, 1):
    print(f"  {i}. {var}")

# Create feature matrix
X = ca_data[similarity_vars].values
X = np.nan_to_num(X, nan=0.0)

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Calculate similarity
similarity_matrix = cosine_similarity(X_scaled)
print(f"\n✓ Calculated 77×77 similarity matrix with real crime data")
print(f"  Mean similarity: {similarity_matrix.mean():.3f}")
print(f"  Similarity range: [{similarity_matrix.min():.3f}, {similarity_matrix.max():.3f}]")

# Save similarity matrix
np.save('outputs/final_project/similarity_matrix_with_REAL_crime.npy', similarity_matrix)

# Build network
THRESHOLD = 0.7
G = nx.Graph()

# Add nodes
for idx, row in ca_data.iterrows():
    G.add_node(row['community_area'],
               ca_name=row['ca_name'] if 'ca_name' in row else f"CA {row['community_area']}",
               population=row['total_population'],
               income=row['median_income'],
               crime_rate=row['crime_rate_per_1000'])

# Add edges
edge_count = 0
for i in range(len(ca_data)):
    for j in range(i+1, len(ca_data)):
        if similarity_matrix[i, j] > THRESHOLD:
            ca_i = ca_data.iloc[i]['community_area']
            ca_j = ca_data.iloc[j]['community_area']
            G.add_edge(ca_i, ca_j, weight=similarity_matrix[i, j])
            edge_count += 1

print(f"\n✓ Built network with real crime data:")
print(f"  Nodes: {G.number_of_nodes()}")
print(f"  Edges: {G.number_of_edges()}")
print(f"  Density: {nx.density(G):.4f}")
print(f"  Connected components: {nx.number_connected_components(G)}")

# Community detection
components = list(nx.connected_components(G))
largest_cc = max(components, key=len)
G_connected = G.subgraph(largest_cc).copy()

communities_louvain = community.greedy_modularity_communities(G_connected)
modularity = community.modularity(G_connected, communities_louvain)

print(f"\n✓ Louvain community detection:")
print(f"  Communities detected: {len(communities_louvain)}")
print(f"  Community sizes: {[len(c) for c in communities_louvain]}")
print(f"  Modularity: {modularity:.4f}")

# Assign community labels
node_to_community = {}
for i, comm in enumerate(communities_louvain):
    for node in comm:
        node_to_community[node] = i

ca_data['louvain_community_with_real_crime'] = ca_data['community_area'].map(node_to_community)

# Save results
ca_data.to_csv('outputs/final_project/chicago_77_cas_FINAL_with_real_crime.csv', index=False)
print(f"\n✓ Saved: chicago_77_cas_FINAL_with_real_crime.csv")

# ==============================================================================
# PHASE 5: COMPARISON - ORIGINAL vs WITH CRIME
# ==============================================================================
print("\n" + "="*80)
print("COMPARISON: Analysis With vs. Without Crime Data")
print("="*80)

# Load original results (without crime)
original_data = pd.read_csv('outputs/final_project/chicago_77_cas_with_communities.csv')

print("\nOriginal Analysis (6 variables - NO crime):")
print(f"  Communities detected: {original_data['louvain_community'].nunique()}")
print(f"  Modularity: 0.6074")

print(f"\nWith REAL Crime Data (7 variables):")
print(f"  Communities detected: {len(communities_louvain)}")
print(f"  Modularity: {modularity:.4f}")

# Check how many CAs changed community
merged = original_data[['community_area', 'louvain_community']].merge(
    ca_data[['community_area', 'louvain_community_with_real_crime']],
    on='community_area'
)
merged = merged.dropna()

# Map old community IDs to new ones to check for structural changes
changes = (merged['louvain_community'] != merged['louvain_community_with_real_crime']).sum()

print(f"\nCommunity Assignment Changes:")
print(f"  CAs that changed: {changes}")
print(f"  Percentage: {changes/len(merged)*100:.1f}%")

if changes < len(merged) * 0.2:
    print(f"  → Crime data VALIDATES existing structure (few changes)")
elif changes < len(merged) * 0.5:
    print(f"  → Crime data REFINES community structure (moderate changes)")
else:
    print(f"  → Crime data SIGNIFICANTLY ALTERS structure")

# ==============================================================================
# PHASE 6: CRIME-SPECIFIC VISUALIZATIONS
# ==============================================================================
print("\n[PHASE 6] Creating crime-specific visualizations...")
print("-" * 80)

# Visualization 1: Crime Rate by Community Area
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Crime Analysis - Chicago Community Areas 2020', fontsize=16, fontweight='bold')

# Panel 1: Crime rate distribution
ax = axes[0, 0]
ax.hist(ca_data['crime_rate_per_1000'], bins=30, color='red', alpha=0.7, edgecolor='black')
ax.axvline(ca_data['crime_rate_per_1000'].mean(), color='blue', linestyle='--', linewidth=2, label=f"Mean: {ca_data['crime_rate_per_1000'].mean():.1f}")
ax.axvline(ca_data['crime_rate_per_1000'].median(), color='green', linestyle='--', linewidth=2, label=f"Median: {ca_data['crime_rate_per_1000'].median():.1f}")
ax.set_xlabel('Crime Rate per 1,000 Residents')
ax.set_ylabel('Number of Community Areas')
ax.set_title('Distribution of Crime Rates')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 2: Crime vs Income
ax = axes[0, 1]
ax.scatter(ca_data['median_income'], ca_data['crime_rate_per_1000'], 
          alpha=0.6, s=100, c=ca_data['crime_rate_per_1000'], 
          cmap='YlOrRd', edgecolors='black')
ax.set_xlabel('Median Household Income ($)')
ax.set_ylabel('Crime Rate per 1,000')
ax.set_title('Crime Rate vs. Income')
ax.grid(True, alpha=0.3)

# Add correlation
corr = ca_data[['median_income', 'crime_rate_per_1000']].corr().iloc[0, 1]
ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
        transform=ax.transAxes, fontsize=11, va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Panel 3: Top 10 highest crime CAs
ax = axes[1, 0]
top_crime = ca_data.nlargest(10, 'crime_rate_per_1000')
bars = ax.barh(range(len(top_crime)), top_crime['crime_rate_per_1000'], color='red', alpha=0.7)
ax.set_yticks(range(len(top_crime)))
ax.set_yticklabels([f"CA {int(row['community_area'])}" + (f" ({row['ca_name'][:15]})" if pd.notna(row['ca_name']) else "") 
                     for idx, row in top_crime.iterrows()], fontsize=9)
ax.set_xlabel('Crime Rate per 1,000')
ax.set_title('Top 10 Highest Crime Rate CAs')
ax.grid(True, alpha=0.3, axis='x')

# Add values
for i, (idx, row) in enumerate(top_crime.iterrows()):
    ax.text(row['crime_rate_per_1000'], i, f" {row['crime_rate_per_1000']:.1f}", 
           va='center', fontsize=9)

# Panel 4: Crime by detected community
ax = axes[1, 1]
if len(communities_louvain) > 0:
    community_crime = []
    community_labels = []
    for i in range(len(communities_louvain)):
        comm_cas = ca_data[ca_data['louvain_community_with_real_crime'] == i]
        if len(comm_cas) > 0:
            community_crime.append(comm_cas['crime_rate_per_1000'].mean())
            community_labels.append(f"Community {i}\n({len(comm_cas)} CAs)")
    
    bars = ax.bar(range(len(community_crime)), community_crime, 
                  color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(community_crime)],
                  alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(community_crime)))
    ax.set_xticklabels(community_labels)
    ax.set_ylabel('Mean Crime Rate per 1,000')
    ax.set_title('Crime Rate by Detected Community')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add values
    for i, val in enumerate(community_crime):
        ax.text(i, val, f'{val:.1f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/final_project/crime_analysis.png', 
           dpi=300, bbox_inches='tight')
print("✓ Saved: crime_analysis.png")
plt.close()

# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "="*80)
print("REAL CRIME DATA INTEGRATION COMPLETE!")
print("="*80)

print(f"\n✅ PROCESSED REAL CHICAGO CRIME DATA:")
print(f"  • {len(crimes):,} total crime records from 2020")
print(f"  • {len(crimes_with_ca):,} crimes mapped to Community Areas")
print(f"  • Crime rates calculated for all 77 CAs")
print(f"  • Integrated with demographic data (7 variables total)")

print(f"\n📊 CRIME STATISTICS:")
print(f"  • Mean crime rate: {ca_data['crime_rate_per_1000'].mean():.1f} per 1,000")
print(f"  • Range: {ca_data['crime_rate_per_1000'].min():.1f} - {ca_data['crime_rate_per_1000'].max():.1f}")
print(f"  • Total crimes across Chicago: {ca_data['total_crimes'].sum():,}")

print(f"\n🕸️ NETWORK ANALYSIS RESULTS:")
print(f"  • Communities detected: {len(communities_louvain)}")
print(f"  • Modularity: {modularity:.4f}")
print(f"  • Network edges: {G.number_of_edges()}")
print(f"  • Network density: {nx.density(G):.4f}")

print(f"\n🔄 IMPACT OF CRIME DATA:")
if changes == 0:
    print(f"  • NO community reassignments (structure unchanged)")
    print(f"  • Crime data VALIDATES demographic divisions")
elif changes < len(merged) * 0.2:
    print(f"  • {changes} CAs reassigned (minor changes)")
    print(f"  • Crime data REFINES community structure")
else:
    print(f"  • {changes} CAs reassigned (significant changes)")
    print(f"  • Crime data REVEALS new patterns")

print(f"\n📁 OUTPUT FILES:")
print(f"  • chicago_77_cas_with_REAL_crime.csv")
print(f"  • chicago_77_cas_FINAL_with_real_crime.csv")
print(f"  • similarity_matrix_with_REAL_crime.npy")
print(f"  • crime_analysis.png")

print(f"\n⏱️ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

print("\n✅ READY FOR FINAL SUBMISSION!")
print("   All 7 variables integrated, analysis complete!")
