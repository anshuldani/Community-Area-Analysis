"""
CS 579 FINAL PROJECT - STATIC MAP VISUALIZATIONS
Creating static PNG maps as backup to Folium interactive maps

Author: Anshul Dani (A20580060)
Date: December 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

print("="*80)
print("CREATING STATIC MAP VISUALIZATIONS")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Load data
ca_data = pd.read_csv('outputs/final_project/chicago_77_cas_FINAL_with_real_crime.csv')

# Add coordinates (same as Folium script)
CA_COORDS = {
    1: [42.01, -87.67], 2: [42.00, -87.69], 3: [41.97, -87.66], 4: [41.97, -87.68],
    5: [41.96, -87.68], 6: [41.95, -87.65], 7: [41.92, -87.64], 8: [41.90, -87.63],
    9: [42.01, -87.81], 10: [42.00, -87.79], 11: [41.98, -87.77], 12: [41.98, -87.79],
    13: [41.98, -87.71], 14: [41.97, -87.71], 15: [41.96, -87.76], 16: [41.95, -87.73],
    17: [41.94, -87.79], 18: [41.93, -87.79], 19: [41.93, -87.75], 20: [41.92, -87.73],
    21: [41.94, -87.71], 22: [41.92, -87.71], 23: [41.90, -87.71], 24: [41.89, -87.68],
    25: [41.90, -87.76], 26: [41.88, -87.73], 27: [41.88, -87.71], 28: [41.88, -87.64],
    29: [41.86, -87.72], 30: [41.86, -87.70], 31: [41.86, -87.66], 32: [41.88, -87.63],
    33: [41.85, -87.62], 34: [41.83, -87.63], 35: [41.82, -87.62], 36: [41.82, -87.60],
    37: [41.81, -87.63], 38: [41.81, -87.62], 39: [41.80, -87.59], 40: [41.79, -87.62],
    41: [41.79, -87.60], 42: [41.78, -87.60], 43: [41.76, -87.58], 44: [41.75, -87.61],
    45: [41.74, -87.59], 46: [41.74, -87.56], 47: [41.73, -87.61], 48: [41.73, -87.59],
    49: [41.72, -87.63], 50: [41.71, -87.61], 51: [41.70, -87.57], 52: [41.71, -87.54],
    53: [41.69, -87.67], 54: [41.69, -87.63], 55: [41.66, -87.55], 56: [41.79, -87.77],
    57: [41.81, -87.73], 58: [41.82, -87.69], 59: [41.83, -87.67], 60: [41.83, -87.64],
    61: [41.81, -87.66], 62: [41.79, -87.71], 63: [41.78, -87.69], 64: [41.78, -87.76],
    65: [41.77, -87.72], 66: [41.76, -87.69], 67: [41.75, -87.67], 68: [41.78, -87.65],
    69: [41.77, -87.61], 70: [41.75, -87.71], 71: [41.74, -87.66], 72: [41.72, -87.67],
    73: [41.71, -87.65], 74: [41.70, -87.67], 75: [41.69, -87.66], 76: [41.98, -87.90],
    77: [42.00, -87.66]
}

ca_data['lat'] = ca_data['community_area'].map(lambda x: CA_COORDS.get(x, [41.88, -87.63])[0])
ca_data['lon'] = ca_data['community_area'].map(lambda x: CA_COORDS.get(x, [41.88, -87.63])[1])

# Create 2x2 grid of maps
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle('Chicago Community Areas - Network Analysis Results', fontsize=18, fontweight='bold')

# Colors for communities
community_colors = {
    0: '#1f77b4',  # Blue
    1: '#ff7f0e',  # Orange
    2: '#2ca02c',  # Green
}

# ==============================================================================
# MAP 1: Community Detection
# ==============================================================================
ax = axes[0, 0]
ax.set_title('Detected Communities (Modularity = 0.607)', fontsize=14, fontweight='bold')

for idx, row in ca_data.iterrows():
    if pd.notna(row['louvain_community_with_real_crime']):
        community = int(row['louvain_community_with_real_crime'])
        color = community_colors.get(community, 'gray')
        
        ax.scatter(row['lon'], row['lat'], 
                  c=color, s=300, alpha=0.7, edgecolors='black', linewidth=1)
        
        # Add CA number
        if row['community_area'] % 5 == 0:  # Label every 5th CA to avoid clutter
            ax.text(row['lon'], row['lat'], str(int(row['community_area'])), 
                   fontsize=7, ha='center', va='center')

ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.grid(True, alpha=0.3)

# Legend
legend_elements = [
    mpatches.Patch(color='#1f77b4', label='Community 0: North/Affluent (28 CAs)'),
    mpatches.Patch(color='#ff7f0e', label='Community 1: South/Low-Income (27 CAs)'),
    mpatches.Patch(color='#2ca02c', label='Community 2: Southwest/Working-Class (20 CAs)')
]
ax.legend(handles=legend_elements, loc='lower left', fontsize=9)

# ==============================================================================
# MAP 2: Income Distribution
# ==============================================================================
ax = axes[0, 1]
ax.set_title('Median Household Income by CA', fontsize=14, fontweight='bold')

# Color by income
income_colors = ca_data['median_income'].values
sc = ax.scatter(ca_data['lon'], ca_data['lat'], 
               c=income_colors, s=300, alpha=0.7, 
               cmap='RdYlGn', edgecolors='black', linewidth=1,
               vmin=ca_data['median_income'].min(), 
               vmax=ca_data['median_income'].max())

ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.grid(True, alpha=0.3)

# Colorbar
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Median Income ($)', rotation=270, labelpad=20)

# ==============================================================================
# MAP 3: Network Connections
# ==============================================================================
ax = axes[1, 0]
ax.set_title('Network Connections (Similarity > 0.7)', fontsize=14, fontweight='bold')

# Load similarity matrix
similarity_matrix = np.load('outputs/final_project/similarity_matrix_77x77.npy')

# Draw edges
THRESHOLD = 0.75  # Stricter for clarity
edge_count = 0
for i in range(len(ca_data)):
    for j in range(i+1, len(ca_data)):
        if similarity_matrix[i, j] > THRESHOLD:
            ca_i = ca_data.iloc[i]
            ca_j = ca_data.iloc[j]
            
            ax.plot([ca_i['lon'], ca_j['lon']], [ca_i['lat'], ca_j['lat']], 
                   'b-', alpha=0.2, linewidth=0.5)
            edge_count += 1

# Draw nodes
for idx, row in ca_data.iterrows():
    if pd.notna(row['louvain_community_with_real_crime']):
        community = int(row['louvain_community_with_real_crime'])
        color = community_colors.get(community, 'gray')
        
        ax.scatter(row['lon'], row['lat'], 
                  c=color, s=200, alpha=0.8, edgecolors='black', linewidth=1)

ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.grid(True, alpha=0.3)
ax.text(0.05, 0.95, f'{edge_count} connections shown', 
        transform=ax.transAxes, fontsize=10, va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ==============================================================================
# MAP 4: Population Density
# ==============================================================================
ax = axes[1, 1]
ax.set_title('Population by Community Area', fontsize=14, fontweight='bold')

# Size by population
pop_sizes = ca_data['total_population'].values

# Handle NaN values in community assignment
colors = []
for idx, row in ca_data.iterrows():
    if pd.notna(row['louvain_community_with_real_crime']):
        community = int(row['louvain_community_with_real_crime'])
        colors.append(community_colors.get(community, 'gray'))
    else:
        colors.append('gray')

sc = ax.scatter(ca_data['lon'], ca_data['lat'], 
               s=pop_sizes/100, alpha=0.6, 
               c=colors, 
               edgecolors='black', linewidth=1)

ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.grid(True, alpha=0.3)

# Add size legend
sizes = [10000, 50000, 90000]
labels = ['10K', '50K', '90K']
legend_elements = [plt.scatter([], [], s=s/100, c='gray', alpha=0.6, edgecolors='black', linewidth=1) 
                  for s in sizes]
ax.legend(legend_elements, labels, scatterpoints=1, title='Population',
         loc='lower left', fontsize=9, title_fontsize=10)

plt.tight_layout()
plt.savefig('outputs/final_project/static_maps_combined.png', 
           dpi=300, bbox_inches='tight')
print("✓ Saved static_maps_combined.png")
plt.close()

# ==============================================================================
# Create Large Single Community Map
# ==============================================================================
fig, ax = plt.subplots(figsize=(14, 12))
fig.suptitle('Chicago Community Areas - Network-Detected Communities', 
            fontsize=18, fontweight='bold', y=0.98)

for idx, row in ca_data.iterrows():
    if pd.notna(row['louvain_community_with_real_crime']):
        community = int(row['louvain_community_with_real_crime'])
        color = community_colors.get(community, 'gray')
        
        ax.scatter(row['lon'], row['lat'], 
                  c=color, s=500, alpha=0.7, edgecolors='black', linewidth=2)
        
        # Add CA number for all
        ax.text(row['lon'], row['lat'], str(int(row['community_area'])), 
               fontsize=9, ha='center', va='center', fontweight='bold')

ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# Enhanced legend
legend_elements = [
    mpatches.Patch(color='#1f77b4', label='Community 0: North/Affluent\n28 CAs, $91,592 avg income'),
    mpatches.Patch(color='#ff7f0e', label='Community 1: South/Low-Income\n27 CAs, $41,050 avg income'),
    mpatches.Patch(color='#2ca02c', label='Community 2: Southwest/Working-Class\n20 CAs, $55,885 avg income')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=11, 
         title='Detected Communities', title_fontsize=12, framealpha=0.9)

# Add stats box
stats_text = f'''Network Statistics:
• Modularity: 0.6074
• Network Density: 0.1548
• Clustering: 0.7023
• Edges: 453
• Official CAs: 77
• Detected Communities: 3'''

ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
       fontsize=10, va='top', ha='left',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('outputs/final_project/static_community_map_large.png', 
           dpi=300, bbox_inches='tight')
print("✓ Saved static_community_map_large.png")
plt.close()

print("\n" + "="*80)
print("STATIC MAP CREATION COMPLETE")
print("="*80)
print("\n✅ Created 2 static map visualizations:")
print("  • static_maps_combined.png (2×2 grid)")
print("  • static_community_map_large.png (detailed community map)")
print("\n📂 Location: outputs/final_project/")
print(f"\n⏱️ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)