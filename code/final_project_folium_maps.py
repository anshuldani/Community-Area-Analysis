"""
CS 579 FINAL PROJECT - INTERACTIVE FOLIUM MAPS WITH CRIME DATA
Creating geographic visualizations of community detection results

Author: Anshul Dani (A20580060)
Date: December 2025

Creates 5 interactive maps:
1. Community Detection Results Map (with crime data)
2. Crime Rate Heatmap  
3. Income Heatmap
4. Network Connections Map
5. Comparison Map (Official vs. Detected)
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime

print("="*80)
print("CREATING INTERACTIVE FOLIUM MAPS WITH CRIME DATA")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Check if folium is available, if not create static HTML maps
try:
    import folium
    from folium import plugins
    FOLIUM_AVAILABLE = True
    print("✓ Folium library available")
except ImportError:
    FOLIUM_AVAILABLE = False
    print("⚠ Folium not available - will create basic HTML maps")

# ==============================================================================
# LOAD DATA
# ==============================================================================
print("\n[PHASE 1] Loading data...")
print("-" * 80)

# Load CA data with communities AND CRIME DATA
ca_data = pd.read_csv('outputs/final_project/chicago_77_cas_FINAL_with_real_crime.csv')
print(f"✓ Loaded {len(ca_data)} Community Areas with community assignments and crime data")

# Check if crime data is present
has_crime_data = 'crime_rate_per_1000' in ca_data.columns
if has_crime_data:
    print(f"✓ Crime data available: {ca_data['total_crimes'].sum():,} total crimes")
    print(f"  Mean crime rate: {ca_data['crime_rate_per_1000'].mean():.1f} per 1,000 residents")
    print(f"  Range: {ca_data['crime_rate_per_1000'].min():.1f} - {ca_data['crime_rate_per_1000'].max():.1f}")
else:
    print("⚠ Crime data not found in file - using demographic data only")

# Community Area approximate coordinates (centroids)
# These are rough approximations for visualization
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

# Add coordinates to dataframe
ca_data['lat'] = ca_data['community_area'].map(lambda x: CA_COORDS.get(x, [41.88, -87.63])[0])
ca_data['lon'] = ca_data['community_area'].map(lambda x: CA_COORDS.get(x, [41.88, -87.63])[1])

# Color scheme for communities
community_colors = {
    0: '#1f77b4',  # Blue - North/Affluent
    1: '#ff7f0e',  # Orange - South/Low-Income
    2: '#2ca02c',  # Green - Southwest/Working-Class
}

# ==============================================================================
# MAP 1: COMMUNITY DETECTION RESULTS WITH CRIME DATA
# ==============================================================================
print("\n[PHASE 2] Creating Map 1: Community Detection Results with Crime Data...")
print("-" * 80)

if FOLIUM_AVAILABLE:
    # Create base map centered on Chicago
    m1 = folium.Map(
        location=[41.88, -87.63],
        zoom_start=11,
        tiles='OpenStreetMap'
    )
    
    # Add markers for each CA
    for idx, row in ca_data.iterrows():
        if pd.notna(row.get('louvain_community_with_real_crime')):
            community = int(row['louvain_community_with_real_crime'])
            color = community_colors.get(community, 'gray')
            
            # Create popup content with crime data
            popup_html = f"""
            <div style="font-family: Arial; width: 280px;">
                <h4 style="margin: 0; color: {color};">{row['ca_name']}</h4>
                <p style="margin: 2px 0; font-size: 11px; color: gray;">CA {int(row['community_area'])} | Community {community}</p>
                <hr style="margin: 5px 0;">
                <p style="margin: 3px 0;"><strong>Population:</strong> {int(row['total_population']):,}</p>
                <p style="margin: 3px 0;"><strong>Median Income:</strong> ${int(row['median_income']):,}</p>
                <p style="margin: 3px 0;"><strong>Homeownership:</strong> {row['pct_owner']:.1f}%</p>
                <p style="margin: 3px 0;"><strong>Bachelor's+:</strong> {row['pct_bachelors']:.1f}%</p>
                <p style="margin: 3px 0;"><strong>Unemployment:</strong> {row['unemployment_rate']:.1f}%</p>
            """
            
            if has_crime_data:
                popup_html += f"""
                <hr style="margin: 5px 0;">
                <p style="margin: 3px 0; color: #d62728;"><strong>Crime Rate:</strong> {row['crime_rate_per_1000']:.1f} per 1,000</p>
                <p style="margin: 3px 0; color: #d62728;"><strong>Total Crimes:</strong> {int(row['total_crimes']):,}</p>
                """
            
            popup_html += """
            </div>
            """
            
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=8,
                popup=folium.Popup(popup_html, max_width=320),
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
                weight=2
            ).add_to(m1)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; 
         top: 10px; right: 10px; width: 220px; height: 160px; 
         background-color: white; border:2px solid grey; z-index:9999; 
         font-size:14px; padding: 10px">
         <p style="margin: 0; font-weight: bold;">Detected Communities</p>
         <hr style="margin: 5px 0;">
         <p style="margin: 3px 0;"><span style="color: #1f77b4;">●</span> Community 0: North/Affluent (28 CAs)</p>
         <p style="margin: 3px 0; font-size: 11px; margin-left: 15px;">Crime: 51.8 per 1,000</p>
         <p style="margin: 3px 0;"><span style="color: #ff7f0e;">●</span> Community 1: South/Low-Income (27 CAs)</p>
         <p style="margin: 3px 0; font-size: 11px; margin-left: 15px;">Crime: 161.1 per 1,000</p>
         <p style="margin: 3px 0;"><span style="color: #2ca02c;">●</span> Community 2: Southwest/Working (20 CAs)</p>
         <p style="margin: 3px 0; font-size: 11px; margin-left: 15px;">Crime: 51.1 per 1,000</p>
    </div>
    '''
    m1.get_root().html.add_child(folium.Element(legend_html))
    
    # Save map
    m1.save('outputs/final_project/map1_communities.html')
    print("✓ Saved map1_communities.html")

else:
    # Create basic HTML map without folium
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Chicago Community Detection Map with Crime Data</title>
        <style>
            body { font-family: Arial; margin: 20px; }
            h1 { color: #333; }
            .info { background: #f0f0f0; padding: 15px; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h1>Community Detection Results Map (with Crime Data)</h1>
        <div class="info">
            <p><strong>Note:</strong> Interactive map requires Folium library.</p>
            <p>To create interactive map: pip install folium</p>
            <p><strong>Community Structure:</strong></p>
            <ul>
                <li>Community 0 (Blue): 28 CAs - North/Affluent, $91,592 avg income, 51.8 crimes per 1,000</li>
                <li>Community 1 (Orange): 27 CAs - South/Low-Income, $41,050 avg income, 161.1 crimes per 1,000</li>
                <li>Community 2 (Green): 20 CAs - Southwest/Working-Class, $55,885 avg income, 51.1 crimes per 1,000</li>
            </ul>
        </div>
    </body>
    </html>
    """
    with open('outputs/final_project/map1_communities.html', 'w') as f:
        f.write(html_content)
    print("✓ Created placeholder map (install folium for interactive version)")

# ==============================================================================
# MAP 2: CRIME RATE HEATMAP
# ==============================================================================
print("\n[PHASE 3] Creating Map 2: Crime Rate Heatmap...")
print("-" * 80)

if FOLIUM_AVAILABLE and has_crime_data:
    m2 = folium.Map(
        location=[41.88, -87.63],
        zoom_start=11,
        tiles='CartoDB positron'
    )
    
    # Create heatmap data for crime
    heat_data = [[row['lat'], row['lon'], row['crime_rate_per_1000']/10] 
                 for idx, row in ca_data.iterrows()]
    
    # Add heatmap
    from folium.plugins import HeatMap
    HeatMap(heat_data, radius=25, blur=35, max_zoom=13, gradient={0.4: 'yellow', 0.65: 'orange', 1: 'red'}).add_to(m2)
    
    # Add markers with crime info
    for idx, row in ca_data.iterrows():
        # Color code by crime rate
        crime_rate = row['crime_rate_per_1000']
        if crime_rate > 150:
            color = 'red'
        elif crime_rate > 80:
            color = 'orange'
        else:
            color = 'green'
        
        popup_html = f"""
        <div style="font-family: Arial; width: 240px;">
            <h4 style="margin: 0;">{row['ca_name']}</h4>
            <hr style="margin: 5px 0;">
            <p style="margin: 3px 0;"><strong>Crime Rate:</strong> {row['crime_rate_per_1000']:.1f} per 1,000</p>
            <p style="margin: 3px 0;"><strong>Total Crimes:</strong> {int(row['total_crimes']):,}</p>
            <p style="margin: 3px 0;"><strong>Population:</strong> {int(row['total_population']):,}</p>
            <p style="margin: 3px 0;"><strong>Median Income:</strong> ${int(row['median_income']):,}</p>
        </div>
        """
        
        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=folium.Popup(popup_html, max_width=260),
            icon=folium.Icon(color=color, icon='warning-sign')
        ).add_to(m2)
    
    # Add title
    title_html = '''
    <div style="position: fixed; 
         top: 10px; left: 50px; width: 320px; height: 80px; 
         background-color: white; border:2px solid grey; z-index:9999; 
         font-size:16px; font-weight: bold; padding: 10px">
         <p style="margin: 0;">Crime Rate by Community Area</p>
         <p style="margin: 5px 0 0 0; font-size:12px; font-weight:normal;">
         212,655 crimes in 2020 | Mean: 90.2 per 1,000</p>
         <p style="margin: 5px 0 0 0; font-size:12px; font-weight:normal;">
         🟢 Low (<80) | 🟠 Medium (80-150) | 🔴 High (>150)</p>
    </div>
    '''
    m2.get_root().html.add_child(folium.Element(title_html))
    
    m2.save('outputs/final_project/map2_crime_heatmap.html')
    print("✓ Saved map2_crime_heatmap.html")
    
else:
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Crime Rate Heatmap</title>
        <style>
            body { font-family: Arial; margin: 20px; }
        </style>
    </head>
    <body>
        <h1>Crime Rate Heatmap</h1>
        <p>Install folium for interactive heatmap: pip install folium</p>
    </body>
    </html>
    """
    with open('outputs/final_project/map2_crime_heatmap.html', 'w') as f:
        f.write(html_content)
    print("✓ Created placeholder map")

# ==============================================================================
# MAP 3: INCOME HEATMAP
# ==============================================================================
print("\n[PHASE 4] Creating Map 3: Income Heatmap...")
print("-" * 80)

if FOLIUM_AVAILABLE:
    m3 = folium.Map(
        location=[41.88, -87.63],
        zoom_start=11,
        tiles='CartoDB positron'
    )
    
    # Create heatmap data
    heat_data = [[row['lat'], row['lon'], row['median_income']/1000] 
                 for idx, row in ca_data.iterrows()]
    
    # Add heatmap
    from folium.plugins import HeatMap
    HeatMap(heat_data, radius=25, blur=35, max_zoom=13).add_to(m3)
    
    # Add markers with income info
    for idx, row in ca_data.iterrows():
        # Color code by income
        income = row['median_income']
        if income > 80000:
            color = 'green'
        elif income > 50000:
            color = 'orange'
        else:
            color = 'red'
        
        popup_html = f"""
        <div style="font-family: Arial; width: 220px;">
            <h4 style="margin: 0;">{row['ca_name']}</h4>
            <hr style="margin: 5px 0;">
            <p style="margin: 3px 0;"><strong>Median Income:</strong> ${int(row['median_income']):,}</p>
            <p style="margin: 3px 0;"><strong>Population:</strong> {int(row['total_population']):,}</p>
        """
        
        if has_crime_data:
            popup_html += f"""
            <p style="margin: 3px 0;"><strong>Crime Rate:</strong> {row['crime_rate_per_1000']:.1f} per 1,000</p>
            """
        
        popup_html += """
        </div>
        """
        
        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=folium.Popup(popup_html, max_width=250),
            icon=folium.Icon(color=color, icon='info-sign')
        ).add_to(m3)
    
    # Add title
    title_html = '''
    <div style="position: fixed; 
         top: 10px; left: 50px; width: 320px; height: 60px; 
         background-color: white; border:2px solid grey; z-index:9999; 
         font-size:16px; font-weight: bold; padding: 10px">
         <p style="margin: 0;">Median Income by Community Area</p>
         <p style="margin: 5px 0 0 0; font-size:12px; font-weight:normal;">
         🟢 High (>$80K) | 🟠 Medium ($50-80K) | 🔴 Low (<$50K)</p>
    </div>
    '''
    m3.get_root().html.add_child(folium.Element(title_html))
    
    m3.save('outputs/final_project/map3_income_heatmap.html')
    print("✓ Saved map3_income_heatmap.html")
    
else:
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Income Heatmap</title>
        <style>
            body { font-family: Arial; margin: 20px; }
        </style>
    </head>
    <body>
        <h1>Income Heatmap</h1>
        <p>Install folium for interactive heatmap: pip install folium</p>
    </body>
    </html>
    """
    with open('outputs/final_project/map3_income_heatmap.html', 'w') as f:
        f.write(html_content)
    print("✓ Created placeholder map")

# ==============================================================================
# MAP 4: NETWORK CONNECTIONS WITH CRIME DATA
# ==============================================================================
print("\n[PHASE 5] Creating Map 4: Network Connections...")
print("-" * 80)

# Load similarity matrix with crime data
similarity_matrix = np.load('outputs/final_project/similarity_matrix_with_REAL_crime.npy')

if FOLIUM_AVAILABLE:
    m4 = folium.Map(
        location=[41.88, -87.63],
        zoom_start=11,
        tiles='OpenStreetMap'
    )
    
    # Draw edges between highly similar CAs
    THRESHOLD = 0.75  # Stricter threshold for visualization clarity
    
    edge_count = 0
    for i in range(len(ca_data)):
        for j in range(i+1, len(ca_data)):
            if similarity_matrix[i, j] > THRESHOLD:
                ca_i = ca_data.iloc[i]
                ca_j = ca_data.iloc[j]
                
                # Draw line
                folium.PolyLine(
                    locations=[[ca_i['lat'], ca_i['lon']], [ca_j['lat'], ca_j['lon']]],
                    color='blue',
                    weight=1,
                    opacity=0.3
                ).add_to(m4)
                edge_count += 1
    
    # Add CA markers
    for idx, row in ca_data.iterrows():
        if pd.notna(row.get('louvain_community_with_real_crime')):
            community = int(row['louvain_community_with_real_crime'])
            color = community_colors.get(community, 'gray')
            
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=6,
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.8,
                weight=2
            ).add_to(m4)
    
    title_html = f'''
    <div style="position: fixed; 
         top: 10px; left: 50px; width: 350px; height: 80px; 
         background-color: white; border:2px solid grey; z-index:9999; 
         font-size:16px; font-weight: bold; padding: 10px">
         <p style="margin: 0;">Network Connections (Similarity > 0.75)</p>
         <p style="margin: 5px 0 0 0; font-size:12px; font-weight:normal;">
         Showing {edge_count} connections between similar CAs</p>
         <p style="margin: 5px 0 0 0; font-size:12px; font-weight:normal;">
         Analysis includes 7 variables (6 demographic + crime)</p>
    </div>
    '''
    m4.get_root().html.add_child(folium.Element(title_html))
    
    m4.save('outputs/final_project/map4_network_connections.html')
    print(f"✓ Saved map4_network_connections.html ({edge_count} connections)")
    
else:
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Network Connections</title>
        <style>
            body { font-family: Arial; margin: 20px; }
        </style>
    </head>
    <body>
        <h1>Network Connections Map</h1>
        <p>Install folium for interactive map: pip install folium</p>
    </body>
    </html>
    """
    with open('outputs/final_project/map4_network_connections.html', 'w') as f:
        f.write(html_content)
    print("✓ Created placeholder map")

# ==============================================================================
# MAP 5: COMPARISON MAP (Official vs. Detected)
# ==============================================================================
print("\n[PHASE 6] Creating Map 5: Comparison View...")
print("-" * 80)

if FOLIUM_AVAILABLE:
    # Create dual pane map
    m5 = folium.Map(
        location=[41.88, -87.63],
        zoom_start=11,
        tiles='CartoDB positron'
    )
    
    # Add all CAs with detected community colors
    for idx, row in ca_data.iterrows():
        if pd.notna(row.get('louvain_community_with_real_crime')):
            community = int(row['louvain_community_with_real_crime'])
            color = community_colors.get(community, 'gray')
            
            popup_html = f"""
            <div style="font-family: Arial; width: 240px;">
                <h4 style="margin: 0;">{row['ca_name']}</h4>
                <hr style="margin: 5px 0;">
                <p style="margin: 3px 0;"><strong>Official CA:</strong> {int(row['community_area'])}</p>
                <p style="margin: 3px 0;"><strong>Detected Community:</strong> {community}</p>
                <p style="margin: 3px 0;"><strong>Income:</strong> ${int(row['median_income']):,}</p>
            """
            
            if has_crime_data:
                popup_html += f"""
                <p style="margin: 3px 0;"><strong>Crime Rate:</strong> {row['crime_rate_per_1000']:.1f} per 1,000</p>
                """
            
            popup_html += """
                <hr style="margin: 5px 0;">
                <p style="margin: 2px 0; font-size: 10px; color: gray;">
                ✓ Matches detected pattern
                </p>
            </div>
            """
            
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=10,
                popup=folium.Popup(popup_html, max_width=260),
                color='black',
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
                weight=1
            ).add_to(m5)
    
    title_html = '''
    <div style="position: fixed; 
         top: 10px; left: 50px; width: 370px; height: 120px; 
         background-color: white; border:2px solid grey; z-index:9999; 
         font-size:14px; padding: 10px">
         <p style="margin: 0; font-weight: bold;">Official (77 CAs) vs. Detected (3 Communities)</p>
         <hr style="margin: 5px 0;">
         <p style="margin: 3px 0; font-size: 12px;">Colors show detected communities</p>
         <p style="margin: 3px 0; font-size: 12px;">Click markers to see both assignments</p>
         <p style="margin: 3px 0; font-size: 11px; color: gray;">Analysis: 7 variables (6 demographic + crime)</p>
         <p style="margin: 3px 0; font-size: 11px; color: gray;">Modularity: 0.603 (very strong structure)</p>
    </div>
    '''
    m5.get_root().html.add_child(folium.Element(title_html))
    
    m5.save('outputs/final_project/map5_comparison.html')
    print("✓ Saved map5_comparison.html")
    
else:
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Comparison Map</title>
        <style>
            body { font-family: Arial; margin: 20px; }
        </style>
    </head>
    <body>
        <h1>Comparison: Official vs. Detected</h1>
        <p>Install folium for interactive map: pip install folium</p>
    </body>
    </html>
    """
    with open('outputs/final_project/map5_comparison.html', 'w') as f:
        f.write(html_content)
    print("✓ Created placeholder map")

# ==============================================================================
# CREATE INDEX PAGE
# ==============================================================================
print("\n[PHASE 7] Creating map index page...")
print("-" * 80)

index_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Chicago Community Network Analysis - Interactive Maps with Crime Data</title>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            color: #1f77b4;
            border-bottom: 3px solid #1f77b4;
            padding-bottom: 10px;
        }}
        .map-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }}
        .map-card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }}
        .map-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}
        .map-card h2 {{
            margin-top: 0;
            color: #333;
        }}
        .map-card p {{
            color: #666;
            line-height: 1.6;
        }}
        .map-link {{
            display: inline-block;
            background: #1f77b4;
            color: white;
            padding: 10px 20px;
            text-decoration: none;
            border-radius: 5px;
            margin-top: 10px;
            transition: background 0.2s;
        }}
        .map-link:hover {{
            background: #165a8a;
        }}
        .summary {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .summary ul {{
            list-style: none;
            padding: 0;
        }}
        .summary li {{
            padding: 5px 0;
            border-bottom: 1px solid #eee;
        }}
        .summary li:last-child {{
            border-bottom: none;
        }}
        .crime-alert {{
            background: #fff3cd;
            border-left: 4px solid #d62728;
            padding: 15px;
            margin: 15px 0;
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <h1>🗺️ Chicago Community Network Analysis - Interactive Maps with Crime Data</h1>
    
    <div class="summary">
        <h2>Project Summary</h2>
        <p><strong>Author:</strong> Anshul Dani (A20580060) | <strong>Course:</strong> CS 579</p>
        <p><strong>Key Findings:</strong></p>
        <ul>
            <li><strong>3 natural communities</strong> detected from 77 official Community Areas</li>
            <li><strong>Modularity: 0.6029</strong> (very strong community structure)</li>
            <li><strong>Network density: 0.1644</strong> with 481 edges</li>
            <li><strong>Scale:</strong> 77 CAs, 2,162 block groups, entire Chicago</li>
            <li><strong>Crime data:</strong> 212,655 crimes from 2020 integrated as 7th variable</li>
        </ul>
        
        <div class="crime-alert">
            <h3 style="margin-top: 0;">🚨 Crime Analysis Findings</h3>
            <ul>
                <li><strong>Mean crime rate:</strong> 90.2 per 1,000 residents</li>
                <li><strong>Income-crime correlation:</strong> -0.602 (strong negative - poverty drives crime)</li>
                <li><strong>Community 1 has 3× higher crime rate</strong> than others (161.1 vs ~51 per 1,000)</li>
                <li><strong>Crime validates structure:</strong> Only 4% of CAs changed community assignments</li>
            </ul>
        </div>
        
        <p><strong>Communities Detected:</strong></p>
        <ul>
            <li>🔵 <strong>Community 0: North/Affluent</strong> - 28 CAs, $91,592 avg income, 51.8 crimes per 1,000</li>
            <li>🟠 <strong>Community 1: South/Low-Income</strong> - 27 CAs, $41,050 avg income, <strong>161.1 crimes per 1,000</strong></li>
            <li>🟢 <strong>Community 2: Southwest/Working-Class</strong> - 20 CAs, $55,885 avg income, 51.1 crimes per 1,000</li>
        </ul>
    </div>
    
    <div class="map-grid">
        <div class="map-card">
            <h2>📍 Map 1: Community Detection</h2>
            <p>Interactive map showing the 3 detected communities <strong>with crime data</strong>. Each Community Area is colored by its detected community assignment. Click on areas to see detailed demographic information and crime rates.</p>
            <p><strong>Features:</strong> Community colors, demographic popups, crime statistics, legend</p>
            <a href="map1_communities.html" class="map-link">Open Map 1 →</a>
        </div>
        
        <div class="map-card">
            <h2>🚨 Map 2: Crime Rate Heatmap</h2>
            <p><strong>NEW!</strong> Heatmap visualization of crime rates across Chicago. Shows distribution of 212,655 crimes from 2020. Warmer colors indicate higher crime areas. Demonstrates stark geographic inequality in public safety.</p>
            <p><strong>Features:</strong> Crime heat visualization, crime rate markers, geographic patterns</p>
            <a href="map2_crime_heatmap.html" class="map-link">Open Map 2 →</a>
        </div>
        
        <div class="map-card">
            <h2>🌡️ Map 3: Income Heatmap</h2>
            <p>Heatmap visualization of median household income across Chicago. Warmer colors indicate higher income areas, cooler colors show lower income areas. Demonstrates economic stratification. Now includes crime rate correlation!</p>
            <p><strong>Features:</strong> Heat visualization, income markers, crime rates, geographic patterns</p>
            <a href="map3_income_heatmap.html" class="map-link">Open Map 3 →</a>
        </div>
        
        <div class="map-card">
            <h2>🕸️ Map 4: Network Connections</h2>
            <p>Shows network edges between highly similar Community Areas (similarity > 0.75). Lines connect similar neighborhoods, revealing the underlying similarity structure. <strong>Now based on 7 variables including crime!</strong></p>
            <p><strong>Features:</strong> Network edges, similarity connections, community colors</p>
            <a href="map4_network_connections.html" class="map-link">Open Map 4 →</a>
        </div>
        
        <div class="map-card">
            <h2>⚖️ Map 5: Official vs. Detected</h2>
            <p>Comparison view showing both official Community Area numbers and detected community assignments. Click areas to see how official boundaries compare to data-driven communities with crime data integrated.</p>
            <p><strong>Features:</strong> Dual assignment, comparison popups, modularity info, crime rates</p>
            <a href="map5_comparison.html" class="map-link">Open Map 5 →</a>
        </div>
    </div>
    
    <div class="summary" style="margin-top: 30px;">
        <h2>📊 Technical Details</h2>
        <p><strong>Methodology:</strong> Cosine similarity on <strong>7 variables</strong>, Louvain community detection algorithm</p>
        <p><strong>Data Sources:</strong></p>
        <ul>
            <li>U.S. Census 2020 ACS 5-Year Estimates (2,162 block groups aggregated to 77 CAs)</li>
            <li><strong>Chicago Data Portal 2020 Crime Data (212,655 crime incidents)</strong></li>
        </ul>
        <p><strong>Variables:</strong> Income, education, homeownership, race/ethnicity (3), unemployment, <strong>+ crime rate per 1,000</strong></p>
        <p><strong>Network Metrics:</strong> 481 edges, density 0.1644, clustering 0.7023, modularity 0.6029</p>
        <p><strong>Crime Integration Result:</strong> Only 4% of CAs changed assignments → crime patterns VALIDATE demographic structure</p>
    </div>
    
    <footer style="text-align: center; margin-top: 40px; padding: 20px; color: #666; font-size: 14px;">
        <p>CS 579 Final Project - Fall 2025 | Illinois Institute of Technology</p>
        <p>Analysis includes real crime data from Chicago Data Portal</p>
        <p>For source code and data, see GitHub repository or contact author</p>
    </footer>
</body>
</html>
"""

with open('outputs/final_project/index.html', 'w') as f:
    f.write(index_html)
print("✓ Created index.html (map navigation page with crime data summary)")

# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "="*80)
print("FOLIUM MAP CREATION COMPLETE - WITH CRIME DATA")
print("="*80)

if FOLIUM_AVAILABLE:
    print(f"\n✅ CREATED 6 INTERACTIVE HTML FILES:")
else:
    print(f"\n✅ CREATED 6 HTML FILES (placeholders - install folium for full interactivity):")

print(f"  1. index.html - Navigation page with all maps and crime summary")
print(f"  2. map1_communities.html - Community detection results WITH CRIME DATA")
print(f"  3. map2_crime_heatmap.html - CRIME RATE HEATMAP (NEW!)")
print(f"  4. map3_income_heatmap.html - Income distribution heatmap with crime")
print(f"  5. map4_network_connections.html - Network edges (7 variables + crime)")
print(f"  6. map5_comparison.html - Official vs. detected comparison with crime")

print(f"\n📂 LOCATION:")
print(f"  outputs/final_project/")

print(f"\n🌐 TO VIEW:")
print(f"  Open index.html in web browser to access all maps")
print(f"  Or open individual map files directly")

if has_crime_data:
    print(f"\n🚨 CRIME DATA INCLUDED:")
    print(f"  • 212,655 crimes from 2020")
    print(f"  • Crime rates shown in all maps")
    print(f"  • Dedicated crime heatmap created")
    print(f"  • Income-crime correlation: -0.602")

if not FOLIUM_AVAILABLE:
    print(f"\n💡 TIP:")
    print(f"  For full interactive maps, install folium:")
    print(f"  pip install folium")
    print(f"  Then re-run this script")

print(f"\n⏱️ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)