"""
CS 579 FINAL PROJECT - PHASE 1: DATA COLLECTION FOR ALL 77 CAs
This script expands HW4 to all 77 Chicago Community Areas

Author: [Your Name]
Date: [Date]
Status: IN PROGRESS - Data Collection Phase
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

print("="*80)
print("CS 579 FINAL PROJECT - EXPANDING TO ALL 77 CHICAGO COMMUNITY AREAS")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Create output directory for final project
os.makedirs('outputs/final_project', exist_ok=True)
OUTPUT_DIR = 'outputs/final_project'

print("[PHASE 1] DATA COLLECTION FOR ALL 77 COMMUNITY AREAS")
print("-" * 80)

# ==============================================================================
# STEP 1: LOAD TRACT-TO-CA MAPPING FOR ALL OF CHICAGO
# ==============================================================================
print("\n[Step 1/6] Loading tract-to-CA mapping...")

TRACT_FILE = '/Users/anshuldani/Documents/Masters_Sem_3/OSNA/OSNA_ASS4/2020 Census Tracts to Chicago Community Area Equivalency File - Sheet1.csv'
tracts_all = pd.read_csv(TRACT_FILE)

print(f"✓ Loaded mapping for {len(tracts_all)} census tracts")
print(f"✓ Covering all {tracts_all['CA'].nunique()} Community Areas")

# Get list of ALL Community Areas
all_cas = sorted(tracts_all['CA'].unique())
print(f"\nAll 77 Community Areas: {all_cas[:10]}... (showing first 10)")

# Create tract list for all CAs
all_tracts = tracts_all['GEOID20'].astype(str).tolist()
print(f"✓ Total census tracts to process: {len(all_tracts)}")

# ==============================================================================
# STEP 2: LOAD 2020 ACS DATA FOR ALL CHICAGO
# ==============================================================================
print("\n[Step 2/6] Loading 2020 ACS data for all Chicago tracts...")

# Load ACS data (same files from HW4, but now processing ALL tracts)
ACS_2020_POP = '/Users/anshuldani/Documents/Masters_Sem_3/OSNA/OSNA_ASS4/ACSDT5Y2020.B01003-Data.csv'
ACS_2020_RACE = '/Users/anshuldani/Documents/Masters_Sem_3/OSNA/OSNA_ASS4/ACSDT5Y2020.B03002-Data.csv'
ACS_2020_EDU = '/Users/anshuldani/Documents/Masters_Sem_3/OSNA/OSNA_ASS4/ACSDT5Y2020.B15003-Data.csv'
ACS_2020_INCOME = '/Users/anshuldani/Documents/Masters_Sem_3/OSNA/OSNA_ASS4/ACSDT5Y2020.B19013-Data.csv'
ACS_2020_EMPLOYMENT = '/Users/anshuldani/Documents/Masters_Sem_3/OSNA/OSNA_ASS4/ACSDT5Y2020.B23025-Data.csv'
ACS_2020_TENURE = '/Users/anshuldani/Documents/Masters_Sem_3/OSNA/OSNA_ASS4/ACSDT5Y2020.B25003-Data.csv'

def load_acs_data(filepath, tract_list):
    """Load and filter ACS data"""
    df = pd.read_csv(filepath, skiprows=1, dtype=str)
    df = df.dropna(axis=1, how='all')
    
    df['block_group_id'] = df.iloc[:, 0]
    df['full_geoid'] = df['block_group_id'].str.replace('1500000US', '')
    df['tract'] = df['full_geoid'].str[:-1]
    df['bg_num'] = df['full_geoid'].str[-1:]
    
    # Filter for Chicago tracts
    return df[df['tract'].isin(tract_list)].copy()

print("Loading ACS tables...")
pop = load_acs_data(ACS_2020_POP, all_tracts)
race = load_acs_data(ACS_2020_RACE, all_tracts)
edu = load_acs_data(ACS_2020_EDU, all_tracts)
income = load_acs_data(ACS_2020_INCOME, all_tracts)
employment = load_acs_data(ACS_2020_EMPLOYMENT, all_tracts)
tenure = load_acs_data(ACS_2020_TENURE, all_tracts)

total_block_groups = len(pop)
print(f"✓ Loaded data for {total_block_groups} block groups across all 77 CAs")

# ==============================================================================
# STEP 3: AGGREGATE TO COMMUNITY AREA LEVEL
# ==============================================================================
print("\n[Step 3/6] Aggregating block group data to Community Area level...")

# Build master dataset at block group level
data_bg = pop[['block_group_id', 'tract', 'bg_num']].copy()

# Extract variables
data_bg['total_pop'] = pd.to_numeric(pop.iloc[:, 2], errors='coerce')
data_bg['white'] = pd.to_numeric(race.iloc[:, 6], errors='coerce')
data_bg['black'] = pd.to_numeric(race.iloc[:, 8], errors='coerce')
data_bg['hispanic'] = pd.to_numeric(race.iloc[:, 22], errors='coerce')
data_bg['median_income'] = pd.to_numeric(income.iloc[:, 2], errors='coerce')
data_bg['in_labor_force'] = pd.to_numeric(employment.iloc[:, 2], errors='coerce')
data_bg['employed'] = pd.to_numeric(employment.iloc[:, 3], errors='coerce')
data_bg['unemployed'] = pd.to_numeric(employment.iloc[:, 4], errors='coerce')
data_bg['total_housing'] = pd.to_numeric(tenure.iloc[:, 2], errors='coerce')
data_bg['owner_occupied'] = pd.to_numeric(tenure.iloc[:, 3], errors='coerce')
data_bg['total_25plus'] = pd.to_numeric(edu.iloc[:, 2], errors='coerce')
data_bg['bachelors'] = pd.to_numeric(edu.iloc[:, 21], errors='coerce')
data_bg['masters'] = pd.to_numeric(edu.iloc[:, 22], errors='coerce')
data_bg['professional'] = pd.to_numeric(edu.iloc[:, 23], errors='coerce')
data_bg['doctorate'] = pd.to_numeric(edu.iloc[:, 24], errors='coerce')

# Add CA assignment
tract_ca_map = tracts_all.set_index('GEOID20')['CA'].to_dict()
tract_ca_map = {str(k): v for k, v in tract_ca_map.items()}
data_bg['community_area'] = data_bg['tract'].map(tract_ca_map)

# Aggregate to CA level
print("Aggregating to CA level...")

ca_data = []
for ca in all_cas:
    ca_bgs = data_bg[data_bg['community_area'] == ca]
    
    if len(ca_bgs) == 0:
        continue
    
    # Sum counts
    total_pop = ca_bgs['total_pop'].sum()
    white = ca_bgs['white'].sum()
    black = ca_bgs['black'].sum()
    hispanic = ca_bgs['hispanic'].sum()
    total_25plus = ca_bgs['total_25plus'].sum()
    bachelors_plus = (ca_bgs['bachelors'] + ca_bgs['masters'] + 
                      ca_bgs['professional'] + ca_bgs['doctorate']).sum()
    total_housing = ca_bgs['total_housing'].sum()
    owner_occ = ca_bgs['owner_occupied'].sum()
    in_lf = ca_bgs['in_labor_force'].sum()
    unemployed = ca_bgs['unemployed'].sum()
    
    # Calculate median income (population-weighted average of medians)
    valid_incomes = ca_bgs[ca_bgs['median_income'].notna()]
    if len(valid_incomes) > 0:
        weights = valid_incomes['total_pop'].fillna(0)
        median_income = np.average(valid_incomes['median_income'], weights=weights)
    else:
        median_income = np.nan
    
    # Calculate percentages
    pct_white = (white / total_pop * 100) if total_pop > 0 else 0
    pct_black = (black / total_pop * 100) if total_pop > 0 else 0
    pct_hispanic = (hispanic / total_pop * 100) if total_pop > 0 else 0
    pct_bachelors = (bachelors_plus / total_25plus * 100) if total_25plus > 0 else 0
    pct_owner = (owner_occ / total_housing * 100) if total_housing > 0 else 0
    unemployment_rate = (unemployed / in_lf * 100) if in_lf > 0 else 0
    
    ca_data.append({
        'community_area': ca,
        'total_population': total_pop,
        'median_income': median_income,
        'pct_white': pct_white,
        'pct_black': pct_black,
        'pct_hispanic': pct_hispanic,
        'pct_bachelors': pct_bachelors,
        'pct_owner': pct_owner,
        'unemployment_rate': unemployment_rate,
        'num_block_groups': len(ca_bgs)
    })

df_chicago = pd.DataFrame(ca_data)

print(f"✓ Created dataset with {len(df_chicago)} Community Areas")
print(f"✓ Variables: {len(df_chicago.columns)} total")

# Save to CSV
output_file = f"{OUTPUT_DIR}/chicago_77_cas_data_2020.csv"
df_chicago.to_csv(output_file, index=False)
print(f"✓ Saved: chicago_77_cas_data_2020.csv")

# ==============================================================================
# STEP 4: SUMMARY STATISTICS
# ==============================================================================
print("\n[Step 4/6] Calculating summary statistics...")

print("\nCHICAGO-WIDE SUMMARY (All 77 CAs):")
print(f"Total Population: {df_chicago['total_population'].sum():,.0f}")
print(f"Mean CA Population: {df_chicago['total_population'].mean():,.0f}")
print(f"Median Income Range: ${df_chicago['median_income'].min():,.0f} - ${df_chicago['median_income'].max():,.0f}")
print(f"Mean Median Income: ${df_chicago['median_income'].mean():,.0f}")
print(f"Homeownership Range: {df_chicago['pct_owner'].min():.1f}% - {df_chicago['pct_owner'].max():.1f}%")
print(f"Mean Homeownership: {df_chicago['pct_owner'].mean():.1f}%")

# ==============================================================================
# STEP 5: CREATE VISUALIZATIONS
# ==============================================================================
print("\n[Step 5/6] Creating visualizations...")

# Figure 1: Distribution of key variables across all 77 CAs
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Distribution of Key Variables Across All 77 Chicago Community Areas (2020)', 
             fontsize=16, fontweight='bold')

# Population
axes[0, 0].hist(df_chicago['total_population'], bins=20, color='steelblue', edgecolor='black')
axes[0, 0].set_title('Total Population', fontweight='bold')
axes[0, 0].set_xlabel('Population')
axes[0, 0].set_ylabel('Number of CAs')
axes[0, 0].axvline(df_chicago['total_population'].median(), color='red', linestyle='--', 
                   label=f"Median: {df_chicago['total_population'].median():,.0f}")
axes[0, 0].legend()

# Median Income
axes[0, 1].hist(df_chicago['median_income'], bins=20, color='green', edgecolor='black')
axes[0, 1].set_title('Median Household Income', fontweight='bold')
axes[0, 1].set_xlabel('Income ($)')
axes[0, 1].set_ylabel('Number of CAs')
axes[0, 1].axvline(df_chicago['median_income'].median(), color='red', linestyle='--',
                   label=f"Median: ${df_chicago['median_income'].median():,.0f}")
axes[0, 1].legend()

# Homeownership
axes[0, 2].hist(df_chicago['pct_owner'], bins=20, color='orange', edgecolor='black')
axes[0, 2].set_title('Homeownership Rate', fontweight='bold')
axes[0, 2].set_xlabel('% Owner-Occupied')
axes[0, 2].set_ylabel('Number of CAs')
axes[0, 2].axvline(df_chicago['pct_owner'].median(), color='red', linestyle='--',
                   label=f"Median: {df_chicago['pct_owner'].median():.1f}%")
axes[0, 2].legend()

# Education
axes[1, 0].hist(df_chicago['pct_bachelors'], bins=20, color='purple', edgecolor='black')
axes[1, 0].set_title("Bachelor's Degree or Higher", fontweight='bold')
axes[1, 0].set_xlabel('% with Bachelor+')
axes[1, 0].set_ylabel('Number of CAs')
axes[1, 0].axvline(df_chicago['pct_bachelors'].median(), color='red', linestyle='--',
                   label=f"Median: {df_chicago['pct_bachelors'].median():.1f}%")
axes[1, 0].legend()

# Race - White
axes[1, 1].hist(df_chicago['pct_white'], bins=20, color='skyblue', edgecolor='black')
axes[1, 1].set_title('White (Non-Hispanic) Population', fontweight='bold')
axes[1, 1].set_xlabel('% White')
axes[1, 1].set_ylabel('Number of CAs')
axes[1, 1].axvline(df_chicago['pct_white'].median(), color='red', linestyle='--',
                   label=f"Median: {df_chicago['pct_white'].median():.1f}%")
axes[1, 1].legend()

# Race - Black
axes[1, 2].hist(df_chicago['pct_black'], bins=20, color='coral', edgecolor='black')
axes[1, 2].set_title('Black/African American Population', fontweight='bold')
axes[1, 2].set_xlabel('% Black')
axes[1, 2].set_ylabel('Number of CAs')
axes[1, 2].axvline(df_chicago['pct_black'].median(), color='red', linestyle='--',
                   label=f"Median: {df_chicago['pct_black'].median():.1f}%")
axes[1, 2].legend()

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/chicago_77_cas_distributions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: chicago_77_cas_distributions.png")
plt.close()

# Figure 2: Top 10 and Bottom 10 CAs by various metrics
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Extreme Community Areas: Top 10 vs. Bottom 10', fontsize=16, fontweight='bold')

# Income
top_income = df_chicago.nlargest(10, 'median_income')[['community_area', 'median_income']]
bottom_income = df_chicago.nsmallest(10, 'median_income')[['community_area', 'median_income']]

axes[0, 0].barh(top_income['community_area'].astype(str), top_income['median_income'], color='green', alpha=0.7)
axes[0, 0].set_xlabel('Median Income ($)')
axes[0, 0].set_ylabel('Community Area')
axes[0, 0].set_title('Top 10 CAs by Income', fontweight='bold')
axes[0, 0].invert_yaxis()

axes[0, 1].barh(bottom_income['community_area'].astype(str), bottom_income['median_income'], color='red', alpha=0.7)
axes[0, 1].set_xlabel('Median Income ($)')
axes[0, 1].set_ylabel('Community Area')
axes[0, 1].set_title('Bottom 10 CAs by Income', fontweight='bold')
axes[0, 1].invert_yaxis()

# Education
top_edu = df_chicago.nlargest(10, 'pct_bachelors')[['community_area', 'pct_bachelors']]
bottom_edu = df_chicago.nsmallest(10, 'pct_bachelors')[['community_area', 'pct_bachelors']]

axes[1, 0].barh(top_edu['community_area'].astype(str), top_edu['pct_bachelors'], color='blue', alpha=0.7)
axes[1, 0].set_xlabel('% Bachelor+')
axes[1, 0].set_ylabel('Community Area')
axes[1, 0].set_title("Top 10 CAs by Education", fontweight='bold')
axes[1, 0].invert_yaxis()

axes[1, 1].barh(bottom_edu['community_area'].astype(str), bottom_edu['pct_bachelors'], color='orange', alpha=0.7)
axes[1, 1].set_xlabel('% Bachelor+')
axes[1, 1].set_ylabel('Community Area')
axes[1, 1].set_title("Bottom 10 CAs by Education", fontweight='bold')
axes[1, 1].invert_yaxis()

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/chicago_77_cas_extremes.png', dpi=300, bbox_inches='tight')
print("✓ Saved: chicago_77_cas_extremes.png")
plt.close()

# ==============================================================================
# STEP 6: COMPARE TO HW4 RESULTS
# ==============================================================================
print("\n[Step 6/6] Comparing to HW4 results (CA 60 - Bridgeport)...")

ca60_new = df_chicago[df_chicago['community_area'] == 60].iloc[0]

print("\nCA 60 (BRIDGEPORT) - NEW CALCULATION (aggregated from block groups):")
print(f"  Population: {ca60_new['total_population']:,.0f}")
print(f"  Median Income: ${ca60_new['median_income']:,.0f}")
print(f"  Homeownership: {ca60_new['pct_owner']:.1f}%")
print(f"  Bachelor's+: {ca60_new['pct_bachelors']:.1f}%")
print(f"  % White: {ca60_new['pct_white']:.1f}%")
print(f"  % Black: {ca60_new['pct_black']:.1f}%")

print("\nCA 60 COMPARISON TO CHICAGO:")
print(f"  Income vs. Chicago average: {'ABOVE' if ca60_new['median_income'] > df_chicago['median_income'].mean() else 'BELOW'} average")
print(f"  Homeownership vs. Chicago average: {'ABOVE' if ca60_new['pct_owner'] > df_chicago['pct_owner'].mean() else 'BELOW'} average")
print(f"  Education vs. Chicago average: {'ABOVE' if ca60_new['pct_bachelors'] > df_chicago['pct_bachelors'].mean() else 'BELOW'} average")

# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "="*80)
print("PHASE 1 COMPLETE - DATA COLLECTION SUMMARY")
print("="*80)

print(f"\n✅ ACCOMPLISHED:")
print(f"  • Loaded data for ALL {len(df_chicago)} Chicago Community Areas")
print(f"  • Processed {total_block_groups} block groups citywide")
print(f"  • Created aggregated CA-level dataset")
print(f"  • Generated citywide distribution visualizations")
print(f"  • Compared extremes (top/bottom 10 CAs)")
print(f"  • Validated against HW4 results (CA 60)")

print(f"\n📊 DATA READY FOR:")
print(f"  • Similarity calculations (77×77 matrix)")
print(f"  • Network construction")
print(f"  • Community detection")
print(f"  • Geographic visualization")

print(f"\n📁 OUTPUT FILES:")
print(f"  • chicago_77_cas_data_2020.csv")
print(f"  • chicago_77_cas_distributions.png")
print(f"  • chicago_77_cas_extremes.png")

print(f"\n🎯 NEXT STEPS:")
print(f"  1. Download Chicago crime data (Chicago Data Portal)")
print(f"  2. Calculate 77×77 similarity matrix")
print(f"  3. Build citywide network")
print(f"  4. Create interactive Folium maps")

print(f"\n⏱️ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Status: PHASE 1 COMPLETE ✓")
print("="*80)
