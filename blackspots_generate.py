"""
STEP 3 — Blackspot Identification
====================================
Input  : crash_data_dbscan.csv   (output of step2)
Output : blackspots.csv          (ranked dangerous cluster table)

Run: python step3_blackspot.py

Dependencies: pip install pandas numpy
"""

import pandas as pd
import numpy as np

# ── Load DBSCAN-labelled data ──────────────────────────────────────────────
df = pd.read_csv("crash_data_dbscan.csv")
print(f"Loaded: {df.shape[0]} rows")

# ── Severity numeric score (re-map in case not present) ───────────────────
sev_map = {'Damage Only': 0, 'Minor': 1, 'Grievous': 2, 'Fatal': 3}
if 'Severity_Score' not in df.columns:
    df['Severity_Score'] = df['Severity'].map(sev_map)

if 'Total_Killed' not in df.columns:
    killed_cols  = [c for c in ['Drivers_Killed','Passengers_Killed','Pedestrians_Killed'] if c in df.columns]
    df['Total_Killed'] = df[killed_cols].sum(axis=1) if killed_cols else 0

if 'Total_Injured' not in df.columns:
    injured_cols = [c for c in [
        'Drivers_Grievous_Injury','Drivers_Minor_Injury',
        'Passengers_Grievous_Injury','Passengers_Minor_Injury',
        'Pedestrians_Grievous_Injury','Pedestrians_Minor_Injury',
    ] if c in df.columns]
    df['Total_Injured'] = df[injured_cols].sum(axis=1) if injured_cols else 0

# ── Keep only clustered rows (ignore noise = -1) ──────────────────────────
clustered = df[df['DBSCAN_Cluster'] >= 0].copy()
print(f"Accidents in clusters: {len(clustered)}")
print(f"Number of clusters   : {clustered['DBSCAN_Cluster'].nunique()}")

# ══════════════════════════════════════════════════════════════════
# AGGREGATE STATS PER CLUSTER
# ══════════════════════════════════════════════════════════════════
cluster_stats = (
    clustered
    .groupby('DBSCAN_Cluster')
    .agg(
        Count          = ('Accident_ID',     'count'),
        Avg_Severity   = ('Severity_Score',  'mean'),
        Fatal_Count    = ('Total_Killed',     'sum'),
        Injured_Count  = ('Total_Injured',    'sum'),
        Lat            = ('Latitude',         'mean'),   # centroid
        Lon            = ('Longitude',        'mean'),   # centroid
        Districts      = ('District',         lambda x: ', '.join(sorted(x.dropna().unique()))),
        Top_Road       = ('Road_Name',        lambda x: x.value_counts().index[0]),
        Top_Collision  = ('Collision_Type',   lambda x: x.value_counts().index[0]),
    )
    .reset_index()
)

# ══════════════════════════════════════════════════════════════════
# BLACKSPOT SCORING
# ══════════════════════════════════════════════════════════════════
#
#  A cluster is more dangerous if:
#    (a) it has many crashes          → weight 40%
#    (b) those crashes are severe     → weight 40%
#    (c) it produced many fatalities  → weight 20%
#
#  All three components are normalised 0–1 before weighting,
#  so they contribute equally on their own scale.

count_max   = cluster_stats['Count'].max()
fatal_max   = cluster_stats['Fatal_Count'].max()

cluster_stats['Score'] = (
    (cluster_stats['Count']        / count_max)           * 0.40 +
    (cluster_stats['Avg_Severity'] / 3.0)                 * 0.40 +
    (cluster_stats['Fatal_Count']  / (fatal_max + 1))     * 0.20
)

# Sort highest → lowest score and assign rank
cluster_stats = cluster_stats.sort_values('Score', ascending=False).reset_index(drop=True)
cluster_stats['Blackspot_Rank'] = cluster_stats.index + 1

# ── Round for readability ─────────────────────────────────────────────────
cluster_stats['Avg_Severity'] = cluster_stats['Avg_Severity'].round(3)
cluster_stats['Score']        = cluster_stats['Score'].round(4)
cluster_stats['Lat']          = cluster_stats['Lat'].round(5)
cluster_stats['Lon']          = cluster_stats['Lon'].round(5)

# ── Save ──────────────────────────────────────────────────────────────────
cluster_stats.to_csv("blackspots.csv", index=False)

print("\n✅  blackspots.csv saved!")
print(f"   Total blackspots : {len(cluster_stats)}")
print(f"\n   Top 10 blackspots:")

top10 = cluster_stats[['Blackspot_Rank','Districts','Top_Road',
                        'Count','Avg_Severity','Fatal_Count','Score']].head(10)
print(top10.to_string(index=False))

# ── Quick summary ─────────────────────────────────────────────────────────
print(f"\n   Scoring formula:")
print(f"   Score = 0.40 × (crashes / max_crashes)")
print(f"         + 0.40 × (avg_severity / 3)")
print(f"         + 0.20 × (fatalities / (max_fatalities + 1))")
print(f"\n   Columns in blackspots.csv:")
print(f"   {list(cluster_stats.columns)}")