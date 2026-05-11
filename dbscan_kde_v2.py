"""
STEP 2 — DBSCAN Spatial Clustering
=====================================
Input  : crash_data_clean.csv   (output of step1)
Output : crash_data_dbscan.csv  (same rows + DBSCAN_Cluster column)
         kde_Z.npy, kde_lat_grid.npy, kde_lon_grid.npy  (KDE surface)

Run: python step2_dbscan_kde.py

Dependencies: pip install pandas numpy scikit-learn scipy
"""

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.stats import gaussian_kde

# ── Load cleaned data ──────────────────────────────────────────────────────
df = pd.read_csv("crash_data_v2.csv")
print(f"Loaded: {df.shape[0]} rows")

# ── Keep only rows with valid coordinates ─────────────────────────────────
df_spatial = df.dropna(subset=['Latitude', 'Longitude']).copy()
print(f"Rows with valid lat/lon: {len(df_spatial)} (dropped {len(df) - len(df_spatial)})")

coords = df_spatial[['Latitude', 'Longitude']].values

# ══════════════════════════════════════════════════════════════════
# DBSCAN CLUSTERING
# ══════════════════════════════════════════════════════════════════
#
#  Why DBSCAN?
#  - Doesn't need you to specify number of clusters upfront
#  - Finds clusters of any shape (road corridors, junctions, etc.)
#  - Handles noise (isolated accidents labelled as -1)
#  - Uses haversine metric → accurate for lat/lon coordinates
#
#  Parameters chosen:
#  - eps = 15 km  →  crashes within 15 km belong to same cluster
#    (tuned by testing 5/8/10/15/20 km; 15 km gave 108 clusters
#     which is meaningful for a Gujarat-wide dataset)
#  - min_samples = 4  →  at least 4 accidents needed to form a cluster

EARTH_RADIUS_KM = 6371.0088
EPS_KM          = 15          # cluster radius in km
MIN_SAMPLES     = 4           # minimum points per cluster

eps_radians = EPS_KM / EARTH_RADIUS_KM

coords_rad = np.radians(coords)

db = DBSCAN(
    eps        = eps_radians,
    min_samples= MIN_SAMPLES,
    algorithm  = 'ball_tree',
    metric     = 'haversine',
)
labels = db.fit_predict(coords_rad)

df_spatial['DBSCAN_Cluster'] = labels

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise    = (labels == -1).sum()
print(f"\nDBSCAN results:")
print(f"  eps         = {EPS_KM} km")
print(f"  min_samples = {MIN_SAMPLES}")
print(f"  Clusters    = {n_clusters}")
print(f"  Noise pts   = {n_noise}")
print(f"  Clustered   = {len(df_spatial) - n_noise}")

# ══════════════════════════════════════════════════════════════════
# KERNEL DENSITY ESTIMATION (KDE)
# ══════════════════════════════════════════════════════════════════
#
#  Why KDE?
#  - Turns individual point locations into a smooth density surface
#  - Each crash contributes a "bump"; overlapping bumps = hot zone
#  - Result is a heatmap showing where crash DENSITY is highest
#  - More informative than raw dots — shows the overall risk shape
#
#  bw_method = 0.08:
#  - Controls smoothness (higher = broader, softer peaks)
#  - 0.08 works well for Gujarat's geographic spread

lat = df_spatial['Latitude'].values
lon = df_spatial['Longitude'].values

# Create a 200×200 grid covering Gujarat's bounding box
lat_grid = np.linspace(lat.min(), lat.max(), 200)
lon_grid = np.linspace(lon.min(), lon.max(), 200)
lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

kde = gaussian_kde(np.vstack([lat, lon]), bw_method=0.08)
Z   = kde(np.vstack([lat_mesh.ravel(), lon_mesh.ravel()])).reshape(lat_mesh.shape)

print(f"\nKDE computed on {len(lat)} points, bw=0.08, grid=200×200")
print(f"  Density range: {Z.min():.6f} – {Z.max():.6f}")

# ── Save outputs ───────────────────────────────────────────────────────────
df_spatial.to_csv("crash_data_dbscan_v2.csv", index=False)
np.save("kde_Z.npy",        Z)
np.save("kde_lat_grid.npy", lat_grid)
np.save("kde_lon_grid.npy", lon_grid)

print("\n✅  Saved:")
print("   crash_data_dbscan.csv")
print("   kde_Z.npy")
print("   kde_lat_grid.npy")
print("   kde_lon_grid.npy")