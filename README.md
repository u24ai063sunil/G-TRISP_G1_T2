# Gujarat Crash Data Analysis Dashboard
**SVNIT Task-2 | Transportation Cell**

An interactive dashboard for road accident analysis covering 1,500 accident records from 10 Gujarat districts (2021–2024).

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the dashboard
streamlit run app.py
```

The dashboard opens at `http://localhost:8501` in your browser.

---

## Project Structure

```
crash_dashboard/
├── app.py                   ← Main Streamlit dashboard
├── requirements.txt         ← Python dependencies
├── crash_data_clean.csv     ← Cleaned full dataset (1500 rows, 35 cols)
├── crash_data_dbscan.csv    ← Dataset with DBSCAN cluster labels (1436 rows)
├── blackspots.csv           ← Ranked blackspot table (108 clusters)
├── kde_Z.npy                ← KDE density surface (200×200 grid)
├── kde_lat_grid.npy         ← Latitude grid for KDE
├── kde_lon_grid.npy         ← Longitude grid for KDE
└── README.md
```

---

## Dashboard Tabs

| Tab | Contents |
|-----|----------|
| **Overview** | KPI metrics, severity bar chart, district bar chart, collision types, weather, traffic violations |
| **Trends** | Monthly trend, hourly distribution, day-of-week, heatmap (hour × day), light condition vs severity, yearly casualty trend |
| **Spatial Map** | Folium interactive map with 3 modes: KDE Heatmap / Crash Markers / DBSCAN Clusters |
| **Blackspots** | Ranked dangerous locations on map, sortable detail table, bubble chart |
| **About** | Methods, parameters, file descriptions |

---

## Methods

### 1. Data Cleaning
- DateTime parsed (`dayfirst=True`)
- 64 rows with missing coordinates dropped for spatial analysis
- Injury/vehicle count nulls → filled with 0
- Categorical nulls (weather, visibility, etc.) → filled with `"Unknown"`
- Added: `Year`, `Month`, `Hour`, `DayOfWeek`, `Time_Category`, `Total_Killed`, `Total_Injured`, `Severity_Score` (0–3)

### 2. DBSCAN Clustering
- **Library**: `sklearn.cluster.DBSCAN`
- **Metric**: Haversine (accurate for geographic coordinates)
- **eps**: 15 km (`15 / 6371.0088` radians)
- **min_samples**: 4
- **Result**: 108 clusters from 1,436 geotagged accidents

### 3. Kernel Density Estimation (KDE)
- **Library**: `scipy.stats.gaussian_kde`
- **Bandwidth**: 0.08
- **Grid**: 200 × 200 over Gujarat bounding box
- Used for the heatmap layer in the Spatial Map tab

### 4. Blackspot Scoring
Each DBSCAN cluster is scored as:

```
Score = 0.4 × (count / max_count)
      + 0.4 × (avg_severity / 3)
      + 0.2 × (fatalities / max_fatalities)
```

Top-ranked blackspots are displayed on the map with size proportional to crash count.

---

## Sidebar Filters
All charts respond live to:
- **Year**: 2021, 2022, 2023, 2024
- **District**: Any of 10 Gujarat districts
- **Severity**: Fatal / Grievous / Minor / Damage Only

---

## Deploy Online (Free)

1. Push this folder to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → select `app.py` → Deploy

---

## Key Findings (from 1500 records)
- **607 Minor**, **358 Grievous**, **316 Damage Only**, **219 Fatal** accidents
- **NH-53** and **SH-6** are the top blackspot corridors
- Most accidents occur at **Night with no street light**
- **Over Speeding** and **Drunk Driving** are the leading violations
- **108 crash clusters** identified across Gujarat

---

*Built for SVNIT Transportation Cell — Task 2 Submission*
