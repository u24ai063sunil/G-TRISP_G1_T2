"""
SVNIT Crash Data Analysis Dashboard
====================================
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
import os, warnings
warnings.filterwarnings("ignore")

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gujarat Crash Data Dashboard",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e2a3a; border-radius: 10px; padding: 16px;
        text-align: center; color: white;
    }
    .metric-num  { font-size: 2rem; font-weight: 700; }
    .metric-lbl  { font-size: 0.85rem; color: #aab8c8; margin-top: 4px; }
    .section-header {
        font-size: 1.1rem; font-weight: 600; color: #e07b39;
        border-left: 4px solid #e07b39; padding-left: 10px; margin: 16px 0 8px;
    }
    div[data-testid="metric-container"] { background:#1e2a3a; border-radius:10px; padding:12px; }

    /* Sticky footer for the whole page */

    /* Prevent content from being hidden behind the footer */
    .stApp { padding-bottom: 48px !important; }
</style>
""", unsafe_allow_html=True)

# ─── Load Data ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.dirname(__file__)
    df      = pd.read_csv(os.path.join(base, "crash_data_clean.csv"))
    df_sp   = pd.read_csv(os.path.join(base, "crash_data_dbscan.csv"))
    bs      = pd.read_csv(os.path.join(base, "blackspots.csv"))
    Z       = np.load(os.path.join(base, "kde_Z.npy"))
    lat_g   = np.load(os.path.join(base, "kde_lat_grid.npy"))
    lon_g   = np.load(os.path.join(base, "kde_lon_grid.npy"))
    return df, df_sp, bs, Z, lat_g, lon_g

df_full, df_sp, blackspots, kde_Z, lat_grid, lon_grid = load_data()

# ─── Sidebar Filters ─────────────────────────────────────────────────────────
st.sidebar.image("svnit_logo.jpg", width=80)
st.sidebar.title("🔍 Filters")

years = sorted(df_full['Year'].dropna().unique())
sel_years = st.sidebar.multiselect("Year", years, default=years)

districts = sorted(df_full['District'].dropna().unique())
sel_dist = st.sidebar.multiselect("District", districts, default=districts)

severities = sorted(df_full['Severity'].dropna().unique())
sel_sev = st.sidebar.multiselect("Severity", severities, default=severities)

# Apply filters
df = df_full[
    df_full['Year'].isin(sel_years) &
    df_full['District'].isin(sel_dist) &
    df_full['Severity'].isin(sel_sev)
].copy()

dfs = df_sp[
    df_sp['Year'].isin(sel_years) &
    df_sp['District'].isin(sel_dist) &
    df_sp['Severity'].isin(sel_sev)
].copy()

st.sidebar.markdown(f"**{len(df):,}** records selected")

# ─── Header ──────────────────────────────────────────────────────────────────

# ─── Centered Header ─────────────────────────────────────────────
st.markdown(
    '''
    <div style="text-align:center; margin-top: -30px; margin-bottom: 0;">
        <h1 style="font-size:2.5rem; margin-bottom:0.2em;">🚨 Gujarat Crash Data Analysis Dashboard</h1>
        <div style="font-size:1.1rem; color:#e07b39; font-weight:500;">SVNIT Transportation Cell | Task-2 | Data: 2021–2024</div>
    </div>
    ''', unsafe_allow_html=True
)
# Add this at the very end of the file for the footer

# ─── Footer (Sticky, for whole page) ─────────────────────────────


# ═══════════════════════════════════════════════════════════════
# TAB LAYOUT
# ═══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "📈 Trends",
    "🗺️ Spatial Map",
    "🔴 Blackspots",
    "🛈 About"
])


# ─────────────────────────────────────────────────────────────────
# TAB 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────────
with tab1:
    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    total      = len(df)
    fatal      = (df['Severity'] == 'Fatal').sum()
    killed     = int(df['Total_Killed'].sum())
    injured    = int(df['Total_Injured'].sum())
    fatal_pct  = round(fatal / total * 100, 1) if total else 0

    c1.metric("Total Accidents",  f"{total:,}")
    c2.metric("Fatal Accidents",  f"{fatal:,}",  f"{fatal_pct}% of total")
    c3.metric("Total Killed",     f"{killed:,}")
    c4.metric("Total Injured",    f"{injured:,}")
    c5.metric("Avg Severity",     f"{df['Severity_Score'].mean():.2f} / 3")

    st.divider()

    # Row 1
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.markdown('<div class="section-header">Accidents by Severity</div>', unsafe_allow_html=True)
        sev_order = ['Fatal','Grievous','Minor','Damage Only']
        sev_colors = {'Fatal':'#e63946','Grievous':'#f4a261','Minor':'#2a9d8f','Damage Only':'#457b9d'}
        sev_df = df['Severity'].value_counts().reindex(sev_order).reset_index()
        sev_df.columns = ['Severity','Count']
        fig = px.bar(sev_df, x='Severity', y='Count', color='Severity',
                     color_discrete_map=sev_colors, text='Count')
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False, height=320, margin=dict(t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with r1c2:
        st.markdown('<div class="section-header">Accidents by District</div>', unsafe_allow_html=True)
        dist_df = df.groupby('District').agg(
            Count=('Accident_ID','count'),
            Fatal=('Severity', lambda x: (x=='Fatal').sum())
        ).reset_index().sort_values('Count', ascending=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(y=dist_df['District'], x=dist_df['Count'],
                              orientation='h', name='Total', marker_color='#457b9d'))
        fig2.add_trace(go.Bar(y=dist_df['District'], x=dist_df['Fatal'],
                              orientation='h', name='Fatal', marker_color='#e63946'))
        fig2.update_layout(barmode='overlay', height=320, margin=dict(t=10,b=0),
                           legend=dict(orientation='h', y=1.05))
        st.plotly_chart(fig2, use_container_width=True)

    # Row 2
    r2c1, r2c2, r2c3 = st.columns(3)

    with r2c1:
        st.markdown('<div class="section-header">Collision Types</div>', unsafe_allow_html=True)
        ct = df['Collision_Type'].value_counts().head(8)
        fig3 = px.pie(values=ct.values, names=ct.index, hole=0.45,
                      color_discrete_sequence=px.colors.qualitative.Set2)
        fig3.update_layout(height=300, margin=dict(t=0,b=0,l=0,r=0),
                           legend=dict(font_size=10))
        st.plotly_chart(fig3, use_container_width=True)

    with r2c2:
        st.markdown('<div class="section-header">Weather Conditions</div>', unsafe_allow_html=True)
        wc = df[df['Weather_Condition'] != 'Unknown']['Weather_Condition'].value_counts()
        fig4 = px.pie(values=wc.values, names=wc.index, hole=0.45,
                      color_discrete_sequence=px.colors.qualitative.Pastel)
        fig4.update_layout(height=300, margin=dict(t=0,b=0,l=0,r=0),
                           legend=dict(font_size=10))
        st.plotly_chart(fig4, use_container_width=True)

    with r2c3:
        st.markdown('<div class="section-header">Top Traffic Violations</div>', unsafe_allow_html=True)
        tv = df[df['Traffic_Violation'] != 'Unknown']['Traffic_Violation'].value_counts().head(8)
        fig5 = px.bar(x=tv.values, y=tv.index, orientation='h',
                      color=tv.values, color_continuous_scale='Reds', text=tv.values)
        fig5.update_traces(textposition='outside')
        fig5.update_layout(height=300, margin=dict(t=0,b=0), showlegend=False,
                           coloraxis_showscale=False, yaxis_title='')
        st.plotly_chart(fig5, use_container_width=True)


# ─────────────────────────────────────────────────────────────────
# TAB 2 — TRENDS
# ─────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">Monthly Accident Trend by Year</div>', unsafe_allow_html=True)
    df['YearMonth'] = pd.to_datetime(df[['Year','Month']].assign(day=1))
    monthly = df.groupby(['YearMonth','Severity'])['Accident_ID'].count().reset_index()
    monthly.columns = ['YearMonth','Severity','Count']
    fig_trend = px.line(monthly, x='YearMonth', y='Count', color='Severity',
                        color_discrete_map={'Fatal':'#e63946','Grievous':'#f4a261',
                                            'Minor':'#2a9d8f','Damage Only':'#457b9d'},
                        markers=True)
    fig_trend.update_layout(height=340, margin=dict(t=10,b=0))
    st.plotly_chart(fig_trend, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Accidents by Hour of Day</div>', unsafe_allow_html=True)
        hourly = df.groupby('Hour')['Accident_ID'].count().reset_index()
        hourly.columns = ['Hour','Count']
        fig_h = px.bar(hourly, x='Hour', y='Count',
                       color='Count', color_continuous_scale='OrRd',
                       labels={'Hour':'Hour of Day','Count':'Accidents'})
        fig_h.update_layout(height=300, margin=dict(t=10,b=0), coloraxis_showscale=False)
        st.plotly_chart(fig_h, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Accidents by Day of Week</div>', unsafe_allow_html=True)
        dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        dow = df['DayOfWeek'].value_counts().reindex(dow_order).reset_index()
        dow.columns = ['Day','Count']
        fig_d = px.bar(dow, x='Day', y='Count',
                       color='Count', color_continuous_scale='Blues',
                       labels={'Day':'','Count':'Accidents'})
        fig_d.update_layout(height=300, margin=dict(t=10,b=0), coloraxis_showscale=False)
        st.plotly_chart(fig_d, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-header">Heatmap: Hour vs Day of Week</div>', unsafe_allow_html=True)
        pivot = df.pivot_table(index='DayOfWeek', columns='Hour',
                               values='Accident_ID', aggfunc='count', fill_value=0)
        pivot = pivot.reindex(dow_order)
        fig_hw = px.imshow(pivot, color_continuous_scale='YlOrRd',
                           labels=dict(x='Hour of Day', y='', color='Accidents'),
                           aspect='auto')
        fig_hw.update_layout(height=300, margin=dict(t=10,b=0))
        st.plotly_chart(fig_hw, use_container_width=True)

    with col4:
        st.markdown('<div class="section-header">Light Condition vs Severity</div>', unsafe_allow_html=True)
        lc = df.groupby(['Light_Condition','Severity'])['Accident_ID'].count().reset_index()
        lc.columns = ['Light_Condition','Severity','Count']
        fig_lc = px.bar(lc, x='Light_Condition', y='Count', color='Severity',
                        barmode='stack',
                        color_discrete_map={'Fatal':'#e63946','Grievous':'#f4a261',
                                            'Minor':'#2a9d8f','Damage Only':'#457b9d'})
        fig_lc.update_layout(height=300, margin=dict(t=10,b=0),
                             xaxis_tickangle=-25)
        st.plotly_chart(fig_lc, use_container_width=True)

    st.markdown('<div class="section-header">Yearly Casualty Trend</div>', unsafe_allow_html=True)
    yr = df.groupby('Year').agg(
        Accidents=('Accident_ID','count'),
        Killed=('Total_Killed','sum'),
        Injured=('Total_Injured','sum')
    ).reset_index()
    fig_yr = make_subplots(specs=[[{"secondary_y": True}]])
    fig_yr.add_trace(go.Bar(x=yr['Year'], y=yr['Accidents'], name='Accidents',
                            marker_color='#457b9d'), secondary_y=False)
    fig_yr.add_trace(go.Scatter(x=yr['Year'], y=yr['Killed'], mode='lines+markers',
                                name='Killed', line=dict(color='#e63946', width=2)), secondary_y=True)
    fig_yr.add_trace(go.Scatter(x=yr['Year'], y=yr['Injured'], mode='lines+markers',
                                name='Injured', line=dict(color='#f4a261', width=2)), secondary_y=True)
    fig_yr.update_layout(height=300, margin=dict(t=10,b=0),
                         legend=dict(orientation='h',y=1.1))
    st.plotly_chart(fig_yr, use_container_width=True)


# ─────────────────────────────────────────────────────────────────
# TAB 3 — SPATIAL MAP
# ─────────────────────────────────────────────────────────────────
with tab3:
    map_mode = st.radio("Map type", ["KDE Heatmap", "Crash Markers", "DBSCAN Clusters"],
                        horizontal=True)

    m = folium.Map(location=[22.5, 71.5], zoom_start=7,
                   tiles='CartoDB dark_matter')

    if map_mode == "KDE Heatmap":
        st.info("🌡️ Brighter = higher crash density (KDE). Computed on full spatial dataset.")
        # Sample coords for HeatMap plugin
        heat_data = [[row.Latitude, row.Longitude, 1]
                     for _, row in dfs[['Latitude','Longitude']].dropna().iterrows()]
        HeatMap(heat_data, radius=18, blur=25, max_zoom=10,
                gradient={0.2:'blue',0.5:'yellow',1.0:'red'}).add_to(m)

    elif map_mode == "Crash Markers":
        st.info("📍 Individual crash points. Colour = severity.")
        sev_color = {'Fatal':'red','Grievous':'orange','Minor':'blue','Damage Only':'green'}
        cluster = MarkerCluster().add_to(m)
        for _, row in dfs.dropna(subset=['Latitude','Longitude']).iterrows():
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=5,
                color=sev_color.get(row['Severity'],'gray'),
                fill=True, fill_opacity=0.7,
                popup=folium.Popup(
                    f"<b>{row['Severity']}</b><br>{row['Road_Name']}<br>"
                    f"{row['District']}<br>{row['Accident_DateTime']}", max_width=200)
            ).add_to(cluster)

    else:  # DBSCAN Clusters
        st.info("🔵 Each colour = one DBSCAN cluster. Grey = noise points.")
        import random
        random.seed(42)
        clustered = dfs[dfs['DBSCAN_Cluster'] >= 0].dropna(subset=['Latitude','Longitude'])
        noise     = dfs[dfs['DBSCAN_Cluster'] == -1].dropna(subset=['Latitude','Longitude'])

        # Noise points
        for _, row in noise.sample(min(200, len(noise))).iterrows():
            folium.CircleMarker([row['Latitude'], row['Longitude']],
                                radius=3, color='gray', fill=True,
                                fill_opacity=0.3).add_to(m)

        # Cluster points (sample top 30 clusters for speed)
        top_cl = clustered['DBSCAN_Cluster'].value_counts().head(30).index
        colors = ['#e63946','#f4a261','#2a9d8f','#457b9d','#6d6875',
                  '#b5e48c','#ffd60a','#48cae4','#c77dff','#ff6b6b'] * 3
        for i, cl_id in enumerate(top_cl):
            cl_rows = clustered[clustered['DBSCAN_Cluster'] == cl_id]
            for _, row in cl_rows.iterrows():
                folium.CircleMarker([row['Latitude'], row['Longitude']],
                                    radius=5, color=colors[i], fill=True,
                                    fill_opacity=0.7).add_to(m)

    st_folium(m, width=None, height=520, returned_objects=[])


# ─────────────────────────────────────────────────────────────────
# TAB 4 — BLACKSPOTS
# ─────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("### 🔴 Top Accident Blackspots (DBSCAN + Severity Scoring)")
    st.caption("Blackspot Score = 40% crash count + 40% avg severity + 20% fatalities (all normalised)")

    top_n = st.slider("Show top N blackspots", 5, 30, 15)
    bs_show = blackspots.head(top_n).copy()

    # Map of blackspots
    bm = folium.Map(location=[22.5, 71.5], zoom_start=7, tiles='CartoDB positron')

    for _, row in bs_show.iterrows():
        rank   = int(row['Blackspot_Rank'])
        color  = '#e63946' if rank <= 3 else ('#f4a261' if rank <= 8 else '#2a9d8f')
        folium.CircleMarker(
            location=[row['Lat'], row['Lon']],
            radius=max(10, min(30, int(row['Count'] / 3))),
            color=color, fill=True, fill_opacity=0.7,
            popup=folium.Popup(
                f"<b>Rank #{rank}</b><br>"
                f"Road: {row['Top_Road']}<br>"
                f"Crashes: {int(row['Count'])}<br>"
                f"Deaths: {int(row['Fatal_Count'])}<br>"
                f"Score: {row['Score']:.3f}", max_width=200)
        ).add_to(bm)
        folium.Marker(
            location=[row['Lat'], row['Lon']],
            icon=folium.DivIcon(
                html=f'<div style="font-size:11px;font-weight:bold;color:white;'
                     f'background:{color};border-radius:50%;width:22px;height:22px;'
                     f'display:flex;align-items:center;justify-content:center;">{rank}</div>',
                icon_size=(22, 22), icon_anchor=(11, 11))
        ).add_to(bm)

    st_folium(bm, width=None, height=460, returned_objects=[])

    # Blackspot table
    st.markdown('<div class="section-header">Blackspot Details Table</div>', unsafe_allow_html=True)
    display_cols = ['Blackspot_Rank','Districts','Top_Road','Top_Collision',
                    'Count','Avg_Severity','Fatal_Count','Injured_Count','Score']
    bs_display = bs_show[display_cols].copy()
    bs_display.columns = ['Rank','Districts','Top Road','Top Collision',
                          'Crashes','Avg Severity','Deaths','Injuries','Score']
    bs_display['Avg Severity'] = bs_display['Avg Severity'].round(2)
    bs_display['Score']        = bs_display['Score'].round(3)

    st.dataframe(
        bs_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=1, format="%.3f"),
            "Crashes": st.column_config.NumberColumn("Crashes", format="%d"),
            "Deaths":  st.column_config.NumberColumn("Deaths",  format="%d"),
        }
    )

    # Bar chart of top blackspots
    col1, col2 = st.columns(2)
    with col1:
        fig_bs = px.bar(bs_show.head(15), x='Blackspot_Rank', y='Count',
                        color='Avg_Severity', color_continuous_scale='Reds',
                        labels={'Count':'Crashes','Blackspot_Rank':'Rank'},
                        title='Crashes per Blackspot')
        fig_bs.update_layout(height=300, margin=dict(t=40,b=0))
        st.plotly_chart(fig_bs, use_container_width=True)
    with col2:
        fig_sc = px.scatter(bs_show.head(20), x='Count', y='Avg_Severity',
                            size='Fatal_Count', color='Score',
                            color_continuous_scale='OrRd',
                            hover_data=['Top_Road','Districts'],
                            labels={'Count':'Crash Count','Avg_Severity':'Avg Severity'},
                            title='Count vs Severity (bubble = fatalities)')
        fig_sc.update_layout(height=300, margin=dict(t=40,b=0))
        st.plotly_chart(fig_sc, use_container_width=True)


# ─────────────────────────────────────────────────────────────────
# TAB 5 — ABOUT
# ─────────────────────────────────────────────────────────────────
with tab5:
    st.markdown("""
### About This Dashboard

**Project**: SVNIT Task-2 — Crash Data Dashboard Development  
**Data**: Gujarat accident records, 2021–2024 (1500 records, 25 fields)

#### Methods Used

| Step | Method | Detail |
|------|--------|--------|
| Cleaning | Pandas | Nulls filled, datetime parsed, severity scored 0-3 |
| Descriptive | Plotly | Severity, district, collision type, weather, violations |
| Trend Analysis | Plotly | Monthly, hourly, day-of-week, yearly casualty trends |
| Spatial | Folium + HeatMap | KDE crash density overlay on interactive map |
| Clustering | DBSCAN (scikit-learn) | eps=15km, min_samples=4, haversine metric |
| KDE | scipy gaussian_kde | bw=0.08, 200×200 grid |
| Blackspot ID | Custom Scoring | 40% count + 40% avg severity + 20% fatalities |

#### DBSCAN Parameters
- **eps = 15 km** — crashes within 15 km are considered the same cluster
- **min_samples = 4** — minimum 4 accidents to form a cluster
- **Metric** — haversine (accurate for lat/lon coordinates)
- **Result** — 108 clusters identified from 1,436 geotagged accidents

#### Files
- `crash_data_clean.csv` — cleaned full dataset (1500 rows)
- `crash_data_dbscan.csv` — with DBSCAN cluster labels (1436 rows)
- `blackspots.csv` — ranked blackspot table (108 clusters)
- `kde_Z.npy`, `kde_lat_grid.npy`, `kde_lon_grid.npy` — KDE surface arrays

#### How to Use
1. Use sidebar **filters** (Year / District / Severity) — all charts update live
2. **Overview tab** — KPI metrics and distribution charts
3. **Trends tab** — temporal patterns
4. **Spatial Map tab** — switch between KDE / Markers / Clusters
5. **Blackspots tab** — ranked dangerous locations with interactive map
    """)
