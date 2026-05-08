"""
STEP 1 — Data Cleaning
========================
Input  : accident_dummy_data__1_.xlsx
Output : crash_data_clean.csv

Run: python step1_data_cleaning.py
"""

import pandas as pd
import numpy as np

# ── Load raw Excel ─────────────────────────────────────────────────────────
df = pd.read_excel("accident_dummy_data.xlsx")
print(f"Raw data loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# ── 1. Parse DateTime ──────────────────────────────────────────────────────
df['Accident_DateTime'] = pd.to_datetime(df['Accident_DateTime'], dayfirst=True, errors='coerce')

# Extract time parts
df['Date']       = df['Accident_DateTime'].dt.date
df['Year']       = df['Accident_DateTime'].dt.year
df['Month']      = df['Accident_DateTime'].dt.month
df['Hour']       = df['Accident_DateTime'].dt.hour
df['DayOfWeek']  = df['Accident_DateTime'].dt.day_name()
df['Month_Name'] = df['Accident_DateTime'].dt.strftime('%b')

print(f"Year range   : {df['Year'].min()} – {df['Year'].max()}")

# ── 2. Fill numeric nulls with 0 (injury / vehicle counts) ────────────────
count_cols = [
    'No_of_Vehicles',
    'Drivers_Killed', 'Drivers_Grievous_Injury', 'Drivers_Minor_Injury',
    'Passengers_Killed', 'Passengers_Grievous_Injury', 'Passengers_Minor_Injury',
    'Pedestrians_Killed', 'Pedestrians_Grievous_Injury', 'Pedestrians_Minor_Injury',
]
# Only fill columns that actually exist in the dataset
count_cols = [c for c in count_cols if c in df.columns]
df[count_cols] = df[count_cols].fillna(0).astype(int)

# ── 3. Fill categorical nulls with 'Unknown' ──────────────────────────────
cat_cols = ['Weather_Condition', 'Collision_Feature', 'Visibility', 'Traffic_Violation']
cat_cols = [c for c in cat_cols if c in df.columns]
for c in cat_cols:
    df[c] = df[c].fillna('Unknown')

# ── 4. Severity numeric score ─────────────────────────────────────────────
#   Damage Only = 0 | Minor = 1 | Grievous = 2 | Fatal = 3
sev_map = {'Damage Only': 0, 'Minor': 1, 'Grievous': 2, 'Fatal': 3}
df['Severity_Score'] = df['Severity'].map(sev_map)

# ── 5. Total casualties per accident ──────────────────────────────────────
killed_cols  = [c for c in ['Drivers_Killed', 'Passengers_Killed', 'Pedestrians_Killed'] if c in df.columns]
injured_cols = [c for c in [
    'Drivers_Grievous_Injury', 'Drivers_Minor_Injury',
    'Passengers_Grievous_Injury', 'Passengers_Minor_Injury',
    'Pedestrians_Grievous_Injury', 'Pedestrians_Minor_Injury',
] if c in df.columns]

df['Total_Killed']  = df[killed_cols].sum(axis=1)  if killed_cols  else 0
df['Total_Injured'] = df[injured_cols].sum(axis=1) if injured_cols else 0

# ── 6. Time-of-day category ───────────────────────────────────────────────
def time_category(h):
    if pd.isna(h):       return 'Unknown'
    if 6  <= h < 12:     return 'Morning'
    elif 12 <= h < 18:   return 'Afternoon'
    elif 18 <= h < 22:   return 'Evening'
    else:                return 'Night'

df['Time_Category'] = df['Hour'].apply(time_category)

# ── 7. Save ───────────────────────────────────────────────────────────────
df.to_csv("crash_data_clean.csv", index=False)

print("\n✅  crash_data_clean.csv saved!")
print(f"   Shape            : {df.shape}")
print(f"   Severity counts  :\n{df['Severity'].value_counts().to_string()}")
print(f"   Null lat/lon rows: {df[['Latitude','Longitude']].isnull().any(axis=1).sum()}")
print(f"   New columns added: Date, Year, Month, Hour, DayOfWeek, Month_Name,")
print(f"                      Severity_Score, Total_Killed, Total_Injured, Time_Category")