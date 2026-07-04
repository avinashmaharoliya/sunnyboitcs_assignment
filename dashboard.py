from __future__ import annotations

from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from sunnybotics.config import CONDITION_COLORS_HEX, FARM_CONFIG


RESULTS_PATH = Path("outputs/results.csv")


@st.cache_data
def load_results(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


st.set_page_config(page_title="Sunnybotics Inspection", layout="wide")
st.title("Sunnybotics Inspection")

if not RESULTS_PATH.exists():
    st.warning("Run `python main.py` to generate outputs/results.csv.")
    st.stop()

df = load_results(str(RESULTS_PATH))
conditions = sorted(df["condition"].dropna().unique().tolist())

with st.sidebar:
    selected_conditions = st.multiselect("Condition", conditions, default=conditions)
    min_priority = st.slider("Minimum Priority", 0, 100, 0)
    gps_statuses = sorted(df["gps_status"].dropna().unique().tolist())
    selected_gps = st.multiselect("GPS Status", gps_statuses, default=gps_statuses)

filtered = df[
    df["condition"].isin(selected_conditions)
    & (df["cleaning_priority_score"] >= min_priority)
    & df["gps_status"].isin(selected_gps)
].copy()

metric_cols = st.columns(4)
metric_cols[0].metric("Images", len(filtered))
metric_cols[1].metric("Critical", int((filtered["cleaning_priority_score"] >= 70).sum()))
metric_cols[2].metric("Review", int(filtered["condition"].isin(["uncertain", "glare", "shadowed"]).sum()))
metric_cols[3].metric("Avg Priority", round(float(filtered["cleaning_priority_score"].mean()), 1) if len(filtered) else 0)

map_col, table_col = st.columns([1.1, 1.4], gap="large")

with map_col:
    center_lat = float(filtered["latitude"].mean()) if len(filtered) else FARM_CONFIG.center_latitude
    center_lon = float(filtered["longitude"].mean()) if len(filtered) else FARM_CONFIG.center_longitude
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=19, tiles="OpenStreetMap")

    for _, row in filtered.iterrows():
        color = CONDITION_COLORS_HEX.get(str(row["condition"]), "#737373")
        popup = folium.Popup(
            "<b>%s</b><br>Condition: %s<br>Confidence: %.2f<br>Priority: %s<br>GPS: %s<br>Issues: %s"
            % (
                row["panel_id"],
                row["condition"],
                row["confidence"],
                row["cleaning_priority_score"],
                row["gps_status"],
                row["detected_issues"],
            ),
            max_width=320,
        )
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=popup,
        ).add_to(fmap)

    st_folium(fmap, use_container_width=True, height=620)

with table_col:
    display_cols = [
        "panel_id",
        "condition",
        "confidence",
        "cleaning_priority_score",
        "gps_status",
        "detected_issues",
        "annotated_image_path",
        "timestamp",
    ]
    st.dataframe(
        filtered.sort_values("cleaning_priority_score", ascending=False)[display_cols],
        use_container_width=True,
        hide_index=True,
    )
