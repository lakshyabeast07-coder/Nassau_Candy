# app.py

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Logistics Dashboard", layout="wide")

st.title("🚚 Logistics & Shipping Performance Dashboard")

# --------------------------------------------------
# DATA GENERATION (SYNTHETIC)
# --------------------------------------------------
@st.cache_data
def generate_data(n=3000):
    np.random.seed(42)

    regions = {
        "North": ["Delhi", "Punjab", "Haryana"],
        "South": ["Karnataka", "Tamil Nadu", "Telangana"],
        "West": ["Maharashtra", "Gujarat", "Rajasthan"],
        "East": ["West Bengal", "Odisha", "Bihar"]
    }

    ship_modes = ["Standard", "Express", "Same Day"]

    # Approximate lat/lon centers
    geo_map = {
        "Delhi": (28.61, 77.20),
        "Punjab": (31.14, 75.34),
        "Haryana": (29.06, 76.08),
        "Karnataka": (15.31, 75.71),
        "Tamil Nadu": (11.12, 78.65),
        "Telangana": (18.11, 79.01),
        "Maharashtra": (19.75, 75.71),
        "Gujarat": (22.25, 71.19),
        "Rajasthan": (27.02, 74.21),
        "West Bengal": (22.98, 87.85),
        "Odisha": (20.95, 85.09),
        "Bihar": (25.09, 85.31),
    }

    base_date = datetime(2024, 1, 1)
    data = []

    for i in range(n):
        region = np.random.choice(list(regions.keys()))
        state = np.random.choice(regions[region])
        ship_mode = np.random.choice(ship_modes, p=[0.6, 0.3, 0.1])

        ship_date = base_date + timedelta(days=np.random.randint(0, 365))

        # Mode-based lead time logic
        if ship_mode == "Same Day":
            lead_time = np.random.randint(0, 2)
        elif ship_mode == "Express":
            lead_time = np.random.randint(1, 4)
        else:
            lead_time = np.random.randint(3, 10)

        # Inject delays
        if np.random.rand() < 0.15:
            lead_time += np.random.randint(3, 6)

        delivery_date = ship_date + timedelta(days=lead_time)

        lat, lon = geo_map[state]
        lat += np.random.normal(0, 0.5)
        lon += np.random.normal(0, 0.5)

        route = f"{state}-{region}"

        data.append([
            f"ORD-{10000+i}",
            ship_date,
            delivery_date,
            region,
            state,
            route,
            ship_mode,
            lead_time,
            lat,
            lon
        ])

    df = pd.DataFrame(data, columns=[
        "Order ID", "Ship Date", "Delivery Date",
        "Region", "State", "Route", "Ship Mode",
        "Lead Time", "Latitude", "Longitude"
    ])

    return df


df = generate_data()

# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------
st.sidebar.header("🔎 Filters")

date_range = st.sidebar.date_input(
    "Date Range",
    [df["Ship Date"].min(), df["Ship Date"].max()]
)

selected_regions = st.sidebar.multiselect(
    "Region",
    df["Region"].unique(),
    default=df["Region"].unique()
)

selected_modes = st.sidebar.multiselect(
    "Ship Mode",
    df["Ship Mode"].unique(),
    default=df["Ship Mode"].unique()
)

lead_threshold = st.sidebar.slider(
    "Minimum Lead Time (Days)",
    0, 15, 0
)

# --------------------------------------------------
# APPLY FILTERS
# --------------------------------------------------
filtered_df = df[
    (df["Ship Date"] >= pd.to_datetime(date_range[0])) &
    (df["Ship Date"] <= pd.to_datetime(date_range[1])) &
    (df["Region"].isin(selected_regions)) &
    (df["Ship Mode"].isin(selected_modes)) &
    (df["Lead Time"] >= lead_threshold)
]

# --------------------------------------------------
# KPI METRICS
# --------------------------------------------------
st.subheader("📊 Key Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Total Orders", len(filtered_df))
col2.metric("Avg Lead Time", round(filtered_df["Lead Time"].mean(), 2))
col3.metric("Delayed Shipments (>7 days)",
            len(filtered_df[filtered_df["Lead Time"] > 7]))

# --------------------------------------------------
# 1. ROUTE EFFICIENCY OVERVIEW
# --------------------------------------------------
st.header("📍 Route Efficiency Overview")

route_perf = filtered_df.groupby("Route")["Lead Time"].mean().reset_index()

col1, col2 = st.columns(2)

fig1 = bar(
    route_perf.sort_values("Lead Time", ascending=False),
    x="Route", y="Lead Time",
    title="Average Lead Time by Route"
)
col1.plotly_chart(fig1, use_container_width=True)

fig2 = bar(
    route_perf.sort_values("Lead Time"),
    x="Lead Time", y="Route",
    orientation="h",
    title="Route Performance Leaderboard"
)
col2.plotly_chart(fig2, use_container_width=True)

# --------------------------------------------------
# 2. GEOGRAPHIC SHIPPING MAP
# --------------------------------------------------
st.header("🗺 Geographic Shipping Map")

map_df = filtered_df.copy()

map_df["Delay Level"] = pd.cut(
    map_df["Lead Time"],
    bins=[0, 3, 7, 20],
    labels=["Fast", "Moderate", "Delayed"]
)

fig_map = scatter_mapbox(
    map_df,
    lat="Latitude",
    lon="Longitude",
    color="Lead Time",
    size="Lead Time",
    hover_data=["State", "Route", "Ship Mode"],
    color_continuous_scale="YlOrRd",
    zoom=3,
    title="Shipping Efficiency Heatmap"
)

fig_map.update_layout(mapbox_style="carto-positron")
st.plotly_chart(fig_map, use_container_width=True)

# --------------------------------------------------
# 3. SHIP MODE COMPARISON
# --------------------------------------------------
st.header("🚢 Ship Mode Comparison")

col1, col2 = st.columns(2)

fig3 = box(
    filtered_df,
    x="Ship Mode",
    y="Lead Time",
    color="Ship Mode",
    title="Lead Time Distribution"
)
col1.plotly_chart(fig3, use_container_width=True)

mode_avg = filtered_df.groupby("Ship Mode")["Lead Time"].mean().reset_index()

fig4 = bar(
    mode_avg,
    x="Ship Mode",
    y="Lead Time",
    color="Ship Mode",
    title="Average Lead Time by Mode"
)
col2.plotly_chart(fig4, use_container_width=True)

# --------------------------------------------------
# 4. ROUTE DRILL-DOWN
# --------------------------------------------------
st.header("🔍 Route Drill-Down")

selected_state = st.selectbox(
    "Select State",
    filtered_df["State"].unique()
)

drill_df = filtered_df[filtered_df["State"] == selected_state]

col1, col2 = st.columns(2)

# State performance
state_perf = drill_df.groupby("Region")["Lead Time"].mean().reset_index()

fig5 = bar(
    state_perf,
    x="Region",
    y="Lead Time",
    title=f"Performance in {selected_state}"
)
col1.plotly_chart(fig5, use_container_width=True)

# Gantt-style timeline
timeline_df = drill_df.copy()

fig6 = timeline(
    timeline_df.sort_values("Ship Date").head(50),
    x_start="Ship Date",
    x_end="Delivery Date",
    y="Order ID",
    color="Ship Mode",
    title="Shipment Timeline (Top 50 Orders)"
)

fig6.update_yaxes(autorange="reversed")
col2.plotly_chart(fig6, use_container_width=True)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.caption("Built with Streamlit • Production-ready Logistics Analytics Dashboard")


