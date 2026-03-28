# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nassau Candy — Logistics Dashboard",
    page_icon="🍬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# THEME COLOURS
# ──────────────────────────────────────────────────────────────
COLORS = {
    "Standard Class": "#1F3864",
    "Second Class":   "#375623",
    "First Class":    "#843C0C",
    "Same Day":       "#4A235A",
}
FLAG_COLORS = {
    "HIGH RISK":        "#E24B4A",
    "High Lead Time":   "#EF9F27",
    "High Volume":      "#378ADD",
    "Normal":           "#88807C",
}
REGION_COLORS = {
    "Interior": "#EF9F27",
    "Atlantic": "#378ADD",
    "Pacific":  "#639922",
    "Gulf":     "#534AB7",
}

# ──────────────────────────────────────────────────────────────
# STATE / FACTORY COORDINATES
# ──────────────────────────────────────────────────────────────
STATE_COORDS = {
    "Alabama": (32.81, -86.79), "Alaska": (61.37, -152.40),
    "Arizona": (33.73, -111.43), "Arkansas": (34.97, -92.37),
    "California": (36.12, -119.68), "Colorado": (39.06, -105.31),
    "Connecticut": (41.60, -72.76), "Delaware": (39.32, -75.51),
    "District of Columbia": (38.90, -77.03),
    "Florida": (27.77, -81.69), "Georgia": (33.04, -83.64),
    "Hawaii": (21.09, -157.50), "Idaho": (44.24, -114.48),
    "Illinois": (40.35, -88.99), "Indiana": (39.85, -86.26),
    "Iowa": (42.01, -93.21), "Kansas": (38.53, -96.73),
    "Kentucky": (37.67, -84.67), "Louisiana": (31.17, -91.87),
    "Maine": (44.69, -69.38), "Maryland": (39.06, -76.80),
    "Massachusetts": (42.23, -71.53), "Michigan": (43.33, -84.54),
    "Minnesota": (45.69, -93.90), "Mississippi": (32.74, -89.68),
    "Missouri": (38.46, -92.29), "Montana": (46.92, -110.45),
    "Nebraska": (41.13, -98.27), "Nevada": (38.31, -117.06),
    "New Hampshire": (43.45, -71.56), "New Jersey": (40.30, -74.52),
    "New Mexico": (34.84, -106.25), "New York": (42.17, -74.95),
    "North Carolina": (35.63, -79.81), "North Dakota": (47.53, -99.78),
    "Ohio": (40.39, -82.76), "Oklahoma": (35.57, -96.93),
    "Oregon": (44.57, -122.07), "Pennsylvania": (40.59, -77.21),
    "Rhode Island": (41.68, -71.51), "South Carolina": (33.86, -80.95),
    "South Dakota": (44.30, -99.44), "Tennessee": (35.75, -86.69),
    "Texas": (31.05, -97.56), "Utah": (40.15, -111.86),
    "Vermont": (44.05, -72.71), "Virginia": (37.77, -78.17),
    "Washington": (47.40, -121.49), "West Virginia": (38.49, -80.95),
    "Wisconsin": (44.27, -89.62), "Wyoming": (42.76, -107.30),
    "Alberta": (55.00, -114.99), "British Columbia": (53.73, -127.65),
    "Manitoba": (53.76, -98.81), "New Brunswick": (46.57, -66.46),
    "Newfoundland and Labrador": (53.14, -57.66),
    "Nova Scotia": (44.68, -63.74), "Ontario": (51.25, -85.32),
    "Prince Edward Island": (46.51, -63.42),
    "Quebec": (52.94, -73.55), "Saskatchewan": (52.94, -106.45),
}

FACTORY_COORDS = {
    "Savannah, Georgia, USA":            (32.076176, -81.088371, "Wicked Choccy's"),
    "Phoenix, Arizona, USA":             (32.881893, -111.768036, "Lot's O' Nuts"),
    "Milan, Illionis, USA":              (41.446333, -90.565487, "Secret Factory"),
    "Memphis, Tennessee, USA":           (35.117500, -89.971107, "The Other Factory"),
    "Thief River Falls, Minnesota, USA": (48.119140, -96.181150, "Sugar Shack"),
}

# ──────────────────────────────────────────────────────────────
# DATA LOADING & PROCESSING
# ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_excel(
        "Nassau_Candy_Distributor_Work_csv.xlsx",
        sheet_name="Nassau Candy Distributor",
    )
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df["Ship Date"]  = pd.to_datetime(df["Ship Date"])
    df["Shipping Lead Time (Days)"] = (df["Ship Date"] - df["Order Date"]).dt.days
    df["Profit Margin (%)"] = (df["Gross Profit"] / df["Sales"] * 100).round(2)
    df["Category"] = df["Ship Mode"].apply(
        lambda x: "Standard" if x == "Standard Class" else "Expedited"
    )
    df["Route"] = df["Factory Location"] + " → " + df["State/Province"]
    df["Region Route"] = df["Factory Location"] + " → " + df["Region"]

    # Congestion scoring
    state_g = df.groupby("State/Province")["Shipping Lead Time (Days)"].agg(
        count="count", avg="mean"
    ).reset_index()
    high_lt  = state_g["avg"].quantile(0.75)
    high_vol = state_g["count"].quantile(0.75)

    lt_min, lt_max   = state_g["avg"].min(),   state_g["avg"].max()
    vol_min, vol_max = state_g["count"].min(), state_g["count"].max()
    state_g["LT_Score"]        = ((state_g["avg"]   - lt_min)  / (lt_max  - lt_min)  * 100).round(1)
    state_g["Vol_Score"]       = ((state_g["count"] - vol_min) / (vol_max - vol_min) * 100).round(1)
    state_g["Congestion_Score"]= (state_g["LT_Score"] * 0.6 + state_g["Vol_Score"] * 0.4).round(1)

    def flag(r):
        if r["avg"] > high_lt and r["count"] > high_vol: return "HIGH RISK"
        elif r["avg"] > high_lt:                          return "High Lead Time"
        elif r["count"] > high_vol:                       return "High Volume"
        return "Normal"
    state_g["Flag"] = state_g.apply(flag, axis=1)

    # Attach lat/lon to state-level table
    state_g["Lat"] = state_g["State/Province"].map(lambda s: STATE_COORDS.get(s, (None, None))[0])
    state_g["Lon"] = state_g["State/Province"].map(lambda s: STATE_COORDS.get(s, (None, None))[1])
    state_g = state_g.dropna(subset=["Lat", "Lon"])

    return df, state_g, high_lt, high_vol

df, state_geo, HIGH_LT, HIGH_VOL = load_data()

# ──────────────────────────────────────────────────────────────
# SIDEBAR FILTERS
# ──────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/emoji/96/candy.png", width=60)
st.sidebar.title("🍬 Nassau Candy")
st.sidebar.markdown("**Logistics Performance Dashboard**")
st.sidebar.markdown("---")

st.sidebar.header("🔎 Filters")

order_dates = st.sidebar.date_input(
    "Order Date Range",
    [df["Order Date"].min().date(), df["Order Date"].max().date()],
)

selected_regions = st.sidebar.multiselect(
    "Region", sorted(df["Region"].unique()), default=sorted(df["Region"].unique())
)

selected_modes = st.sidebar.multiselect(
    "Ship Mode", sorted(df["Ship Mode"].unique()), default=sorted(df["Ship Mode"].unique())
)

selected_divisions = st.sidebar.multiselect(
    "Division", sorted(df["Division"].unique()), default=sorted(df["Division"].unique())
)

selected_factories = st.sidebar.multiselect(
    "Factory", sorted(df["Factory"].unique()), default=sorted(df["Factory"].unique())
)

lead_range = st.sidebar.slider(
    "Lead Time Range (Days)", int(df["Shipping Lead Time (Days)"].min()),
    int(df["Shipping Lead Time (Days)"].max()),
    (int(df["Shipping Lead Time (Days)"].min()), int(df["Shipping Lead Time (Days)"].max())),
)

# ──────────────────────────────────────────────────────────────
# APPLY FILTERS
# ──────────────────────────────────────────────────────────────
fdf = df[
    (df["Order Date"].dt.date >= order_dates[0]) &
    (df["Order Date"].dt.date <= order_dates[1]) &
    (df["Region"].isin(selected_regions)) &
    (df["Ship Mode"].isin(selected_modes)) &
    (df["Division"].isin(selected_divisions)) &
    (df["Factory"].isin(selected_factories)) &
    (df["Shipping Lead Time (Days)"] >= lead_range[0]) &
    (df["Shipping Lead Time (Days)"] <= lead_range[1])
].copy()

global_avg = df["Shipping Lead Time (Days)"].mean()

# ──────────────────────────────────────────────────────────────
# TITLE
# ──────────────────────────────────────────────────────────────
st.title("🍬 Nassau Candy — Logistics & Shipping Performance Dashboard")
st.caption(f"Showing **{len(fdf):,}** of {len(df):,} total orders · Filtered by active sidebar selections")
st.markdown("---")

# ──────────────────────────────────────────────────────────────
# KPI METRICS
# ──────────────────────────────────────────────────────────────
st.subheader("📊 Key Performance Indicators")

k1, k2, k3, k4, k5, k6 = st.columns(6)

total_sales   = fdf["Sales"].sum()
total_profit  = fdf["Gross Profit"].sum()
avg_lead      = fdf["Shipping Lead Time (Days)"].mean()
delayed       = len(fdf[fdf["Shipping Lead Time (Days)"] > 1400])
margin        = (total_profit / total_sales * 100) if total_sales > 0 else 0
avg_order_val = fdf["Sales"].mean()

k1.metric("Total Orders",         f"{len(fdf):,}")
k2.metric("Total Sales",          f"${total_sales:,.0f}")
k3.metric("Total Profit",         f"${total_profit:,.0f}")
k4.metric("Profit Margin",        f"{margin:.1f}%")
k5.metric("Avg Lead Time",        f"{avg_lead:.1f} days")
k6.metric("Delayed (> 1,400d)",   f"{delayed:,}")

st.markdown("---")

# ──────────────────────────────────────────────────────────────
# SECTION 1 — SHIP MODE COMPARISON
# ──────────────────────────────────────────────────────────────
st.header("🚢 Ship Mode Performance")

mode_order = ["Standard Class", "Second Class", "First Class", "Same Day"]

sm = fdf.groupby("Ship Mode").agg(
    Shipments    =("Row ID",                    "count"),
    Avg_Lead     =("Shipping Lead Time (Days)", "mean"),
    Std_Lead     =("Shipping Lead Time (Days)", "std"),
    Total_Sales  =("Sales",                     "sum"),
    Avg_Sales    =("Sales",                     "mean"),
    Avg_Cost     =("Cost",                      "mean"),
    Avg_Profit   =("Gross Profit",              "mean"),
    Total_Profit =("Gross Profit",              "sum"),
).reset_index()
sm["Margin_Pct"] = (sm["Total_Profit"] / sm["Total_Sales"] * 100).round(2)
sm["Vol_Share"]  = (sm["Shipments"] / sm["Shipments"].sum() * 100).round(1)
sm = sm[sm["Ship Mode"].isin(mode_order)].set_index("Ship Mode").loc[
    [m for m in mode_order if m in sm["Ship Mode"].values]
].reset_index()
sm_colors = [COLORS.get(m, "#888") for m in sm["Ship Mode"]]

col1, col2, col3 = st.columns(3)

# Volume donut
fig_donut = go.Figure(go.Pie(
    labels=sm["Ship Mode"], values=sm["Shipments"],
    hole=0.55, marker_colors=sm_colors,
    textinfo="label+percent", textfont_size=11,
))
fig_donut.update_layout(title="Shipment Volume Share", showlegend=False,
                         height=320, margin=dict(t=40, b=10, l=10, r=10))
col1.plotly_chart(fig_donut, use_container_width=True)

# Avg lead time bar
fig_lt = px.bar(
    sm.sort_values("Avg_Lead"),
    x="Avg_Lead", y="Ship Mode", orientation="h",
    color="Ship Mode", color_discrete_map=COLORS,
    text=sm.sort_values("Avg_Lead")["Avg_Lead"].round(1).astype(str) + "d",
    title="Avg Lead Time (days) — Fastest → Slowest",
)
fig_lt.add_vline(x=global_avg, line_dash="dash", line_color="#E24B4A",
                  annotation_text=f"Global avg: {global_avg:.0f}d",
                  annotation_position="top right")
fig_lt.update_traces(textposition="outside")
fig_lt.update_layout(showlegend=False, height=320, xaxis_range=[1290, 1360],
                      margin=dict(t=40, b=10, l=10, r=60),
                      xaxis_title="Days", yaxis_title="")
col2.plotly_chart(fig_lt, use_container_width=True)

# Profit margin bar
fig_mg = px.bar(
    sm, x="Ship Mode", y="Margin_Pct",
    color="Ship Mode", color_discrete_map=COLORS,
    text=sm["Margin_Pct"].round(1).astype(str) + "%",
    title="Profit Margin (%) by Mode",
)
fig_mg.update_traces(textposition="outside")
fig_mg.update_layout(showlegend=False, height=320,
                      yaxis_range=[64, 67.5],
                      xaxis_title="", yaxis_title="Margin (%)",
                      margin=dict(t=40, b=10, l=10, r=10))
col3.plotly_chart(fig_mg, use_container_width=True)

# Box plots + grouped financials
col4, col5 = st.columns(2)

fig_box = px.box(
    fdf, x="Ship Mode", y="Shipping Lead Time (Days)",
    color="Ship Mode", color_discrete_map=COLORS,
    category_orders={"Ship Mode": mode_order},
    title="Lead Time Distribution by Ship Mode",
)
fig_box.add_hline(y=global_avg, line_dash="dash", line_color="#E24B4A",
                   annotation_text=f"Global avg: {global_avg:.0f}d")
fig_box.update_layout(showlegend=False, height=360,
                       xaxis_title="", yaxis_title="Lead Time (days)",
                       margin=dict(t=40, b=10, l=10, r=10))
col4.plotly_chart(fig_box, use_container_width=True)

# Grouped: Sales / Cost / Profit
fig_fin = go.Figure()
fig_fin.add_trace(go.Bar(name="Avg Sales",  x=sm["Ship Mode"], y=sm["Avg_Sales"].round(2),
                          marker_color="#1F3864", text=sm["Avg_Sales"].round(2),
                          texttemplate="$%{text}", textposition="outside"))
fig_fin.add_trace(go.Bar(name="Avg Cost",   x=sm["Ship Mode"], y=sm["Avg_Cost"].round(2),
                          marker_color="#843C0C", text=sm["Avg_Cost"].round(2),
                          texttemplate="$%{text}", textposition="outside"))
fig_fin.add_trace(go.Bar(name="Avg Profit", x=sm["Ship Mode"], y=sm["Avg_Profit"].round(2),
                          marker_color="#375623", text=sm["Avg_Profit"].round(2),
                          texttemplate="$%{text}", textposition="outside"))
fig_fin.update_layout(barmode="group", title="Avg Sales / Cost / Profit per Order",
                       height=360, yaxis_title="Amount ($)",
                       xaxis_title="", legend=dict(orientation="h", y=1.1),
                       margin=dict(t=60, b=10, l=10, r=10))
col5.plotly_chart(fig_fin, use_container_width=True)

st.markdown("---")

# ──────────────────────────────────────────────────────────────
# SECTION 2 — STANDARD vs EXPEDITED
# ──────────────────────────────────────────────────────────────
st.header("⚡ Standard vs Expedited — Head-to-Head")

cat = fdf.groupby("Category").agg(
    Shipments    =("Row ID",                    "count"),
    Avg_Lead     =("Shipping Lead Time (Days)", "mean"),
    Total_Sales  =("Sales",                     "sum"),
    Avg_Sales    =("Sales",                     "mean"),
    Avg_Cost     =("Cost",                      "mean"),
    Avg_Profit   =("Gross Profit",              "mean"),
    Total_Profit =("Gross Profit",              "sum"),
).reset_index()
cat["Margin_Pct"] = (cat["Total_Profit"] / cat["Total_Sales"] * 100).round(2)
cat["Share_Pct"]  = (cat["Shipments"] / cat["Shipments"].sum() * 100).round(1)
cat_colors = [("#1F3864" if c == "Standard" else "#843C0C") for c in cat["Category"]]

c1, c2, c3, c4 = st.columns(4)

fig_cv = px.bar(cat, x="Category", y="Shipments",
                color="Category", color_discrete_map={"Standard":"#1F3864","Expedited":"#843C0C"},
                text=cat["Shipments"].map(lambda v: f"{v:,}"),
                title="Total Shipments")
fig_cv.update_traces(textposition="outside")
fig_cv.update_layout(showlegend=False, height=300, xaxis_title="", yaxis_title="Orders",
                      margin=dict(t=40, b=10, l=10, r=10))
c1.plotly_chart(fig_cv, use_container_width=True)

fig_cl = px.bar(cat, x="Category", y="Avg_Lead",
                color="Category", color_discrete_map={"Standard":"#1F3864","Expedited":"#843C0C"},
                text=cat["Avg_Lead"].round(1).astype(str) + "d",
                title="Avg Lead Time (days)")
fig_cl.update_traces(textposition="outside")
fig_cl.update_layout(showlegend=False, height=300, yaxis_range=[1300, 1345],
                      xaxis_title="", yaxis_title="Days",
                      margin=dict(t=40, b=10, l=10, r=10))
c2.plotly_chart(fig_cl, use_container_width=True)

fig_cm = px.bar(cat, x="Category", y="Margin_Pct",
                color="Category", color_discrete_map={"Standard":"#1F3864","Expedited":"#843C0C"},
                text=cat["Margin_Pct"].round(1).astype(str) + "%",
                title="Profit Margin (%)")
fig_cm.update_traces(textposition="outside")
fig_cm.update_layout(showlegend=False, height=300, yaxis_range=[64, 67],
                      xaxis_title="", yaxis_title="Margin (%)",
                      margin=dict(t=40, b=10, l=10, r=10))
c3.plotly_chart(fig_cm, use_container_width=True)

fig_cp = px.bar(cat, x="Category", y="Avg_Profit",
                color="Category", color_discrete_map={"Standard":"#1F3864","Expedited":"#843C0C"},
                text=cat["Avg_Profit"].round(2).map(lambda v: f"${v:.2f}"),
                title="Avg Profit / Order ($)")
fig_cp.update_traces(textposition="outside")
fig_cp.update_layout(showlegend=False, height=300,
                      xaxis_title="", yaxis_title="Profit ($)",
                      margin=dict(t=40, b=10, l=10, r=10))
c4.plotly_chart(fig_cp, use_container_width=True)

# Cost-time tradeoff scatter
st.subheader("Cost-Time Tradeoff — Lead Time vs Avg Profit (bubble = shipment volume)")
fig_scatter = go.Figure()
for _, row in sm.iterrows():
    fig_scatter.add_trace(go.Scatter(
        x=[round(row["Avg_Lead"], 1)], y=[round(row["Avg_Profit"], 2)],
        mode="markers+text",
        marker=dict(size=row["Shipments"]/30, color=COLORS.get(row["Ship Mode"], "#888"),
                    opacity=0.85, line=dict(color="white", width=1.5)),
        text=[row["Ship Mode"]], textposition="top center",
        name=row["Ship Mode"],
    ))
fig_scatter.add_hline(y=fdf["Gross Profit"].mean(), line_dash="dash", line_color="#378ADD",
                       annotation_text="Avg profit/order", annotation_position="right")
fig_scatter.add_vline(x=global_avg, line_dash="dash", line_color="#E24B4A",
                       annotation_text="Avg lead time", annotation_position="top")
fig_scatter.update_layout(height=380, showlegend=True,
                           xaxis_title="Avg Lead Time (days)",
                           yaxis_title="Avg Profit per Order ($)",
                           margin=dict(t=20, b=40, l=60, r=20))
st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")

# ──────────────────────────────────────────────────────────────
# SECTION 3 — ROUTE EFFICIENCY
# ──────────────────────────────────────────────────────────────
st.header("📍 Route Efficiency")

route_perf = fdf.groupby(["Route", "Factory Location", "State/Province"]).agg(
    Shipments=("Row ID", "count"),
    Avg_Lead =("Shipping Lead Time (Days)", "mean"),
    Avg_Sales=("Sales", "mean"),
).reset_index()
route_perf["Avg_Lead"] = route_perf["Avg_Lead"].round(1)
route_perf["Rank"] = route_perf["Avg_Lead"].rank(method="first").astype(int)
route_perf_sorted = route_perf.sort_values("Avg_Lead")

col_r1, col_r2 = st.columns(2)

# Top 10 fastest
top10 = route_perf_sorted.head(10)
fig_top = px.bar(
    top10, x="Avg_Lead", y="Route", orientation="h",
    color="Avg_Lead", color_continuous_scale="Greens_r",
    text=top10["Avg_Lead"].astype(str) + "d",
    title="🥇 Top 10 Fastest Routes",
)
fig_top.update_traces(textposition="outside")
fig_top.update_layout(coloraxis_showscale=False, showlegend=False,
                       height=380, xaxis_title="Avg Lead Time (days)",
                       yaxis_title="", margin=dict(t=40, b=10, l=10, r=80))
col_r1.plotly_chart(fig_top, use_container_width=True)

# Bottom 10 slowest
bot10 = route_perf_sorted.tail(10).sort_values("Avg_Lead", ascending=False)
fig_bot = px.bar(
    bot10, x="Avg_Lead", y="Route", orientation="h",
    color="Avg_Lead", color_continuous_scale="Reds",
    text=bot10["Avg_Lead"].astype(str) + "d",
    title="🔴 Bottom 10 Slowest Routes",
)
fig_bot.update_traces(textposition="outside")
fig_bot.update_layout(coloraxis_showscale=False, showlegend=False,
                       height=380, xaxis_title="Avg Lead Time (days)",
                       yaxis_title="", margin=dict(t=40, b=10, l=10, r=80))
col_r2.plotly_chart(fig_bot, use_container_width=True)

# Region routes heatmap
region_route = fdf.groupby(["Factory", "Region"])["Shipping Lead Time (Days)"].mean().round(1).reset_index()
pivot = region_route.pivot(index="Factory", columns="Region", values="Shipping Lead Time (Days)")
fig_heat = px.imshow(
    pivot, color_continuous_scale="YlOrRd",
    title="Factory × Region — Avg Lead Time Heatmap",
    text_auto=True, aspect="auto",
)
fig_heat.update_layout(height=340, margin=dict(t=40, b=20, l=10, r=10),
                        coloraxis_colorbar_title="Days")
st.plotly_chart(fig_heat, use_container_width=True)

st.markdown("---")

# ──────────────────────────────────────────────────────────────
# SECTION 4 — CONGESTION MAP
# ──────────────────────────────────────────────────────────────
st.header("🗺️ Congestion & Shipping Map")

# Recompute congestion from filtered data
st_filt = fdf.groupby("State/Province")["Shipping Lead Time (Days)"].agg(
    count="count", avg="mean"
).reset_index()
flt_min, flt_max   = st_filt["avg"].min(),   st_filt["avg"].max()
fvl_min, fvl_max   = st_filt["count"].min(), st_filt["count"].max()
st_filt["LT_Score"]  = ((st_filt["avg"]   - flt_min) / max(flt_max - flt_min, 1)   * 100).round(1)
st_filt["Vol_Score"] = ((st_filt["count"] - fvl_min) / max(fvl_max - fvl_min, 1) * 100).round(1)
st_filt["Congestion_Score"] = (st_filt["LT_Score"] * 0.6 + st_filt["Vol_Score"] * 0.4).round(1)

fhlt  = st_filt["avg"].quantile(0.75)
fhvol = st_filt["count"].quantile(0.75)
def flag_fn(r):
    if r["avg"] > fhlt and r["count"] > fhvol: return "HIGH RISK"
    elif r["avg"] > fhlt:                        return "High Lead Time"
    elif r["count"] > fhvol:                     return "High Volume"
    return "Normal"
st_filt["Flag"] = st_filt.apply(flag_fn, axis=1)
st_filt["Lat"]  = st_filt["State/Province"].map(lambda s: STATE_COORDS.get(s, (None, None))[0])
st_filt["Lon"]  = st_filt["State/Province"].map(lambda s: STATE_COORDS.get(s, (None, None))[1])
st_filt = st_filt.dropna(subset=["Lat", "Lon"])
st_filt["Flag_Color"] = st_filt["Flag"].map(FLAG_COLORS)

map_tab1, map_tab2 = st.tabs(["🔴 Congestion Heatmap", "🏭 Factory Shipping Map"])

with map_tab1:
    fig_map = px.scatter_mapbox(
        st_filt,
        lat="Lat", lon="Lon",
        size="Congestion_Score",
        color="Flag",
        color_discrete_map=FLAG_COLORS,
        hover_name="State/Province",
        hover_data={"count": True, "avg": ":.1f", "Congestion_Score": ":.1f",
                    "Lat": False, "Lon": False},
        labels={"count": "Shipments", "avg": "Avg Lead Time (days)", "Congestion_Score": "Congestion Score"},
        zoom=3, center={"lat": 39.5, "lon": -98.5},
        title="State Congestion Map — Bubble Size = Congestion Score",
        mapbox_style="carto-positron",
    )
    fig_map.update_layout(height=520, margin=dict(t=40, b=10, l=10, r=10),
                           legend_title="Classification")
    st.plotly_chart(fig_map, use_container_width=True)

with map_tab2:
    # Factory locations
    fac_df = pd.DataFrame([
        {"Factory Location": loc, "Lat": c[0], "Lon": c[1],
         "Factory": c[2], "Shipments": len(fdf[fdf["Factory Location"] == loc])}
        for loc, c in FACTORY_COORDS.items()
    ])
    fac_df = fac_df[fac_df["Shipments"] > 0]

    # Customer destinations
    cust_df = fdf.groupby("State/Province").agg(
        Shipments=("Row ID", "count"),
        Avg_Lead =("Shipping Lead Time (Days)", "mean"),
    ).reset_index()
    cust_df["Lat"] = cust_df["State/Province"].map(lambda s: STATE_COORDS.get(s, (None, None))[0])
    cust_df["Lon"] = cust_df["State/Province"].map(lambda s: STATE_COORDS.get(s, (None, None))[1])
    cust_df = cust_df.dropna(subset=["Lat", "Lon"])

    fig_fmap = go.Figure()
    fig_fmap.add_trace(go.Scattermapbox(
        lat=cust_df["Lat"], lon=cust_df["Lon"],
        mode="markers",
        marker=dict(size=cust_df["Shipments"] / 60, color=cust_df["Avg_Lead"],
                    colorscale="YlOrRd", showscale=True,
                    colorbar=dict(title="Avg Lead Time")),
        text=cust_df["State/Province"],
        hovertemplate="<b>%{text}</b><br>Shipments: %{customdata[0]:,}<br>Avg Lead: %{customdata[1]:.1f}d<extra></extra>",
        customdata=cust_df[["Shipments", "Avg_Lead"]].values,
        name="Destinations",
    ))
    fig_fmap.add_trace(go.Scattermapbox(
        lat=fac_df["Lat"], lon=fac_df["Lon"],
        mode="markers+text",
        marker=dict(size=18, color="#1F3864", symbol="star"),
        text=fac_df["Factory"],
        textposition="top right",
        hovertemplate="<b>%{text}</b><br>%{customdata[0]}<br>Shipments: %{customdata[1]:,}<extra></extra>",
        customdata=fac_df[["Factory Location", "Shipments"]].values,
        name="Factories",
    ))
    fig_fmap.update_layout(
        mapbox=dict(style="carto-positron", zoom=3, center={"lat": 39.5, "lon": -98.5}),
        height=520, margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    st.plotly_chart(fig_fmap, use_container_width=True)

st.markdown("---")

# ──────────────────────────────────────────────────────────────
# SECTION 5 — REGION ANALYSIS
# ──────────────────────────────────────────────────────────────
st.header("🌎 Region Performance")

rgn = fdf.groupby("Region").agg(
    Shipments   =("Row ID",                    "count"),
    Avg_Lead    =("Shipping Lead Time (Days)", "mean"),
    Std_Lead    =("Shipping Lead Time (Days)", "std"),
    Total_Sales =("Sales",                     "sum"),
    Avg_Sales   =("Sales",                     "mean"),
    Avg_Profit  =("Gross Profit",              "mean"),
    Total_Profit=("Gross Profit",              "sum"),
).reset_index()
rgn["Margin_Pct"] = (rgn["Total_Profit"] / rgn["Total_Sales"] * 100).round(2)
rgn["vs_Global"]  = (rgn["Avg_Lead"] - global_avg).round(1)
rgn_colors = [REGION_COLORS.get(r, "#888") for r in rgn["Region"]]

col_rg1, col_rg2, col_rg3 = st.columns(3)

fig_rl = px.bar(
    rgn.sort_values("Avg_Lead", ascending=False),
    x="Region", y="Avg_Lead",
    color="Region", color_discrete_map=REGION_COLORS,
    text=rgn.sort_values("Avg_Lead", ascending=False)["Avg_Lead"].round(1).astype(str) + "d",
    title="Avg Lead Time by Region",
)
fig_rl.add_hline(y=global_avg, line_dash="dash", line_color="#E24B4A")
fig_rl.update_traces(textposition="outside")
fig_rl.update_layout(showlegend=False, height=320, yaxis_range=[1290, 1340],
                      xaxis_title="", yaxis_title="Days",
                      margin=dict(t=40, b=10, l=10, r=10))
col_rg1.plotly_chart(fig_rl, use_container_width=True)

fig_rv = px.pie(
    rgn, values="Shipments", names="Region",
    color="Region", color_discrete_map=REGION_COLORS,
    title="Shipment Volume Share by Region",
    hole=0.4,
)
fig_rv.update_layout(height=320, margin=dict(t=40, b=10, l=10, r=10))
col_rg2.plotly_chart(fig_rv, use_container_width=True)

fig_rm = px.bar(
    rgn, x="Region", y="Margin_Pct",
    color="Region", color_discrete_map=REGION_COLORS,
    text=rgn["Margin_Pct"].round(1).astype(str) + "%",
    title="Profit Margin by Region",
)
fig_rm.update_traces(textposition="outside")
fig_rm.update_layout(showlegend=False, height=320, yaxis_range=[64, 68],
                      xaxis_title="", yaxis_title="Margin (%)",
                      margin=dict(t=40, b=10, l=10, r=10))
col_rg3.plotly_chart(fig_rm, use_container_width=True)

# Region × Ship Mode heatmap
rgn_mode = fdf.groupby(["Region", "Ship Mode"])["Shipping Lead Time (Days)"].mean().round(1).reset_index()
pivot_rm  = rgn_mode.pivot(index="Ship Mode", columns="Region", values="Shipping Lead Time (Days)")
pivot_rm  = pivot_rm.loc[[m for m in mode_order if m in pivot_rm.index]]
fig_rhm = px.imshow(
    pivot_rm, color_continuous_scale="YlOrRd",
    title="Region × Ship Mode — Avg Lead Time (days)",
    text_auto=True, aspect="auto",
)
fig_rhm.update_layout(height=300, margin=dict(t=40, b=20, l=10, r=10))
st.plotly_chart(fig_rhm, use_container_width=True)

st.markdown("---")

# ──────────────────────────────────────────────────────────────
# SECTION 6 — STATE DRILL-DOWN
# ──────────────────────────────────────────────────────────────
st.header("🔍 State & Congestion Drill-Down")

col_dd1, col_dd2 = st.columns([1, 2])

with col_dd1:
    state_list = sorted(fdf["State/Province"].unique())
    sel_state  = st.selectbox("Select a State / Province", state_list)

state_df = fdf[fdf["State/Province"] == sel_state]

with col_dd2:
    s_ship = len(state_df)
    s_lead = state_df["Shipping Lead Time (Days)"].mean()
    s_sales= state_df["Sales"].sum()
    s_margin = (state_df["Gross Profit"].sum() / state_df["Sales"].sum() * 100) if s_sales > 0 else 0
    congestion_row = st_filt[st_filt["State/Province"] == sel_state]
    congestion_val = congestion_row["Congestion_Score"].values[0] if len(congestion_row) else "N/A"
    flag_val       = congestion_row["Flag"].values[0] if len(congestion_row) else "N/A"
    flag_color     = FLAG_COLORS.get(flag_val, "#888")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Shipments",     f"{s_ship:,}")
    m2.metric("Avg Lead Time", f"{s_lead:.1f}d")
    m3.metric("Total Sales",   f"${s_sales:,.0f}")
    m4.metric("Margin",        f"{s_margin:.1f}%")
    m5.metric("Congestion",    f"{congestion_val}" if isinstance(congestion_val, str) else f"{congestion_val:.1f}",
              delta=flag_val, delta_color="off")

col_s1, col_s2 = st.columns(2)

# Ship mode breakdown for state
sm_state = state_df.groupby("Ship Mode").agg(
    Shipments=("Row ID","count"),
    Avg_Lead =("Shipping Lead Time (Days)","mean"),
).reset_index()
fig_ss = px.bar(
    sm_state, x="Ship Mode", y="Shipments",
    color="Ship Mode", color_discrete_map=COLORS,
    text="Shipments",
    title=f"Shipments by Mode — {sel_state}",
)
fig_ss.update_traces(textposition="outside")
fig_ss.update_layout(showlegend=False, height=320, xaxis_title="", yaxis_title="Orders",
                      margin=dict(t=40, b=10, l=10, r=10))
col_s1.plotly_chart(fig_ss, use_container_width=True)

# Factory breakdown for state
fac_state = state_df.groupby("Factory").agg(
    Shipments=("Row ID","count"),
    Avg_Lead =("Shipping Lead Time (Days)","mean"),
    Avg_Sales=("Sales","mean"),
).reset_index()
fig_fs = px.bar(
    fac_state, x="Factory", y="Avg_Lead",
    color="Factory", text=fac_state["Avg_Lead"].round(1).astype(str)+"d",
    title=f"Avg Lead Time by Factory → {sel_state}",
)
fig_fs.update_traces(textposition="outside")
fig_fs.update_layout(showlegend=False, height=320, xaxis_title="", yaxis_title="Days",
                      margin=dict(t=40, b=10, l=10, r=10))
col_s2.plotly_chart(fig_fs, use_container_width=True)

# Congestion leaderboard
st.subheader("📊 State Congestion Leaderboard")
cong_display = st_filt[["State/Province","count","avg","Congestion_Score","Flag"]].copy()
cong_display.columns = ["State","Shipments","Avg Lead Time","Congestion Score","Classification"]
cong_display = cong_display.sort_values("Congestion Score", ascending=False).reset_index(drop=True)
cong_display.index += 1
cong_display["Avg Lead Time"] = cong_display["Avg Lead Time"].round(1)
cong_display["Congestion Score"] = cong_display["Congestion Score"].round(1)

def color_flag(val):
    c = FLAG_COLORS.get(val, "#888")
    return f"background-color: {c}22; color: {c}; font-weight: bold"

st.dataframe(
    cong_display.style.applymap(color_flag, subset=["Classification"]),
    use_container_width=True, height=420,
)

st.markdown("---")

# ──────────────────────────────────────────────────────────────
# SECTION 7 — PRODUCT & DIVISION
# ──────────────────────────────────────────────────────────────
st.header("🍫 Product Division Performance")

div = fdf.groupby("Division").agg(
    Shipments   =("Row ID",      "count"),
    Total_Sales =("Sales",       "sum"),
    Avg_Sales   =("Sales",       "mean"),
    Total_Profit=("Gross Profit","sum"),
    Avg_Profit  =("Gross Profit","mean"),
    Avg_Lead    =("Shipping Lead Time (Days)","mean"),
).reset_index()
div["Margin_Pct"] = (div["Total_Profit"] / div["Total_Sales"] * 100).round(2)

col_d1, col_d2, col_d3 = st.columns(3)

fig_dv = px.pie(div, values="Shipments", names="Division",
                title="Orders by Division", hole=0.4,
                color_discrete_sequence=["#1F3864","#843C0C","#375623"])
fig_dv.update_layout(height=300, margin=dict(t=40,b=10,l=10,r=10))
col_d1.plotly_chart(fig_dv, use_container_width=True)

fig_ds = px.bar(div, x="Division", y="Total_Sales",
                color="Division", text=div["Total_Sales"].map(lambda v: f"${v:,.0f}"),
                title="Total Sales by Division",
                color_discrete_sequence=["#1F3864","#843C0C","#375623"])
fig_ds.update_traces(textposition="outside")
fig_ds.update_layout(showlegend=False, height=300, xaxis_title="", yaxis_title="Sales ($)",
                      margin=dict(t=40,b=10,l=10,r=10))
col_d2.plotly_chart(fig_ds, use_container_width=True)

fig_dm = px.bar(div, x="Division", y="Margin_Pct",
                color="Division", text=div["Margin_Pct"].round(1).astype(str)+"%",
                title="Profit Margin by Division",
                color_discrete_sequence=["#1F3864","#843C0C","#375623"])
fig_dm.update_traces(textposition="outside")
fig_dm.update_layout(showlegend=False, height=300, xaxis_title="", yaxis_title="Margin (%)",
                      margin=dict(t=40,b=10,l=10,r=10))
col_d3.plotly_chart(fig_dm, use_container_width=True)

st.markdown("---")

# ──────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style='text-align:center; color:#888; font-size:12px; padding:10px 0'>
    🍬 Nassau Candy Distributor · Logistics Analytics Dashboard ·
    Built with <strong>Streamlit</strong> & <strong>Plotly</strong> ·
    Data: Nassau Candy Distributor Work Dataset
    </div>
    """,
    unsafe_allow_html=True,
)
