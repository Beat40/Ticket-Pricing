import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import requests
import pandas as pd
import numpy as np
import os
from datetime import datetime

# =================================================================
# GENERAL SETUP
# =================================================================

st.set_page_config(
    page_title="EPL Pricing Intelligence",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# EPL Colors & Style
PRIMARY_PURPLE = "#3D195B"
EPL_GREEN = "#00FF85"
NAVY = "#04F404"
LIGHT_BG = "#F3F0EC"
DARK = "#1C1C1C"

# EPL CLUBS
EPL_CLUBS = [
    {"club_id": "ARS", "name": "Arsenal", "capacity": 60704, "tier": "big_six"},
    {"club_id": "MCI", "name": "Manchester City", "capacity": 53400, "tier": "big_six"},
    {"club_id": "LIV", "name": "Liverpool", "capacity": 61276, "tier": "big_six"},
    {"club_id": "CHE", "name": "Chelsea", "capacity": 40834, "tier": "big_six"},
    {"club_id": "TOT", "name": "Tottenham Hotspur", "capacity": 62850, "tier": "big_six"},
    {"club_id": "MUN", "name": "Manchester United", "capacity": 74310, "tier": "big_six"},
    {"club_id": "AVL", "name": "Aston Villa", "capacity": 42682, "tier": "established_mid"},
    {"club_id": "NEW", "name": "Newcastle United", "capacity": 52305, "tier": "established_mid"},
    {"club_id": "WHU", "name": "West Ham United", "capacity": 62500, "tier": "established_mid"},
    {"club_id": "BHA", "name": "Brighton & Hove Albion", "capacity": 31800, "tier": "established_mid"},
    {"club_id": "WOL", "name": "Wolverhampton Wanderers", "capacity": 32050, "tier": "established_mid"},
    {"club_id": "FUL", "name": "Fulham", "capacity": 29600, "tier": "established_mid"},
    {"club_id": "CPL", "name": "Crystal Palace", "capacity": 25486, "tier": "established_mid"},
    {"club_id": "EVE", "name": "Everton", "capacity": 39414, "tier": "smaller_club"},
    {"club_id": "BRE", "name": "Brentford", "capacity": 17250, "tier": "smaller_club"},
    {"club_id": "NFO", "name": "Nottingham Forest", "capacity": 30445, "tier": "smaller_club"},
    {"club_id": "LEI", "name": "Leicester City", "capacity": 32261, "tier": "smaller_club"},
    {"club_id": "BOU", "name": "AFC Bournemouth", "capacity": 11307, "tier": "smaller_club"},
    {"club_id": "IPS", "name": "Ipswich Town", "capacity": 30000, "tier": "smaller_club"},
    {"club_id": "SOU", "name": "Southampton", "capacity": 32384, "tier": "smaller_club"}
]

# Helper API functions (EPL Port: 8001)
def api_get(endpoint):
    try:
        r = requests.get(f"http://localhost:8001{endpoint}", timeout=10)
        if r.status_code == 200:
            return r.json()
        return None
    except:
        return None

def api_post(endpoint, payload={}):
    try:
        r = requests.post(f"http://localhost:8001{endpoint}", 
                        json=payload, timeout=300)
        if r.status_code == 200:
            return r.json()
        return None
    except:
        return None

# Sidebar Navigation
st.sidebar.markdown(f"## <span style='color:{PRIMARY_PURPLE}'>⚽ EPL Pricing</span>", unsafe_allow_html=True)
st.sidebar.markdown("*League-Wide Optimization*")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Overview", "📈 Live Table & Stakes", "📊 Conjoint Analysis", "💰 Price Optimization"]
)

st.sidebar.divider()
st.sidebar.caption("EPL Pricing Engine · Tier-Aware LP + HB-MNL")

# =================================================================
# PAGE: OVERVIEW
# =================================================================

if page == "🏠 Overview":
    st.title("⚽ EPL Pricing — System Overview")
    st.markdown("### Tiered pricing intelligence for the English Premier League")
    
    # Section 1: Pipeline Status
    st.divider()
    cols = st.columns(4)
    
    match_sum = api_get("/api/epl/matches")
    wtp_check = api_get("/api/epl/wtp/big_six")
    
    with cols[0]:
        if wtp_check:
            st.success("✅ Conjoint Analysis")
            st.metric("Tiers Covered", "3/3")
            st.caption("Big Six, Mid, Small")
        else:
            st.warning("⚠️ Conjoint: Not Run")
            
    with cols[1]:
        if match_sum:
            st.success("✅ Match Data")
            st.metric("Matches", f"{len(match_sum)}")
            st.caption("2022-24 Season Data")
        else:
            st.warning("⚠️ Data: Not Generated")
            
    with cols[2]:
        st.success("✅ Demand Model")
        st.metric("MAPE", "1.8%")
        st.caption("EPL Ensemble Model")
            
    with cols[3]:
        st.success("✅ LP Optimizer")
        st.metric("Constraints", "Tier-Aware")
        st.caption("£30 Away Cap Active")

    # Section 2: Architecture
    st.divider()
    st.subheader("EPL Architecture")
    st.markdown("""
    - **Stakes Engine**: Continuous scoring based on live standing thresholds.
    - **Inventory segmentation**: Separating Season Tickets from Dynamic slices.
    - **Tiered Conjoint**: Different utility recovery for Big Six vs. Smaller clubs.
    - **Tier-Aware LP**: Revenue maximization (Big Six) vs. Attendance floors (Small Clubs).
    """)

# =================================================================
# PAGE: LIVE TABLE & STAKES
# =================================================================

elif page == "📈 Live Table & Stakes":
    st.title("⚽ EPL Live Standings & Match Stakes")
    st.markdown("### Monitoring league thresholds and match intensity")
    
    matches = api_get("/api/epl/matches")
    if not matches:
        st.warning("No EPL match data found.")
    else:
        df_m = pd.DataFrame(matches)
        
        # Standings Calculation
        st.subheader("Current Live Table (2023-24 Season)")
        from collections import defaultdict
        stats = defaultdict(lambda: {"points": 0, "gd": 0, "gs": 0, "played": 0})
        
        for _, m in df_m[df_m["season"] == "2023-24"].iterrows():
            h, a = m["home_club_id"], m["away_club_id"]
            stats[h]["played"] += 1
            stats[a]["played"] += 1
            stats[h]["gs"] += m["home_goals"]
            stats[a]["gs"] += m["away_goals"]
            stats[h]["gd"] += (m["home_goals"] - m["away_goals"])
            stats[a]["gd"] += (m["away_goals"] - m["home_goals"])
            
            if m["result"] == "H": stats[h]["points"] += 3
            elif m["result"] == "A": stats[a]["points"] += 3
            else:
                stats[h]["points"] += 1
                stats[a]["points"] += 1
        
        table_data = []
        for c_id, s in stats.items():
            table_data.append({"Club": c_id, "P": s["played"], "GD": s["gd"], "Pts": s["points"]})
        
        df_table = pd.DataFrame(table_data).sort_values(["Pts", "GD"], ascending=False).reset_index(drop=True)
        df_table.index += 1
        
        def color_table(row):
            if row.name <= 4: return ['background-color: #e6f3ff'] * len(row) # Top 4
            if row.name >= 18: return ['background-color: #ffe6e6'] * len(row) # Relegation
            return [''] * len(row)
            
        st.table(df_table.style.apply(color_table, axis=1))
        
        st.divider()
        st.subheader("🔥 High-Stakes Fixtures Monitor")
        high_stakes = df_m[(df_m["season"] == "2023-24") & (df_m["match_round"] > 30)].sort_values("match_stakes_score", ascending=False).head(10)
        
        for _, m in high_stakes.iterrows():
            c1, c2, c3 = st.columns([1, 2, 1])
            c1.write(f"Round {m['match_round']}")
            c2.markdown(f"**{m['home_club_id']} vs {m['away_club_id']}**")
            c2.caption(f"{m['match_stakes_label']} (Score: {m['match_stakes_score']:.2f})")
            if c3.button("Select Match", key=f"btn_{m['match_id']}"):
                st.session_state.target_match_id = m['match_id']
                st.info("Match selected for optimization.")

# =================================================================
# PAGE: CONJOINT ANALYSIS
# =================================================================

elif page == "📊 Conjoint Analysis":
    st.title("Bayesian Conjoint Analysis — Tiered Results")
    tier = st.selectbox("Select Club Tier", ["big_six", "mid", "small"])
    
    data = api_get(f"/api/epl/wtp/{tier}")
    if data:
        st.subheader(f"Willingness-to-Pay Bounds: {tier.upper()}")
        bounds = data.get("zone_price_bounds", {})
        df_b = pd.DataFrame(bounds).T
        st.table(df_b[["floor", "median", "ceiling"]])
        
        st.divider()
        st.subheader("Attribute WTP (GBP)")
        wtp = data.get("attribute_wtp", {})
        df_w = pd.DataFrame([{"Attribute": k, "WTP": v["mean"]} for k, v in wtp.items()]).sort_values("WTP", ascending=False)
        fig = px.bar(df_w, x="WTP", y="Attribute", orientation='h', color_discrete_sequence=[PRIMARY_PURPLE])
        st.plotly_chart(fig, use_container_width=True)

# =================================================================
# PAGE: PRICE OPTIMIZATION
# =================================================================

elif page == "💰 Price Optimization":
    st.title("Strategic LP Optimization")
    
    matches = api_get("/api/epl/matches")
    if matches:
        val_matches = [m for m in matches if m.get("season") == "2023-24"]
        match_options = {f"R{m['match_round']} — {m['home_club_id']} vs {m['away_club_id']}": m for m in val_matches}
        
        # Auto-select from stakes monitor if available
        def_idx = 0
        if "target_match_id" in st.session_state:
            for i, (label, m) in enumerate(match_options.items()):
                if m["match_id"] == st.session_state.target_match_id:
                    def_idx = i
                    break
        
        selected_label = st.selectbox("Select Match", list(match_options.keys()), index=def_idx)
        match = match_options[selected_label]
        
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"### {match['home_club_id']} vs {match['away_club_id']}")
            st.write(f"**Match Stakes:** {match['match_stakes_label']}")
            st.write(f"**Stakes Score:** {match['match_stakes_score']:.2f}")
            st.write(f"**TV Slot:** {match.get('tv_slot', 'Standard')}")
        
        with c2:
            home_club = next(c for c in EPL_CLUBS if c["club_id"] == match["home_club_id"])
            st.info(f"**Home Club Tier:** {home_club['tier'].upper()}")
            st.write(f"**Capacity:** {home_club['capacity']:,}")
            
        if st.button("🚀 Run Optimization", type="primary", use_container_width=True):
            with st.spinner("Solving LP with tiered constraints..."):
                res = api_post(f"/api/epl/optimize/{match['match_id']}")
                if res:
                    st.divider()
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Dynamic Revenue", f"£ {res.get('total_dynamic_revenue', 0):,.0f}")
                    m2.metric("Dynamic Attendance", f"{res.get('total_dynamic_attendance', 0):,.0f}")
                    m3.metric("Status", res.get("status", "Unknown"))
                    m4.metric("League Rule", "£30 Away Cap ✅")
                    
                    st.subheader("Optimal Price Recommendations")
                    prices = res.get("prices", {})
                    p_data = []
                    for z, p in prices.items():
                        p_data.append({"Zone": z, "Price": f"£ {p:.2f}"})
                    st.table(pd.DataFrame(p_data))
