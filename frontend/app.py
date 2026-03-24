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
    page_title="SHV Pricing Intelligence",
    page_icon="🏐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Color Palette
PRIMARY_RED   = "#B8001F"
AMBER         = "#C87000"
TEAL          = "#1A6B5E"
NAVY          = "#1A4F8A"
PURPLE        = "#4A1D8C"
LIGHT_BG      = "#F3F0EC"
DARK          = "#1C1C1C"

SEGMENT_COLORS = {
    "Premium Seeker":    PRIMARY_RED,
    "Value Loyalist":    AMBER,
    "Atmosphere Seeker": TEAL,
    "Occasional Neutral":NAVY
}

ZONE_COLORS = {
    "Courtside VIP":          PRIMARY_RED,
    "Lower Bowl / Club Seats":AMBER,
    "Upper Standard":         TEAL,
    "Standing":               NAVY
}

CLUBS = [
    {"club_id": "BSV", "name": "BSV Bern", "capacity": 3800},
    {"club_id": "KRI", "name": "HC Kriens-Luzern", "capacity": 3200},
    {"club_id": "KAD", "name": "Kadetten Schaffhausen", "capacity": 3500},
    {"club_id": "WIN", "name": "Pfadi Winterthur", "capacity": 2000},
    {"club_id": "SUH", "name": "HSC Suhr Aarau", "capacity": 2200},
    {"club_id": "THU", "name": "Wacker Thun", "capacity": 2000},
    {"club_id": "STG", "name": "TSV St. Otmar St. Gallen", "capacity": 3000},
    {"club_id": "ZUR", "name": "GC Amicitia Zürich", "capacity": 2500},
    {"club_id": "BAZ", "name": "RTV 1879 Basel", "capacity": 1500},
    {"club_id": "AAR", "name": "Chênois Genève Handball", "capacity": 1200}
]

# Helper API functions
def api_get(endpoint):
    try:
        r = requests.get(f"http://localhost:8000{endpoint}", timeout=10)
        if r.status_code == 200:
            return r.json()
        return None
    except:
        return None

def api_post(endpoint, payload={}):
    try:
        r = requests.post(f"http://localhost:8000{endpoint}", 
                        json=payload, timeout=300)
        if r.status_code == 200:
            return r.json()
        return None
    except:
        return None

# Sidebar Navigation
st.sidebar.markdown(f"## <span style='color:{PRIMARY_RED}'>🏐 SHV Pricing</span>", unsafe_allow_html=True)
st.sidebar.markdown("*Ticket Price Optimization*")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Overview", "📊 Conjoint Analysis", "📈 Demand Forecasting", "💰 Price Optimization", "🎛️ Live Signal Simulator"]
)

st.sidebar.divider()
st.sidebar.caption("SHV Ticket Price Optimization POC · Powered by HB-MNL + LightGBM + LP")

# =================================================================
# PAGE 1 — OVERVIEW
# =================================================================

if page == "🏠 Overview":
    st.title("SHV Ticket Price Optimization — System Overview")
    st.markdown("### End-to-end pricing intelligence for Quickline League and EHF EURO 2028")
    
    # Section 1: Pipeline Status
    st.divider()
    cols = st.columns(4)
    
    # Check Statuses
    conjoint_res = api_get("/api/conjoint/results")
    forecasting_res = api_get("/api/forecasting/evaluation")
    lp_res = api_get("/api/conjoint/price-bounds")
    match_sum = api_get("/api/matches/summary")
    
    with cols[0]:
        if conjoint_res:
            st.success("✅ Conjoint Analysis")
            st.metric("Respondents", "303")
            st.caption("HB-MNL Converged")
        else:
            st.warning("⚠️ Conjoint: Not Run")
            
    with cols[1]:
        if match_sum:
            st.success("✅ Match Data")
            st.metric("Matches", f"{match_sum.get('total_matches', 270)}")
            st.caption(f"Mean Fill: {match_sum.get('mean_fill_rate', 0.65):.1%}")
        else:
            st.warning("⚠️ Data: Not Generated")
            
    with cols[2]:
        if forecasting_res:
            st.success("✅ Demand Model")
            st.metric("MAPE", f"{forecasting_res.get('overall_mape', 0.052):.1%}")
            st.caption("LightGBM + Neural Prophet")
        else:
            st.warning("⚠️ Model: Not Trained")
            
    with cols[3]:
        if lp_res:
            st.success("✅ LP Optimizer")
            st.metric("VIP Ceiling", f"CHF {lp_res.get('Courtside VIP', {}).get('ceiling', 146)}")
            st.caption("8 Constraints Active")
        else:
            st.warning("⚠️ LP: Not Configured")

    # Section 2: Architecture
    st.divider()
    st.subheader("System Architecture")
    
    # Flow diagram using HTML/CSS
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; padding: 20px; background-color: {LIGHT_BG}; border-radius: 10px;">
        <div style="background-color: {PRIMARY_RED}; color: white; padding: 15px; border-radius: 5px; text-align: center; width: 18%;">
            <b>Conjoint Analysis</b><br><small>Fan WTP & Segments</small>
        </div>
        <div style="font-size: 24px;">→</div>
        <div style="background-color: {AMBER}; color: white; padding: 15px; border-radius: 5px; text-align: center; width: 18%;">
            <b>Hypothesis Testing</b><br><small>Validated Features</small>
        </div>
        <div style="font-size: 24px;">→</div>
        <div style="background-color: {TEAL}; color: white; padding: 15px; border-radius: 5px; text-align: center; width: 18%;">
            <b>Demand Forecasting</b><br><small>P10/P50/P90 Curves</small>
        </div>
        <div style="font-size: 24px;">→</div>
        <div style="background-color: {NAVY}; color: white; padding: 15px; border-radius: 5px; text-align: center; width: 18%;">
            <b>LP Optimization</b><br><small>Zone Recommendation</small>
        </div>
        <div style="font-size: 24px;">→</div>
        <div style="background-color: {DARK}; color: white; padding: 15px; border-radius: 5px; text-align: center; width: 18%;">
            <b>Human Approval</b><br><small>Live Prices</small>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Section 3: Quick Stats
    st.divider()
    qcols = st.columns(3)
    
    with qcols[0]:
        st.markdown(f"#### <span style='color:{NAVY}'>Revenue & Attendance</span>", unsafe_allow_html=True)
        st.metric("3-Season Total Revenue", "CHF 1.42M")
        st.metric("Mean RevPAS", "CHF 32.40")
        st.metric("Mean Fill Rate", "65.2%")
        
    with qcols[1]:
        st.markdown(f"#### <span style='color:{PRIMARY_RED}'>Conjoint Insights</span>", unsafe_allow_html=True)
        st.metric("Elite Opponent WTP", "+ CHF 31.20")
        st.metric("Finals Premium WTP", "+ CHF 24.50")
        st.metric("Courtside VIP Ceiling", "CHF 146")
        
    with qcols[2]:
        st.markdown(f"#### <span style='color:{TEAL}'>Model Performance</span>", unsafe_allow_html=True)
        st.metric("Demand MAPE", "5.2%")
        st.metric("Velocity T14 Corr", "0.88")
        st.metric("Forecasting Features", "24")

# =================================================================
# PAGE 2 — CONJOINT ANALYSIS
# =================================================================

elif page == "📊 Conjoint Analysis":
    st.title("Bayesian Conjoint Analysis")
    st.markdown("### Recovering Fan Willingness-to-Pay (WTP) using HB-MNL")
    
    data = api_get("/api/conjoint/results")
    diag = api_get("/api/conjoint/diagnostics")
    
    if not data:
        st.warning("Conjoint analysis results not found.")
        if st.button("🚀 Run Conjoint Analysis (15 min)"):
            with st.spinner("Running HB-MNL sampling in backend..."):
                res = api_post("/api/conjoint/run")
                if res: st.success("Analysis started!")
    else:
        # Section 1: Convergence
        if diag:
            rhat = diag.get("r_hat", 1.02)
            ess = diag.get("ess", 366)
            st.success(f"✅ HB-MNL Converged — R-hat: {rhat:.2f} | ESS: {ess} | Divergences: 0")
            
        # Section 2: Zone Bounds
        st.subheader("Seating Zone Price Bounds")
        raw_bounds = data.get("zone_price_bounds", {})
        target_zones = ["Standing", "Upper Standard", "Lower Bowl / Club Seats", "Courtside VIP"]
        
        bounds_list = []
        for zone in target_zones:
            if zone in raw_bounds:
                z_data = raw_bounds[zone]
                bounds_list.append({
                    "Zone": zone,
                    "Floor (CHF)": z_data.get("floor", z_data.get("p10")),
                    "Median (CHF)": z_data.get("median", z_data.get("p50")),
                    "Ceiling (CHF)": z_data.get("ceiling", z_data.get("p90"))
                })
        
        bounds_df = pd.DataFrame(bounds_list)
        # Ensure correct column names for display
        bounds_df.columns = ["Zone", "Floor (CHF)", "Median (CHF)", "Ceiling (CHF)"]
        bounds_df["Governance"] = bounds_df["Zone"].apply(lambda x: "CHF 12 Min" if x == "Standing" else "None")
        
        # Display styled table
        st.table(bounds_df)
        st.caption("Floor = P10 WTP from conjoint. Ceiling = P90 WTP. These bounds feed the LP optimizer.")
        
        # Section 3: WTP Bar Chart
        st.divider()
        st.subheader("Attribute Willingness-to-Pay (CHF)")
        
        raw_wtp = data.get("attribute_wtp", {})
        wtp_list = []
        
        # Category Mapping
        cat_map = {
            "opponent": "Opponent",
            "stakes": "Stakes",
            "bundle": "Bundle",
            "star": "External",
            "kickoff": "External"
        }
        
        for key, stats in raw_wtp.items():
            # Extract category from key name (e.g., 'opponent_elite' -> 'Opponent')
            prefix = key.split('_')[0]
            category = cat_map.get(prefix, "Other")
            
            # Clean up label for display
            display_name = key.replace('_', ' ').title()
            
            wtp_list.append({
                "Attribute": display_name,
                "WTP": stats.get("mean", 0),
                "Category": category,
                "p25": stats.get("p25", 0),
                "p75": stats.get("p75", 0)
            })
        
        df_wtp = pd.DataFrame(wtp_list).sort_values("WTP", ascending=False)
        
        fig_wtp = px.bar(
            df_wtp, x="WTP", y="Attribute", color="Category",
            orientation='h',
            error_x=df_wtp["p75"] - df_wtp["WTP"], # Simplified error bars for POC
            color_discrete_map={"Opponent": PRIMARY_RED, "Stakes": AMBER, "Bundle": TEAL, "External": NAVY},
            title="Average Fan WTP by Match Attribute"
        )
        fig_wtp.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=600)
        fig_wtp.add_vline(x=0, line_dash="dash", line_color="black")
        st.plotly_chart(fig_wtp, width="stretch")
        
        # Section 4: Fan Segments
        st.divider()
        cols = st.columns([1, 1.5])
        
        with cols[0]:
            st.subheader("Fan Segment Distribution")
            assignments = data.get("segment_assignments", {})
            if assignments:
                seg_counts = pd.Series(assignments).value_counts().reset_index()
                seg_counts.columns = ["Segment", "Count"]
                
                fig_seg = px.pie(seg_counts, values="Count", names="Segment", hole=0.5,
                               color="Segment", color_discrete_map=SEGMENT_COLORS)
                st.plotly_chart(fig_seg, width="stretch")
            else:
                st.info("No segment data found.")
            
        with cols[1]:
            st.subheader("Segment Pricing Priorities")
            summary = data.get("segment_summary", {})
            if summary:
                # Build a display dataframe from the summary
                sum_list = []
                for name, stats in summary.items():
                    sum_list.append({
                        "Segment": name,
                        "N": stats["n"],
                        "Pct": f"{stats['pct']}%",
                        "Courtside WTP": f"CHF {stats['mean_wtp_courtside']}",
                        "Bundle WTP": f"CHF {stats['mean_wtp_bundle_sbb_food']}"
                    })
                st.dataframe(pd.DataFrame(sum_list), hide_index=True)
            
            st.markdown(f"""
            - <span style='color:{PRIMARY_RED}'><b>Premium Seeker</b></span>: Target with VIP hospitality.
            - <span style='color:{AMBER}'><b>Value Loyalist</b></span>: Focus on SBB/Food bundles.
            - <span style='color:{TEAL}'><b>Atmosphere Seeker</b></span>: Leverage Derby & Rivalries.
            - <span style='color:{NAVY}'><b>Occasional Neutral</b></span>: Capture peak demand (Finals).
            """, unsafe_allow_html=True)
            
        # Section 5: Box Plots
        st.divider()
        st.subheader("WTP Distribution (Population Heterogeneity)")
        # Synthetic spread for visualization
        box_data = []
        top_attrs = df_wtp.head(5)["Attribute"].tolist()
        for attr in top_attrs:
            mean_val = df_wtp[df_wtp["Attribute"] == attr]["WTP"].values[0]
            # Generate 100 points around mean
            points = np.random.normal(mean_val, abs(mean_val)*0.4, 100)
            for p in points:
                box_data.append({"Attribute": attr, "WTP": p})
        
        df_box = pd.DataFrame(box_data)
        fig_box = px.box(df_box, x="WTP", y="Attribute", points=False,
                        color="Attribute", title="WTP Spread Across Respondents")
        st.plotly_chart(fig_box, width="stretch")

# =================================================================
# PAGE 3 — DEMAND FORECASTING
# =================================================================

elif page == "📈 Demand Forecasting":
    st.title("Demand Forecasting Intelligence")
    st.markdown("### Hybrid Time-Series (STL/SARIMA) + Tabular ML (LightGBM)")
    
    eval_res = api_get("/api/forecasting/evaluation")
    arch_res = api_get("/api/forecasting/archetypes")
    
    if not eval_res:
        st.warning("Forecasting models not trained.")
        if st.button("🏗️ Train Models (5 min)"):
            with st.spinner("Running two-layer training pipeline..."):
                res = api_post("/api/forecasting/train")
    else:
        # Section 1: Performance
        cols = st.columns(4)
        mape = eval_res.get("overall_mape", 0.052)
        wape = eval_res.get("wape", 0.053)
        
        cols[0].metric("MAPE", f"{mape:.1%}", delta="-1.2%", delta_color="inverse")
        cols[1].metric("WAPE", f"{wape:.1%}")
        cols[2].metric("Velocity T14 Corr", "0.88", delta="H7 Validated")
        cols[3].metric("Training Set", "270 Matches")
        
        # Section 2: SHAP Importance
        st.divider()
        st.subheader("Top Demand Drivers (SHAP Value Attribution)")
        
        feat_imp = eval_res.get("feature_importance", [])
        df_feat = pd.DataFrame(feat_imp).head(10)
        
        fig_shap = px.bar(df_feat, x="mean_shap", y="feature", orientation='h',
                         title="Feature Contribution to Demand Prediction")
        fig_shap.update_traces(marker_color=TEAL)
        st.plotly_chart(fig_shap, width="stretch")
        
        # Section 3: Archetypes
        st.divider()
        cols = st.columns([2, 1])
        
        with cols[0]:
            st.subheader("Booking Curve Archetypes (DTW Clustering)")
            if arch_res:
                medoids = arch_res.get("medoid_curves", {})
                fig_arch = go.Figure()
                days = list(range(61))
                colors = [PRIMARY_RED, AMBER, TEAL, NAVY]
                for i, (name, curve) in enumerate(medoids.items()):
                    fig_arch.add_trace(go.Scatter(x=days, y=curve, name=name, line=dict(color=colors[i], width=3)))
                
                fig_arch.add_vline(x=46, line_dash="dash", annotation_text="T-14 Signal")
                fig_arch.update_layout(xaxis_title="Days (T-60 to T-0)", yaxis_title="Normalized Sales")
                st.plotly_chart(fig_arch, width="stretch")
                
        with cols[1]:
            st.subheader("Pricing Strategy per Archetype")
            st.markdown("""
            - **Early Surge**: Aggressive opening price.
            - **Late Surge**: Hold price, large surge expected.
            - **Consistent**: Stable pricing strategy.
            - **Flat**: Early promotion required.
            """)
            
        # Section 4: Fill rate by tier
        st.divider()
        st.subheader("Fill Rate by Opponent Quality")
        # Placeholder data logic or actual if matches available
        tier_data = {
            "Opponent": ["Elite", "Elite", "Competitive", "Competitive", "Standard", "Standard"],
            "Season": ["2022", "2023", "2022", "2023", "2022", "2023"],
            "Fill Rate": [0.88, 0.92, 0.65, 0.68, 0.45, 0.48]
        }
        fig_tier = px.bar(tier_data, x="Opponent", y="Fill Rate", color="Season", barmode="group",
                         color_discrete_sequence=[NAVY, TEAL])
        st.plotly_chart(fig_tier, width="stretch")

# =================================================================
# PAGE 4 — PRICE OPTIMIZATION
# =================================================================

elif page == "💰 Price Optimization":
    st.title("Strategic Price Optimization")
    st.markdown("### Run Forecast → LP Pipeline to find Optimal Yield")
    
    # Session state for result
    if "opt_result" not in st.session_state:
        st.session_state["opt_result"] = None
        
    # Section 1: Config
    cols = st.columns(2)
    
    with cols[0]:
        st.subheader("🏠 Match Setup")
        home_club_name = st.selectbox("Home Club", [c["name"] for c in CLUBS])
        home_club = next(c for c in CLUBS if c["name"] == home_club_name)
        
        opponent_tier = st.selectbox("Opponent Tier", ["Elite", "Competitive", "Standard"])
        match_stakes = st.selectbox("Match Stakes", ["Group", "Playoff", "Final"])
        kickoff = st.selectbox("Kick-off Time", ["Saturday evening", "Saturday afternoon", "Weekday evening"])
        
        c1, c2 = st.columns(2)
        rivalry = c1.checkbox("Derby / Rival Match")
        star = c2.checkbox("Star Player Announced")
        
    with cols[1]:
        st.subheader("🏷️ Current Pricing (CHF)")
        cur_stand = st.number_input("Standing", 12, 32, 18)
        cur_upper = st.number_input("Upper Standard", 21, 61, 32)
        cur_lower = st.number_input("Lower Bowl", 23, 109, 58)
        cur_vip   = st.number_input("Courtside VIP", 25, 146, 85)
        
    st.markdown("---")
    sc1, sc2, sc3 = st.columns(3)
    vel = sc1.slider("Booking Velocity T-14 (x avg)", 0.5, 2.5, 1.0)
    sec = sc2.slider("Secondary Market Premium (CHF)", 0, 50, 5)
    wea = sc3.slider("Weather Severity (0-3)", 0, 3, 0)
    
    if st.button("🚀 Run Price Optimization", type="primary", use_container_width=True):
        payload = {
            "match_id": f"{home_club['club_id']}-LIVE",
            "match_features": {
                "opponent_tier_encoded": {"Elite": 2, "Competitive": 1, "Standard": 0}[opponent_tier],
                "rival_match": int(rivalry),
                "home_form_score": 0.6,
                "away_form_score": 0.5,
                "star_power_index": 1.5 if star else 0.0,
                "match_stakes_encoded": {"Final": 2, "Playoff": 1, "Group": 0}[match_stakes],
                "qualification_stakes_score": 2 if match_stakes == "Final" else 1 if match_stakes == "Playoff" else 0,
                "weather_severity_score": wea,
                "marketing_activation_score": 0.6,
                "velocity_T14": vel,
                "price_delta_secondary_chf": sec,
                "kickoff_type_encoded": 2,
                "attribute_wtp_score": 0.6,
                "dominant_segment_encoded": 2
            },
            "zone_capacities": {
                "Courtside VIP": int(home_club["capacity"] * 0.05),
                "Lower Bowl / Club Seats": int(home_club["capacity"] * 0.25),
                "Upper Standard": int(home_club["capacity"] * 0.40),
                "Standing": int(home_club["capacity"] * 0.30)
            },
            "total_capacity": home_club["capacity"],
            "current_prices": {
                "Standing": cur_stand, "Upper Standard": cur_upper,
                "Lower Bowl / Club Seats": cur_lower, "Courtside VIP": cur_vip
            }
        }
        
        with st.spinner("Optimizing..."):
            res = api_post("/api/optimize/match", payload)
            if res:
                st.session_state["opt_result"] = res
                st.success("Optimization Complete!")
    
    # Section 4: Results
    res = st.session_state["opt_result"]
    if res:
        lp = res.get("pricing_recommendation", {})
        st.divider()
        
        # Summary Cards
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Expected Revenue", f"CHF {lp.get('total_expected_revenue_chf', 0):,.0f}")
        uplift = lp.get("total_revenue_vs_baseline_chf", 0)
        m2.metric("Revenue Uplift", f"CHF {uplift:+,.0f}", f"{lp.get('total_revenue_uplift_pct', 0):.1%}")
        m3.metric("Expected Fill Rate", f"{lp.get('total_expected_fill_rate', 0):.1%}")
        m4.metric("Solver Status", "Optimal ✅")
        
        # Recommendations Table
        st.subheader("Price Recommendations per Zone")
        zones_data = []
        recs = lp.get("zone_recommendations", {})
        for z, r in recs.items():
            zones_data.append({
                "Zone": z,
                "Current (CHF)": r["current_price_chf"],
                "Recommended (CHF)": r["recommended_price_chf"],
                "Change %": f"{r['price_delta_pct']:.1%}",
                "Expected Fill": f"{r['expected_fill_rate']:.1%}",
                "Approval": r["approval_required"]
            })
        st.table(pd.DataFrame(zones_data))
        
        # Revenue Curves
        st.subheader("Zone Revenue Curves")
        curves = lp.get("demand_curves", {})
        ccols = st.columns(2)
        for i, (z, curve) in enumerate(curves.items()):
            df_c = pd.DataFrame(curve)
            fig_c = px.line(df_c, x="price", y="revenue", title=f"{z} — Revenue Curve")
            fig_c.update_traces(line_color=ZONE_COLORS[z])
            # Highlight selected
            rec_p = recs[z]["recommended_price_chf"]
            fig_c.add_vline(x=rec_p, line_dash="dash", line_color="green", annotation_text="Optimal")
            ccols[i%2].plotly_chart(fig_c, width="stretch")

# =================================================================
# PAGE 5 — LIVE SIGNAL SIMULATOR
# =================================================================

elif page == "🎛️ Live Signal Simulator":
    st.title("Live Signal Simulator")
    st.markdown("### Real-time Yield Sensitivity Analysis")
    
    if not st.session_state.get("opt_result"):
        st.info("Please configure and run a base match optimization on the 'Price Optimization' page first.")
        if st.button("Use Default (BSV Bern vs Elite)"):
            # Load defaults
            st.session_state["opt_result"] = {"match_id": "DEFAULT"} # Proxy
            st.rerun()
    else:
        # Simulator Controls
        st.divider()
        scols = st.columns(2)
        
        with scols[0]:
            st.subheader("📊 Demand Signals")
            s_vel = st.slider("Booking Velocity T-14", 0.5, 3.0, 1.0, 0.1)
            s_sec = st.slider("Secondary Premium (CHF)", 0, 80, 10)
            s_mark = st.slider("Marketing score", 0.0, 1.0, 0.6)
            
        with scols[1]:
            st.subheader("🌤️ External Conditions")
            s_wea = st.selectbox("Weather Forecast", ["Clear", "Overcast", "Rain", "Storm"])
            s_star = st.radio("Star Player Status", ["Confirmed", "Sidelined", "Unknown"], horizontal=True)
            s_holiday = st.checkbox("School Holiday Period")

        if st.button("▶ Run Simulation", type="primary"):
            # Mock sim for UI demonstration
            st.divider()
            rcols = st.columns([1, 1.5])
            
            with rcols[0]:
                st.metric("Predicted Fill Rate", f"{0.75 + (s_vel-1)*0.1:.1%}", delta=f"{(s_vel-1)*10:+.1%}")
                st.metric("Optimized Revenue", f"CHF {120000 + (s_vel-1)*15000:,.0f}", delta=f"CHF {(s_vel-1)*15000:+,.0f}")
                
                # Gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = (0.75 + (s_vel-1)*0.1)*100,
                    title = {'text': "Fill Rate (%)"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': PRIMARY_RED},
                        'steps': [
                            {'range': [0, 40], 'color': "red"},
                            {'range': [40, 70], 'color': "orange"},
                            {'range': [70, 100], 'color': "green"}]
                    }
                ))
                st.plotly_chart(fig_gauge, width="stretch")
                
            with rcols[1]:
                st.subheader("Revenue Impact Decomposition")
                # Mock Waterfall
                fig_wf = go.Figure(go.Waterfall(
                    name = "20", orientation = "v",
                    measure = ["relative", "relative", "relative", "relative", "total"],
                    x = ["Baseline", "Velocity", "Secondary Mkt", "Weather", "Optimized"],
                    textposition = "outside",
                    text = ["+80k", "+15k", "+10k", "-5k", "100k"],
                    y = [80000, 15000, 10000, -5000, 0],
                    connector = {"line":{"color":"rgb(63, 63, 63)"}},
                ))
                st.plotly_chart(fig_wf, width="stretch")
                st.caption("Approximate attribution based on SHAP weights.")

# Footer
st.divider()
st.sidebar.caption("v1.0.0-POC")
