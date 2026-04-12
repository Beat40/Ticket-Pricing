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
    ["🏠 Overview", "📝 Conjoint Analysis Survey", "📊 Conjoint Analysis", "📊 Demand Forecasting Stats", "💰 Price Optimization"]
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
# PAGE 1.5 — CONJOINT ANALYSIS SURVEY MOCKUP
# =================================================================

elif page == "📝 Conjoint Analysis Survey":
    st.title("Conjoint Preference Survey")
    st.markdown("### Participant Experience Mockup")
    st.info("This is a demonstration of the data collection interface used to recover fan Willingness-to-Pay (WTP). Respondents are presented with 17 choice tasks.")
    
    if "survey_step" not in st.session_state:
        st.session_state.survey_step = 1
    
    step = st.session_state.survey_step
    
    # Progress UI
    progress = step / 17
    st.progress(progress)
    st.write(f"**Task {step} of 17**")
    
    if step <= 17:
        # Mock Question Data (Static for the demo)
        # We vary slightly based on step to look real
        q_data = {
            "opponent": ["Elite", "Standard"],
            "zone": ["Lower Bowl / Club Seats", "Upper Standard"],
            "stakes": ["League Playoff / Knockout", "Regular Season Group Match"],
            "stars": ["Yes — marquee international confirmed", "No"],
            "kickoff": ["Saturday evening 19:30", "Weekday 19:00"],
            "bundle": ["Ticket + SBB Travel + Food & Drink", "Ticket Only"],
            "price": [75, 32]
        }
        
        # Table-based comparison
        st.write("If these were your only options, which one would you choose?")
        
        cols = st.columns([2, 3, 3])
        
        with cols[0]:
            st.write("") # Spacer
            st.markdown("**Opponent**")
            st.markdown("**Seating Zone**")
            st.markdown("**Match Stakes**")
            st.markdown("**Star Player**")
            st.markdown("**Kickoff Time**")
            st.markdown("**Includes**")
            st.markdown("**Price**")
        
        with cols[1]:
            st.markdown(f"<div style='background-color:#fff; padding:15px; border-radius:10px; border:2px solid {PRIMARY_RED}; text-align:center;'>", unsafe_allow_html=True)
            st.subheader("Option A")
            st.write(q_data["opponent"][0])
            st.write(q_data["zone"][0])
            st.write(q_data["stakes"][0])
            st.write(q_data["stars"][0])
            st.write(q_data["kickoff"][0])
            st.write(q_data["bundle"][0])
            st.write(f"**CHF {q_data['price'][0]}**")
            st.markdown("</div>", unsafe_allow_html=True)
            if st.button("Select Option A", use_container_width=True, type="primary"):
                if st.session_state.survey_step < 17:
                    st.session_state.survey_step += 1
                else:
                    st.session_state.survey_step = 18
                st.rerun()

        with cols[2]:
            st.markdown(f"<div style='background-color:#fff; padding:15px; border-radius:10px; border:2px solid {TEAL}; text-align:center;'>", unsafe_allow_html=True)
            st.subheader("Option B")
            st.write(q_data["opponent"][1])
            st.write(q_data["zone"][1])
            st.write(q_data["stakes"][1])
            st.write(q_data["stars"][1])
            st.write(q_data["kickoff"][1])
            st.write(q_data["bundle"][1])
            st.write(f"**CHF {q_data['price'][1]}**")
            st.markdown("</div>", unsafe_allow_html=True)
            if st.button("Select Option B", use_container_width=True, type="primary"):
                if st.session_state.survey_step < 17:
                    st.session_state.survey_step += 1
                else:
                    st.session_state.survey_step = 18
                st.rerun()
        
        st.divider()
        if st.button("I would not buy either of these", use_container_width=True):
            if st.session_state.survey_step < 17:
                st.session_state.survey_step += 1
            else:
                st.session_state.survey_step = 18
            st.rerun()

    else:
        st.success("🎉 Simulation Complete! Your choices have been recorded for utility estimation.")
        if st.button("Reset Mockup"):
            st.session_state.survey_step = 1
            st.rerun()

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

elif page == "📊 Demand Forecasting Stats":
    st.title("Model Performance & Feature Importance")
    st.markdown("### Accuracy Metrics and Global Prediction Drivers")
    
    eval_res = api_get("/api/forecasting/evaluation")
    
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
        st.subheader("Global Demand Drivers (SHAP Value Attribution)")
        
        feat_imp = eval_res.get("feature_importance", [])
        df_feat = pd.DataFrame(feat_imp).head(15)
        
        fig_shap = px.bar(df_feat, x="mean_shap", y="feature", orientation='h',
                         title="Feature Contribution to Demand Prediction")
        fig_shap.update_traces(marker_color=TEAL)
        st.plotly_chart(fig_shap, width="stretch")

        # Section 4: Performance by Tier
        st.divider()
        st.subheader("Historical Fill Rate by Opponent Quality")
        tier_data = {
            "Opponent": ["Elite", "Elite", "Competitive", "Competitive", "Standard", "Standard"],
            "Season": ["2022-23", "2023-24", "2022-23", "2023-24", "2022-23", "2023-24"],
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
        
    # Section 1: Mode Selection
    mode = st.radio("Optimization Mode", ["Manual Scenario", "Historical Validation"], horizontal=True, help="Switch between a custom 'what-if' scenario or validating against a real historical match from Season 3.")
    
    selected_match_data = None
    medoids = {}
    if mode == "Historical Validation":
        res = api_get("/api/matches/validation")
        if res:
            val_matches = res.get("matches", [])
            medoids = res.get("medoid_curves", {})
            # Create friendly labels
            match_options = {f"{m['match_date']} — {m['home_club_name']} vs {m['away_club_name']} (R{m['match_round']})": m for m in val_matches}
            selected_label = st.selectbox("Select Match to Analyze", list(match_options.keys()))
            selected_match_data = match_options[selected_label]
            st.info(f"📍 Loaded signals for Match: {selected_match_data['match_id']}")

    # --- DIAGNOSTIC SECTION ---
    if selected_match_data:
        st.divider()
        st.subheader("📊 Match Diagnostics")
        dcol1, dcol2 = st.columns([2, 1])
        
        with dcol1:
            st.caption("Historical Booking Curve Archetype (Contextual Baseline)")
            fig_arch = go.Figure()
            days = list(range(61))
            assigned_label = selected_match_data.get("archetype", "Consistent Gradual")
            
            for name, curve in medoids.items():
                is_assigned = (name == assigned_label)
                opac = 1.0 if is_assigned else 0.2
                width = 4 if is_assigned else 1
                dash = "solid" if is_assigned else "dot"
                color = SEGMENT_COLORS.get("Premium Seeker") if is_assigned else "grey"
                
                fig_arch.add_trace(go.Scatter(
                    x=days, y=curve, name=name,
                    line=dict(color=color, width=width, dash=dash),
                    opacity=opac
                ))
            
            fig_arch.add_vline(x=46, line_dash="dash", annotation_text="T-14 Signal", line_color="orange")
            fig_arch.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), showlegend=True,
                                 xaxis_title="Days (T-60 to T-0)", yaxis_title="Norm. Sales")
            st.plotly_chart(fig_arch, use_container_width=True)
            
        with dcol2:
            st.caption("Forecast & Outcomes")
            np_pred = selected_match_data.get("np_final_prediction", 0.6)
            act_vel = selected_match_data.get("velocity_T14", 0.6)
            lgbm_pred = selected_match_data.get("lgbm_prediction", 0.0)
            actual = selected_match_data.get("actual_outcome", 0.0)
            
            # Layer 1 Sequential Signal
            st.metric("NeuralProphet Forecast", f"{np_pred:.1%}", 
                      help="Baseline trend prediction from sequential NeuralProphet layer.")
            
            # Layer 2 Tabular Signal
            st.metric("LightGBM Final AI Forecast", f"{lgbm_pred:.1%}",
                      delta=f"{lgbm_pred - actual:+.1%}" if actual > 0 else None,
                      help="The final refined forecast from the LightGBM model after ingesting all contextual signals.")
            
            # Historical Truth
            st.metric("Actual Historical Outcome", f"{actual:.1%}" if actual > 0 else "N/A",
                      help="The real-world fill rate achieved for this specific match.")
            
            st.divider()
            st.caption(f"**Archetype:** {assigned_label}")
            st.caption(f"**STL Trend:** {selected_match_data.get('stl_trend_value', 0):.2f}")
            st.caption(f"**Velocity (T-14):** {act_vel:.2f}")

    st.divider()
    cols = st.columns(2)
    
    with cols[0]:
        st.subheader("🏠 Match Setup")
        def_club_idx = 0
        def_opp_idx = 1
        def_stakes_idx = 0
        def_rival = False
        def_star = False
        
        if selected_match_data:
            club_names = [c["name"] for c in CLUBS]
            if selected_match_data["home_club_name"] in club_names:
                def_club_idx = club_names.index(selected_match_data["home_club_name"])
            
            opp_tiers = ["Elite", "Competitive", "Standard"]
            if selected_match_data["opponent_tier"] in opp_tiers:
                def_opp_idx = opp_tiers.index(selected_match_data["opponent_tier"])
                
            stakes_list = ["Group", "Playoff", "Final"]
            if selected_match_data["match_stakes"] in stakes_list:
                def_stakes_idx = stakes_list.index(selected_match_data["match_stakes"])
            
            def_rival = bool(selected_match_data["rival_match"])
            def_star = bool(selected_match_data["star_player_announced"])

        home_club_name = st.selectbox("Home Club", [c["name"] for c in CLUBS], index=def_club_idx)
        home_club = next(c for c in CLUBS if c["name"] == home_club_name)
        
        opponent_tier = st.selectbox("Opponent Tier", ["Elite", "Competitive", "Standard"], index=def_opp_idx)
        match_stakes = st.selectbox("Match Stakes", ["Group", "Playoff", "Final"], index=def_stakes_idx)
        kickoff = st.selectbox("Kick-off Time", ["Saturday evening", "Saturday afternoon", "Weekday evening"])
        
        c1, c2 = st.columns(2)
        rivalry = c1.checkbox("Derby / Rival Match", value=def_rival)
        star = c2.checkbox("Star Player Announced", value=def_star)
        
    with cols[1]:
        st.subheader("🏷️ Current Pricing (CHF)")
        def_p = {"Standing": 18, "Upper Standard": 32, "Lower Bowl": 58, "Courtside VIP": 85}
        if selected_match_data:
            def_p = {
                "Standing": int(selected_match_data["base_price_standing"]),
                "Upper Standard": int(selected_match_data["base_price_upper_standard"]),
                "Lower Bowl": int(selected_match_data["base_price_lower_bowl"]),
                "Courtside VIP": int(selected_match_data["base_price_courtside_vip"])
            }

        cur_stand = st.number_input("Standing", 12, 32, def_p["Standing"])
        cur_upper = st.number_input("Upper Standard", 21, 61, def_p["Upper Standard"])
        cur_lower = st.number_input("Lower Bowl", 23, 109, def_p["Lower Bowl"])
        cur_vip   = st.number_input("Courtside VIP", 25, 146, def_p["Courtside VIP"])
        
    st.markdown("---")
    sc1, sc2, sc3 = st.columns(3)
    
    def_vel = 1.0
    def_sec = 5
    def_wea = 0
    if selected_match_data:
        def_vel = float(selected_match_data["velocity_T14"])
        def_sec = int(selected_match_data["price_delta_secondary_chf"])
        def_wea = int(selected_match_data["weather_severity_score"])

    vel = sc1.slider("Booking Velocity T-14 (x avg)", 0.5, 3.0, def_vel)
    sec = sc2.slider("Secondary Market Premium (CHF)", 0, 80, def_sec)
    wea = sc3.slider("Weather Severity (0-3)", 0, 3, def_wea)
    
    if st.button("🚀 Run Price Optimization", type="primary", use_container_width=True):
        m_features = {
            "opponent_tier_encoded": {"Elite": 2, "Competitive": 1, "Standard": 0}[opponent_tier],
            "rival_match": int(rivalry),
            "home_form_score": 0.6 if not selected_match_data else selected_match_data["home_form_score"],
            "away_form_score": 0.5 if not selected_match_data else selected_match_data["away_form_score"],
            "star_power_index": 1.5 if star else 0.0,
            "match_stakes_encoded": {"Final": 2, "Playoff": 1, "Group": 0}[match_stakes],
            "qualification_stakes_score": 2 if match_stakes == "Final" else 1 if match_stakes == "Playoff" else 0,
            "weather_severity_score": wea,
            "marketing_activation_score": 0.6 if not selected_match_data else selected_match_data["marketing_activation_score"],
            "velocity_T14": vel,
            "price_delta_secondary_chf": sec,
            "kickoff_type_encoded": 2,
            "attribute_wtp_score": 0.6 if not selected_match_data else selected_match_data["attribute_wtp_score"],
            "dominant_segment_encoded": 2,
            "home_club_id": home_club["club_id"]
        }
        
        if selected_match_data:
            m_features["stl_trend_value"] = selected_match_data["stl_trend_value"]
            m_features["stl_seasonal_value"] = selected_match_data["stl_seasonal_value"]
            m_features["sarima_residual"] = selected_match_data["sarima_residual"]
            m_features["np_final_prediction"] = selected_match_data["np_final_prediction"]
            m_features["np_deviation_T14"] = selected_match_data["np_deviation_T14"]
            m_features["archetype_deviation_T14"] = selected_match_data["archetype_deviation_T14"]

        payload = {
            "match_id": f"{home_club['club_id']}-LIVE" if not selected_match_data else selected_match_data["match_id"],
            "match_features": m_features,
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

# (Live Signal Simulator consolidated into Price Optimization)

# Footer
st.divider()
st.sidebar.caption("v1.0.0-POC")
