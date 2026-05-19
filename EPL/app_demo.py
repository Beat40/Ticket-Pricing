import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pulp

# --- CONFIG & STYLING ---
st.set_page_config(page_title="EPL Pricing Intelligence", layout="wide", initial_sidebar_state="expanded")

C_PURPLE = "#3D195B"
C_GREEN = "#00FF85"
C_LIGHT = "#F3F0EC"
C_DARK = "#1C1C1C"

st.markdown(f"""
    <style>
    .stApp {{ background-color: {C_LIGHT}; }}
    h1, h2, h3, h4, h5, h6, .css-10trblm {{ color: {C_PURPLE} !important; }}
    .stButton>button {{ background-color: {C_PURPLE}; color: white; border: none; }}
    .stButton>button:hover {{ background-color: {C_GREEN}; color: {C_PURPLE}; border: none; }}
    .css-1d391kg {{ background-color: {C_DARK}; }}
    .st-eb {{ background-color: white; }}
    .metric-card {{ background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-left: 5px solid {C_GREEN}; }}
    </style>
""", unsafe_allow_html=True)

# --- DATA REGISTRY ---
CLUBS = {
    "ARS": {"name": "Arsenal", "tier": "big_six", "capacity": 60704, "dynamic_pct": 0.0545, "base_prob": 0.8},
    "TOT": {"name": "Tottenham Hotspur", "tier": "big_six", "capacity": 62850, "dynamic_pct": 0.0917, "base_prob": 0.75},
    "MUN": {"name": "Manchester United", "tier": "big_six", "capacity": 74310, "dynamic_pct": 0.1290, "base_prob": 0.75},
    "LIV": {"name": "Liverpool", "tier": "big_six", "capacity": 61276, "dynamic_pct": 0.2114, "base_prob": 0.8},
    "MCI": {"name": "Manchester City", "tier": "big_six", "capacity": 53400, "dynamic_pct": 0.0996, "base_prob": 0.85},
    "CHE": {"name": "Chelsea", "tier": "big_six", "capacity": 40834, "dynamic_pct": 0.0708, "base_prob": 0.75},
    "AVL": {"name": "Aston Villa", "tier": "established_mid", "capacity": 42682, "dynamic_pct": 0.1067, "base_prob": 0.6},
    "NEW": {"name": "Newcastle United", "tier": "established_mid", "capacity": 52305, "dynamic_pct": 0.0966, "base_prob": 0.6},
    "WHU": {"name": "West Ham United", "tier": "established_mid", "capacity": 62500, "dynamic_pct": 0.2520, "base_prob": 0.55},
    "BHA": {"name": "Brighton & Hove Albion", "tier": "established_mid", "capacity": 31800, "dynamic_pct": 0.1157, "base_prob": 0.55},
    "WOL": {"name": "Wolverhampton Wanderers", "tier": "established_mid", "capacity": 32050, "dynamic_pct": 0.1864, "base_prob": 0.5},
    "FUL": {"name": "Fulham", "tier": "established_mid", "capacity": 29600, "dynamic_pct": 0.2230, "base_prob": 0.5},
    "CPL": {"name": "Crystal Palace", "tier": "established_mid", "capacity": 25486, "dynamic_pct": 0.1800, "base_prob": 0.5},
    "EVE": {"name": "Everton", "tier": "smaller_club", "capacity": 39414, "dynamic_pct": 0.1701, "base_prob": 0.4},
    "BRE": {"name": "Brentford", "tier": "smaller_club", "capacity": 17250, "dynamic_pct": 0.2100, "base_prob": 0.45},
    "NFO": {"name": "Nottingham Forest", "tier": "smaller_club", "capacity": 30445, "dynamic_pct": 0.2015, "base_prob": 0.4},
    "LEI": {"name": "Leicester City", "tier": "smaller_club", "capacity": 32261, "dynamic_pct": 0.1870, "base_prob": 0.45},
    "BOU": {"name": "AFC Bournemouth", "tier": "smaller_club", "capacity": 11307, "dynamic_pct": 0.2230, "base_prob": 0.4},
    "IPS": {"name": "Ipswich Town", "tier": "smaller_club", "capacity": 30000, "dynamic_pct": 0.2400, "base_prob": 0.3},
    "SOU": {"name": "Southampton", "tier": "smaller_club", "capacity": 32384, "dynamic_pct": 0.2474, "base_prob": 0.3}
}

CLUB_RIVALS = {
    "ARS": ["TOT", "CHE"], "TOT": ["ARS", "CHE"], "MCI": ["MUN", "LIV"],
    "MUN": ["MCI", "LIV"], "LIV": ["EVE", "MUN", "MCI"], "CHE": ["ARS", "TOT"]
}

# --- LEAGUE SIMULATION LOGIC ---
@st.cache_data
def simulate_season():
    # Deterministic simulation for consistency
    np.random.seed(42)
    club_ids = list(CLUBS.keys())
    schedule = []
    # simple double round robin
    for i in range(len(club_ids)):
        for j in range(len(club_ids)):
            if i != j:
                schedule.append((club_ids[i], club_ids[j]))
    
    np.random.shuffle(schedule)
    matches_per_mw = 10
    
    standings_history = []
    full_schedule = []
    points = {c: 0 for c in club_ids}
    gd = {c: 0 for c in club_ids}
    gs = {c: 0 for c in club_ids}
    
    for mw in range(1, 39):
        mw_matches = schedule[(mw-1)*matches_per_mw : mw*matches_per_mw]
        for home, away in mw_matches:
            full_schedule.append({
                "Matchweek": mw,
                "Home Club": CLUBS[home]["name"],
                "Away Club": CLUBS[away]["name"],
                "Home_ID": home,
                "Away_ID": away
            })
            # Result prob
            h_prob = CLUBS[home]["base_prob"]
            a_prob = CLUBS[away]["base_prob"] * 0.8
            norm = h_prob + a_prob + 0.3 # draw
            draw_prob = 0.3 / norm
            h_win = h_prob / norm
            
            r = np.random.rand()
            if r < h_win:
                points[home] += 3
                h_g = np.random.randint(1, 4)
                a_g = np.random.randint(0, h_g)
            elif r < h_win + draw_prob:
                points[home] += 1
                points[away] += 1
                h_g = np.random.randint(0, 3)
                a_g = h_g
            else:
                points[away] += 3
                a_g = np.random.randint(1, 4)
                h_g = np.random.randint(0, a_g)
                
            gd[home] += (h_g - a_g)
            gd[away] += (a_g - h_g)
            gs[home] += h_g
            gs[away] += a_g
            
        # Snapshot
        mw_table = []
        for c in club_ids:
            mw_table.append({
                "Club": CLUBS[c]["name"],
                "ID": c,
                "Points": points[c],
                "GD": gd[c],
                "GS": gs[c]
            })
        mw_table.sort(key=lambda x: (x["Points"], x["GD"], x["GS"]), reverse=True)
        for idx, row in enumerate(mw_table):
            row["Pos"] = idx + 1
        standings_history.append(mw_table)
        
    return standings_history, pd.DataFrame(full_schedule)

history, schedule_df = simulate_season()

# --- STATE INIT ---
def init_state(key, default):
    if key not in st.session_state:
        st.session_state[key] = default

init_state("mw", 20)
default_match = schedule_df[schedule_df["Matchweek"]==st.session_state.mw].iloc[0]
init_state("home_club", default_match["Home_ID"])
init_state("away_club", default_match["Away_ID"])
init_state("tv_slot", 1)
init_state("is_derby", False)
init_state("festive", False)
init_state("fatigue", False)
init_state("new_manager", False)
init_state("distance", 150)
init_state("home_form", 0.8)
init_state("away_form", 0.7)
init_state("star_power", 8)

# --- PRESETS ---
def apply_preset(preset):
    if preset == "Title":
        m = schedule_df[(schedule_df["Home_ID"].isin(["ARS","MCI"])) & (schedule_df["Away_ID"].isin(["ARS","MCI"]))].iloc[-1]
        st.session_state.home_form = 0.9
        st.session_state.away_form = 0.9
        st.session_state.star_power = 9
    elif preset == "Rel":
        m = schedule_df[(schedule_df["Home_ID"].isin(["IPS","SOU"])) & (schedule_df["Away_ID"].isin(["IPS","SOU"]))].iloc[-1]
        st.session_state.home_form = 0.3
        st.session_state.away_form = 0.2
        st.session_state.star_power = 2
    elif preset == "Std":
        m = schedule_df[(schedule_df["Home_ID"].isin(["WOL","FUL"])) & (schedule_df["Away_ID"].isin(["WOL","FUL"]))].iloc[-1]
        st.session_state.home_form = 0.5
        st.session_state.away_form = 0.5
        st.session_state.star_power = 4
        
    st.session_state.mw = int(m["Matchweek"])
    st.session_state.home_club = m["Home_ID"]
    st.session_state.away_club = m["Away_ID"]

st.title("EPL Pricing Intelligence Live Season Simulator")
st.markdown("##### PwC Client Presentation Prototype | Commercial Director Edition")

col1, col2, col3 = st.columns(3)
if col1.button("🏆 Title Race (ARS vs MCI)", use_container_width=True): apply_preset("Title")
if col2.button("⚠️ Relegation Six-Pointer (IPS vs SOU)", use_container_width=True): apply_preset("Rel")
if col3.button("⚽ Standard Fixture (WOL vs FUL)", use_container_width=True): apply_preset("Std")

st.markdown("---")
st.markdown("#### Season Schedule & Selection")

st.markdown("**Matchweek Timeline:**")
selected_mw = st.radio("Matchweek Timeline", range(1, 39), index=st.session_state.mw-1, horizontal=True, label_visibility="collapsed")
if selected_mw != st.session_state.mw:
    st.session_state.mw = selected_mw
    df_mw = schedule_df[schedule_df["Matchweek"] == st.session_state.mw]
    if not ((df_mw["Home_ID"] == st.session_state.home_club) & (df_mw["Away_ID"] == st.session_state.away_club)).any():
        st.session_state.home_club = df_mw.iloc[0]["Home_ID"]
        st.session_state.away_club = df_mw.iloc[0]["Away_ID"]

df_filtered = schedule_df[schedule_df["Matchweek"] == st.session_state.mw].reset_index(drop=True)

def highlight_selected_match(row):
    if row["Home Club"] == CLUBS[st.session_state.home_club]["name"] and row["Away Club"] == CLUBS[st.session_state.away_club]["name"]:
        return ['background-color: rgba(0, 255, 133, 0.3)'] * len(row)
    return [''] * len(row)

event = st.dataframe(
    df_filtered[["Matchweek", "Home Club", "Away Club"]].style.apply(highlight_selected_match, axis=1),
    on_select="rerun",
    selection_mode="single-row",
    use_container_width=True,
    hide_index=True
)

if hasattr(event, 'selection') and event.selection is not None and len(event.selection.rows) > 0:
    selected_idx = event.selection.rows[0]
    row = df_filtered.iloc[selected_idx]
    st.session_state.home_club = row["Home_ID"]
    st.session_state.away_club = row["Away_ID"]

# --- SIDEBAR ---
with st.sidebar:
    with st.expander("Model Info", expanded=True):
        st.markdown("""
        **Training Data:** 760 matches (22/23 train, 23/24 val)  
        **Overall MAPE:** 21.12%  
        **Architecture:** STL(19) + SARIMA(38) + LightGBM(21 features)
        """)

# --- STAKES ENGINE LOGIC ---
def compute_stakes(h_pos, a_pos, h_pts, a_pts, mw, table):
    urgency = 0.1 + (mw - 1) * (0.9 / 37)
    games_remaining = 38 - mw + 1
    max_pts = games_remaining * 3
    
    leader_pts = table[0]["Points"]
    fourth_pts = table[3]["Points"]
    seventh_pts = table[6]["Points"]
    drop_pts = table[17]["Points"]
    
    h_title_gap = leader_pts - h_pts
    a_title_gap = leader_pts - a_pts
    title_score = 0.0
    if h_title_gap <= max_pts * 0.5 and a_title_gap <= max_pts * 0.5:
        closeness = 1 - (h_title_gap + a_title_gap) / (max_pts * 2 + 1)
        title_score = closeness * urgency * 1.0
        
    h_t4_gap = fourth_pts - h_pts
    a_t4_gap = fourth_pts - a_pts
    t4_score = 0.0
    if (h_pos <= 6 and h_t4_gap <= max_pts * 0.6) or (a_pos <= 6 and a_t4_gap <= max_pts * 0.6):
        rel = 1 - min(max(0, h_t4_gap), max(0, a_t4_gap)) / (max_pts + 1)
        t4_score = rel * urgency * 0.8
        
    h_rel_gap = h_pts - drop_pts
    a_rel_gap = a_pts - drop_pts
    rel_score = 0.0
    h_threat = h_pos >= 14 and h_rel_gap <= max_pts * 0.5
    a_threat = a_pos >= 14 and a_rel_gap <= max_pts * 0.5
    if h_threat and a_threat:
        closeness = 1 - (abs(h_rel_gap) + abs(a_rel_gap)) / (max_pts * 2 + 1)
        rel_score = closeness * urgency * 0.85
    elif h_threat or a_threat:
        closeness = 1 - min(abs(h_rel_gap), abs(a_rel_gap)) / (max_pts + 1)
        rel_score = closeness * urgency * 0.65
        
    h_eur_gap = seventh_pts - h_pts
    a_eur_gap = seventh_pts - a_pts
    eur_score = 0.0
    if (6 <= h_pos <= 10 and h_eur_gap <= max_pts * 0.5) or (6 <= a_pos <= 10 and a_eur_gap <= max_pts * 0.5):
        rel = 1 - min(max(0, h_eur_gap), max(0, a_eur_gap)) / (max_pts + 1)
        eur_score = rel * urgency * 0.55
        
    final_score = float(np.clip(max(title_score, t4_score, rel_score, eur_score), 0, 1))
    
    if final_score >= 0.75:
        label = "Title Decider" if title_score == final_score else ("Relegation Six-Pointer" if rel_score == final_score else "Top Four Decider")
    elif final_score >= 0.5:
        label = "High Stakes"
    elif final_score >= 0.3:
        label = "European Spot"
    else:
        label = "Standard"
        
    return {"title": title_score, "top4": t4_score, "rel": rel_score, "euro": eur_score, "final": final_score, "label": label, "urgency": urgency}


current_table = history[st.session_state.mw - 1]
h_data = next(r for r in current_table if r["ID"] == st.session_state.home_club)
a_data = next(r for r in current_table if r["ID"] == st.session_state.away_club)

stakes = compute_stakes(h_data["Pos"], a_data["Pos"], h_data["Points"], a_data["Points"], st.session_state.mw, current_table)

archs = {
    "Immediate Sellout": {"t_mid": 5, "k": 0.40},
    "Early Surge": {"t_mid": 35, "k": 0.15},
    "Consistent Gradual": {"t_mid": 45, "k": 0.10},
    "Late Surge": {"t_mid": 52, "k": 0.18},
    "Flat/Slow": {"t_mid": 48, "k": 0.06}
}

h_tier = CLUBS[st.session_state.home_club]["tier"]
if h_tier == "big_six" and stakes["final"] > 0.6: sel_arch = "Immediate Sellout"
elif stakes["final"] > 0.4: sel_arch = "Early Surge"
elif "Relegation" in stakes["label"]: sel_arch = "Late Surge"
elif h_tier == "smaller_club" and stakes["final"] < 0.2: sel_arch = "Flat/Slow"
else: sel_arch = "Consistent Gradual"

cap = CLUBS[st.session_state.home_club]["capacity"]
dyn_pct = CLUBS[st.session_state.home_club]["dynamic_pct"]
dyn_seats = int(cap * dyn_pct)

# --- EXPANDERS ---
with st.expander("SECTION 1 — LEAGUE TABLE & STAKES ENGINE", expanded=True):
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Live League Table")
        df_table = pd.DataFrame(current_table)[["Pos", "Club", "Points", "GD"]]
        
        def highlight_rows(row):
            if row['Pos'] <= 4: return ['background-color: rgba(4, 245, 255, 0.2)'] * len(row)
            elif row['Pos'] <= 7: return ['background-color: rgba(255, 165, 0, 0.2)'] * len(row)
            elif row['Pos'] >= 18: return ['background-color: rgba(255, 40, 130, 0.2)'] * len(row)
            return [''] * len(row)
            
        st.dataframe(df_table.style.apply(highlight_rows, axis=1), height=400, use_container_width=True)
        
    with col2:
        st.markdown("#### Stakes Engine Panel")
        st.markdown(f"**Fixture:** {CLUBS[st.session_state.home_club]['name']} vs {CLUBS[st.session_state.away_club]['name']}")
        
        fig = go.Figure()
        fig.add_trace(go.Bar(y=['Dimensions'], x=[stakes['title']], name='Title', marker_color='gold', orientation='h'))
        fig.add_trace(go.Bar(y=['Dimensions'], x=[stakes['top4']], name='Top 4', marker_color='#04F5FF', orientation='h'))
        fig.add_trace(go.Bar(y=['Dimensions'], x=[stakes['euro']], name='Euro', marker_color='orange', orientation='h'))
        fig.add_trace(go.Bar(y=['Dimensions'], x=[stakes['rel']], name='Relegation', marker_color='#FF2882', orientation='h'))
        fig.update_layout(barmode='group', height=150, margin=dict(l=0, r=0, t=0, b=0), plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Match Stakes Score", f"{stakes['final']:.2f}")
        c2.metric("Urgency Multiplier", f"{stakes['urgency']:.2f}")
        c3.markdown(f"<div style='margin-top:25px; padding:10px; background-color:{C_PURPLE}; color:white; border-radius:5px; text-align:center; font-weight:bold;'>{stakes['label']}</div>", unsafe_allow_html=True)
        
        st.info("This score directly feeds into the demand forecast as a feature, and into the LP optimizer's price ceiling.")

with st.expander("SECTION 2 — TICKET SALE LIFECYCLE & T-45 TRIGGER", expanded=False):
    st.markdown("#### Swimlane Timeline")
    fig = go.Figure()
    fig.add_shape(type="rect", x0=-60, x1=-45, y0=0, y1=1, fillcolor="lightgrey", line_width=0, opacity=0.5)
    fig.add_annotation(x=-52.5, y=0.5, text="Member Ballot Window", showarrow=False)
    
    fig.add_shape(type="rect", x0=-45, x1=-7, y0=0, y1=1, fillcolor=C_GREEN, line_width=0, opacity=0.3)
    fig.add_annotation(x=-26, y=0.5, text="General Sale Window", showarrow=False)
    
    fig.add_shape(type="rect", x0=-7, x1=0, y0=0, y1=1, fillcolor=C_PURPLE, line_width=0, opacity=0.3)
    fig.add_annotation(x=-3.5, y=0.5, text="Final Window", showarrow=False, font=dict(color="white"))
    
    fig.add_shape(type="line", x0=-45, x1=-45, y0=0, y1=1.2, line=dict(color="red", width=3, dash="dash"))
    fig.add_annotation(x=-45, y=1.3, text="<b>LP Optimizer Triggers Here</b>", showarrow=False, font=dict(color="red"))
    
    fig.update_layout(xaxis=dict(range=[-60, 5], title="Days to Kickoff"), yaxis=dict(showticklabels=False, range=[0, 1.5]), height=150, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("The optimizer waits 15 days to observe how many members are competing in the ballot. This velocity signal (velocity_T14, velocity_T7) is the 6th most important feature in the demand model. Only then does it set the price for the public sale window — once, not dynamically mid-sale.")

    st.markdown("#### Booking Curve Archetype")
    
    t_vals = np.linspace(0, 60, 100)
    fig = go.Figure()
    for name, params in archs.items():
        y = 1 / (1 + np.exp(-params["k"] * (t_vals - params["t_mid"])))
        width = 4 if name == sel_arch else 1
        color = C_PURPLE if name == sel_arch else "lightgrey"
        fig.add_trace(go.Scatter(x=60-t_vals, y=y, name=name, line=dict(color=color, width=width)))
        
    fig.add_shape(type="line", x0=45, x1=45, y0=0, y1=1, line=dict(color="red", width=2, dash="dash"))
    fig.update_layout(xaxis=dict(autorange="reversed", title="Days to Kickoff"), yaxis=dict(title="% Sold"), height=300)
    st.plotly_chart(fig, use_container_width=True)

with st.expander("SECTION 3 — DEMAND FORECAST: ALL INPUTS & ENGINE", expanded=False):
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.markdown("#### Feature Inputs")
        
        st.markdown("**Match Context (Auto-populated)**")
        a_tier_encoded = {"smaller_club": 0, "established_mid": 1, "big_six": 2}[CLUBS[st.session_state.away_club]["tier"]]
        st.caption(f"`opponent_tier_encoded`: {a_tier_encoded} | `home_pos`: {h_data['Pos']} | `away_pos`: {a_data['Pos']} | `games_remaining`: {38 - st.session_state.mw}")
        
        st.markdown("**EPL-Specific Features**")
        st.toggle("is_derby", key="is_derby")
        st.selectbox("tv_slot_encoded", [0, 1, 2], format_func=lambda x: ["Sat 15:00 Std", "Sat 12:30 / Sun Prime", "Mon/Fri Night"][x], key="tv_slot")
        tv_effects = {0: "−3%", 1: "+4%", 2: "−2%"}
        st.caption(f"Saturday 3pm matches are subject to a UK broadcast blackout — no live domestic TV, which slightly reduces casual interest. Effect on fill rate: {tv_effects[st.session_state.tv_slot]}")
        
        col_a, col_b = st.columns(2)
        col_a.toggle("festive_fixture", key="festive")
        col_b.toggle("european_midweek_fatigue", key="fatigue")
        col_a.toggle("manager_new_bonus", key="new_manager")
        col_b.slider("away_distance_km", 50, 500, key="distance")
        
        st.markdown("**Club Momentum**")
        st.slider("home_form_score", 0.0, 1.0, key="home_form")
        st.slider("away_form_score", 0.0, 1.0, key="away_form")
        st.caption("Away fan propensity affects atmosphere and broadcasted crowd quality.")
        st.slider("star_power_index", 0, 10, key="star_power")

    with c2:
        st.markdown("#### Time-Series Layer")
        stl_val = 0.98 if h_tier == "big_six" else (0.8 if h_tier == "established_mid" else 0.65)
        st.metric("stl_trend", f"{stl_val:.2f}", delta="Historical Baseline")
        st.metric("sarima_residual", "0.012", delta="League Dev")
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = (1 / (1 + np.exp(-archs[sel_arch]["k"] * (15 - archs[sel_arch]["t_mid"])))) * 100,
            title = {'text': "velocity_T14 (% sold)"},
            gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': C_GREEN}}
        ))
        fig.update_layout(height=180, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("% of dynamic inventory sold in member ballot window.")
        
        st.markdown("#### Feature Importance Chart")
        fi = pd.DataFrame({
            "Feature": ["stl_seasonal", "stl_trend", "star_power", "stakes", "home_pos", "velocity_T14", "is_derby"],
            "Importance": [27.08, 12.19, 0.70, 0.10, 0.05, 0.04, 0.03],
            "Type": ["Time-Series", "Time-Series", "Interactive", "Context", "Context", "Interactive", "Interactive"]
        })
        color_map = {"Time-Series": C_PURPLE, "Context": C_GREEN, "Interactive": "orange"}
        fig = px.bar(fi, x="Importance", y="Feature", orientation='h', color="Type", color_discrete_map=color_map)
        fig.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0), yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    # Compute Output
    base_fr = 0.96 if h_tier == "big_six" else (0.82 if h_tier == "established_mid" else 0.68)
    adj = 0
    if st.session_state.is_derby: adj += 0.08
    if st.session_state.festive: adj += 0.06
    if st.session_state.fatigue: adj -= 0.09
    if st.session_state.new_manager: adj += 0.08
    adj += (st.session_state.star_power / 10.0) * 0.12
    adj += (st.session_state.home_form - 0.5) * 0.15
    adj += (st.session_state.away_form - 0.5) * 0.05
    if st.session_state.tv_slot == 0: adj -= 0.03
    elif st.session_state.tv_slot == 1: adj += 0.04
    elif st.session_state.tv_slot == 2: adj -= 0.02
    
    fr = np.clip(base_fr + adj + (stakes["final"] * 0.1), 0.30, 0.99)
    p10, p50, p90 = fr - 0.08, fr, min(fr + 0.06, 0.99)
    
    st.markdown("#### Output: Predicted Fill Rate (P10/P50/P90)")
    zones = ["Dynamic Lower", "Dynamic Upper", "Dynamic General", "Hospitality"]
    fig = go.Figure()
    fig.add_trace(go.Bar(name='P10', y=zones, x=[p10]*4, orientation='h', marker_color='lightgrey'))
    fig.add_trace(go.Bar(name='P50', y=zones, x=[p50]*4, orientation='h', marker_color=C_PURPLE))
    fig.add_trace(go.Bar(name='P90', y=zones, x=[p90]*4, orientation='h', marker_color=C_GREEN))
    fig.update_layout(barmode='group', height=250)
    st.plotly_chart(fig, use_container_width=True)

with st.expander("SECTION 4 — DYNAMIC INVENTORY BREAKDOWN", expanded=False):
    st.markdown("#### League-wide Dynamic Inventory %")
    df_dyn = pd.DataFrame([{"Club": CLUBS[c]["name"], "ID": c, "Dynamic%": CLUBS[c]["dynamic_pct"]*100} for c in CLUBS])
    df_dyn = df_dyn.sort_values("Dynamic%")
    colors = [C_GREEN if row["ID"] == st.session_state.home_club else "lightgrey" for _, row in df_dyn.iterrows()]
    fig = go.Figure(go.Bar(x=df_dyn["Club"], y=df_dyn["Dynamic%"], marker_color=colors))
    fig.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.info("Liverpool can dynamically price 4× as many seats as Arsenal — the solution must be club-specific, not league-wide.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"#### {CLUBS[st.session_state.home_club]['name']} Capacity Breakdown")
        away_pct = min(3000, 0.10 * cap) / cap
        hosp_pct = 0.05
        mem_pct = 0.05
        st_pct = 1.0 - (dyn_pct + away_pct + hosp_pct + mem_pct)
        
        labels = ["Season Tickets", "Hospitality", "Away End", "Member Ballot", "Dynamic"]
        values = [st_pct, hosp_pct, away_pct, mem_pct, dyn_pct]
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4, marker_colors=["#e2e8f0", "#94a3b8", "#cbd5e1", "#64748b", C_PURPLE])])
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)
        
    with c2:
        st.markdown("#### Dynamic Inventory Split")
        df_split = pd.DataFrame({
            "Zone": ["Lower Tier", "Upper Tier", "General"],
            "Multiplier": ["55%", "35%", "10%"],
            "Seats": [int(dyn_seats * 0.55), int(dyn_seats * 0.35), int(dyn_seats * 0.10)]
        })
        st.dataframe(df_split, hide_index=True, use_container_width=True)
        
        if st.session_state.is_derby:
            away_end_fill = 1.0
        else:
            away_end_fill = max(0.3, 1.0 - (st.session_state.distance - 50) / 600)
        st.metric("Away End Fill Rate", f"{away_end_fill*100:.1f}%")

with st.expander("SECTION 5 — LP OPTIMIZER & PRICE RECOMMENDATION", expanded=False):
    c1, c2 = st.columns([1, 1.5])
    
    with c1:
        st.markdown("#### Optimizer Setup")
        
        route = "Revenue Maximisation Route (No Floor)" if h_tier == "big_six" else "Attendance Protected Route (65% Floor)"
        st.markdown(f"**Tier Routing:** `{route}`")
        st.markdown("**Monotonicity:** Hospitality ≥ Lower ≥ Upper ≥ General")
        
        st.markdown("**Elasticity Coefficients**")
        st.table(pd.DataFrame({
            "Zone": ["Hospitality", "Lower Tier", "Upper Tier", "General Sale", "Away End 🔒"],
            "Elasticity": ["-0.10", "-0.55", "-0.75", "-1.10", "Fixed £30"]
        }))
        
        st.markdown("**WTP Bounds (£)**")
        bounds = {}
        if h_tier == "big_six": bounds = {"Lower": (30,40), "Upper": (30,40), "Hosp": (35,265)}
        elif h_tier == "established_mid": bounds = {"Lower": (37,47), "Upper": (30,40), "Hosp": (30,40)}
        else: bounds = {"Lower": (38,43), "Upper": (30,40), "Hosp": (30,40)}
        st.json(bounds)

    with c2:
        st.markdown("#### Optimization Output")
        
        # Simple Optimization inline
        # Demand: Q = Q_base * (1 + elast * (P - P_med) / P_med)
        zones_lp = ["Hosp", "Lower", "Upper"]
        elasts = {"Hosp": -0.10, "Lower": -0.55, "Upper": -0.75}
        caps = {"Hosp": int(cap*0.05), "Lower": int(dyn_seats*0.55), "Upper": int(dyn_seats*0.45)} # grouped general into upper for demo
        
        prices_flat = {}
        prices_opt = {}
        rev_flat = 0
        rev_opt = 0
        
        out_table = []
        for z in zones_lp:
            b_min, b_max = bounds[z]
            # Stakes uplift
            b_min += int(stakes["final"] * 20)
            b_max += int(stakes["final"] * 20)
            
            p_med = (b_min + b_max) / 2
            prices_flat[z] = p_med
            
            # Simple grid search 10 pts
            pts = np.linspace(b_min, b_max, 10)
            best_r = 0
            best_p = 0
            best_q = 0
            q_base = caps[z] * p50
            
            for p in pts:
                q = q_base * (1 + elasts[z] * (p - p_med)/p_med)
                q = min(caps[z], max(0, q))
                r = p * q
                if r > best_r:
                    best_r = r
                    best_p = p
                    best_q = q
                    
            prices_opt[z] = best_p
            rev_opt += best_r
            q_flat = q_base * (1 + elasts[z] * (p_med - p_med)/p_med)
            rev_flat += p_med * q_flat
            
            out_table.append({
                "Zone": z, "Floor": f"£{b_min}", "Optimal Price": f"£{best_p:.0f}", "Ceiling": f"£{b_max}",
                "Elasticity": elasts[z], "Est. Seats": int(best_q), "Revenue": f"£{best_r:,.0f}"
            })
            
        m1, m2, m3 = st.columns(3)
        m1.metric("Dynamic Revenue (LP Optimised)", f"£{rev_opt:,.0f}", f"+£{rev_opt - rev_flat:,.0f} vs Flat")
        m2.metric("Dynamic Revenue (Static Flat Pricing)", f"£{rev_flat:,.0f}")
        m3.metric("Uplift %", f"{((rev_opt - rev_flat)/rev_flat)*100:.1f}%")
        
        st.dataframe(pd.DataFrame(out_table), hide_index=True, use_container_width=True)
        
        st.markdown("#### Revenue Curve (Lower Tier)")
        b_min, b_max = bounds["Lower"][0] + int(stakes["final"] * 20), bounds["Lower"][1] + int(stakes["final"] * 20)
        p_med = (b_min + b_max) / 2
        p_range = np.linspace(b_min - 10, b_max + 10, 50)
        q_base = caps["Lower"] * p50
        revs = [p * min(caps["Lower"], max(0, q_base * (1 + elasts["Lower"] * (p - p_med)/p_med))) for p in p_range]
        
        fig = go.Figure(go.Scatter(x=p_range, y=revs, mode='lines', name='Revenue Curve', line=dict(color=C_PURPLE)))
        fig.add_trace(go.Scatter(x=[p_med], y=[p_med * q_base], mode='markers', name='Static Price', marker=dict(color='grey', size=12)))
        opt_p = prices_opt["Lower"]
        opt_q = min(caps["Lower"], max(0, q_base * (1 + elasts["Lower"] * (opt_p - p_med)/p_med)))
        fig.add_trace(go.Scatter(x=[opt_p], y=[opt_p * opt_q], mode='markers', name='LP Optimal', marker=dict(color=C_GREEN, size=14)))
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0), xaxis_title="Price (£)", yaxis_title="Revenue (£)")
        st.plotly_chart(fig, use_container_width=True)
