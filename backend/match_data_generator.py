import json
import csv
import os
import random
import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# --- STEP 1: LEAGUE STRUCTURE ---

CLUBS = [
    {
        "club_id": "BSV",
        "name": "BSV Bern",
        "city": "Bern",
        "canton": "BE",
        "venue": "PostFinance Arena (Handball Hall)",
        "capacity": 3800,
        "tier": "top",
        "base_fanbase": 0.72,
        "rival_club_ids": ["RTV", "HSC"]
    },
    {
        "club_id": "RTV",
        "name": "RTV Basel",
        "city": "Basel",
        "canton": "BS",
        "venue": "St. Jakobshalle",
        "capacity": 4200,
        "tier": "top",
        "base_fanbase": 0.68,
        "rival_club_ids": ["BSV", "GC"]
    },
    {
        "club_id": "GC",
        "name": "Grasshopper Club Zürich",
        "city": "Zürich",
        "canton": "ZH",
        "venue": "Sporthalle Deutweg",
        "capacity": 3200,
        "tier": "top",
        "base_fanbase": 0.65,
        "rival_club_ids": ["RTV", "KAD"]
    },
    {
        "club_id": "KAD",
        "name": "Kadetten Schaffhausen",
        "city": "Schaffhausen",
        "canton": "SH",
        "venue": "Kammgarnhalle",
        "capacity": 2800,
        "tier": "mid",
        "base_fanbase": 0.60,
        "rival_club_ids": ["GC", "WIN"]
    },
    {
        "club_id": "WIN",
        "name": "TV Winterthur",
        "city": "Winterthur",
        "canton": "ZH",
        "venue": "Eulachpark",
        "capacity": 2200,
        "tier": "mid",
        "base_fanbase": 0.55,
        "rival_club_ids": ["KAD", "GC"]
    },
    {
        "club_id": "HSC",
        "name": "HSC Suhr Aarau",
        "city": "Aarau",
        "canton": "AG",
        "venue": "Swiss Arena Bubble",
        "capacity": 2600,
        "tier": "mid",
        "base_fanbase": 0.58,
        "rival_club_ids": ["BSV", "LUZ"]
    },
    {
        "club_id": "LUZ",
        "name": "STV Willisau",
        "city": "Willisau",
        "canton": "LU",
        "venue": "Dreifachhalle Willisau",
        "capacity": 1800,
        "tier": "small",
        "base_fanbase": 0.50,
        "rival_club_ids": ["HSC", "THU"]
    },
    {
        "club_id": "THU",
        "name": "TSV Oftringen",
        "city": "Oftringen",
        "canton": "AG",
        "venue": "Athletik Zentrum Oftringen",
        "capacity": 1600,
        "tier": "small",
        "base_fanbase": 0.48,
        "rival_club_ids": ["LUZ", "HSC"]
    },
    {
        "club_id": "WIN2",
        "name": "Pfadi Winterthur",
        "city": "Winterthur",
        "canton": "ZH",
        "venue": "Sportpark Deutweg",
        "capacity": 2000,
        "tier": "mid",
        "base_fanbase": 0.52,
        "rival_club_ids": ["WIN", "GC"]
    },
    {
        "club_id": "THG",
        "name": "Handball Thurgau",
        "city": "Kreuzlingen",
        "canton": "TG",
        "venue": "Bodenseearena",
        "capacity": 1500,
        "tier": "small",
        "base_fanbase": 0.45,
        "rival_club_ids": ["WIN", "WIN2"]
    }
]

# --- STEP 2: SEASON CONFIG ---

SEASON_CONFIG = {
    "2021-22": {"start": "2021-09-04", "multiplier": 1.0, "notes": "Post-COVID recovery"},
    "2022-23": {"start": "2022-09-03", "multiplier": 1.0, "notes": "Normal season"},
    "2023-24": {"start": "2023-09-02", "multiplier": 1.05, "notes": "Growth season"}
}

ZONE_SPLIT = {
    "Courtside VIP": 0.08,
    "Lower Bowl / Club Seats": 0.25,
    "Upper Standard": 0.35,
    "Standing": 0.32
}

BASE_PRICES = {
    "Standing": 18,
    "Upper Standard": 32,
    "Lower Bowl / Club Seats": 58,
    "Courtside VIP": 85
}

class MatchDataGenerator:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.wtp_results = self._load_json("wtp_results.json")
        self.fan_segments = self._load_json("fan_segments.json")
        self.matches = []
        
    def _load_json(self, filename):
        path = os.path.join(self.data_dir, filename)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def generate(self):
        self.matches = []
        
        # Clubs state
        club_stats = {c["club_id"]: {
            "form_score": 0.5,
            "wins": 0,
            "points": 0,
            "goals_scored": 0,
            "goals_against": 0,
            "stars": self._initialize_stars(c),
            "dominant_segment": self._get_dominant_segment(c)
        } for c in CLUBS}
        
        # H2H state
        h2h_state = {} # (home, away) -> winrate list

        for season_name, config in SEASON_CONFIG.items():
            season_matches = self._create_fixtures(season_name, config["start"])
            
            # Reset season stats
            for c_id in club_stats:
                club_stats[c_id]["wins"] = 0
                club_stats[c_id]["points"] = 0
                club_stats[c_id]["goals_scored"] = 0
                club_stats[c_id]["goals_against"] = 0
            
            for match in season_matches:
                # Add dynamic attributes
                self._assign_match_attributes(match, club_stats, h2h_state, season_name)
                
                # Attendance & Revenue
                self._calculate_attendance_and_revenue(match, season_name)
                
                # Booking Curve
                self._generate_booking_curve(match)
                
                # Secondary Market
                self._simulate_secondary_market(match)
                
                # Simulate Outcome & Update State
                self._resolve_match(match, club_stats, h2h_state)
                
                self.matches.append(match)
                
        self.save_outputs()
        return self.get_validation_report()

    def _initialize_stars(self, club):
        stars = []
        n_stars = 0
        if club["tier"] == "top": n_stars = random.randint(2, 3)
        elif club["tier"] == "mid": n_stars = 1
        
        for _ in range(n_stars):
            stars.append({
                "follower_count": random.randint(100000, 800000)
            })
        return stars

    def _get_dominant_segment(self, club):
        # top-tier: alternate Premium Seeker / Value Loyalist
        # mid-tier: alternate Value Loyalist / Atmosphere Seeker
        # small: Occasional Neutral
        tier = club["tier"]
        idx = CLUBS.index(club)
        if tier == "top":
            return "Premium Seeker" if idx % 2 == 0 else "Value Loyalist"
        elif tier == "mid":
            return "Value Loyalist" if idx % 2 == 0 else "Atmosphere Seeker"
        else:
            return "Occasional Neutral"

    def _create_fixtures(self, season_name, start_date_str):
        c_ids = [c["club_id"] for c in CLUBS]
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        
        # Double Round Robin: Leg 1 (45 matches) + Leg 2 (45 matches)
        leg1_rounds = self._get_round_robin_schedule(c_ids)
        leg2_rounds = self._get_round_robin_schedule(c_ids)
        
        # For Leg 2, swap home/away to ensure every team plays twice (once home, once away)
        for r in leg2_rounds:
            for i in range(len(r)):
                r[i] = (r[i][1], r[i][0])
                
        all_rounds = leg1_rounds + leg2_rounds
        all_matches = []
        
        for r_idx, round_pairs in enumerate(all_rounds):
            # Matches spread over weekends
            round_date = start_date + timedelta(weeks=r_idx)
            for home_id, away_id in round_pairs:
                # Timing variation
                rand = random.random()
                kickoff_type = "Saturday evening"
                kickoff_time = "19:30"
                match_date = round_date # Saturday
                
                if rand < 0.20:
                    kickoff_type = "Saturday afternoon"
                    kickoff_time = "17:00"
                elif rand < 0.35:
                    kickoff_type = "Sunday afternoon"
                    kickoff_time = "16:00"
                    match_date = round_date + timedelta(days=1)
                elif rand < 0.40:
                     kickoff_type = "Weekday evening"
                     kickoff_time = "19:30"
                     match_date = round_date - timedelta(days=3)
                
                home_club = next(c for c in CLUBS if c["club_id"] == home_id)
                away_club = next(c for c in CLUBS if c["club_id"] == away_id)
                
                m = {
                    "match_id": f"{home_id}_{season_name[:4]}_R{r_idx+1:02d}_{away_id}",
                    "season": season_name,
                    "match_round": r_idx + 1,
                    "match_date": match_date.strftime("%Y-%m-%d"),
                    "kickoff_time": kickoff_time,
                    "kickoff_type": kickoff_type,
                    "home_club_id": home_id,
                    "home_club_name": home_club["name"],
                    "home_city": home_club["city"],
                    "home_canton": home_club["canton"],
                    "venue": home_club["venue"],
                    "total_capacity": home_club["capacity"],
                    "away_club_id": away_id,
                    "away_club_name": away_club["name"]
                }
                all_matches.append(m)
        
        return sorted(all_matches, key=lambda x: x["match_date"])

    def _get_round_robin_schedule(self, teams):
        if len(teams) % 2:
            teams.append(None)
        n = len(teams)
        rounds = []
        for i in range(n - 1):
            round_pairs = []
            for j in range(n // 2):
                if teams[j] and teams[n - 1 - j]:
                    round_pairs.append((teams[j], teams[n - 1 - j]))
            teams.insert(1, teams.pop())
            rounds.append(round_pairs)
        
        # Ensure home/away balance by swapping every other round
        for i in range(len(rounds)):
            if i % 2 == 0:
                rounds[i] = [(p[1], p[0]) for p in rounds[i]]
        
        return rounds

    def _assign_match_attributes(self, match, club_stats, h2h_state, season_name):
        home_id = match["home_club_id"]
        away_id = match["away_club_id"]
        home_club = next(c for c in CLUBS if c["club_id"] == home_id)
        away_club = next(c for c in CLUBS if c["club_id"] == away_id)
        
        # Opponent Tier
        match["opponent_tier"] = "Elite" if away_club["tier"] == "top" else \
                                 "Competitive" if away_club["tier"] == "mid" else "Standard"
        
        # Rival Match
        match["rival_match"] = away_id in home_club["rival_club_ids"]
        
        # H2H Winrate
        key = tuple(sorted([home_id, away_id]))
        if key not in h2h_state:
            h2h_state[key] = [random.uniform(0.3, 0.7)]
        match["head_to_head_home_winrate"] = round(sum(h2h_state[key]) / len(h2h_state[key]), 2)
        
        # Form
        match["home_form_score"] = round(club_stats[home_id]["form_score"], 2)
        match["away_form_score"] = round(club_stats[away_id]["form_score"], 2)
        
        # Stars
        home_stars = club_stats[home_id]["stars"]
        away_stars = club_stats[away_id]["stars"]
        
        confirmed_followers = 0
        star_announced = False
        for s in home_stars + away_stars:
            if random.random() > 0.15: # 15% absence
                confirmed_followers += s["follower_count"]
                star_announced = True
        
        match["star_player_announced"] = star_announced
        match["star_power_index"] = round(confirmed_followers / 1000000, 2)
        
        # Stakes
        # Simplified standings logic
        standings = sorted(club_stats.items(), key=lambda x: x[1]["points"], reverse=True)
        home_rank = next(i for i, (c_id, _) in enumerate(standings) if c_id == home_id) + 1
        
        match_stakes = "Group"
        qual_score = 1
        
        # Round 15-18 logic
        if match["match_round"] >= 15:
            if home_rank <= 3:
                match_stakes = "Playoff"
                qual_score = 2
            if random.random() < 0.1: # Simulate a cup final/final 4
                match_stakes = "Final"
                qual_score = 3
        
        match["match_stakes"] = match_stakes
        match["qualification_stakes_score"] = qual_score
        
        # External Factors
        month = datetime.strptime(match["match_date"], "%Y-%m-%d").month
        match["weather_severity_score"] = self._get_weather(month)
        
        penalty = 0.0
        if random.random() < 0.20:
            events = [0.12, 0.08, 0.05] # CL, Super League, Local
            penalty = random.choice(events)
        match["competing_event_penalty"] = penalty
        
        match["is_school_holiday"] = self._is_holiday(match["match_date"], home_club["canton"])
        
        beta_a, beta_b = (2.5, 2.5) if home_club["tier"] == "top" else (2, 3)
        match["marketing_activation_score"] = round(np.random.beta(beta_a, beta_b), 2)
        
        # Conjoint
        match["dominant_segment"] = club_stats[home_id]["dominant_segment"]
        match["attribute_wtp_score"] = self._calculate_wtp_score(match)

    def _get_weather(self, month):
        # 9: 0-1, 10-11: 0-2, 12-1: 1-3, 2: 1-3, 3-4: 0-2
        if month == 9: return random.randint(0, 1)
        if month in [10, 11, 3, 4]: return random.randint(0, 2)
        if month in [12, 1, 2]: return random.randint(1, 3)
        return 0

    def _is_holiday(self, date_str, canton):
        d = datetime.strptime(date_str, "%Y-%m-%d")
        # Rough Swiss holiday simplified
        if d.month == 10 and 5 <= d.day <= 20: return True # Autumn
        if (d.month == 12 and d.day >= 22) or (d.month == 1 and d.day <= 8): return True # Xmas
        if d.month == 2 and 10 <= d.day <= 18: return True # Sport
        if d.month == 4 and 10 <= d.day <= 25: return True # Spring
        return False

    def _calculate_wtp_score(self, match):
        if not self.wtp_results: return 0.5
        wtp = self.wtp_results["attribute_wtp"]
        
        opp_wtp = wtp["opponent_elite"]["mean"] if match["opponent_tier"] == "Elite" else \
                  wtp["opponent_competitive"]["mean"] if match["opponent_tier"] == "Competitive" else 0
        
        stakes_wtp = wtp["stakes_final"]["mean"] if match["match_stakes"] == "Final" else \
                     wtp["stakes_playoff"]["mean"] if match["match_stakes"] == "Playoff" else 0
        
        star_wtp = wtp["star_confirmed"]["mean"] if match["star_player_announced"] else 0
        
        # Normalized (rough max is ~150-200)
        score = (opp_wtp + stakes_wtp + star_wtp) / 200.0
        return round(min(1.0, score), 2)

    def _calculate_attendance_and_revenue(self, match, season_name):
        home_id = match["home_club_id"]
        home_club = next(c for c in CLUBS if c["club_id"] == home_id)
        base_rate = home_club["base_fanbase"]
        
        # Multipliers
        opp_m = 1.35 if match["opponent_tier"] == "Elite" else 1.10 if match["opponent_tier"] == "Competitive" else 1.0
        stakes_m = 1.80 if match["match_stakes"] == "Final" else 1.35 if match["match_stakes"] == "Playoff" else 1.0
        rival_m = 1.25 if match["rival_match"] else 1.0
        star_m = 1.18 if match["star_player_announced"] else 1.0
        
        w_scores = {0: 1.0, 1: 0.97, 2: 0.92, 3: 0.85}
        weather_m = w_scores[match["weather_severity_score"]]
        
        comp_m = 1.0 - match["competing_event_penalty"]
        weekday_m = 0.88 if "Weekday" in match["kickoff_type"] else 1.0
        
        # Season multiplier
        season_m = 1.0
        if season_name == "2021-22" and dt.month >= 9:
            season_m = 0.85
        elif season_name == "2023-24":
            season_m = 1.05
        
        # Round multiplier
        r = match["match_round"]
        round_m = 1.0
        if r <= 3: round_m = 1.08
        elif 13 <= r <= 15: round_m = 1.05
        elif r >= 16: round_m = 1.15
        
        match["season_multiplier"] = season_m
        match["round_position_multiplier"] = round_m
        
        # FIX 4: Expected tickets for velocity denominator (Leakage Fix)
        # We use a baseline derived from fanbase and opponent tier, NOT the actual final_tickets.
        match["expected_tickets"] = int(round(base_rate * opp_m * home_club["capacity"]))
        
        combined_rate = base_rate * opp_m * stakes_m * rival_m * star_m * weather_m * comp_m * weekday_m * season_m * round_m
        final_rate = np.clip(combined_rate + np.random.normal(0, 0.04), 0.20, 1.00)
        
        match["overall_fill_rate"] = round(float(final_rate), 3)
        
        # Zones
        match["attendance"] = {}
        total_tickets = 0
        total_rev = 0
        
        zone_sens = {
            "Courtside VIP": 1.05,
            "Lower Bowl / Club Seats": 1.00,
            "Upper Standard": 0.98,
            "Standing": 0.95
        }
        
        match["zone_capacities"] = {}
        for zone, cap_pct in ZONE_SPLIT.items():
            z_cap = int(home_club["capacity"] * cap_pct)
            match["zone_capacities"][zone] = z_cap
            
            z_fill = np.clip(final_rate * zone_sens[zone], 0.0, 1.0)
            z_sold = int(round(z_fill * z_cap))
            # noise
            z_sold = int(np.clip(z_sold * (1 + random.uniform(-0.03, 0.03)), 0, z_cap))
            
            rev = z_sold * BASE_PRICES[zone]
            
            match["attendance"][zone] = {
                "capacity": z_cap,
                "tickets_sold": z_sold,
                "fill_rate": round(float(z_sold / z_cap), 3) if z_cap > 0 else 0,
                "revenue_chf": rev
            }
            total_tickets += z_sold
            total_rev += rev
            
        match["total_tickets_sold"] = total_tickets
        match["total_revenue_chf"] = total_rev
        match["revpas"] = round(total_rev / home_club["capacity"], 2)
        match["base_price_standing"] = BASE_PRICES["Standing"]
        match["base_price_upper_standard"] = BASE_PRICES["Upper Standard"]
        match["base_price_lower_bowl"] = BASE_PRICES["Lower Bowl / Club Seats"]
        match["base_price_courtside_vip"] = BASE_PRICES["Courtside VIP"]

    def _generate_booking_curve(self, match):
        rate = match["overall_fill_rate"]
        opp = match["opponent_tier"]
        rival = match["rival_match"]
        stakes = match["match_stakes"]
        qual = match["qualification_stakes_score"]
        
        if stakes in ["Final", "Playoff"] or qual >= 2:
            match_archetype = "Late Surge"
        elif (opp == "Elite" or rival) and rate > 0.75:
            match_archetype = "Early Surge"
        elif opp == "Competitive" and 0.50 < rate < 0.75:
            match_archetype = "Consistent Gradual"
        else:
            match_archetype = "Flat"
        
        match["booking_curve_archetype"] = match_archetype
        
        # Archtype parameters: t_mid (day of 50% sales), k (steepness)
        base_params = {
            "Early Surge": {"t_mid": 35, "k": 0.15},
            "Consistent Gradual": {"t_mid": 45, "k": 0.10},
            "Late Surge": {"t_mid": 52, "k": 0.18},
            "Flat": {"t_mid": 48, "k": 0.06}
        }
        
        full_match_curve = np.zeros(61)
        match["zone_booking_curves"] = {}
        
        for zone in ZONE_SPLIT:
            z_sold = match["attendance"][zone]["tickets_sold"]
            
            # FIX 5: Zone-specific dynamics
            if zone == "Courtside VIP":
                # Fills earliest, premium commitment
                z_archetype = "Early Surge"
                p = {"t_mid": 30, "k": 0.16}
            elif zone == "Standing":
                # Most elastic, fills latest
                z_archetype = "Late Surge"
                p = {"t_mid": 54, "k": 0.20}
            elif zone == "Upper Standard":
                # Slightly delayed vs overall
                z_archetype = match_archetype
                p = base_params[z_archetype].copy()
                p["t_mid"] += 2 
            else: # Lower Bowl
                # Follows overall benchmark
                z_archetype = match_archetype
                p = base_params[z_archetype]
                
            z_curve = []
            for t in range(61):
                val = z_sold / (1 + np.exp(-p["k"] * (t - p["t_mid"])))
                z_curve.append(val)
            
            # Scale and add noise
            current_final = z_curve[-1]
            scale = z_sold / current_final if current_final > 0 else 0
            
            final_z_curve = []
            last_val = 0
            for val in z_curve:
                noisy_val = max(last_val, int(val * scale + random.uniform(-0.01, 0.01) * z_sold))
                final_z_curve.append(noisy_val)
                last_val = noisy_val
            
            final_z_curve[-1] = z_sold
            match["zone_booking_curves"][zone] = final_z_curve
            full_match_curve += np.array(final_z_curve)
            
            # Zone Velocities (using expected zone tickets as denominator to avoid leakage)
            expected_z = int(match["expected_tickets"] * ZONE_SPLIT[zone])
            match[f"velocity_T14_{zone}"] = round(final_z_curve[46] / expected_z, 2) if expected_z > 0 else 0
            match[f"velocity_T7_{zone}"] = round(final_z_curve[53] / expected_z, 2) if expected_z > 0 else 0

        match["booking_curve"] = full_match_curve.tolist()
        
        # Global Velocity (Leakage Fix: use expected_tickets, not total_tickets_sold)
        exp = match["expected_tickets"]
        match["velocity_T30"] = round(full_match_curve[30] / exp, 2) if exp > 0 else 0
        match["velocity_T14"] = round(full_match_curve[46] / exp, 2) if exp > 0 else 0
        match["velocity_T7"] = round(full_match_curve[53] / exp, 2) if exp > 0 else 0

    def _simulate_secondary_market(self, match):
        base_premium = 0.05
        add = 0
        if match["opponent_tier"] == "Elite": add += 0.12
        if match["rival_match"]: add += 0.08
        if match["match_stakes"] == "Final": add += 0.20
        if match["star_player_announced"]: add += 0.10
        
        rate = match["overall_fill_rate"]
        if rate > 0.90: add += 0.15
        elif rate > 0.80: add += 0.08
        
        total_p = np.clip(base_premium + add + np.random.normal(0, 0.03), 0.0, 0.60)
        match["secondary_premium_pct"] = round(float(total_p), 3)
        
        # Average delta across zones
        avg_price = sum(BASE_PRICES.values()) / 4
        match["price_delta_secondary_chf"] = round(avg_price * total_p, 1)

    def _resolve_match(self, match, club_stats, h2h_state):
        home_id = match["home_club_id"]
        away_id = match["away_club_id"]
        
        home_form = club_stats[home_id]["form_score"]
        away_form = club_stats[away_id]["form_score"]
        
        p_home = 0.55 * home_form / (home_form + away_form) + 0.10
        p_home = np.clip(p_home, 0.25, 0.75)
        
        home_win = random.random() < p_home
        
        h_goals = int(np.clip(np.random.normal(26, 4), 18, 38))
        a_goals = int(np.clip(np.random.normal(24, 4), 16, 36))
        
        if home_win:
            h_goals = max(h_goals, a_goals + 1)
            club_stats[home_id]["points"] += 2
        elif h_goals == a_goals:
            club_stats[home_id]["points"] += 1
            club_stats[away_id]["points"] += 1
        else:
            a_goals = max(a_goals, h_goals + 1)
            club_stats[away_id]["points"] += 2
            
        match["home_goals"] = h_goals
        match["away_goals"] = a_goals
        match["home_win"] = home_win
        
        # Update Form
        res = 1.0 if home_win else (0.5 if h_goals == a_goals else 0.0)
        club_stats[home_id]["form_score"] = home_form * 0.8 + res * 0.2
        club_stats[away_id]["form_score"] = away_form * 0.8 + (1.0 - res) * 0.2
        
        # Standings (snapshot for next match)
        standings = sorted(club_stats.items(), key=lambda x: x[1]["points"], reverse=True)
        match["league_position_home"] = next(i for i, (c_id, _) in enumerate(standings) if c_id == home_id) + 1
        match["league_position_away"] = next(i for i, (c_id, _) in enumerate(standings) if c_id == away_id) + 1
        match["points_home"] = club_stats[home_id]["points"]
        match["points_away"] = club_stats[away_id]["points"]
        
        # H2H update
        key = tuple(sorted([home_id, away_id]))
        h2h_state[key].append(res)
        if len(h2h_state[key]) > 5: h2h_state[key].pop(0)

    def _clean_numpy(self, obj):
        if isinstance(obj, dict):
            return {k: self._clean_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_numpy(v) for v in obj]
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return self._clean_numpy(obj.tolist())
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        else:
            return obj

    def save_outputs(self):
        os.makedirs(self.data_dir, exist_ok=True)
        
        clean_matches = self._clean_numpy(self.matches)
        
        # JSON
        with open(os.path.join(self.data_dir, "match_data.json"), "w", encoding="utf-8") as f:
            json.dump(clean_matches, f, indent=2)
            
        # CSV
        df = pd.DataFrame(self.matches)
        # Flatten attendance
        for zone in ZONE_SPLIT:
            df[f"sold_{zone}"] = df["attendance"].apply(lambda x: x[zone]["tickets_sold"])
            df[f"fill_{zone}"] = df["attendance"].apply(lambda x: x[zone]["fill_rate"])
        
        df_flat = df.drop(columns=["attendance", "booking_curve", "zone_capacities", "zone_booking_curves"])
        df_flat.to_csv(os.path.join(self.data_dir, "match_data.csv"), index=False)

    def get_validation_report(self):
        df = pd.DataFrame(self.matches)
        
        report = {
            "total_matches": len(self.matches),
            "matches_per_season": df["season"].value_counts().to_dict(),
            "overall_fill_rate_mean": round(df["overall_fill_rate"].mean(), 3),
            "overall_fill_rate_std": round(df["overall_fill_rate"].std(), 3),
            "fill_rate_by_season": df.groupby("season")["overall_fill_rate"].mean().round(3).to_dict(),
            "fill_rate_by_opponent_tier": df.groupby("opponent_tier")["overall_fill_rate"].mean().round(3).to_dict(),
            "archetype_distribution": df["booking_curve_archetype"].value_counts().to_dict(),
            "revenue_stats": {
                "mean_match_revenue_chf": round(df["total_revenue_chf"].mean(), 2),
                "total_revenue_3_seasons_chf": int(df["total_revenue_chf"].sum()),
                "mean_revpas": round(df["revpas"].mean(), 2)
            },
            "velocity_stats": {
                "mean_velocity_T14": round(df["velocity_T14"].mean(), 3),
                "correlation_velocity_T14_fill_rate": round(df["velocity_T14"].corr(df["overall_fill_rate"]), 3)
            },
            "secondary_market_stats": {
                "mean_premium_pct": round(df["secondary_premium_pct"].mean(), 3),
                "matches_with_premium_above_20pct": int((df["secondary_premium_pct"] > 0.20).sum())
            }
        }
        return report

if __name__ == "__main__":
    gen = MatchDataGenerator()
    report = gen.generate()
    print(json.dumps(report, indent=2))
