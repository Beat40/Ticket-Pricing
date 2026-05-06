import json
import csv
import os
import random
import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from EPL.stakes_engine import StakesEngine

# --- SECTION 1: CLUB REGISTRY ---

CLUBS = [
    # BIG SIX
    {"club_id": "ARS", "name": "Arsenal", "city": "London", "venue": "Emirates Stadium", "capacity": 60704, "season_ticket_pct": 0.88, "member_priority_pct": 0.05, "hospitality_pct": 0.04, "dynamic_inventory_pct": 0.03, "base_demand": 0.97, "tier": "big_six", "rivals": ["TOT", "CHE", "MCI"], "expected_finish_range": [1, 4], "stars": 4, "global_fanbase_millions": 45},
    {"club_id": "MCI", "name": "Manchester City", "city": "Manchester", "venue": "Etihad Stadium", "capacity": 53400, "season_ticket_pct": 0.82, "member_priority_pct": 0.07, "hospitality_pct": 0.05, "dynamic_inventory_pct": 0.06, "base_demand": 0.95, "tier": "big_six", "rivals": ["MUN", "LIV"], "expected_finish_range": [1, 3], "stars": 4, "global_fanbase_millions": 35},
    {"club_id": "LIV", "name": "Liverpool", "city": "Liverpool", "venue": "Anfield", "capacity": 61276, "season_ticket_pct": 0.90, "member_priority_pct": 0.05, "hospitality_pct": 0.03, "dynamic_inventory_pct": 0.02, "base_demand": 0.98, "tier": "big_six", "rivals": ["EVE", "MUN", "MCI"], "expected_finish_range": [1, 5], "stars": 4, "global_fanbase_millions": 50},
    {"club_id": "CHE", "name": "Chelsea", "city": "London", "venue": "Stamford Bridge", "capacity": 40834, "season_ticket_pct": 0.80, "member_priority_pct": 0.08, "hospitality_pct": 0.06, "dynamic_inventory_pct": 0.06, "base_demand": 0.92, "tier": "big_six", "rivals": ["ARS", "TOT", "FUL"], "expected_finish_range": [3, 8], "stars": 3, "global_fanbase_millions": 30},
    {"club_id": "TOT", "name": "Tottenham Hotspur", "city": "London", "venue": "Tottenham Hotspur Stadium", "capacity": 62850, "season_ticket_pct": 0.83, "member_priority_pct": 0.07, "hospitality_pct": 0.05, "dynamic_inventory_pct": 0.05, "base_demand": 0.94, "tier": "big_six", "rivals": ["ARS", "CHE", "MUN"], "expected_finish_range": [3, 8], "stars": 3, "global_fanbase_millions": 25},
    {"club_id": "MUN", "name": "Manchester United", "city": "Manchester", "venue": "Old Trafford", "capacity": 74310, "season_ticket_pct": 0.85, "member_priority_pct": 0.06, "hospitality_pct": 0.05, "dynamic_inventory_pct": 0.04, "base_demand": 0.96, "tier": "big_six", "rivals": ["MCI", "LIV", "ARS"], "expected_finish_range": [3, 8], "stars": 3, "global_fanbase_millions": 75},
    
    # ESTABLISHED MID
    {"club_id": "AVL", "name": "Aston Villa", "city": "Birmingham", "venue": "Villa Park", "capacity": 42682, "season_ticket_pct": 0.62, "member_priority_pct": 0.08, "hospitality_pct": 0.04, "dynamic_inventory_pct": 0.26, "base_demand": 0.78, "tier": "established_mid", "rivals": ["WOL", "BIR"], "expected_finish_range": [4, 10], "stars": 2, "global_fanbase_millions": 8},
    {"club_id": "NEW", "name": "Newcastle United", "city": "Newcastle", "venue": "St. James Park", "capacity": 52305, "season_ticket_pct": 0.70, "member_priority_pct": 0.08, "hospitality_pct": 0.04, "dynamic_inventory_pct": 0.18, "base_demand": 0.88, "tier": "established_mid", "rivals": ["SUN"], "expected_finish_range": [4, 12], "stars": 2, "global_fanbase_millions": 10},
    {"club_id": "WHU", "name": "West Ham United", "city": "London", "venue": "London Stadium", "capacity": 62500, "season_ticket_pct": 0.58, "member_priority_pct": 0.07, "hospitality_pct": 0.05, "dynamic_inventory_pct": 0.30, "base_demand": 0.72, "tier": "established_mid", "rivals": ["MIL", "TOT"], "expected_finish_range": [8, 15], "stars": 2, "global_fanbase_millions": 7},
    {"club_id": "BHA", "name": "Brighton & Hove Albion", "city": "Brighton", "venue": "Amex Stadium", "capacity": 31800, "season_ticket_pct": 0.65, "member_priority_pct": 0.10, "hospitality_pct": 0.04, "dynamic_inventory_pct": 0.21, "base_demand": 0.80, "tier": "established_mid", "rivals": ["CPL", "MIL"], "expected_finish_range": [6, 12], "stars": 2, "global_fanbase_millions": 4},
    {"club_id": "WOL", "name": "Wolverhampton Wanderers", "city": "Wolverhampton", "venue": "Molineux", "capacity": 32050, "season_ticket_pct": 0.60, "member_priority_pct": 0.08, "hospitality_pct": 0.04, "dynamic_inventory_pct": 0.28, "base_demand": 0.74, "tier": "established_mid", "rivals": ["AVL", "BIR"], "expected_finish_range": [8, 16], "stars": 1, "global_fanbase_millions": 5},
    {"club_id": "FUL", "name": "Fulham", "city": "London", "venue": "Craven Cottage", "capacity": 29600, "season_ticket_pct": 0.55, "member_priority_pct": 0.08, "hospitality_pct": 0.05, "dynamic_inventory_pct": 0.32, "base_demand": 0.70, "tier": "established_mid", "rivals": ["CHE", "QPR"], "expected_finish_range": [10, 17], "stars": 1, "global_fanbase_millions": 4},
    {"club_id": "CPL", "name": "Crystal Palace", "city": "London", "venue": "Selhurst Park", "capacity": 25486, "season_ticket_pct": 0.60, "member_priority_pct": 0.08, "hospitality_pct": 0.04, "dynamic_inventory_pct": 0.28, "base_demand": 0.72, "tier": "established_mid", "rivals": ["BHA", "MIL"], "expected_finish_range": [10, 17], "stars": 1, "global_fanbase_millions": 3},
    
    # SMALLER CLUB
    {"club_id": "EVE", "name": "Everton", "city": "Liverpool", "venue": "Goodison Park", "capacity": 39414, "season_ticket_pct": 0.65, "member_priority_pct": 0.08, "hospitality_pct": 0.03, "dynamic_inventory_pct": 0.24, "base_demand": 0.75, "tier": "smaller_club", "rivals": ["LIV"], "expected_finish_range": [12, 19], "stars": 1, "global_fanbase_millions": 6},
    {"club_id": "BRE", "name": "Brentford", "city": "London", "venue": "Gtech Community Stadium", "capacity": 17250, "season_ticket_pct": 0.55, "member_priority_pct": 0.10, "hospitality_pct": 0.04, "dynamic_inventory_pct": 0.31, "base_demand": 0.68, "tier": "smaller_club", "rivals": ["FUL", "QPR"], "expected_finish_range": [10, 18], "stars": 1, "global_fanbase_millions": 2},
    {"club_id": "NFO", "name": "Nottingham Forest", "city": "Nottingham", "venue": "City Ground", "capacity": 30445, "season_ticket_pct": 0.58, "member_priority_pct": 0.08, "hospitality_pct": 0.04, "dynamic_inventory_pct": 0.30, "base_demand": 0.70, "tier": "smaller_club", "rivals": ["LEI", "DER"], "expected_finish_range": [12, 19], "stars": 1, "global_fanbase_millions": 3},
    {"club_id": "LEI", "name": "Leicester City", "city": "Leicester", "venue": "King Power Stadium", "capacity": 32261, "season_ticket_pct": 0.60, "member_priority_pct": 0.08, "hospitality_pct": 0.04, "dynamic_inventory_pct": 0.28, "base_demand": 0.72, "tier": "smaller_club", "rivals": ["NFO", "AVL"], "expected_finish_range": [10, 19], "stars": 1, "global_fanbase_millions": 5},
    {"club_id": "BOU", "name": "AFC Bournemouth", "city": "Bournemouth", "venue": "Vitality Stadium", "capacity": 11307, "season_ticket_pct": 0.50, "member_priority_pct": 0.10, "hospitality_pct": 0.05, "dynamic_inventory_pct": 0.35, "base_demand": 0.65, "tier": "smaller_club", "rivals": ["SOU", "POR"], "expected_finish_range": [12, 19], "stars": 0, "global_fanbase_millions": 1},
    {"club_id": "IPS", "name": "Ipswich Town", "city": "Ipswich", "venue": "Portman Road", "capacity": 30000, "season_ticket_pct": 0.55, "member_priority_pct": 0.08, "hospitality_pct": 0.03, "dynamic_inventory_pct": 0.34, "base_demand": 0.62, "tier": "smaller_club", "rivals": ["NOR", "CIT"], "expected_finish_range": [14, 20], "stars": 0, "global_fanbase_millions": 1},
    {"club_id": "SOU", "name": "Southampton", "city": "Southampton", "venue": "St. Mary's Stadium", "capacity": 32384, "season_ticket_pct": 0.55, "member_priority_pct": 0.08, "hospitality_pct": 0.03, "dynamic_inventory_pct": 0.34, "base_demand": 0.60, "tier": "smaller_club", "rivals": ["BOU", "POR"], "expected_finish_range": [15, 20], "stars": 0, "global_fanbase_millions": 2}
]

# --- SECTION 2: SEASON CONFIG ---

SEASON_DATES = {
    "2022-23": {"start": "2022-08-06", "end": "2023-05-28"},
    "2023-24": {"start": "2023-08-12", "end": "2024-05-19"}
}

class EPLMatchDataGenerator:
    def __init__(self, data_dir="EPL/data"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.stakes_engine = StakesEngine()
        self.matches = []
        
    def generate(self):
        self.matches = []
        
        for season_name, dates in SEASON_DATES.items():
            print(f"Generating Season {season_name}...")
            
            # 1. Create fixtures (sequential matchweeks)
            fixtures = self._create_fixtures(season_name, dates["start"])
            
            # 2. Maintain live standings
            club_stats = {c["club_id"]: {
                "points": 0, "goals_scored": 0, "goals_against": 0, 
                "form_score": 0.5, "form_history": [0.5] * 5,
                "tenure": random.randint(1, 36) # Months
            } for c in CLUBS}
            
            standings = self._get_standings(club_stats)
            
            for mw in range(1, 39):
                mw_matches = [m for m in fixtures if m["match_round"] == mw]
                
                for match in mw_matches:
                    home_id = match["home_club_id"]
                    away_id = match["away_club_id"]
                    
                    # A. Dynamic Stakes
                    home_stats = club_stats[home_id]
                    away_stats = club_stats[away_id]
                    home_pos = self._get_position(home_id, standings)
                    away_pos = self._get_position(away_id, standings)
                    
                    match["home_position"] = home_pos
                    match["away_position"] = away_pos
                    match["home_points"] = home_stats["points"]
                    match["away_points"] = away_stats["points"]

                    stakes = self.stakes_engine.compute_stakes_score(
                        home_position=home_pos,
                        away_position=away_pos,
                        home_points=home_stats["points"],
                        away_points=away_stats["points"],
                        home_gd=home_stats["goals_scored"] - home_stats["goals_against"],
                        away_gd=away_stats["goals_scored"] - away_stats["goals_against"],
                        matchweek=mw,
                        standings=standings
                    )
                    match.update(stakes)
                    
                    # B. Add Attributes (TV Slot, Weather, Fatigue, etc.)
                    self._add_match_attributes(match, club_stats, season_name)
                    
                    # C. Inventory Segmentation & Attendance
                    self._calculate_attendance(match)
                    
                    # D. Booking Curve
                    self._generate_booking_curve(match)
                    
                    # E. Simulate Result & Update State
                    self._resolve_match(match, club_stats)
                    
                    self.matches.append(match)
                
                # Update standings after each matchweek
                standings = self._get_standings(club_stats)
                
        self.save_outputs()
        return self.get_summary()

    def _create_fixtures(self, season_name, start_date_str):
        c_ids = [c["club_id"] for c in CLUBS]
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        
        # Double Round Robin
        rounds = self._get_round_robin_schedule(c_ids)
        all_fixtures = []
        
        for mw, round_pairs in enumerate(rounds):
            mw_date = start_date + timedelta(weeks=mw)
            for home_id, away_id in round_pairs:
                home_club = next(c for c in CLUBS if c["club_id"] == home_id)
                away_club = next(c for c in CLUBS if c["club_id"] == away_id)
                
                match = {
                    "match_id": f"{home_id}_{away_id}_{season_name}",
                    "season": season_name,
                    "match_round": mw + 1,
                    "match_date": mw_date.strftime("%Y-%m-%d"), # Will be refined by TV slot
                    "home_club_id": home_id,
                    "away_club_id": away_id,
                    "home_tier": home_club["tier"],
                    "away_tier": away_club["tier"],
                    "venue": home_club["venue"],
                    "total_capacity": home_club["capacity"],
                    "is_derby": away_id in home_club["rivals"]
                }
                all_fixtures.append(match)
        return all_fixtures

    def _get_round_robin_schedule(self, teams):
        if len(teams) % 2: teams.append(None)
        n = len(teams)
        rounds = []
        t = list(teams)
        for i in range(n - 1):
            round_pairs = []
            for j in range(n // 2):
                if t[j] and t[n-1-j]:
                    if i % 2 == 0: round_pairs.append((t[j], t[n-1-j]))
                    else: round_pairs.append((t[n-1-j], t[j]))
            t.insert(1, t.pop())
            rounds.append(round_pairs)
        
        # Second half of season: swap home/away
        second_half = []
        for r in rounds:
            second_half.append([(p[1], p[0]) for p in r])
        return rounds + second_half

    def _get_standings(self, club_stats):
        s = []
        for c_id, stats in club_stats.items():
            s.append({
                "club_id": c_id,
                "points": stats["points"],
                "gd": stats["goals_scored"] - stats["goals_against"],
                "gs": stats["goals_scored"]
            })
        # Sort by points, then GD, then goals scored
        s.sort(key=lambda x: (x["points"], x["gd"], x["gs"]), reverse=True)
        for i, item in enumerate(s):
            item["position"] = i + 1
        return s

    def _get_position(self, club_id, standings):
        return next(item["position"] for item in standings if item["club_id"] == club_id)

    def _add_match_attributes(self, match, club_stats, season_name):
        home_id = match["home_club_id"]
        away_id = match["away_club_id"]
        home_club = next(c for c in CLUBS if c["club_id"] == home_id)
        away_club = next(c for c in CLUBS if c["club_id"] == away_id)
        
        # 1. TV Slot (Section 2)
        match.update(self._assign_tv_slot(match))
        
        # 2. Weather (Section 5)
        dt = datetime.strptime(match["match_date"], "%Y-%m-%d")
        match["weather_severity_score"] = self._get_weather(dt.month)
        
        # 3. Festive Flag
        match["festive_fixture"] = match["match_date"] in [f"{dt.year}-12-26", f"{dt.year}-01-01"]
        
        # 4. Fatigue (Section 5)
        # Randomly assign European participation for Big Six and some established mid
        match["european_midweek_fatigue"] = False
        if home_club["tier"] == "big_six" or (home_club["tier"] == "established_mid" and random.random() < 0.3):
            if random.random() < 0.25: # Roughly 1 in 4 matches have midweek fatigue
                match["european_midweek_fatigue"] = True
        
        # 5. Form
        match["home_form_score"] = club_stats[home_id]["form_score"]
        match["away_form_score"] = club_stats[away_id]["form_score"]
        
        # 6. Manager Bonus
        match["manager_new_bonus"] = club_stats[home_id]["tenure"] < 3
        
        # 7. Star Power
        match["star_power_index"] = home_club["stars"] + away_club["stars"]
        
        # 8. Distance
        match["away_distance_km"] = random.randint(5, 450) # Mock distance
        
        # 9. Normalised WTP Score (Normalise for tier)
        # We'll use a placeholder since actual recovery is in Step 3
        match["attribute_wtp_score"] = random.uniform(0.4, 0.9)

    def _assign_tv_slot(self, match):
        h_tier = match["home_tier"]
        a_tier = match["away_tier"]
        stakes = match["match_stakes_label"]
        
        # Rules from Section 2
        if h_tier == "big_six" and a_tier == "big_six":
            slot = random.choice(["Saturday 12:30", "Sunday 16:30"])
            code = 4 if slot == "Saturday 12:30" else 3
        elif (h_tier == "big_six" or a_tier == "big_six") and stakes in ["Title Decider", "Top Four Decider", "High Stakes"]:
            slot = "Sunday 14:00"
            code = 2 # Sunday 14:00 is not broadcast selected but top-half stakes
        elif stakes == "Relegation Six-Pointer":
            slot = random.choice(["Saturday 17:30", "Sunday 14:00"])
            code = 3 if slot == "Saturday 17:30" else 2
        else:
            rand = random.random()
            if rand < 0.40: slot, code = "Saturday 15:00", 1
            elif rand < 0.55: slot, code = "Sunday 14:00", 2
            elif rand < 0.65: slot, code = "Sunday 16:30", 3
            elif rand < 0.72: slot, code = "Monday 20:00", 2
            elif rand < 0.80: slot, code = "Saturday 12:30", 4
            elif rand < 0.88: slot, code = "Saturday 17:30", 3
            elif rand < 0.95: slot, code = "Midweek Tue/Wed 20:00", 2
            else: slot, code = "Friday 20:00", 2
            
        return {"tv_slot": slot, "tv_slot_encoded": code}

    def _get_weather(self, month):
        # Aug-Sep: 0-1, Oct-Nov: 1-2, Dec-Feb: 2-4, Mar-Apr: 1-2, May: 0-1
        if month in [8, 9, 5]: return random.randint(0, 1)
        if month in [10, 11, 3, 4]: return random.randint(1, 2)
        if month in [12, 1, 2]: return random.randint(2, 4)
        return 0

    def _calculate_attendance(self, match):
        home_id = match["home_club_id"]
        home_club = next(c for c in CLUBS if c["club_id"] == home_id)
        
        # 1. Base Demand
        demand = home_club["base_demand"]
        
        # 2. Multipliers (Section 5)
        # Opponent
        if match["away_tier"] == "big_six": demand *= 1.25
        elif match["away_tier"] == "established_mid": demand *= 1.05
        else: demand *= 0.90
        
        # Stakes
        stakes_m = {
            "Title Decider": 1.40, "Top Four Decider": 1.25, "Relegation Six-Pointer": 1.20,
            "High Stakes": 1.15, "European Spot": 1.08, "Standard": 1.00
        }
        demand *= stakes_m.get(match["match_stakes_label"], 1.0)
        
        # Derby
        if match["is_derby"]: demand *= 1.30
        
        # TV
        tv_m = {"Saturday 12:30": 1.10, "Sunday 16:30": 1.08, "Monday 20:00": 0.92, "Saturday 15:00": 1.00}
        demand *= tv_m.get(match["tv_slot"], 1.02)
        
        # Festive
        if match["festive_fixture"]: demand *= 1.15
        
        # Fatigue
        if match["european_midweek_fatigue"]:
            if home_club["tier"] == "big_six": demand *= 1.05
            else: demand *= 0.93
            
        # Weather
        w_m = {0: 1.0, 1: 0.98, 2: 0.94, 3: 0.88, 4: 0.82}
        demand *= w_m[match["weather_severity_score"]]
        
        # Manager
        if match["manager_new_bonus"]: demand *= 1.08
        
        final_rate = np.clip(demand, 0.40, 1.10) # 1.10 means excess demand
        match["overall_fill_rate"] = float(round(min(1.0, final_rate), 3))
        match["excess_demand_ratio"] = float(round(final_rate, 3))
        
        # 3. Inventory Segmentation (Section 4)
        cap = home_club["capacity"]
        inv_pct = home_club["dynamic_inventory_pct"]
        
        match["inventory"] = {
            "Away End": {"capacity": int(cap * 0.05), "fixed_price": 30},
            "Hospitality": {"capacity": int(cap * 0.08), "fixed_price": 200}, # Placeholder
            "Dynamic Lower": {"capacity": int(cap * inv_pct * 0.55)},
            "Dynamic Upper": {"capacity": int(cap * inv_pct * 0.35)},
            "Dynamic General": {"capacity": int(cap * inv_pct * 0.10)}
        }
        
        # Fill zones
        total_sold = 0
        for zone, zdata in match["inventory"].items():
            # Apply zone-specific sensitivity? 
            # Hospitality is less sensitive, General more.
            # For simulation, we'll just distribute the overall fill rate with some noise.
            z_fill = np.clip(final_rate + random.uniform(-0.05, 0.05), 0, 1)
            sold = int(zdata["capacity"] * z_fill)
            zdata["tickets_sold"] = sold
            total_sold += sold
            
        # Add the non-dynamic portions (Season Tickets)
        st_sold = int(cap * home_club["season_ticket_pct"])
        match["total_tickets_sold"] = total_sold + st_sold
        match["st_tickets_sold"] = st_sold

    def _generate_booking_curve(self, match):
        # Archetypes from Section 9
        h_tier = match["home_tier"]
        a_tier = match["away_tier"]
        stakes = match["match_stakes_label"]
        score = match["match_stakes_score"]
        
        if h_tier == "big_six" and (a_tier == "big_six" or match["is_derby"] or score > 0.70):
            arch = "Immediate Sellout"
            p = {"t_mid": 5, "k": 0.40}
        elif a_tier == "big_six" or (h_tier == "established_mid" and score > 0.50):
            arch = "Early Surge"
            p = {"t_mid": 35, "k": 0.15}
        elif stakes == "Relegation Six-Pointer" or (match["match_round"] >= 33 and score > 0.40):
            arch = "Late Surge"
            p = {"t_mid": 52, "k": 0.18}
        elif a_tier == "smaller_club" and score < 0.30:
            arch = "Flat/Slow"
            p = {"t_mid": 48, "k": 0.06}
        else:
            arch = "Consistent Gradual"
            p = {"t_mid": 45, "k": 0.10}
            
        match["booking_curve_archetype"] = arch
        
        # Generate 61-day curve
        curve = []
        total = match["total_tickets_sold"]
        for t in range(61):
            val = total / (1 + np.exp(-p["k"] * (t - p["t_mid"])))
            curve.append(int(val))
        curve[-1] = total
        match["booking_curve"] = curve
        
        # Velocity signals
        match["velocity_T14"] = round(curve[46] / total, 3) if total > 0 else 0
        match["velocity_T7"] = round(curve[53] / total, 3) if total > 0 else 0

    def _resolve_match(self, match, club_stats):
        h_id = match["home_club_id"]
        a_id = match["away_club_id"]
        
        # Simple probabilistic result based on form and tier
        tiers = {"big_six": 3, "established_mid": 2, "smaller_club": 1}
        h_score = club_stats[h_id]["form_score"] * tiers[match["home_tier"]]
        a_score = club_stats[a_id]["form_score"] * tiers[match["away_tier"]]
        
        prob_h = h_score / (h_score + a_score + 0.1)
        res = random.random()
        if res < prob_h: # Home win
            club_stats[h_id]["points"] += 3
            match["result"] = "H"
        elif res < prob_h + 0.25: # Draw
            club_stats[h_id]["points"] += 1
            club_stats[a_id]["points"] += 1
            match["result"] = "D"
        else: # Away win
            club_stats[a_id]["points"] += 3
            match["result"] = "A"
            
        # Update form
        h_res = 1.0 if match["result"] == "H" else (0.5 if match["result"] == "D" else 0.0)
        club_stats[h_id]["form_score"] = club_stats[h_id]["form_score"] * 0.7 + h_res * 0.3
        club_stats[a_id]["form_score"] = club_stats[a_id]["form_score"] * 0.7 + (1.0 - h_res) * 0.3
        
        # Simple goal simulation for GD
        match["home_goals"] = random.randint(0, 4) if match["result"] == "H" else random.randint(0, 2)
        match["away_goals"] = random.randint(0, 4) if match["result"] == "A" else random.randint(0, 2)
        if match["result"] == "D": match["home_goals"] = match["away_goals"] = random.randint(0, 3)
        
        club_stats[h_id]["goals_scored"] += match["home_goals"]
        club_stats[h_id]["goals_against"] += match["away_goals"]
        club_stats[a_id]["goals_scored"] += match["away_goals"]
        club_stats[a_id]["goals_against"] += match["home_goals"]

    def save_outputs(self):
        with open(os.path.join(self.data_dir, "epl_match_data.json"), "w") as f:
            json.dump(self.matches, f, indent=2)
            
        df = pd.DataFrame(self.matches)
        df_flat = df.drop(columns=["inventory", "booking_curve"])
        df_flat.to_csv(os.path.join(self.data_dir, "epl_match_data.csv"), index=False)

    def get_summary(self):
        df = pd.DataFrame(self.matches)
        return {
            "total_matches": len(self.matches),
            "seasons": df["season"].unique().tolist(),
            "avg_stakes_score": float(df["match_stakes_score"].mean()),
            "stakes_by_round": df.groupby("match_round")["match_stakes_score"].mean().to_dict(),
            "archetype_dist": df["booking_curve_archetype"].value_counts().to_dict()
        }

if __name__ == "__main__":
    gen = EPLMatchDataGenerator()
    summary = gen.generate()
    print(json.dumps(summary, indent=2))
