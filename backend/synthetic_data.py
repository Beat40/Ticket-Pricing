import json
import uuid
import random
import numpy as np
import hashlib
from datetime import datetime, timedelta
import os

# --- STEP 1: DEFINE GROUND TRUTH UTILITIES ---

SEGMENTS = {
    "Premium Seeker": {
        "n": 56,
        "utilities": {
            "opponent_competitive": 0.8,
            "opponent_elite": 2.2,
            "zone_upper_standard": 1.0,
            "zone_lower_bowl": 2.2,
            "zone_courtside_vip": 3.8,
            "stakes_playoff": 1.2,
            "stakes_final": 2.8,
            "price_coef": -0.035,
            "bundle_sbb": 0.4,
            "bundle_sbb_food": 0.6,
            "star_uncertain": 0.5,
            "star_confirmed": 1.4,
            "kickoff_sat_afternoon": 0.5,
            "kickoff_sat_evening": 0.9
        }
    },
    "Value Loyalist": {
        "n": 152,
        "utilities": {
            "opponent_competitive": 0.7,
            "opponent_elite": 1.4,
            "zone_upper_standard": 0.8,
            "zone_lower_bowl": 1.8,
            "zone_courtside_vip": 2.4,
            "stakes_playoff": 0.9,
            "stakes_final": 1.8,
            "price_coef": -0.055,
            "bundle_sbb": 1.2,
            "bundle_sbb_food": 1.9,
            "star_uncertain": 0.3,
            "star_confirmed": 0.9,
            "kickoff_sat_afternoon": 0.6,
            "kickoff_sat_evening": 1.0
        }
    },
    "Atmosphere Seeker": {
        "n": 80,
        "utilities": {
            "opponent_competitive": 1.1,
            "opponent_elite": 1.6,
            "zone_upper_standard": 0.5,
            "zone_lower_bowl": 1.0,
            "zone_courtside_vip": 1.2,
            "stakes_playoff": 1.4,
            "stakes_final": 2.2,
            "price_coef": -0.065,
            "bundle_sbb": 0.5,
            "bundle_sbb_food": 0.7,
            "star_uncertain": 0.6,
            "star_confirmed": 1.6,
            "kickoff_sat_afternoon": 0.4,
            "kickoff_sat_evening": 0.8
        }
    },
    "Occasional Neutral": {
        "n": 112,
        "utilities": {
            "opponent_competitive": 0.4,
            "opponent_elite": 0.8,
            "zone_upper_standard": 0.6,
            "zone_lower_bowl": 1.1,
            "zone_courtside_vip": 1.4,
            "stakes_playoff": 0.5,
            "stakes_final": 1.1,
            "price_coef": -0.085,
            "bundle_sbb": 0.3,
            "bundle_sbb_food": 0.5,
            "star_uncertain": 0.2,
            "star_confirmed": 0.5,
            "kickoff_sat_afternoon": 0.3,
            "kickoff_sat_evening": 0.5
        }
    }
}

FIRST_NAMES = [
    "Luca", "Noah", "Elias", "Leon", "David", "Jonas", "Simon", "Finn", 
    "Tobias", "Lukas", "Anna", "Laura", "Sara", "Lea", "Julia", "Sophie", "Mia", "Emma", 
    "Lena", "Jana", "Markus", "Stefan", "Daniel", "Michael", "Thomas", "Andreas", "Christian",
    "Patrick", "Fabian", "Raphael"
]

SURNAMES = [
    "Müller", "Meier", "Schmid", "Keller", "Weber", "Huber", "Schneider",
    "Fischer", "Zimmermann", "Brunner", "Steiner", "Moser", "Frei", "Gerber", "Bucher",
    "Lehmann", "Wenger", "Lüthi", "Bosshard", "Kunz", "Maurer", "Baumann", "Roth", "Suter",
    "Hofer", "Arnold", "Wüthrich", "Flückiger", "Schär", "Zbinden"
]

DISPLAY_STRINGS = {
    "opponent": {"Elite": "Elite", "Competitive": "Competitive", "Standard": "Standard"},
    "seat_zone": {
        "Courtside_VIP": "Courtside VIP",
        "Lower_Bowl": "Lower Bowl / Club Seats",
        "Upper_Standard": "Upper Standard",
        "Standing": "Standing"
    },
    "stakes": {
        "Final": "EHF EURO / Cup Final",
        "Playoff": "League Playoff / Knockout",
        "Group": "Regular Season Group Match"
    },
    "bundle": {
        "Ticket_Only": "Ticket Only",
        "SBB": "Ticket + SBB Travel",
        "SBB_Food": "Ticket + SBB Travel + Food & Drink"
    },
    "star_player": {
        "Confirmed": "Yes — marquee international confirmed",
        "Uncertain": "Uncertain — squad announced, no star confirmed",
        "No": "No"
    },
    "kickoff": {
        "Sat_Evening": "Saturday evening 19:30",
        "Sat_Afternoon": "Saturday afternoon 15:00",
        "Weekday": "Weekday 19:00"
    }
}

ZONE_PRICES = {
    "Standing": [12, 18, 25, 32],
    "Upper_Standard": [25, 38, 52, 68],
    "Lower_Bowl": [40, 58, 75, 95],
    "Courtside_VIP": [70, 95, 120, 150]
}

class SyntheticDataGenerator:
    def __init__(self):
        self.respondents = []
        self.sigma = 0.30
        self.utility_scale = 1.0
        self.data_dir = os.path.join(os.getcwd(), "data")
        os.makedirs(self.data_dir, exist_ok=True)

    def save_ground_truth(self):
        path = os.path.join(self.data_dir, "ground_truth_utilities.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(SEGMENTS, f, indent=2)

    def generate_individual_utilities(self, segment_name, segment_data):
        utilities = segment_data["utilities"]
        individual_betas = {}
        for key, val in utilities.items():
            sigma_val = max(0.05, self.sigma * abs(val))
            noise = np.random.normal(0, sigma_val)
            beta = val + noise
            if key == "price_coef":
                while beta >= 0:
                    beta = val + np.random.normal(0, sigma_val)
            individual_betas[key] = beta
        return individual_betas

    def generate_option(self):
        zone = random.choice(list(ZONE_PRICES.keys()))
        return {
            "opponent": random.choice(["Standard", "Competitive", "Elite"]),
            "seat_zone": zone,
            "stakes": random.choice(["Group", "Playoff", "Final"]),
            "price": random.choice(ZONE_PRICES[zone]),
            "bundle": random.choice(["Ticket_Only", "SBB", "SBB_Food"]),
            "star_player": random.choice(["No", "Uncertain", "Confirmed"]),
            "kickoff": random.choice(["Weekday", "Sat_Afternoon", "Sat_Evening"])
        }

    def compute_utility(self, option, betas):
        u = 0.0
        # Scale non-price utilities
        scale = self.utility_scale
        
        if option["opponent"] == "Competitive": u += betas["opponent_competitive"] * scale
        elif option["opponent"] == "Elite": u += betas["opponent_elite"] * scale
        
        if option["seat_zone"] == "Upper_Standard": u += betas["zone_upper_standard"] * scale
        elif option["seat_zone"] == "Lower_Bowl": u += betas["zone_lower_bowl"] * scale
        elif option["seat_zone"] == "Courtside_VIP": u += betas["zone_courtside_vip"] * scale
        
        if option["stakes"] == "Playoff": u += betas["stakes_playoff"] * scale
        elif option["stakes"] == "Final": u += betas["stakes_final"] * scale
        
        if option["bundle"] == "SBB": u += betas["bundle_sbb"] * scale
        elif option["bundle"] == "SBB_Food": u += betas["bundle_sbb_food"] * scale
        
        if option["star_player"] == "Uncertain": u += betas["star_uncertain"] * scale
        elif option["star_player"] == "Confirmed": u += betas["star_confirmed"] * scale
        
        if option["kickoff"] == "Sat_Afternoon": u += betas["kickoff_sat_afternoon"] * scale
        elif option["kickoff"] == "Sat_Evening": u += betas["kickoff_sat_evening"] * scale
        
        u += option["price"] * betas["price_coef"]
        return u

    def make_choice(self, opt_a, opt_b, betas):
        v_a = self.compute_utility(opt_a, betas)
        v_b = self.compute_utility(opt_b, betas)
        
        exp_a = np.exp(v_a)
        exp_b = np.exp(v_b)
        denom = exp_a + exp_b + 1.0 # 1.0 is exp(0.0) for "neither"
        
        p_a = exp_a / denom
        p_b = exp_b / denom
        p_n = 1.0 / denom
        
        return random.choices(["A", "B", "neither"], weights=[p_a, p_b, p_n])[0]

    def generate_data(self):
        self.save_ground_truth()
        self.sigma = 0.30
        self.utility_scale = 1.0
        
        while True:
            all_respondents = []
            neither_count = 0
            total_choices = 0
            
            for segment_name, data in SEGMENTS.items():
                for _ in range(data["n"]):
                    res_id = str(uuid.uuid4())
                    name = f"{random.choice(FIRST_NAMES)} {random.choice(SURNAMES)}"
                    
                    seed_val = int(hashlib.md5(res_id.encode()).hexdigest(), 16) % (2**32)
                    np.random.seed(seed_val)
                    random.seed(seed_val)
                    
                    betas = self.generate_individual_utilities(segment_name, data)
                    
                    tasks = []
                    # Generate 15 distinct base tasks
                    base_tasks = []
                    for i in range(15):
                        opt_a = self.generate_option()
                        while True:
                            opt_b = self.generate_option()
                            diff = sum(1 for k in opt_a if opt_a[k] != opt_b[k])
                            if diff >= 3:
                                break
                        base_tasks.append((opt_a, opt_b))
                    
                    # Construct 17 tasks: 1-13 + 14(dup 1) + 15(dup 2) + 16, 17? 
                    # User says: Tasks 14 and 15 are exact duplicates of tasks 1 and 2 respectively.
                    # Index 0, 1 ... 12 (Task 1 to 13)
                    # Index 13 (Task 14) = Index 0
                    # Index 14 (Task 15) = Index 1
                    # Index 15 (Task 16)
                    # Index 16 (Task 17)
                    
                    final_tasks = []
                    for i in range(13):
                        final_tasks.append(base_tasks[i])
                    final_tasks.append(base_tasks[0]) # Task 14
                    final_tasks.append(base_tasks[1]) # Task 15
                    final_tasks.append(base_tasks[13]) # Task 16
                    final_tasks.append(base_tasks[14]) # Task 17
                    
                    responses = []
                    for i, (opt_a, opt_b) in enumerate(final_tasks):
                        choice = self.make_choice(opt_a, opt_b, betas)
                        if choice == "neither":
                            neither_count += 1
                        total_choices += 1
                        
                        responses.append({
                            "task_index": i,
                            "option_chosen": choice,
                            "option_a": {k: DISPLAY_STRINGS.get(k, {}).get(v, v) for k, v in opt_a.items()},
                            "option_b": {k: DISPLAY_STRINGS.get(k, {}).get(v, v) for k, v in opt_b.items()}
                        })
                    
                    # Consistency check
                    c1 = responses[13]["option_chosen"] == responses[0]["option_chosen"]
                    c2 = responses[14]["option_chosen"] == responses[1]["option_chosen"]
                    consistency_flag = c1 and c2
                    
                    # Timestamp
                    start_date = datetime(2025, 1, 10)
                    random_days = random.randint(0, 59)
                    random_seconds = random.randint(0, 86400)
                    ts = start_date + timedelta(days=random_days, seconds=random_seconds)
                    
                    all_respondents.append({
                        "respondent_id": res_id,
                        "name": name,
                        "segment_true": segment_name,
                        "consistency_flag": consistency_flag,
                        "submitted_at": ts.isoformat() + "Z",
                        "responses": responses,
                        "_betas": betas # Hidden for processing
                    })
            
            # Check neither rate
            neither_rate = neither_count / total_choices
            if neither_rate > 0.15:
                print(f"Neither rate too high: {neither_rate:.2%}. Scaling utilities (current scale: {self.utility_scale:.2f})...")
                self.utility_scale *= 1.2
                continue
            
            # Check consistency rate
            c_rate = sum(1 for r in all_respondents if r["consistency_flag"]) / len(all_respondents)
            if c_rate < 0.75:
                print(f"Consistency too low: {c_rate:.2%}. (Scale: {self.utility_scale:.2f}, Sigma: {self.sigma:.2f})")
                if self.sigma > 0.20:
                    print("Reducing sigma to 0.20...")
                    self.sigma = 0.20
                else:
                    print("Increasing utility scale to boost consistency...")
                    self.utility_scale *= 1.1
                continue
            elif c_rate > 0.90:
                print(f"Consistency too high: {c_rate:.2%}. Increasing sigma...")
                self.sigma = 0.35
                continue
            
            # Success
            self.respondents = all_respondents
            self.save_files()
            return self.get_report()

    def save_files(self):
        # Save survey responses (without betas)
        resp_output = []
        for r in self.respondents:
            clean_r = {k: v for k, v in r.items() if k != "_betas"}
            resp_output.append(clean_r)
        
        with open(os.path.join(self.data_dir, "survey_responses.json"), "w", encoding="utf-8") as f:
            json.dump(resp_output, f, indent=2)
            
        # Save individual utilities
        indiv_utils = []
        for r in self.respondents:
            indiv_utils.append({
                "respondent_id": r["respondent_id"],
                "segment": r["segment_true"],
                "utilities": r["_betas"]
            })
        with open(os.path.join(self.data_dir, "individual_utilities.json"), "w", encoding="utf-8") as f:
            json.dump(indiv_utils, f, indent=2)

    def get_report(self):
        total = len(self.respondents)
        segments_count = {}
        for r in self.respondents:
            s = r["segment_true"]
            segments_count[s] = segments_count.get(s, 0) + 1
            
        c_rate = sum(1 for r in self.respondents if r["consistency_flag"]) / total
        
        all_choices = [resp["option_chosen"] for r in self.respondents for resp in r["responses"]]
        n_a = all_choices.count("A")
        n_b = all_choices.count("B")
        n_n = all_choices.count("neither")
        total_c = len(all_choices)
        
        report = {
            "total_respondents": total,
            "segment_distribution": {
                s: {"n": count, "pct": round(count/total*100, 1)} 
                for s, count in segments_count.items()
            },
            "consistency_rate": round(c_rate, 3),
            "neither_rate": round(n_n / total_c, 3),
            "choice_rates": {
                "A": round(n_a / total_c, 3),
                "B": round(n_b / total_c, 3),
                "neither": round(n_n / total_c, 3)
            },
            "ground_truth_check": {}
        }
        
        for s_name, s_data in SEGMENTS.items():
            s_resps = [r for r in self.respondents if r["segment_true"] == s_name]
            avg_elite = np.mean([r["_betas"]["opponent_elite"] for r in s_resps])
            avg_price = np.mean([r["_betas"]["price_coef"] for r in s_resps])
            
            truth_elite = s_data["utilities"]["opponent_elite"]
            truth_price = s_data["utilities"]["price_coef"]
            
            # Use utility_scale for comparison if we scaled them? 
            # Actually betas are generated from SEGMENTS directly. utility_scale is only used in compute_utility.
            # So recovery should be compared to the original ground truth values used for beta generation.
            
            diff_elite = abs(avg_elite - truth_elite) / abs(truth_elite)
            diff_price = abs(avg_price - truth_price) / abs(truth_price)
            
            status = "OK"
            if diff_elite > 0.15 or diff_price > 0.15:
                status = "WARN"
                
            report["ground_truth_check"][s_name] = {
                "elite_utility_mean": round(float(avg_elite), 3),
                "elite_utility_truth": truth_elite,
                "price_coef_mean": round(float(avg_price), 4),
                "price_coef_truth": truth_price,
                "status": status
            }
            
        return report

if __name__ == "__main__":
    gen = SyntheticDataGenerator()
    report = gen.generate_data()
    print(json.dumps(report, indent=2))
