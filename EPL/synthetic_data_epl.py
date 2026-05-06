import json
import uuid
import random
import numpy as np
import hashlib
from datetime import datetime, timedelta
import os

# --- DOMAIN CONSTANTS (EPL) ---

# BIG SIX TIER
BIG_SIX_SEGMENTS = {
    "Tourist/Global Fan": {
        "n": 105,
        "utilities": {
            "opponent_big_six": 3.2,
            "opponent_mid": 1.1,
            "zone_lower": 2.8,
            "zone_upper": 1.4,
            "zone_hospitality": 4.5,
            "stakes_title": 3.5,
            "stakes_top4": 1.8,
            "price_coef": -0.025,
            "derby": 1.5,
            "hospitality": 2.1,
            "tv_prime": 0.8
        }
    },
    "Corporate/Hospitality": {
        "n": 75,
        "utilities": {
            "opponent_big_six": 2.1,
            "opponent_mid": 0.8,
            "zone_lower": 1.5,
            "zone_upper": 0.8,
            "zone_hospitality": 5.5,
            "stakes_title": 2.0,
            "stakes_top4": 1.0,
            "price_coef": -0.015,
            "derby": 1.0,
            "hospitality": 4.2,
            "tv_prime": 0.5
        }
    },
    "Local Season Ticket": {
        "n": 90,
        "utilities": {
            "opponent_big_six": 2.5,
            "opponent_mid": 1.5,
            "zone_lower": 3.5,
            "zone_upper": 2.0,
            "zone_hospitality": 1.2,
            "stakes_title": 2.8,
            "stakes_top4": 2.0,
            "price_coef": -0.055,
            "derby": 2.8,
            "hospitality": 0.6,
            "tv_prime": 0.3
        }
    },
    "Matchday Casual": {
        "n": 30,
        "utilities": {
            "opponent_big_six": 1.8,
            "opponent_mid": 0.5,
            "zone_lower": 1.5,
            "zone_upper": 1.0,
            "zone_hospitality": 0.5,
            "stakes_title": 2.5,
            "stakes_top4": 1.2,
            "price_coef": -0.090,
            "derby": 1.8,
            "hospitality": 0.3,
            "tv_prime": 0.2
        }
    }
}

# MID-TABLE TIER
MID_TABLE_SEGMENTS = {
    "Local Loyal": {
        "n": 120,
        "utilities": {
            "opponent_big_six": 2.8,
            "opponent_mid": 1.2,
            "zone_lower": 3.2,
            "zone_upper": 1.8,
            "zone_hospitality": 1.5,
            "stakes_top4": 2.5,
            "stakes_relegation": 2.8,
            "stakes_euro": 1.5,
            "price_coef": -0.065,
            "derby": 2.5,
            "tv_prime": 0.4
        }
    },
    "Family": {
        "n": 90,
        "utilities": {
            "opponent_big_six": 2.2,
            "opponent_mid": 1.0,
            "zone_lower": 2.0,
            "zone_upper": 2.5,
            "zone_hospitality": 0.8,
            "stakes_top4": 1.8,
            "stakes_relegation": 2.0,
            "stakes_euro": 1.2,
            "price_coef": -0.075,
            "derby": 1.5,
            "tv_prime": -0.3
        }
    },
    "Occasional": {
        "n": 60,
        "utilities": {
            "opponent_big_six": 3.0,
            "opponent_mid": 0.5,
            "zone_lower": 1.8,
            "zone_upper": 1.2,
            "zone_hospitality": 0.5,
            "stakes_top4": 1.5,
            "stakes_relegation": 1.0,
            "stakes_euro": 0.8,
            "price_coef": -0.095,
            "derby": 2.0,
            "tv_prime": 0.2
        }
    },
    "Away Traveller": {
        "n": 30,
        "utilities": {
            "opponent_big_six": 1.5,
            "opponent_mid": 1.8,
            "zone_lower": 2.5,
            "zone_upper": 1.5,
            "zone_hospitality": 1.0,
            "stakes_top4": 1.2,
            "stakes_relegation": 1.5,
            "stakes_euro": 1.8,
            "price_coef": -0.070,
            "derby": 1.0,
            "tv_prime": 0.5
        }
    }
}

# SMALLER CLUB TIER
SMALL_CLUB_SEGMENTS = {
    "Die-Hard Local": {
        "n": 135,
        "utilities": {
            "opponent_big_six": 3.5,
            "opponent_mid": 1.5,
            "zone_lower": 3.8,
            "zone_upper": 2.0,
            "zone_hospitality": 1.0,
            "stakes_relegation": 4.0,
            "stakes_top4": 2.0,
            "stakes_euro": 1.5,
            "price_coef": -0.075,
            "derby": 3.5,
            "tv_prime": 0.2
        }
    },
    "Community Fan": {
        "n": 90,
        "utilities": {
            "opponent_big_six": 2.5,
            "opponent_mid": 1.0,
            "zone_lower": 2.0,
            "zone_upper": 2.5,
            "zone_hospitality": 0.5,
            "stakes_relegation": 3.0,
            "stakes_top4": 1.5,
            "stakes_euro": 1.0,
            "price_coef": -0.095,
            "derby": 2.5,
            "tv_prime": -0.2
        }
    },
    "Neutral Observer": {
        "n": 60,
        "utilities": {
            "opponent_big_six": 3.8,
            "opponent_mid": 0.8,
            "zone_lower": 1.5,
            "zone_upper": 1.0,
            "zone_hospitality": 0.5,
            "stakes_relegation": 2.0,
            "stakes_top4": 1.5,
            "stakes_euro": 0.8,
            "price_coef": -0.110,
            "derby": 1.0,
            "tv_prime": 0.4
        }
    },
    "Away Day Fan": {
        "n": 15,
        "utilities": {
            "opponent_big_six": 1.0,
            "opponent_mid": 1.5,
            "zone_lower": 2.0,
            "zone_upper": 1.5,
            "zone_hospitality": 0.5,
            "stakes_relegation": 2.5,
            "stakes_top4": 1.0,
            "stakes_euro": 2.0,
            "price_coef": -0.085,
            "derby": 0.5,
            "tv_prime": 0.8
        }
    }
}

# --- GENERATOR CLASSES ---

class BaseSyntheticGenerator:
    def __init__(self, tier_name, segments, price_points, data_dir="EPL/data"):
        self.tier_name = tier_name
        self.segments = segments
        self.price_points = price_points
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.sigma = 0.15 # Reduced noise
        self.utility_scale = 3.0 # Boosted signal
        
    def generate_individual_utilities(self, segment_data):
        utilities = segment_data["utilities"]
        individual_betas = {}
        for key, val in utilities.items():
            sigma_val = max(0.02, self.sigma * abs(val))
            noise = np.random.normal(0, sigma_val)
            beta = val + noise
            if key == "price_coef":
                while beta >= 0:
                    beta = val + np.random.normal(0, sigma_val)
            individual_betas[key] = beta
        return individual_betas

    def generate_option(self):
        zone = random.choice(["Lower Tier", "Upper Tier", "Hospitality"])
        prices = self.price_points[zone]
        
        opp_tiers = ["Standard", "Mid", "Big Six"]
        stakes = ["Standard", "Top 4", "Title Decider"] if self.tier_name == "big_six" else \
                 ["Standard", "Top 4", "Relegation", "Euro"]
        
        return {
            "opponent": random.choice(opp_tiers),
            "seat_zone": zone,
            "stakes": random.choice(stakes),
            "price": random.choice(prices),
            "derby": random.choice(["Yes", "No"]),
            "hospitality_inclusion": random.choice(["Yes", "No"]),
            "tv_prime": random.choice(["Yes", "No"])
        }

    def compute_utility(self, option, betas):
        u = 0.0
        s = self.utility_scale
        
        # Opponent
        if option["opponent"] == "Mid": u += betas.get("opponent_mid", 0) * s
        elif option["opponent"] == "Big Six": u += betas.get("opponent_big_six", 0) * s
        
        # Zone
        if option["seat_zone"] == "Lower Tier": u += betas.get("zone_lower", 0) * s
        elif option["seat_zone"] == "Upper Tier": u += betas.get("zone_upper", 0) * s
        elif option["seat_zone"] == "Hospitality": u += betas.get("zone_hospitality", 0) * s
        
        # Stakes
        if option["stakes"] == "Title Decider": u += betas.get("stakes_title", 0) * s
        elif option["stakes"] == "Top 4": u += betas.get("stakes_top4", 0) * s
        elif option["stakes"] == "Relegation": u += betas.get("stakes_relegation", 0) * s
        elif option["stakes"] == "Euro": u += betas.get("stakes_euro", 0) * s
        
        # Other
        if option["derby"] == "Yes": u += betas.get("derby", 0) * s
        if option["hospitality_inclusion"] == "Yes": u += betas.get("hospitality", 0) * s
        if option["tv_prime"] == "Yes": u += betas.get("tv_prime", 0) * s
        
        # Price (Price coef already reflects sensitivity, scale it too if needed, 
        # but usually price coef is relative to the choice scale)
        u += option["price"] * betas["price_coef"] * s
        return u

    def make_choice(self, opt_a, opt_b, betas):
        v_a = self.compute_utility(opt_a, betas)
        v_b = self.compute_utility(opt_b, betas)
        
        # Logit probabilities
        # Prevent overflow
        m = max(v_a, v_b, 0)
        exp_a = np.exp(v_a - m)
        exp_b = np.exp(v_b - m)
        exp_n = np.exp(0 - m)
        denom = exp_a + exp_b + exp_n
        
        p_a = exp_a / denom
        p_b = exp_b / denom
        p_n = exp_n / denom
        
        return random.choices(["A", "B", "neither"], weights=[p_a, p_b, p_n])[0]

    def generate_data(self):
        all_respondents = []
        for segment_name, data in self.segments.items():
            for _ in range(data["n"]):
                res_id = str(uuid.uuid4())
                
                seed_val = int(hashlib.md5(res_id.encode()).hexdigest(), 16) % (2**32)
                np.random.seed(seed_val)
                random.seed(seed_val)
                
                betas = self.generate_individual_utilities(data)
                
                responses = []
                for i in range(13): # 1-13
                    opt_a = self.generate_option()
                    opt_b = self.generate_option()
                    choice = self.make_choice(opt_a, opt_b, betas)
                    
                    responses.append({
                        "task_index": i,
                        "option_chosen": choice,
                        "option_a": opt_a,
                        "option_b": opt_b
                    })
                
                # Duplicate tasks for consistency (Task 14 and 15 are dups of 1 and 2)
                r1_choice = responses[0]["option_chosen"]
                r2_choice = responses[1]["option_chosen"]
                
                # Task 14 (dup 1)
                responses.append({
                    "task_index": 13,
                    "option_chosen": self.make_choice(responses[0]["option_a"], responses[0]["option_b"], betas),
                    "option_a": responses[0]["option_a"],
                    "option_b": responses[0]["option_b"]
                })
                # Task 15 (dup 2)
                responses.append({
                    "task_index": 14,
                    "option_chosen": self.make_choice(responses[1]["option_a"], responses[1]["option_b"], betas),
                    "option_a": responses[1]["option_a"],
                    "option_b": responses[1]["option_b"]
                })
                
                # Add two more tasks to make 17 as per original logic? 
                # Briefing says "Generate 15 distinct base tasks" in original code but here just "15 tasks".
                # Let's add 16 and 17.
                for i in range(15, 17):
                    opt_a = self.generate_option()
                    opt_b = self.generate_option()
                    choice = self.make_choice(opt_a, opt_b, betas)
                    responses.append({
                        "task_index": i,
                        "option_chosen": choice,
                        "option_a": opt_a,
                        "option_b": opt_b
                    })
                
                consistency = (responses[13]["option_chosen"] == r1_choice and 
                               responses[14]["option_chosen"] == r2_choice)
                
                all_respondents.append({
                    "respondent_id": res_id,
                    "segment_true": segment_name,
                    "consistency_flag": consistency,
                    "responses": responses,
                    "_betas": betas
                })
        
        self.save_files(all_respondents)
        return all_respondents

    def save_files(self, respondents):
        output = []
        for r in respondents:
            clean_r = {k: v for k, v in r.items() if k != "_betas"}
            output.append(clean_r)
        
        filename = f"epl_survey_{self.tier_name}.json"
        with open(os.path.join(self.data_dir, filename), "w") as f:
            json.dump(output, f, indent=2)

class BigSixSyntheticGenerator(BaseSyntheticGenerator):
    def __init__(self):
        price_points = {
            "Lower Tier": [45, 65, 85, 105, 125],
            "Upper Tier": [35, 50, 65, 80, 95],
            "Hospitality": [150, 200, 250, 300, 350]
        }
        super().__init__("big_six", BIG_SIX_SEGMENTS, price_points)

class MidTableSyntheticGenerator(BaseSyntheticGenerator):
    def __init__(self):
        price_points = {
            "Lower Tier": [25, 35, 45, 55, 65],
            "Upper Tier": [18, 25, 32, 40, 48],
            "Hospitality": [80, 110, 140, 170, 200]
        }
        super().__init__("mid", MID_TABLE_SEGMENTS, price_points)

class SmallClubSyntheticGenerator(BaseSyntheticGenerator):
    def __init__(self):
        price_points = {
            "Lower Tier": [20, 27, 34, 41, 48],
            "Upper Tier": [14, 20, 26, 32, 38],
            "Hospitality": [60, 80, 100, 120, 140]
        }
        super().__init__("small", SMALL_CLUB_SEGMENTS, price_points)

if __name__ == "__main__":
    for gen_class in [BigSixSyntheticGenerator, MidTableSyntheticGenerator, SmallClubSyntheticGenerator]:
        gen = gen_class()
        resps = gen.generate_data()
        c_rate = sum(1 for r in resps if r["consistency_flag"]) / len(resps)
        choices = [resp["option_chosen"] for r in resps for resp in r["responses"]]
        n_rate = choices.count("neither") / len(choices)
        print(f"Tier: {gen.tier_name}, Consistency Rate: {c_rate:.2%}, Neither Rate: {n_rate:.2%}")
