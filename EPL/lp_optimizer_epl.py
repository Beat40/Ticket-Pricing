import pulp
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime

class EPLLPOptimizer:
    def __init__(self, data_dir="EPL/data"):
        self.data_dir = data_dir
        # Load conjoint price bounds
        self.wtp_data = {
            "big_six": self._load_json(f"epl_wtp_big_six.json"),
            "mid": self._load_json(f"epl_wtp_mid.json"),
            "small": self._load_json(f"epl_wtp_small.json")
        }

    def _load_json(self, filename):
        path = os.path.join(self.data_dir, filename)
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return {}

    def optimize(self, match_record: dict, club_tier: str) -> dict:
        """
        Main entry point for EPL optimization.
        Uses tier-based routing.
        """
        # 1. Prepare Inventory
        inventory = match_record["inventory"]
        dynamic_zones = ["Dynamic Lower", "Dynamic Upper", "Dynamic General"]
        
        # 2. Prep Bounds from Conjoint
        wtp = self.wtp_data.get(club_tier, {})
        bounds = wtp.get("zone_price_bounds", {})
        
        # 3. Formulate and Solve
        prob = pulp.LpProblem(f"EPL_Pricing_{match_record['match_id']}", pulp.LpMaximize)
        
        # Decision variables: binary for each price point in each zone
        price_points = {}
        for zone in dynamic_zones:
            # Map dynamic zone name to conjoint zone name
            c_zone = zone.replace("Dynamic ", "")
            if c_zone == "General": 
                c_zone = "Upper Tier"
            else:
                c_zone = c_zone + " Tier"
            
            z_bounds = bounds.get(c_zone, {"floor": 20, "ceiling": 100})
            pts = np.linspace(z_bounds["floor"], z_bounds["ceiling"], 10)
            price_points[zone] = pts
            
        # Also dynamic hospitality if available
        if "Hospitality" in inventory:
            z_bounds = bounds.get("Hospitality", {"floor": 100, "ceiling": 400})
            price_points["Hospitality"] = np.linspace(z_bounds["floor"], z_bounds["ceiling"], 10)
            dynamic_zones.append("Hospitality")

        vars = {}
        for zone in dynamic_zones:
            vars[zone] = pulp.LpVariable.dicts(f"x_{zone.replace(' ', '_')}", range(10), cat=pulp.LpBinary)
            # Exactly one price point per zone
            prob += pulp.lpSum([vars[zone][i] for i in range(10)]) == 1

        # Elasticity logic (Section 8)
        elasticities = {
            "Hospitality": -0.10,
            "Dynamic Lower": -0.55,
            "Dynamic Upper": -0.75,
            "Dynamic General": -1.10
        }
        
        # Objective: Revenue from dynamic zones
        total_revenue = 0
        total_attendance = 0
        
        for zone in dynamic_zones:
            cap = inventory[zone]["capacity"]
            # Base demand from forecasting engine (match_record['overall_fill_rate'])
            # But here we use a simplified linear demand curve for the LP points
            base_q = cap * match_record["overall_fill_rate"]
            median_p = (price_points[zone][0] + price_points[zone][-1]) / 2
            
            for i in range(10):
                p = price_points[zone][i]
                # Q = Q_base * (1 + epsilon * (P - P_median)/P_median)
                q = base_q * (1 + elasticities.get(zone, -0.5) * (p - median_p) / median_p)
                q = pulp.lpSum([q]) # Ensure it's a linear expression if needed
                
                total_revenue += vars[zone][i] * p * q
                total_attendance += vars[zone][i] * q

        prob += total_revenue

        # CONSTRAINTS
        
        # 1. Tier Routing (Section 8)
        if club_tier == "big_six":
            # No attendance floor, purely revenue maximisation
            pass 
        else:
            # Attendance floor constraint (65% of dynamic inventory)
            prob += total_attendance >= 0.65 * sum(inventory[z]["capacity"] for z in dynamic_zones)

        # 2. Monotonicity: Hospitality > Lower > Upper > General
        # (Simplified: Hospitality > Lower Tier > Upper Tier)
        z_order = ["Hospitality", "Dynamic Lower", "Dynamic Upper", "Dynamic General"]
        actual_zones = [z for z in z_order if z in dynamic_zones]
        for i in range(len(actual_zones) - 1):
            z_hi = actual_zones[i]
            z_lo = actual_zones[i+1]
            p_hi = pulp.lpSum([vars[z_hi][j] * price_points[z_hi][j] for j in range(10)])
            p_lo = pulp.lpSum([vars[z_lo][j] * price_points[z_lo][j] for j in range(10)])
            prob += p_hi >= p_lo

        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # Extract Results
        final_prices = {}
        for zone in dynamic_zones:
            for i in range(10):
                if pulp.value(vars[zone][i]) == 1:
                    final_prices[zone] = float(price_points[zone][i])

        # Add fixed prices
        final_prices["Away End"] = 30.0
        
        return {
            "match_id": match_record["match_id"],
            "prices": final_prices,
            "status": pulp.LpStatus[prob.status],
            "total_dynamic_revenue": pulp.value(total_revenue),
            "total_dynamic_attendance": pulp.value(total_attendance)
        }
