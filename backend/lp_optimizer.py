import json
import os
import numpy as np
import pulp

# Zone specific elasticities from H5 hypothesis test
ZONE_ELASTICITIES = {
    "Standing": -1.30,
    "Upper Standard": -0.90,
    "Lower Bowl / Club Seats": -0.60,
    "Courtside VIP": -0.20
}

# Base prices used during high-level training/sim
BASE_PRICES = {
    "Standing": 18.0,
    "Upper Standard": 32.0,
    "Lower Bowl / Club Seats": 58.0,
    "Courtside VIP": 85.0
}

# Fallback bounds if wtp_results.json is missing
FALLBACK_BOUNDS = {
    "Standing": {"floor": 12.0, "median": 22.0, "ceiling": 32.0},
    "Upper Standard": {"floor": 21.0, "median": 41.0, "ceiling": 61.0},
    "Lower Bowl / Club Seats": {"floor": 23.0, "median": 66.0, "ceiling": 109.0},
    "Courtside VIP": {"floor": 25.0, "median": 85.5, "ceiling": 146.0}
}

class LPOptimizer:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.results_path = os.path.join(data_dir, "wtp_results.json")
        self.last_opt_path = os.path.join(data_dir, "last_optimization.json")
        self.price_bounds = self._load_price_bounds()

    def _load_price_bounds(self):
        """Loads floor/median/ceiling from conjoint results or returns fallbacks."""
        if os.path.exists(self.results_path):
            try:
                with open(self.results_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Extract zone_price_bounds section
                    return data.get("zone_price_bounds", FALLBACK_BOUNDS)
            except:
                return FALLBACK_BOUNDS
        return FALLBACK_BOUNDS

    def optimize(self, 
                 match_id: str,
                 zone_capacities: dict,
                 total_capacity: int,
                 current_prices: dict,
                 demand_model_predictions: dict,
                 min_fill_rate: float = 0.60
                 ) -> dict:
        """
        Main entry point for optimization.
        demand_model_predictions: {zone: base_fill_rate_at_base_price}
        """
        # Step 1: Construct Demand Curves
        demand_curves = self._build_demand_curves(demand_model_predictions, zone_capacities)
        
        # Step 2: Solve LP
        solve_result = self._solve_lp(demand_curves, zone_capacities, total_capacity, current_prices, min_fill_rate)
        
        # If infeasible, relax and retry
        if solve_result["status"] != 1: # pulp.constants.LpStatusOptimal
            print(f"Match {match_id} infeasible at 60% fill. Relaxing to 45%...")
            solve_result = self._solve_lp(demand_curves, zone_capacities, total_capacity, current_prices, 0.45)
        
        if solve_result["status"] != 1:
            return {"match_id": match_id, "solver_status": "Infeasible", "error": "Solver failed even after relaxation."}

        # Step 3: Sensitivity Analysis
        sensitivity = self._sensitivity_analysis(demand_curves, solve_result["selected_indices"])
        
        # Step 4: Final Format
        output = self._format_output(solve_result, demand_curves, current_prices, match_id, sensitivity, total_capacity)
        
        # Save last result
        with open(self.last_opt_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
            
        return output

    def _build_demand_curves(self, demand_model_predictions, zone_capacities):
        """Builds 10 price points per zone using elasticity formula."""
        curves = {}
        for zone in ["Standing", "Upper Standard", "Lower Bowl / Club Seats", "Courtside VIP"]:
            bounds = self.price_bounds.get(zone, FALLBACK_BOUNDS[zone])
            base_fill = demand_model_predictions.get(zone, 0.5)
            epsilon = ZONE_ELASTICITIES[zone]
            base_price = BASE_PRICES[zone]
            capacity = zone_capacities[zone]
            
            price_points = np.linspace(bounds["floor"], bounds["ceiling"], 10)
            points = []
            for p in price_points:
                # adjusted_fill_rate(P) = base_fill_rate * (P / base_price) ^ epsilon
                adj_fill = base_fill * (p / base_price)**epsilon
                adj_fill = min(1.0, max(0.01, adj_fill)) # Clip realistically
                
                tickets = int(round(adj_fill * capacity))
                rev = p * tickets
                
                points.append({
                    "price": float(p),
                    "fill_rate": float(adj_fill),
                    "tickets_sold": tickets,
                    "revenue": float(rev)
                })
            curves[zone] = points
        return curves

    def _solve_lp(self, demand_curves, zone_capacities, total_capacity, current_prices, min_fill_rate):
        """Piecewise linear LP solve using PuLP."""
        prob = pulp.LpProblem("Revenue_Optimization", pulp.LpMaximize)
        
        # Variables: x[zone][k] is binary, 1 if price point k is selected
        vars = {}
        zones = list(demand_curves.keys())
        for z in zones:
            vars[z] = [pulp.LpVariable(f"x_{z.replace(' ', '_')}_{k}", cat=pulp.LpBinary) for k in range(10)]
            
        # Objective: Maximize total revenue
        prob += pulp.lpSum([vars[z][k] * demand_curves[z][k]["revenue"] for z in zones for k in range(10)])
        
        # Constraint 1: One price point per zone
        for z in zones:
            prob += pulp.lpSum([vars[z][k] for k in range(10)]) == 1
            
        # Constraint 2 & 3: Capacities
        # Venue
        prob += pulp.lpSum([vars[z][k] * demand_curves[z][k]["tickets_sold"] for z in zones for k in range(10)]) <= total_capacity
        # Per Zone (Implicitly handled by demand curve building, but added for safety)
        for z in zones:
            prob += pulp.lpSum([vars[z][k] * demand_curves[z][k]["tickets_sold"] for k in range(10)]) <= zone_capacities[z]
            
        # Constraint 4: Attendance floor
        prob += pulp.lpSum([vars[z][k] * demand_curves[z][k]["tickets_sold"] for z in zones for k in range(10)]) >= min_fill_rate * total_capacity
        
        # Constraint 5: Price Monotonicity
        # Standing <= Upper Standard <= Lower Bowl <= Courtside VIP
        ordered = ["Standing", "Upper Standard", "Lower Bowl / Club Seats", "Courtside VIP"]
        for i in range(len(ordered)-1):
            lo_z = ordered[i]
            hi_z = ordered[i+1]
            prob += pulp.lpSum([vars[lo_z][k] * demand_curves[lo_z][k]["price"] for k in range(10)]) <= \
                    pulp.lpSum([vars[hi_z][k] * demand_curves[hi_z][k]["price"] for k in range(10)])
                    
        # Solve
        status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        result = {"status": status, "selected_indices": {}}
        if status == 1: # Optimal
            for z in zones:
                for k in range(10):
                    if pulp.value(vars[z][k]) == 1:
                        result["selected_indices"][z] = k
        
        return result

    def _sensitivity_analysis(self, demand_curves, selected_indices):
        """Calculates revenue impact of moving ±1 step for each zone."""
        sensitivity = {}
        for z, k in selected_indices.items():
            opt_rev = demand_curves[z][k]["revenue"]
            
            up_delta = None
            if k < 9:
                up_delta = demand_curves[z][k+1]["revenue"] - opt_rev
                
            down_delta = None
            if k > 0:
                down_delta = demand_curves[z][k-1]["revenue"] - opt_rev
                
            sensitivity[z] = {
                "one_step_up_revenue_delta_chf": up_delta,
                "one_step_down_revenue_delta_chf": down_delta
            }
        return sensitivity

    def _format_output(self, solve_result, demand_curves, current_prices, match_id, sensitivity, total_capacity):
        """Prepares the final recommendation JSON."""
        indices = solve_result["selected_indices"]
        
        total_revenue = sum(demand_curves[z][k]["revenue"] for z, k in indices.items())
        total_tickets = sum(demand_curves[z][k]["tickets_sold"] for z, k in indices.items())
        
        # FIX 1: Baseline revenue calculation
        # Use the demand curve point closest to current price for each zone
        baseline_revenue = 0
        for z in indices:
            cur_p = current_prices.get(z, BASE_PRICES[z])
            closest = min(demand_curves[z], key=lambda pt: abs(pt["price"] - cur_p))
            baseline_revenue += closest["revenue"]
        
        recs = {}
        for z, k in indices.items():
            p_rec = demand_curves[z][k]["price"]
            p_cur = current_prices.get(z, BASE_PRICES[z])
            delta_pct = (p_rec - p_cur) / p_cur if p_cur > 0 else 0
            
            # Approval
            approval = "AUTO_APPLY"
            if abs(delta_pct) > 0.30: approval = "VP_REQUIRED"
            elif abs(delta_pct) > 0.20: approval = "MANAGER_REQUIRED"
            
            recs[z] = {
                "recommended_price_chf": round(p_rec, 2),
                "current_price_chf": round(p_cur, 2),
                "price_delta_pct": float(delta_pct),
                "expected_tickets_sold": demand_curves[z][k]["tickets_sold"],
                "expected_fill_rate": demand_curves[z][k]["fill_rate"],
                "expected_revenue_chf": round(demand_curves[z][k]["revenue"], 2),
                "approval_required": approval,
                "explanation": self._gen_explanation(z, p_rec, p_cur, delta_pct, 
                                                   demand_curves[z][k]["tickets_sold"],
                                                   demand_curves[z][k]["fill_rate"],
                                                   demand_curves[z][k]["revenue"],
                                                   ZONE_ELASTICITIES[z],
                                                   approval),
                "sensitivity": sensitivity[z]
            }
            
        return {
            "match_id": match_id,
            "solver_status": "Optimal",
            "total_expected_revenue_chf": round(total_revenue, 2),
            "total_expected_tickets": total_tickets,
            "total_expected_fill_rate": round(total_tickets / total_capacity, 3), # FIX 2
            "total_revenue_vs_baseline_chf": round(total_revenue - baseline_revenue, 2),
            "total_revenue_uplift_pct": float((total_revenue - baseline_revenue) / baseline_revenue) if baseline_revenue > 0 else 0,
            "zone_recommendations": recs,
            "demand_curves": {z: demand_curves[z] for z in demand_curves}
            # FIX 3: Removed constraints_active
        }

    def _gen_explanation(self, zone, p_rec, p_cur, delta_pct, tickets, fill, rev, epsilon, approval):
        return (
            f"{zone}: Recommended CHF {p_rec:.0f} "
            f"(current CHF {p_cur:.0f}, "
            f"delta {delta_pct:+.1%}). "
            f"Expected {tickets} tickets sold "
            f"({fill:.1%} fill), "
            f"revenue CHF {rev:,.0f}. "
            f"Price elasticity for this zone: {epsilon}. "
            f"Approval required: {approval}."
        )
