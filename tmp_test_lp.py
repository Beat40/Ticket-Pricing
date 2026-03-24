import json
from backend.lp_optimizer import LPOptimizer

def test_optimization():
    opt = LPOptimizer()
    match_id = "VERIFY_FINAL"
    zone_capacities = {
        "Standing": 1200, 
        "Upper Standard": 1500, 
        "Lower Bowl / Club Seats": 1000, 
        "Courtside VIP": 300
    }
    total_capacity = 4000
    current_prices = {
        "Standing": 18, 
        "Upper Standard": 32, 
        "Lower Bowl / Club Seats": 58, 
        "Courtside VIP": 85
    }
    demand_preds = {
        "Standing": 0.7, 
        "Upper Standard": 0.6, 
        "Lower Bowl / Club Seats": 0.5, 
        "Courtside VIP": 0.4
    }
    
    res = opt.optimize(match_id, zone_capacities, total_capacity, current_prices, demand_preds)
    
    print(f"Match: {res['match_id']}")
    print(f"Solver Status: {res['solver_status']}")
    print(f"Revenue: {res['total_expected_revenue_chf']}")
    print(f"Uplift: {res['total_revenue_uplift_pct']:.2%}")
    print(f"Fill Rate: {res['total_expected_fill_rate']}")
    
    # Check baseline revenue logic (Fix 1)
    # total_revenue_vs_baseline_chf = total_expected_revenue - baseline_rev
    baseline_rev = res['total_expected_revenue_chf'] - res['total_revenue_vs_baseline_chf']
    print(f"Estimated Baseline Rev: {baseline_rev:.2f}")

if __name__ == "__main__":
    test_optimization()
