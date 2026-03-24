import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def test_system():
    # 1. Conjoint Analysis
    print("\n=== [1/4] CONJOINT ANALYSIS TEST ===")
    r_conj = requests.get(f"{BASE_URL}/api/conjoint/results")
    if r_conj.status_code == 200:
        data = r_conj.json()
        print(f"Status: SUCCESS")
        print(f"Price Bounds (Standing): {data['zone_price_bounds']['Standing']}")
        # Flat structure check
        wtp_elite = data['attribute_wtp'].get('opponent_elite', {}).get('mean', 0)
        print(f"Elite Opponent WTP: {wtp_elite:.2f} CHF")
        print(f"Segments Found: {list(data['segment_summary'].keys())}")
    else:
        print(f"Status: FAILED ({r_conj.status_code})")

    # 2. Demand Forecasting
    print("\n=== [2/4] DEMAND FORECASTING TEST ===")
    r_fore = requests.get(f"{BASE_URL}/api/forecasting/evaluation")
    if r_fore.status_code == 200:
        data = r_fore.json()
        print(f"Status: SUCCESS")
        print(f"Overall MAPE: {data['overall_mape']:.2%}")
        print(f"Top Driver: {data['feature_importance'][0]['feature']}")
    else:
        print(f"Status: FAILED ({r_fore.status_code})")

    # 3. Price Optimization (Standard Match)
    print("\n=== [3/4] PRICE OPTIMIZATION TEST ===")
    payload = {
        "match_id": "TEST_MATCH_FINAL",
        "match_features": {
            "opponent_tier_encoded": 2, # Elite
            "rival_match": 1,
            "match_stakes_encoded": 1, # Playoff
            "velocity_T14": 1.0, 
            "price_delta_secondary_chf": 0,
            "weather_severity_score": 0,
            "home_form_score": 0.6,
            "away_form_score": 0.5,
            "star_power_index": 1.2
        },
        "zone_capacities": {
            "Standing": 1200, "Upper Standard": 1500, 
            "Lower Bowl / Club Seats": 1000, "Courtside VIP": 300
        },
        "total_capacity": 4000,
        "current_prices": {
            "Standing": 18, "Upper Standard": 32, 
            "Lower Bowl / Club Seats": 58, "Courtside VIP": 85
        }
    }
    r_opt = requests.post(f"{BASE_URL}/api/optimize/match", json=payload)
    if r_opt.status_code == 200:
        data = r_opt.json()["pricing_recommendation"]
        print(f"Status: SUCCESS")
        print(f"Total Expected Revenue: {data['total_expected_revenue_chf']:.2f} CHF")
        print(f"Revenue Uplift: {data['total_revenue_uplift_pct']:.2%}")
        print(f"Standing Price: {data['zone_recommendations']['Standing']['recommended_price_chf']} CHF")
    else:
        print(f"Status: FAILED ({r_opt.status_code})")

    # 4. Live Signal Simulator (Surge Pricing)
    print("\n=== [4/4] LIVE SIGNAL SIMULATOR TEST ===")
    payload["match_features"]["velocity_T14"] = 2.5 # Violent Surge
    payload["match_features"]["price_delta_secondary_chf"] = 30.0 # High demand
    
    r_sim = requests.post(f"{BASE_URL}/api/optimize/match", json=payload)
    if r_sim.status_code == 200:
        data = r_sim.json()["pricing_recommendation"]
        print(f"Status: SUCCESS (Demand Surge Simulated)")
        print(f"Total Expected Revenue: {data['total_expected_revenue_chf']:.2f} CHF")
        print(f"Revenue Uplift: {data['total_revenue_uplift_pct']:.2%}")
        print(f"Standing Price: {data['zone_recommendations']['Standing']['recommended_price_chf']} CHF")
    else:
        print(f"Status: FAILED ({r_sim.status_code})")

if __name__ == "__main__":
    test_system()
