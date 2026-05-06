from lp_optimizer_epl import EPLLPOptimizer
import json

def test_lp():
    optimizer = EPLLPOptimizer()
    
    # Mock Big Six Match
    match_big = {
        "match_id": "ARS_MCI_2024",
        "overall_fill_rate": 0.98,
        "inventory": {
            "Dynamic Lower": {"capacity": 1000},
            "Dynamic Upper": {"capacity": 600},
            "Dynamic General": {"capacity": 200},
            "Away End": {"capacity": 3000},
            "Hospitality": {"capacity": 200}
        }
    }
    
    res_big = optimizer.optimize(match_big, "big_six")
    print("Big Six Optimization Result:")
    print(json.dumps(res_big, indent=2))
    assert res_big["prices"]["Away End"] == 30.0
    
    # Mock Small Club Match (Low demand)
    match_small = {
        "match_id": "BOU_IPS_2024",
        "overall_fill_rate": 0.75,
        "inventory": {
            "Dynamic Lower": {"capacity": 2000},
            "Dynamic Upper": {"capacity": 1200},
            "Dynamic General": {"capacity": 400},
            "Away End": {"capacity": 3000},
            "Hospitality": {"capacity": 100}
        }
    }
    
    res_small = optimizer.optimize(match_small, "small")
    print("\nSmall Club Optimization Result:")
    print(json.dumps(res_small, indent=2))
    assert res_small["prices"]["Away End"] == 30.0

if __name__ == "__main__":
    test_lp()
