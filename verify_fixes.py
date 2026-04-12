import asyncio
import json
import os
import pandas as pd
import numpy as np
from backend.forecasting_engine import ForecastingEngine

async def verify():
    engine = ForecastingEngine(data_dir="data")
    
    print("--- 1. Running Training Pipeline ---")
    # This will run STL, SARIMA, NeuralProphet, Archetype Clustering, and Layer 2
    await engine.train()
    
    print("\n--- 2. Verifying Temporal Lookup Persistence ---")
    latest_path = "data/latest_temporal_features.json"
    if os.path.exists(latest_path):
        with open(latest_path, "r") as f:
            lt = json.load(f)
        print(f"Latest temporal features saved for {len(lt)} clubs.")
        # Check BSV specifically
        if "BSV" in lt:
            print(f"BSV Latest STL Trend: {lt['BSV']['latest_stl_trend']}")
    else:
        print("ERROR: latest_temporal_features.json missing!")

    print("\n--- 3. Verifying Inference Injection ---")
    test_features = {
        "home_club_id": "BSV",
        "opponent_tier": "Elite",
        "match_date": "2024-05-01",
        "rival_match": False,
        "home_form_score": 0.5,
        "away_form_score": 0.5,
        "star_power_index": 1.0,
        "match_stakes": "Group",
        "qualification_stakes_score": 1,
        "weather_severity_score": 0,
        "competing_event_penalty": 0,
        "marketing_activation_score": 0.5,
        "is_school_holiday": 0,
        "kickoff_type": "Saturday evening",
        "attribute_wtp_score": 0.5,
        "dominant_segment": "Value Loyalist",
        "velocity_T14": 1.0,
        "velocity_T7": 1.0,
        "price_delta_secondary_chf": 0,
        "zone_capacities": {
            "Courtside VIP": 300,
            "Lower Bowl / Club Seats": 1000,
            "Upper Standard": 1500,
            "Standing": 1200
        }
    }
    
    # We'll run it and check if it crashes (it won't if columns match)
    prediction = engine.predict(test_features)
    print("Prediction successful. Coherent zone outputs:")
    for zone, val in prediction["zones"].items():
        print(f"  {zone}: {val['p50_fill_rate']:.2f} fill rate")

    print("\n--- 4. Verifying Leakage Fix (Correlation Check) ---")
    with open("data/match_data.json", "r") as f:
        matches = json.load(f)
    df = pd.DataFrame(matches)
    corr = df["velocity_T14"].corr(df["overall_fill_rate"])
    print(f"Correlation (velocity_T14 vs overall_fill_rate): {corr:.3f}")
    if 0.50 <= corr <= 0.70:
        print("SUCCESS: Correlation is in the healthy range (0.50 - 0.70).")
    else:
        print(f"WARNING: Correlation {corr:.3f} is outside the target range!")

if __name__ == "__main__":
    asyncio.run(verify())
