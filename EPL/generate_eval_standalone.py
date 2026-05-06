import os
import json
import joblib
import pandas as pd
import numpy as np

def generate_evaluation_json():
    data_dir = "EPL/data"
    match_data_path = os.path.join(data_dir, "epl_match_data.json")
    models_dir = os.path.join(data_dir, "models")
    
    with open(match_data_path, "r") as f:
        matches = json.load(f)
    
    df = pd.DataFrame(matches)
    df["match_date"] = pd.to_datetime(df["match_date"])
    df = df.sort_values("match_date").reset_index(drop=True)
    
    # Replicate _prepare_layer2_data logic
    tier_map = {"smaller_club": 0, "established_mid": 1, "big_six": 2}
    stakes_map = {"Standard": 0, "European Spot": 1, "High Stakes": 2, "Relegation Six-Pointer": 3, "Top Four Decider": 4, "Title Decider": 5}
    
    data = []
    for i, row in df.iterrows():
        feat = {
            "opponent_tier_encoded": tier_map.get(row["away_tier"], 0),
            "is_derby": int(row["is_derby"]),
            "home_form_score": row["home_form_score"],
            "away_form_score": row["away_form_score"],
            "star_power_index": row["star_power_index"],
            "match_stakes_score": row["match_stakes_score"],
            "match_stakes_label_encoded": stakes_map.get(row["match_stakes_label"], 0),
            "home_position": row["home_position"],
            "away_position": row["away_position"],
            "games_remaining": row["games_remaining"],
            "weather_severity_score": row["weather_severity_score"],
            "tv_slot_encoded": row["tv_slot_encoded"],
            "festive_fixture": int(row["festive_fixture"]),
            "european_midweek_fatigue": int(row["european_midweek_fatigue"]),
            "away_distance_km": row["away_distance_km"],
            "manager_new_bonus": int(row["manager_new_bonus"]),
            "stl_trend": row.get("stl_trend", 0.0), # Use .get for safety
            "stl_seasonal": row.get("stl_seasonal", 0.0),
            "sarima_residual": row.get("sarima_residual", 0.0),
            "velocity_T14": row["velocity_T14"],
            "velocity_T7": row["velocity_T7"],
            "target": row["overall_fill_rate"],
            "season": row["season"],
            "home_tier": row["home_tier"] # For MAPE per tier
        }
        data.append(feat)
    
    df_all = pd.DataFrame(data)
    feature_cols = [c for c in df_all.columns if c not in ["target", "season", "home_tier"]]
    X = df_all[feature_cols]
    y = df_all["target"]
    
    # Load trained model
    models = joblib.load(os.path.join(models_dir, "epl_lgb.joblib"))
    model = models["total"]
    
    val_idx = df_all[df_all["season"] == "2023-24"].index
    X_val = X.iloc[val_idx]
    y_val = y.iloc[val_idx]
    
    y_pred = model.predict(X_val)
    
    # Compute MAPE per tier
    results = {
        "overall_mape": float(np.mean(np.abs((y_val - y_pred) / y_val))),
        "tier_mape": {},
        "feature_importance": []
    }
    
    for tier in ["big_six", "established_mid", "smaller_club"]:
        tier_idx = df_all[(df_all["season"] == "2023-24") & (df_all["home_tier"] == tier)].index
        if len(tier_idx) > 0:
            m = np.mean(np.abs((y_val.loc[tier_idx] - model.predict(X_val.loc[tier_idx])) / y_val.loc[tier_idx]))
            results["tier_mape"][tier] = float(m)
            
    # LightGBM native importance
    importance = model.feature_importance(importance_type='gain')
    feat_importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": importance
    }).sort_values("importance", ascending=False).head(10)
    
    for _, row in feat_importance.iterrows():
        results["feature_importance"].append({
            "feature": row["feature"],
            "importance": float(row["importance"])
        })
        
    with open("EPL/data/forecasting_evaluation.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Generated EPL/data/forecasting_evaluation.json")

if __name__ == "__main__":
    generate_evaluation_json()
