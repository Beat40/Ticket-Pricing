import os
import json
import joblib
import pandas as pd
import numpy as np
from forecasting_engine_epl import EPLForecastingEngine

async def generate_evaluation_json():
    engine = EPLForecastingEngine()
    # Mock training results or run actual training
    # Since we need actual numbers, let's run the evaluate logic
    with open(engine.match_data_path, "r") as f:
        matches = json.load(f)
    
    df = pd.DataFrame(matches)
    df["match_date"] = pd.to_datetime(df["match_date"])
    df = df.sort_values("match_date").reset_index(drop=True)
    
    X, y, df_all = engine._prepare_layer2_data(df)
    
    # Load trained model
    models = joblib.load(os.path.join(engine.models_dir, "epl_lgb.joblib"))
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
        tier_idx = df_all[(df_all["season"] == "2023-24") & (df["home_tier"] == tier)].index
        if len(tier_idx) > 0:
            m = np.mean(np.abs((y.iloc[tier_idx] - model.predict(X.iloc[tier_idx])) / y.iloc[tier_idx]))
            results["tier_mape"][tier] = float(m)
            
    # LightGBM native importance
    importance = model.feature_importance(importance_type='gain')
    feat_importance = pd.DataFrame({
        "feature": X.columns,
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
    import asyncio
    asyncio.run(generate_evaluation_json())
