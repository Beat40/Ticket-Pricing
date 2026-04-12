from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from backend.synthetic_data import SyntheticDataGenerator
from backend.conjoint_engine import ConjointEngine
from backend.match_data_generator import MatchDataGenerator
from backend.forecasting_engine import ForecastingEngine
from backend.lp_optimizer import LPOptimizer

app = FastAPI(title="Price Optimization API")

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = os.path.join(os.getcwd(), "data")
GROUND_TRUTH_PATH = os.path.join(DATA_DIR, "ground_truth_utilities.json")

@app.post("/api/data/generate")
async def generate_data():
    """
    Runs the full synthetic data generation pipeline.
    Overwrites survey_responses.json and individual_utilities.json.
    """
    try:
        generator = SyntheticDataGenerator()
        report = generator.generate_data()
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/data/ground-truth")
async def get_ground_truth():
    """
    Returns the contents of ground_truth_utilities.json.
    """
    if not os.path.exists(GROUND_TRUTH_PATH):
        raise HTTPException(status_code=404, detail="Ground truth data not found. Please generate data first.")
    
    try:
        with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/conjoint/run")
async def run_conjoint():
    """
    Instantiates ConjointEngine and runs the full analysis pipeline.
    """
    try:
        engine = ConjointEngine(data_dir=DATA_DIR)
        result = await engine.run()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/conjoint/results")
async def get_conjoint_results():
    """
    Returns combined wtp_results.json + fan_segments.json + validation_report.json.
    """
    wtp_path = os.path.join(DATA_DIR, "wtp_results.json")
    segments_path = os.path.join(DATA_DIR, "fan_segments.json")
    val_path = os.path.join(DATA_DIR, "validation_report.json")
    
    if not all(os.path.exists(p) for p in [wtp_path, segments_path, val_path]):
        raise HTTPException(status_code=404, detail="Run analysis first")
        
    try:
        results = {}
        with open(wtp_path, "r") as f: results.update(json.load(f))
        with open(segments_path, "r") as f: results.update(json.load(f))
        with open(val_path, "r") as f: results["validation"] = json.load(f)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/conjoint/price-bounds")
async def get_price_bounds():
    """
    Returns only zone_price_bounds from wtp_results.json.
    """
    wtp_path = os.path.join(DATA_DIR, "wtp_results.json")
    if not os.path.exists(wtp_path):
        raise HTTPException(status_code=404, detail="wtp_results.json missing")
    
    try:
        with open(wtp_path, "r") as f:
            data = json.load(f)
        return data.get("zone_price_bounds")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/conjoint/diagnostics")
async def get_diagnostics():
    """
    Returns hb_diagnostics.json.
    """
    diag_path = os.path.join(DATA_DIR, "hb_diagnostics.json")
    if not os.path.exists(diag_path):
        raise HTTPException(status_code=404, detail="hb_diagnostics.json missing")
    
    try:
        with open(diag_path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/conjoint/individual-utilities")
async def get_estimated_utilities():
    """
    Returns estimated_individual_utilities.json.
    """
    path = os.path.join(DATA_DIR, "estimated_individual_utilities.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="estimated_individual_utilities.json missing")
    
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/matches/generate")
async def generate_matches():
    """
    Runs MatchDataGenerator().generate()
    Overwrites /data/match_data.json and /data/match_data.csv
    Returns validation report
    """
    try:
        generator = MatchDataGenerator(data_dir=DATA_DIR)
        report = generator.generate()
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/matches/data")
async def get_matches_data():
    """
    Returns full match_data.json
    """
    path = os.path.join(DATA_DIR, "match_data.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Match data not found. Please generate first.")
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/matches/validation")
async def get_validation_matches():
    """
    Returns matches from Season 3 with merged features for optimization analysis.
    """
    path = os.path.join(DATA_DIR, "match_data.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Match data not found.")
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            matches = json.load(f)
            
        # Load all supplemental features
        def load_json(filename):
            p = os.path.join(DATA_DIR, filename)
            if os.path.exists(p):
                with open(p, "r") as f: return json.load(f)
            return {}

        stl = load_json("stl_features.json")
        sarima = load_json("sarima_features.json")
        np_feat = load_json("neural_prophet_features.json")
        arch_full = load_json("archetype_results.json")
        arch = arch_full.get("match_archetype_assignments", {})
        medoids = arch_full.get("medoid_curves", {})

        validation_matches = []
        
        # Initialize engine to get LightGBM predictions
        engine = ForecastingEngine(data_dir=DATA_DIR)
        engine.load_models()

        opp_map = {"Elite": 2, "Competitive": 1, "Standard": 0}
        stakes_map = {"Final": 2, "Playoff": 1, "Group": 0}

        for m in matches:
            if m["season"] == "2023-24":
                m_id = m["match_id"]
                # Merge base features
                m["stl_trend_value"] = stl.get(m_id, {}).get("stl_trend_value", 0)
                m["stl_seasonal_value"] = stl.get(m_id, {}).get("stl_seasonal_value", 0)
                m["sarima_residual"] = sarima.get(m_id, 0)
                
                np_data = np_feat.get(m_id, {})
                m["np_final_prediction"] = np_data.get("np_final_prediction", 0)
                m["np_deviation_T14"] = np_data.get("np_deviation_T14", 0)
                
                arch_data = arch.get(m_id, {})
                m["archetype"] = arch_data.get("archetype", "Consistent Gradual")
                m["archetype_deviation_T14"] = arch_data.get("archetype_deviation_T14", 0)
                
                # --- CALCULATE LIGHTGBM PREDICTION ---
                # Construct features for engine.predict()
                m_feat = {
                    "opponent_tier_encoded": opp_map.get(m["opponent_tier"], 1),
                    "rival_match": int(m["rival_match"]),
                    "home_form_score": float(m["home_form_score"]),
                    "away_form_score": float(m["away_form_score"]),
                    "star_power_index": float(m["star_power_index"]),
                    "match_stakes_encoded": stakes_map.get(m["match_stakes"], 0),
                    "qualification_stakes_score": int(m["qualification_stakes_score"]),
                    "weather_severity_score": int(m["weather_severity_score"]),
                    "marketing_activation_score": float(m["marketing_activation_score"]),
                    "velocity_T14": float(m["velocity_T14"]),
                    "price_delta_secondary_chf": float(m["price_delta_secondary_chf"]),
                    "kickoff_type_encoded": 1, 
                    "attribute_wtp_score": float(m["attribute_wtp_score"]),
                    "dominant_segment_encoded": 2,
                    "home_club_id": m["home_club_id"],
                    # Injected L1 signals
                    "stl_trend_value": m["stl_trend_value"],
                    "stl_seasonal_value": m["stl_seasonal_value"],
                    "sarima_residual": m["sarima_residual"],
                    "np_final_prediction": m["np_final_prediction"],
                    "np_deviation_T14": m["np_deviation_T14"],
                    "archetype_deviation_T14": m["archetype_deviation_T14"]
                }
                
                # Run prediction
                pred_res = engine.predict(m_feat)
                # Weighted average fill rate
                total_tickets_pred = 0
                total_capacity = 0
                for z, z_res in pred_res["zones"].items():
                    z_cap = m["zone_capacities"].get(z, 500)
                    total_tickets_pred += (z_res["p50_fill_rate"] * z_cap)
                    total_capacity += z_cap
                
                m["lgbm_prediction"] = total_tickets_pred / total_capacity if total_capacity > 0 else 0
                m["actual_outcome"] = m.get("overall_fill_rate", 0)
                
                validation_matches.append(m)
        
        return {
            "matches": validation_matches,
            "medoid_curves": medoids
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/matches/summary")
async def get_matches_summary():
    """
    Returns validation report stats only
    """
    try:
        generator = MatchDataGenerator(data_dir=DATA_DIR)
        # Instead of re-generating, we reconstruct the report from existing data if possible
        path = os.path.join(DATA_DIR, "match_data.json")
        if not os.path.exists(path):
             raise HTTPException(status_code=404, detail="Match data not found.")
        
        with open(path, "r", encoding="utf-8") as f:
            matches = json.load(f)
            generator.matches = matches
            return generator.get_validation_report()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/matches/{match_id}")
async def get_match_detail(match_id: str):
    """
    Returns single match record including full booking_curve array
    """
    path = os.path.join(DATA_DIR, "match_data.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Match data not found.")
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            matches = json.load(f)
            match = next((m for m in matches if m["match_id"] == match_id), None)
            if not match:
                raise HTTPException(status_code=404, detail="Match not found")
            return match
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/forecasting/train")
async def train_forecasting():
    """
    Runs ForecastingEngine().train()
    Returns evaluation metrics
    """
    try:
        engine = ForecastingEngine(data_dir=DATA_DIR)
        # Run training in a thread pool as it takes ~5-10 mins
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            metrics = await loop.run_in_executor(pool, lambda: asyncio.run(engine.train()))
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/forecasting/evaluation")
async def get_forecasting_evaluation():
    """
    Returns forecasting_evaluation.json
    """
    path = os.path.join(DATA_DIR, "forecasting_evaluation.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Forecasting model not trained yet.")
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/forecasting/predict")
async def predict_demand(match_features: dict):
    """
    Accepts match_features dict and returns P10/P50/P90 demand prediction
    """
    try:
        engine = ForecastingEngine(data_dir=DATA_DIR)
        engine.load_models()
        prediction = engine.predict(match_features)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/forecasting/feature-importance")
async def get_feature_importance():
    """
    Returns top 10 features by SHAP importance from evaluation report
    """
    path = os.path.join(DATA_DIR, "forecasting_evaluation.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Evaluation data missing")
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("feature_importance", [])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/forecasting/archetypes")
async def get_booking_archetypes():
    """
    Returns archetype cluster sizes and medoid curves
    """
    path = os.path.join(DATA_DIR, "archetype_results.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Archetype data missing")
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "cluster_sizes": data.get("cluster_sizes"),
            "medoid_curves": data.get("medoid_curves")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/optimize/match")
async def optimize_match(request: dict):
    """
    Unified Pipeline: 
    1. ForecastingEngine.predict(match_features) -> base fill rates
    2. LPOptimizer.optimize(fill_rates) -> recommendations
    """
    try:
        match_id = request.get("match_id", "MATCH_UNKNOWN")
        match_features = request.get("match_features", {})
        zone_capacities = request.get("zone_capacities")
        total_capacity = request.get("total_capacity")
        current_prices = request.get("current_prices")
        
        if not all([zone_capacities, total_capacity, current_prices]):
            raise HTTPException(status_code=400, detail="Missing capacities or current prices")

        # 1. Forecast
        engine = ForecastingEngine(data_dir=DATA_DIR)
        engine.load_models()
        # predict() will pop zone_capacities, but we need it for LP too
        # So we pass a copy
        forecast = engine.predict(match_features.copy())
        
        # Extract P50 fill rates per zone
        demand_preds = {z: forecast["zones"][z]["p50_fill_rate"] for z in forecast["zones"]}
        
        # 2. Optimize
        optimizer = LPOptimizer(data_dir=DATA_DIR)
        recommendation = optimizer.optimize(
            match_id=match_id,
            zone_capacities=zone_capacities,
            total_capacity=total_capacity,
            current_prices=current_prices,
            demand_model_predictions=demand_preds
        )
        
        return {
            "match_id": match_id,
            "demand_forecast": forecast["zones"],
            "pricing_recommendation": recommendation,
            "pipeline": "ForecastingEngine -> LPOptimizer"
        }
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/optimize/standalone")
async def optimize_standalone(request: dict):
    """Runs LP only using a provided base_fill_rate."""
    try:
        optimizer = LPOptimizer(data_dir=DATA_DIR)
        base_fill = request.get("base_fill_rate", 0.5)
        # Convert single float to dict for zones
        demand_preds = {z: base_fill for z in ["Standing", "Upper Standard", "Lower Bowl / Club Seats", "Courtside VIP"]}
        
        recommendation = optimizer.optimize(
            match_id=request.get("match_id", "STANDALONE_TEST"),
            zone_capacities=request.get("zone_capacities"),
            total_capacity=request.get("total_capacity"),
            current_prices=request.get("current_prices"),
            demand_model_predictions=demand_preds
        )
        return recommendation
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/optimize/price-bounds")
async def get_lp_bounds():
    """Returns price bounds used by the LP optimizer."""
    optimizer = LPOptimizer(data_dir=DATA_DIR)
    return optimizer.price_bounds

@app.get("/api/optimize/elasticities")
async def get_elasticities():
    """Returns zone elasticities used for demand curve construction."""
    from backend.lp_optimizer import ZONE_ELASTICITIES
    return {
        **ZONE_ELASTICITIES,
        "source": "H5 hypothesis test — log-log OLS per zone"
    }

@app.post("/api/optimize/batch")
async def optimize_batch(request: dict):
    """Optimizes multiple matches (mocked for POC)."""
    match_ids = request.get("match_ids", [])
    results = []
    # In a real batch, we'd fetch features for each ID from DB/JSON
    # For now, we'll just return a count
    return {
        "total_matches_optimized": len(match_ids),
        "status": "Nightly batch simulation completed.",
        "results": ["... recommendations would be here ..."]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
