from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from backend.synthetic_data import SyntheticDataGenerator
from backend.conjoint_engine import ConjointEngine

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
