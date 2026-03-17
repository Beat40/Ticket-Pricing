from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import os
from backend.synthetic_data import SyntheticDataGenerator

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
