from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import pandas as pd
from EPL.stakes_engine import StakesEngine
from EPL.lp_optimizer_epl import EPLLPOptimizer
from EPL.inventory_manager import InventoryManager

app = FastAPI(title="EPL Ticket Price Optimization API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = "EPL/data"
stakes_engine = StakesEngine()
optimizer = EPLLPOptimizer(data_dir=DATA_DIR)
inventory_manager = InventoryManager()

@app.get("/api/epl/health")
def health():
    return {"status": "healthy", "engine": "EPL"}

@app.get("/api/epl/matches")
def get_matches():
    path = os.path.join(DATA_DIR, "epl_match_data.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="EPL match data not found. Run generator first.")
    with open(path, "r") as f:
        return json.load(f)

@app.get("/api/epl/clubs")
def get_clubs():
    from EPL.match_data_generator_epl import CLUBS
    return CLUBS

@app.post("/api/epl/optimize/{match_id}")
def optimize_match(match_id: str):
    # Load match
    path = os.path.join(DATA_DIR, "epl_match_data.json")
    with open(path, "r") as f:
        matches = json.load(f)
    
    match = next((m for m in matches if m["match_id"] == match_id), None)
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
        
    # Get club tier
    from EPL.match_data_generator_epl import CLUBS
    home_club = next(c for c in CLUBS if c["club_id"] == match["home_club_id"])
    
    results = optimizer.optimize(match, home_club["tier"])
    return results

@app.get("/api/epl/wtp/{tier}")
def get_wtp(tier: str):
    filename = f"epl_wtp_{tier}.json"
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"WTP data for {tier} not found.")
    with open(path, "r") as f:
        return json.load(f)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
