# SHV Pricing Intelligence Dashboard

Interactive multi-page dashboard for the SHV Ticket Price Optimization system.

## Features
- **🏠 Overview**: Unified system status and pipeline architecture.
- **📊 Conjoint Analysis**: Bayesian preference (HB-MNL) and WTP visualization.
- **📈 Demand Forecasting**: Multi-layer model metrics, SHAP drivers, and booking archetypes.
- **💰 Price Optimization**: Strategic match configuration and revenue-optimal pricing results.
- **🎛️ Live Signal Simulator**: Real-time sensitivity analysis for dynamic demand signals (Velocity, Secondary Market, Weather).

## How to Run

### 1. Start the Backend
```powershell
# Open a new terminal
conda activate shv-pricing
cd c:\Users\sengu\price_optimization
C:\Users\sengu\anaconda3\envs\shv-pricing\python.exe -m uvicorn backend.main:app --reload
```

### 2. Start the Dashboard
```powershell
# Open another terminal
conda activate shv-pricing
cd c:\Users\sengu\price_optimization\frontend
C:\Users\sengu\anaconda3\envs\shv-pricing\python.exe -m streamlit run app.py
```

### 3. Open in Browser
Visit [http://localhost:8501](http://localhost:8501)

## System Requirements
- Streamlit
- Plotly
- Requests
- Pandas
- Numpy
