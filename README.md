# 🏐 SHV Ticket Pricing Optimization System

An AI-driven revenue management platform for the Swiss Handball Federation, combining Bayesian Conjoint, Hybrid Demand Forecasting, and Linear Programming (LP) Optimization.

---

## 🛠️ Environment Setup (using `venv`)

If your system does not have Anaconda, follow these steps to create a clean Python virtual environment.

### **1. Clone and Navigate**
```bash
git clone <your-repo-url>
cd price_optimization
```

### **2. Create & Activate Virtual Environment**
*   **Windows**:
    ```powershell
    python -m venv venv
    .\venv\Scripts\activate
    ```
*   **Mac / Linux**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

### **3. Install Dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Running the System

### **Step 1: Start the Backend (FastAPI)**
In a dedicated terminal (with `venv` activated):
```bash
uvicorn backend.main:app --reload
```
The API will be available at `http://localhost:8000`.

### **Step 2: Start the Dashboard (Streamlit)**
In a **new** terminal (with `venv` activated):
```bash
streamlit run frontend/app.py
```
The Dashboard will be live at `http://localhost:8501`.

---

## 📂 Project Structure
*   `backend/`: FastAPI application, Conjoint Engine, Forecasting Engine, and LP Optimizer.
*   `frontend/`: Streamlit dashboard and Plotly visualizations.
*   `data/`: Persistent storage for match data and model artifacts.
*   `solution_overview.md`: Master documentation for Clients, Managers, and Developers.

---

**System Status**: 🟢 Fully Operational | **Last Verification**: March 24, 2026
