# 🏐 SHV Ticket Pricing Optimization — Solution Overview

This document provides a comprehensive technical and strategic breakdown of the pricing intelligence system built for the **Swiss Handball Federation (SHV)**, focusing on readiness for the **Quickline Handball League** and **EHF EURO 2028**. It is intended as the definitive guide for Developers (Architecture), Managers (Product/Strategic), and Clients (Business Value).

---

## 1. Executive Summary 
The SHV Pricing System is an **AI-driven revenue engine** that replaces static, historical pricing with dynamic, fan-centric optimization.

### **Key Resilience Features**
*   **Feature Alignment Layer**: Automatically handles partial input from "Live Signals" by aligning them with the 23-feature training schema. It uses **Intelligent Fallbacks** (STL-derived averages) for missing time-series residuals.
*   **MinT Reconciliation**: A "Hierarchical Consistency" layer. It uses the **Minimum Trace (MinT)** algorithm to ensure that the sum of the 4 independent zone forecasts (Bottom-Up) always sums perfectly to the overall stadium forecast (Top-Down).
*   **Strategic Elasticity (H5)**: Maps the **Bayesian coefficients** directly to the LP price-demand curve, ensuring the optimization is rooted in the current fan appetite.
*   **Governance Override**: Hardcoded logic for the **Standing Zone** (min CHF 12.00) to protect the traditional fan base.
*   **Atmosphere Guardrail**: Built-in 60% fill-rate constraint ensures the arena stays vibrant for players and broadcasters.
*   **Live Resilience**: Dynamically adjusts prices based on real-time signals like "Booking Velocity" and "Secondary Market Premiums."

### **The Data Foundation: Synthetic Match Simulation**
**Purpose**: To provide high-fidelity training data that mirrors the nuances of real Swiss Handball.

*   **Mechanism**: A 270-match simulation across 3 seasons (2021-2024).
*   **Team Profiles**: Uses **Glicko-style Performance Ratings** to model team strength, rivalries (Derbys), and "League Tier" (Top vs. Bottom).
*   **Demand Vectors**: Synthesizes match-specific demand based on:
    *   **External Factors**: Weekend vs. Weekday, Weather severity, and Competition Stakes (Finals/Playoffs).
    *   **Booking Curves**: Generates 60-day **Sigmoid/Gompertz-style sales curves** that represent real ticket behavior in Switzerland.
    *   **Marketing Impact**: Simulates "Star Power" index and "Marketing Activation" scores on attendance.

---

## 2. Engine A: Bayesian Conjoint Analysis
**Purpose**: To recover the "True Value" of match attributes directly from fan trade-off behavior.

### **Methodology: HB-MNL (Hierarchical Bayes)**
Unlike simple surveys, the system uses a **Hierarchical Bayes Multinomial Logit (HB-MNL)** model estimated via the **NUTS (No-U-Turn Sampler)** algorithm.
*   **Hierarchical Layer**: Accounts for population-level averages while allowing individual-level utility (beta) variation.
*   **NUTS Sampler**: Highly efficient MCMC (Markov Chain Monte Carlo) sampling that ensures convergence even with complex, non-linear prior distributions.
*   **Individual Utilities**: Calculates a unique preference vector for every single respondent, capturing "Fan Heterogeneity."

### **Strategic Outputs**
*   **WTP (Willingness-to-Pay)**: Quantifies attributes in CHF (e.g., an "Elite Opponent" adds **CHF 42.84** to a fan's base utility).
*   **Price Anchors**: Establishes the **Floor (P10)**, **Median (P50)**, and **Ceiling (P90)** prices for all seating zones.
*   **Fan Segmentation**: Uses k-means clustering on the recovered utilities to identify 4 strategic personas:
    *   **Premium Seeker**: High sensitivity to comfort; resistant to VIP price hikes.
    *   **Value Loyalist**: High utility for SBB/Food bundles; highly price-sensitive.
    *   **Atmosphere Seeker**: High utility for rivalry/derby matches.
    *   **Occasional Neutral**: High WTP for "Finals" but low general interest.

---

## 3. Engine B: Demand Forecasting
**Purpose**: To predict exactly how many tickets will sell at any given price point.

### Architecture: The Multi-Step Hybrid Model
#### **Layer 1: Club-Specific Baseline (Sequential Logic)**
*   **Club-Isolated STL Decomposition**: Decomposes data into *Trend*, *Seasonality*, and *Remainder* per club, preventing signal leakage between different performance profiles.
*   **SARIMA (Residual Capture)**: Models the remaining variance (short-term momentum) for each club using validated auto-regressive parameters.
*   **DTW Archetype Clustering**: Identifies 4 "Booking Archetypes" (e.g., *Early Surgers* vs. *Late Surgers*). The UI now supports real-time highlight of these clusters.
*   **NeuralProphet (Temporal Signal)**: Captures sequential selling patterns within the 60-day window to provide a "Momentum Baseline."

#### **Layer 2: ML-Driven Synthesis (The Decision Layer)**
*   **LightGBM Regressor**: The "Final Decision" layer. It synthesizes all Layer 1 signals (STL Trends, SARIMA Residuals, NP Deviations) with real-world **Live Signals** (Booking Velocity, Weather, Secondary Mkt) to predict the final fill rate per zone.
*   **Model Accuracy**: Validated on Season 3 (2023-24) with a **MAPE of 5.2%**, significantly outperforming traditional moving-average heuristics.

---

## 4. Engine C: LP Price Optimizer
**Purpose**: To find the exact price per zone that maximizes total revenue while respecting all technical and social constraints.

### **The 8 Strategic Optimization Constraints**
The **PuLP** Linear Programming solver evaluates thousands of price combinations against these 8 non-negotiable rules:

1.  **Venue Capacity**: Total predicted tickets sold across all 4 zones MUST be $\leq$ Stadium capacity.
2.  **Zone Capacity**: Predicted sales in any specific zone (e.g., VIP) MUST be $\leq$ that zone's capacity.
3.  **The Atmosphere Guardrail (60% Floor)**: Total predicted tickets sold MUST be $\geq$ 60% of venue capacity. The AI is forbidden from maximizing revenue by selling only a few high-priced VIP tickets; it must ensure a "full house."
4.  **Price Monotonicity**: Prices MUST follow the quality hierarchy: $P_{VIP} \geq P_{Lower} \geq P_{Upper} \geq P_{Standing}$.
5.  **Governance Floor (Standing)**: A hardcoded floor of **CHF 12.00** as a "Social Contract" with the core fan base, regardless of high demand.
6.  **Conjoint Anchor (WTP Bounds)**: Recommended prices MUST stay between the **P10 (Floor)** and **P90 (Ceiling)** discovered during the Conjoint Analysis.
7.  **Symmetry Selection**: Exactly one price point must be selected per zone from the 10-point demand curve generated by Engine B.
8.  **Approval Escalation**: Automated risk assessment based on the delta from current prices:
    *   **$\Delta P < 20\%$**: Auto-Apply.
    *   **$20\% \leq \Delta P < 30\%$**: Manager Approval Required.
    *   **$\Delta P \geq 30\%$**: VP Approval Required.

---

## 5. Technical Architecture (For Developers)

### **Stack & Integration**
*   **Backend**: Python 3.11+, FastAPI (REST endpoints), Uvicorn.
*   **Database**: JSON-based persistent storage for match data, WTP results, and model artifacts.
*   **Optimization Layer**: **PuLP (Coin-OR CBC Solver)** using binary variable piecewise linear approximations.
*   **Hierarchical Reconciliation**: Uses the **MinT (Minimum Trace)** algorithm to ensure zone-level forecasts (bottom-up) always sum perfectly to the total venue forecast (top-down).

### **Frontend Implementation**
*   **Streamlit (v1.32+)**: Built as a specialized command center for Pricing Managers.
*   **Unified Price Optimization Panel**: Consolidates simulation and optimization into a single interface.
    *   **Manual Scenario**: For "What-if" planning of future matches.
    *   **Historical Validation**: For retrospective analysis of Season 3 matches (Actual vs. Optimal).
*   **Integrated Match Diagnostics**: Re-invigorated analysis loop featuring:
    *   **Archetype Contrast Curves**: Visualizes the selected match's sales trajectory against historical medoids.
    *   **Momentum Metrics**: Side-by-side comparison of **NeuralProphet Baseline**, **LightGBM AI Forecast**, and **Historical Truth**.
*   **Plotly Integration**: High-resolution WTP distributions and interactive Revenue/Yield curves.

---

**System Status**: 🟢 Fully Operational | **Last Documentation Update**: April 11, 2026
