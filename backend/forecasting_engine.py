import os
import json
import joblib
import numpy as np
import pandas as pd
import asyncio
from datetime import datetime, timedelta
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from dtaidistance import dtw, clustering
from neuralprophet import NeuralProphet
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error
import shap

class ForecastingEngine:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.match_data_path = os.path.join(data_dir, "match_data.json")
        self.models_dir = os.path.join(data_dir, "models")
        self.lgb_models = {}
        self.quantile_models = {}
        self.feature_cols = []
        
        # Ensure directories exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # In-memory storage for models and metadata
        self.stl_results = None
        self.sarima_model = None
        self.archetype_results = None
        self.np_model = None
        self.lgb_models = {} # zone -> model
        self.quantile_models = {} # alpha -> model
        self.evaluation_results = None
        self.feature_importance = None

    async def train(self) -> dict:
        """Orchestrates the full two-layer training pipeline."""
        if not os.path.exists(self.match_data_path):
            raise FileNotFoundError("match_data.json missing. Generate data first.")

        with open(self.match_data_path, "r", encoding="utf-8") as f:
            matches = json.load(f)
        
        df = pd.DataFrame(matches)
        df["match_date"] = pd.to_datetime(df["match_date"])
        df = df.sort_values("match_date").reset_index(drop=True)

        print("Training Layer 1: Time Series components...")
        # 1a: STL
        self._run_stl(df)
        
        # 1b: SARIMA
        self._run_sarima(df)
        
        # 1c: Booking Curve Clustering (DTW)
        self._cluster_archetypes(matches)
        
        # 1d: Neural Prophet
        await self._train_neural_prophet(matches)
        
        print("Training Layer 2: Tabular ML models...")
        # Prepare features for Layer 2
        X, y, df_features = self._prepare_layer2_data(df)
        
        # 2a: LightGBM (Total and per zone)
        self._train_lightgbm(X, y, df_features)
        
        # 2b: Quantile Regression Forest
        self._train_quantile_models(X, y)
        
        # Evaluation & SHAP
        self._evaluate(X, y, df_features)
        
        self.save_models()
        return self.evaluation_results

    def _run_stl(self, df):
        """Layer 1a: STL Decomposition for trend and seasonality extraction."""
        series = df["overall_fill_rate"].values
        # STL with period=18 (full league cycle)
        # robust=True to handle COVID outliers in 2021-22
        res = STL(series, period=18, seasonal=7, robust=True).fit()
        
        # Anomaly detection: residual > 2 * std
        resid_std = np.std(res.resid)
        anomalies = np.abs(res.resid) > 2 * resid_std
        
        direction = "stable"
        if res.trend[-1] > res.trend[0] * 1.05: direction = "increasing"
        elif res.trend[-1] < res.trend[0] * 0.95: direction = "decreasing"

        self.stl_results = {
            "trend": res.trend.tolist(),
            "seasonal": res.seasonal.tolist(),
            "residual": res.resid.tolist(),
            "anomaly_flags": anomalies.tolist(),
            "n_anomalies": int(np.sum(anomalies)),
            "trend_direction": direction,
            "seasonal_amplitude": float(np.max(res.seasonal) - np.min(res.seasonal))
        }
        
        # Add components back to dataframe for following steps
        df["stl_trend_value"] = res.trend
        df["stl_seasonal_value"] = res.seasonal
        
        # FIX 1: Persist STL features for _prepare_layer2_data
        stl_features = {}
        for i, row in df.iterrows():
            stl_features[row["match_id"]] = {
                "stl_trend_value": float(res.trend[i]),
                "stl_seasonal_value": float(res.seasonal[i])
            }
        with open(os.path.join(self.data_dir, "stl_features.json"), "w") as f:
            json.dump(stl_features, f, indent=2)

        with open(os.path.join(self.data_dir, "stl_decomposition.json"), "w") as f:
            json.dump(self.stl_results, f, indent=2)

    def _run_sarima(self, df):
        """Layer 1b: SARIMAX for sequential baseline residuals."""
        series = df["overall_fill_rate"].values
        
        # Exogenous regressors
        opp_map = {"Elite": 2, "Competitive": 1, "Standard": 0}
        stakes_map = {"Final": 2, "Playoff": 1, "Group": 0}
        
        exog = pd.DataFrame({
            "opp": df["opponent_tier"].map(opp_map),
            "stakes": df["match_stakes"].map(stakes_map),
            "weather": df["weather_severity_score"],
            "marketing": df["marketing_activation_score"]
        })
        
        # Fit SARIMAX (1, 0, 1) x (1, 0, 1, 18)
        model = SARIMAX(series, exog=exog, order=(1,0,1), seasonal_order=(1,0,1,18))
        res = model.fit(disp=False)
        
        self.sarima_model = res
        
        # Extract in-sample residuals for LightGBM
        # We use simple subtraction if residuals attribute is tricky, but SARIMAX gives them
        sarima_resid = res.resid
        df["sarima_residual"] = sarima_resid

        # FIX 1: Persist SARIMA features for _prepare_layer2_data
        sarima_features = {}
        for i, row in df.iterrows():
            sarima_features[row["match_id"]] = float(sarima_resid[i])
        with open(os.path.join(self.data_dir, "sarima_features.json"), "w") as f:
            json.dump(sarima_features, f, indent=2)

        results = {
            "aic": float(res.aic),
            "bic": float(res.bic),
            "log_likelihood": float(res.llf),
            "coefficients": res.params.to_dict(),
            "in_sample_mape": float(mean_absolute_percentage_error(series, res.fittedvalues)),
            "residuals": sarima_resid.tolist()
        }
        
        with open(os.path.join(self.data_dir, "sarima_results.json"), "w") as f:
            json.dump(results, f, indent=2)

    def _cluster_archetypes(self, matches):
        """Layer 1c: DTW Clustering of normalized booking curves."""
        curves = []
        match_ids = []
        for m in matches:
            curve = np.array(m["booking_curve"])
            if curve[-1] > 0:
                curves.append(curve / curve[-1])
                match_ids.append(m["match_id"])
        
        curves_np = np.array(curves)
        
        # Clustering with scipy on DTW distance matrix
        print("Computing DTW distance matrix (Layer 1c)...")
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        
        # dtaidistance returns a condensed or full matrix?
        # dtw.distance_matrix_fast returns a full matrix.
        ds = dtw.distance_matrix_fast(curves_np)
        
        # Ensure symmetric and zero diagonal for squareform
        ds = (ds + ds.T) / 2
        np.fill_diagonal(ds, 0)
        
        condensed_ds = squareform(ds)
        linkage_matrix = linkage(condensed_ds, method="ward")
        cluster_assignments = fcluster(linkage_matrix, 4, criterion="maxclust")
        
        # cluster_assignments is 1-indexed (1 to 4)
        flat_assignments = {match_ids[i]: (int(cluster_assignments[i]) - 1) for i in range(len(match_ids))}
        
        # Group indices for medoid calculation
        assignments = [[] for _ in range(4)]
        for i, c_id in enumerate(cluster_assignments):
            assignments[int(c_id) - 1].append(i)
        
        # Label clusters by medoid shape
        medoids = {}
        labels = {}
        for c_id, idxs in enumerate(assignments):
            # Medoid is the one with minimum sum of distances to others in cluster
            sub_matrix = ds[np.ix_(idxs, idxs)]
            medoid_local_idx = np.argmin(np.sum(sub_matrix, axis=0))
            medoid_global_idx = idxs[medoid_local_idx]
            medoid_curve = curves_np[medoid_global_idx]
            
            # Label heuristic
            val_30 = medoid_curve[30]
            val_53 = medoid_curve[53]
            
            if val_30 >= 0.35: label = "Early Surge"
            elif val_30 <= 0.25 and val_53 >= 0.60: label = "Late Surge"
            elif val_53 < 0.55: label = "Flat"
            else: label = "Consistent Gradual"
            
            # Ensure unique labels if heuristics overlap
            if label in labels.values():
                label = f"{label}_{c_id}"
            
            labels[c_id] = label
            medoids[label] = medoid_curve.tolist()

        match_results = {}
        # Calculate deviations
        for m_id, c_id in flat_assignments.items():
            label = labels[c_id]
            # Find the match and its normalized curve
            m_idx = match_ids.index(m_id)
            actual_curve = curves_np[m_idx]
            medoid_curve = np.array(medoids[label])
            
            match_results[m_id] = {
                "archetype": label,
                "archetype_deviation_T14": float(actual_curve[46] - medoid_curve[46]),
                "archetype_deviation_T7": float(actual_curve[53] - medoid_curve[53])
            }

        # Calculate accuracy vs ground truth
        correct = 0
        total = 0
        for m in matches:
            if m["match_id"] in match_results:
                total += 1
                if match_results[m["match_id"]]["archetype"].split("_")[0] == m["booking_curve_archetype"]:
                    correct += 1
        
        self.archetype_results = {
            "n_clusters": 4,
            "cluster_sizes": {label: len(idxs) for label, idxs in zip(labels.values(), assignments)},
            "medoid_curves": medoids,
            "match_archetype_assignments": match_results,
            "recovery_accuracy": round(correct/total, 3) if total > 0 else 0
        }
        
        with open(os.path.join(self.data_dir, "archetype_results.json"), "w") as f:
            json.dump(self.archetype_results, f, indent=2)

    async def _train_neural_prophet(self, matches):
        """Layer 1d: Neural Prophet velocity forecaster."""
        # Long format conversion
        rows = []
        opp_map = {"Elite": 2, "Competitive": 1, "Standard": 0}
        stakes_map = {"Final": 2, "Playoff": 1, "Group": 0}

        for m in matches:
            curve = np.array(m["booking_curve"])
            f_rate = m["overall_fill_rate"]
            norm_curve = curve / curve[-1] if curve[-1] > 0 else curve
            
            # To create a chronological time series for NP, 
            # we assign a dummy date sequence starting from the match_date backwards.
            match_dt = datetime.strptime(m["match_date"], "%Y-%m-%d")
            for t in range(61):
                ds = match_dt - timedelta(days=(60 - t))
                rows.append({
                    "ds": ds,
                    "y": norm_curve[t],
                    "ID": m["match_id"],
                    "opponent_tier_encoded": opp_map.get(m["opponent_tier"], 0),
                    "match_stakes_encoded": stakes_map.get(m["match_stakes"], 0),
                    "star_power_index": m["star_power_index"],
                    "weather_severity_score": m["weather_severity_score"]
                })
        
        np_df = pd.DataFrame(rows)
        np_df["ds"] = pd.to_datetime(np_df["ds"])
        np_df["y"] = np_df["y"].astype(float)
        
        # Training (Global Model with IDs)
        m_np = NeuralProphet(
            n_forecasts=1,
            n_lags=7,
            yearly_seasonality=False,
            weekly_seasonality=True,
            learning_rate=0.01
        )
        
        try:
            m_np.fit(np_df[["ds", "y", "ID"]], freq="D")
            self.np_model = m_np
            print("NeuralProphet trained successfully.")
        except Exception:
            import traceback
            print(f"NeuralProphet training failed. Traceback:\n{traceback.format_exc()}")
            print("Using fallback logic.")
            self.np_model = None
        
        # Extract features for LightGBM
        np_features = {}
        for match in matches:
            m_id = match["match_id"]
            pred_final = match["velocity_T14"] # Fallback
            
            if self.np_model:
                try:
                    # Get T-14 slice (days 0 to 46)
                    match_subset = np_df[np_df["ID"] == m_id].iloc[:47].copy()
                    future = m_np.make_future_dataframe(match_subset, periods=1)
                    forecast = m_np.predict(future)
                    pred_final = float(forecast.iloc[-1]["yhat1"])
                except:
                    pass
            
            np_features[m_id] = {
                "np_final_prediction": pred_final,
                "np_deviation_T14": float(match["velocity_T14"] - pred_final)
            }
            
        with open(os.path.join(self.data_dir, "neural_prophet_features.json"), "w") as f:
            json.dump(np_features, f, indent=2)
            
    def _prepare_layer2_data(self, df):
        """Merges Layer 1 outputs into a feature matrix for LightGBM."""
        # Load extensions
        with open(os.path.join(self.data_dir, "archetype_results.json"), "r") as f:
            archetype_data = json.load(f)["match_archetype_assignments"]
        with open(os.path.join(self.data_dir, "neural_prophet_features.json"), "r") as f:
            np_data = json.load(f)
        
        # FIX 1: Load persisted STL and SARIMA features
        with open(os.path.join(self.data_dir, "stl_features.json"), "r") as f:
            stl_features = json.load(f)
        with open(os.path.join(self.data_dir, "sarima_features.json"), "r") as f:
            sarima_features = json.load(f)

        # Encodings
        opp_map = {"Elite": 2, "Competitive": 1, "Standard": 0}
        stakes_map = {"Final": 2, "Playoff": 1, "Group": 0}
        kickoff_map = {"Weekday evening": 0, "Saturday afternoon": 1, "Saturday evening": 2, "Sunday afternoon": 1}
        segment_map = {"Occasional Neutral": 0, "Atmosphere Seeker": 1, "Value Loyalist": 2, "Premium Seeker": 3}
        
        data = []
        for i, row in df.iterrows():
            m_id = row["match_id"]
            feat = {
                "opponent_tier_encoded": opp_map.get(row["opponent_tier"], 0),
                "rival_match": int(row["rival_match"]),
                "home_form_score": row["home_form_score"],
                "away_form_score": row["away_form_score"],
                "star_power_index": row["star_power_index"],
                "match_stakes_encoded": stakes_map.get(row["match_stakes"], 0),
                "qualification_stakes_score": row["qualification_stakes_score"],
                "weather_severity_score": row["weather_severity_score"],
                "competing_event_penalty": row["competing_event_penalty"],
                "marketing_activation_score": row["marketing_activation_score"],
                "is_school_holiday": int(row["is_school_holiday"]),
                "kickoff_type_encoded": kickoff_map.get(row["kickoff_type"], 2),
                "attribute_wtp_score": row["attribute_wtp_score"],
                "dominant_segment_encoded": segment_map.get(row["dominant_segment"], 0),
                "velocity_T14": row["velocity_T14"],
                "velocity_T7": row["velocity_T7"],
                "price_delta_secondary_chf": row["price_delta_secondary_chf"],
                "stl_trend_value": stl_features.get(m_id, {}).get("stl_trend_value", 0),
                "stl_seasonal_value": stl_features.get(m_id, {}).get("stl_seasonal_value", 0),
                "sarima_residual": sarima_features.get(m_id, 0),
                "archetype_deviation_T14": archetype_data.get(m_id, {}).get("archetype_deviation_T14", 0),
                "np_final_prediction": np_data.get(m_id, {}).get("np_final_prediction", 0),
                "np_deviation_T14": np_data.get(m_id, {}).get("np_deviation_T14", 0),
                "target": row["overall_fill_rate"],
                "season": row["season"]
            }
            # Zone targets
            for zone in ["Courtside VIP", "Lower Bowl / Club Seats", "Upper Standard", "Standing"]:
                feat[f"target_{zone}"] = row["attendance"][zone]["fill_rate"]
            
            data.append(feat)
            
        full_df = pd.DataFrame(data)
        X = full_df.drop(columns=[col for col in full_df.columns if col.startswith("target")] + ["season"])
        y = full_df["target"]
        self.feature_cols = X.columns.tolist()
        return X, y, full_df

    def _train_lightgbm(self, X, y, df_all):
        """Layer 2a: LightGBM for mean demand."""
        # Temporal split: Season 1 & 2 for train, 3 for validation
        train_idx = df_all[df_all["season"] != "2023-24"].index
        val_idx = df_all[df_all["season"] == "2023-24"].index
        
        params = {
            "objective": "regression",
            "metric": "mape",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.85,
            "bagging_freq": 5,
            "min_child_samples": 10,
            "verbose": -1
        }
        
        # Overall model
        dtrain = lgb.Dataset(X.iloc[train_idx], label=y.iloc[train_idx])
        dval = lgb.Dataset(X.iloc[val_idx], label=y.iloc[val_idx], reference=dtrain)
        
        model = lgb.train(params, dtrain, num_boost_round=500, valid_sets=[dval], 
                          callbacks=[lgb.early_stopping(stopping_rounds=50)])
        self.lgb_models["total"] = model
        
        # Zone models
        for zone in ["Courtside VIP", "Lower Bowl / Club Seats", "Upper Standard", "Standing"]:
            y_z = df_all[f"target_{zone}"]
            dz_train = lgb.Dataset(X.iloc[train_idx], label=y_z.iloc[train_idx])
            dz_val = lgb.Dataset(X.iloc[val_idx], label=y_z.iloc[val_idx], reference=dz_train)
            z_model = lgb.train(params, dz_train, num_boost_round=500, valid_sets=[dz_val],
                                callbacks=[lgb.early_stopping(stopping_rounds=50)])
            self.lgb_models[zone] = z_model

    def _train_quantile_models(self, X, y):
        """Layer 2b: Quantile Regression Forest for uncertainty."""
        # Quantile models using Sklearn GBR
        for alpha in [0.1, 0.5, 0.9]:
            model = GradientBoostingRegressor(loss="quantile", alpha=alpha, n_estimators=200, max_depth=4)
            model.fit(X, y)
            self.quantile_models[alpha] = model

    def _evaluate(self, X, y, df_all):
        """Computes MAPE, WAPE, and SHAP importance on validation set."""
        val_idx = df_all[df_all["season"] == "2023-24"].index
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]
        
        y_pred = self.lgb_models["total"].predict(X_val)
        mape = mean_absolute_percentage_error(y_val, y_pred)
        
        # WAPE (weighted by attendance/revenue)
        wape = np.sum(np.abs(y_val - y_pred)) / np.sum(y_val)
        
        # SHAP
        explainer = shap.TreeExplainer(self.lgb_models["total"])
        shap_values = explainer.shap_values(X_val)
        
        importance = []
        for i, col in enumerate(X.columns):
            importance.append({
                "feature": col,
                "mean_shap": float(np.mean(np.abs(shap_values[:, i])))
            })
        importance = sorted(importance, key=lambda x: x["mean_shap"], reverse=True)[:10]
        
        self.evaluation_results = {
            "overall_mape": float(mape),
            "wape": float(wape),
            "feature_importance": importance,
            "zone_mape": {
                z: float(mean_absolute_percentage_error(df_all.loc[val_idx, f"target_{z}"], 
                                                        self.lgb_models[z].predict(X_val)))
                for z in ["Courtside VIP", "Lower Bowl / Club Seats", "Upper Standard", "Standing"]
            }
        }
        
        with open(os.path.join(self.data_dir, "forecasting_evaluation.json"), "w") as f:
            json.dump(self.evaluation_results, f, indent=2)

    def _mint_reconcile(self, total_fill_pred, zone_preds, capacities):
        """Layer 2c: Hierarchical reconciliation to ensure coherence."""
        total_cap = sum(capacities.values())
        total_tickets = total_fill_pred * total_cap
        
        zone_tickets = {z: zone_preds[z] * capacities[z] for z in capacities}
        zone_sum = sum(zone_tickets.values())
        
        scale = total_tickets / zone_sum if zone_sum > 0 else 0
        
        reconciled = {}
        for z in capacities:
            rec_tickets = zone_tickets[z] * scale
            reconciled[z] = {
                "fill_rate": float(rec_tickets / capacities[z]) if capacities[z] > 0 else 0,
                "tickets_sold": int(round(rec_tickets))
            }
        return reconciled

    def predict(self, match_features: dict) -> dict:
        """Runs the two-layer pipeline for a single match prediction."""
        # FIX 3: Pop zone_capacities before converting to DataFrame
        capacities = match_features.pop("zone_capacities", {
            "Courtside VIP": 300,
            "Lower Bowl / Club Seats": 1000,
            "Upper Standard": 1500,
            "Standing": 1200
        })

        # Convert dict to DataFrame row and align with training features
        X_row = pd.DataFrame([match_features])
        if self.feature_cols:
            # Add missing columns with 0, drop extra columns, and maintain order
            X_row = X_row.reindex(columns=self.feature_cols, fill_value=0)
        
        # 1. Total Prediction (P50)
        p50_total = float(self.lgb_models["total"].predict(X_row)[0])
        
        # 2. Quantiles
        p10 = float(self.quantile_models[0.1].predict(X_row)[0])
        p90 = float(self.quantile_models[0.9].predict(X_row)[0])
        
        # 3. Zone Predictions (unreconciled)
        zone_preds = {z: float(self.lgb_models[z].predict(X_row)[0]) for z in 
                      ["Courtside VIP", "Lower Bowl / Club Seats", "Upper Standard", "Standing"]}
        
        # 4. Reconciliation
        reconciled = self._mint_reconcile(p50_total, zone_preds, capacities)
        
        # SHAP for explanation
        explainer = shap.TreeExplainer(self.lgb_models["total"])
        shap_val = explainer.shap_values(X_row)[0]
        
        drivers = []
        for i, col in enumerate(X_row.columns):
            drivers.append({
                "feature": col,
                "shap_value": float(shap_val[i]),
                "direction": "increases demand" if shap_val[i] > 0 else "decreases demand"
            })
        drivers = sorted(drivers, key=lambda x: abs(x["shap_value"]), reverse=True)[:5]

        return {
            "overall": {
                "p10_fill_rate": p10,
                "p50_fill_rate": p50_total,
                "p90_fill_rate": p90
            },
            "zones": {z: {
                "p50_fill_rate": reconciled[z]["fill_rate"],
                "p50_tickets_sold": reconciled[z]["tickets_sold"]
            } for z in reconciled},
            "shap_explanation": {
                "top_5_drivers": drivers
            }
        }

    def save_models(self):
        joblib.dump(self.lgb_models, os.path.join(self.models_dir, "lgb_models.joblib"))
        joblib.dump(self.quantile_models, os.path.join(self.models_dir, "quantile_models.joblib"))
        joblib.dump(self.feature_cols, os.path.join(self.models_dir, "feature_cols.joblib"))
        # NeuralProphet has special save logic
        # self.np_model.save(...) 

    def load_models(self):
        self.lgb_models = joblib.load(os.path.join(self.models_dir, "lgb_models.joblib"))
        self.quantile_models = joblib.load(os.path.join(self.models_dir, "quantile_models.joblib"))
        feat_path = os.path.join(self.models_dir, "feature_cols.joblib")
        if os.path.exists(feat_path):
            self.feature_cols = joblib.load(feat_path)
