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
        self._train_quantile_models(X, y, df_features)
        
        # Evaluation & SHAP
        self._evaluate(X, y, df_features)
        
        self.save_models()
        return self.evaluation_results

    def _run_stl(self, df):
        """Layer 1a: STL Decomposition per club for venue-specific dynamics."""
        df["stl_trend_value"] = 0.0
        df["stl_seasonal_value"] = 0.0
        
        clubs = df["home_club_id"].unique()
        stl_summary = {}

        for club in clubs:
            club_df = df[df["home_club_id"] == club].sort_values("match_date")
            if len(club_df) < 18: # Need at least 2 seasons for decent decomposition
                continue
            
            series = club_df["overall_fill_rate"].values
            # period=9 (one full season of home matches)
            res = STL(series, period=9, seasonal=7, robust=True).fit()
            
            # Map back to main DF
            df.loc[club_df.index, "stl_trend_value"] = res.trend
            df.loc[club_df.index, "stl_seasonal_value"] = res.seasonal
            
            # Summary for debugging/logging
            resid_std = np.std(res.resid)
            anomalies = np.abs(res.resid) > 2 * resid_std
            
            stl_summary[club] = {
                "n_anomalies": int(np.sum(anomalies)),
                "seasonal_amplitude": float(np.max(res.seasonal) - np.min(res.seasonal))
            }

        self.stl_results = stl_summary
        
        # Save latest temporal features per club for inference lookup
        latest_temporal = {}
        for club in clubs:
            club_rows = df[df["home_club_id"] == club].sort_values("match_date")
            if len(club_rows) > 0:
                latest_temporal[club] = {
                    "latest_stl_trend": float(df.loc[club_rows.index[-1], "stl_trend_value"]),
                    "latest_stl_seasonal": float(df.loc[club_rows.index[-1], "stl_seasonal_value"]),
                    "sarima_residual_default": 0.0 # Neutral assumption for future
                }
        
        with open(os.path.join(self.data_dir, "latest_temporal_features.json"), "w") as f:
            json.dump(latest_temporal, f, indent=2)
        
        # Persist features for Layer 2
        stl_features = {}
        for _, row in df.iterrows():
            stl_features[row["match_id"]] = {
                "stl_trend_value": float(row["stl_trend_value"]),
                "stl_seasonal_value": float(row["stl_seasonal_value"])
            }
        
        with open(os.path.join(self.data_dir, "stl_features.json"), "w") as f:
            json.dump(stl_features, f, indent=2)

        with open(os.path.join(self.data_dir, "stl_decomposition.json"), "w") as f:
            json.dump(self.stl_results, f, indent=2)

    def _run_sarima(self, df):
        """Layer 1b: SARIMAX per club for sequential baseline residuals."""
        df["sarima_residual"] = 0.0
        
        clubs = df["home_club_id"].unique()
        opp_map = {"Elite": 2, "Competitive": 1, "Standard": 0}
        stakes_map = {"Final": 2, "Playoff": 1, "Group": 0}
        sarima_summary = {}

        for club in clubs:
            club_df = df[df["home_club_id"] == club].sort_values("match_date")
            if len(club_df) < 18:
                continue
            
            series = club_df["overall_fill_rate"].values
            exog = pd.DataFrame({
                "opp": club_df["opponent_tier"].map(opp_map),
                "stakes": club_df["match_stakes"].map(stakes_map),
                "weather": club_df["weather_severity_score"],
                "marketing": club_df["marketing_activation_score"]
            })
            
            # Fit SARIMAX (1, 0, 1) x (1, 0, 1, 9)
            try:
                # Simplify to AR(1) with exog, remove seasonal components (handled by STL)
                # This increases degrees of freedom for the small (n=27) dataset
                model = SARIMAX(series, exog=exog, order=(1,0,0), seasonal_order=(0,0,0,0))
                res = model.fit(disp=False)
                # Fill initial NaNs (first max(p,q) obs) with 0.0
                filled_resid = pd.Series(res.resid).fillna(0).values
                df.loc[club_df.index, "sarima_residual"] = filled_resid
                
                sarima_summary[club] = {
                    "aic": float(res.aic),
                    "in_sample_mape": float(mean_absolute_percentage_error(series, res.fittedvalues))
                }
            except Exception as e:
                print(f"SARIMA failed for club {club}: {e}")
        
        # Persist features for Layer 2
        sarima_features = {}
        for _, row in df.iterrows():
            sarima_features[row["match_id"]] = float(row["sarima_residual"])
            
        with open(os.path.join(self.data_dir, "sarima_features.json"), "w") as f:
            json.dump(sarima_features, f, indent=2)

        with open(os.path.join(self.data_dir, "sarima_results.json"), "w") as f:
            json.dump(sarima_summary, f, indent=2)

    def _cluster_archetypes(self, matches):
        """Layer 1c: DTW Clustering of normalized booking curves with train/val split."""
        # Split into Train (S1, S2) and Transform (S3) to avoid leakage
        train_matches = [m for m in matches if m["season"] != "2023-24"]
        val_matches = [m for m in matches if m["season"] == "2023-24"]
        
        def get_norm_curves(ms):
            c_list, ids = [], []
            for m in ms:
                curve = np.array(m["booking_curve"])
                if curve[-1] > 0:
                    c_list.append(curve / curve[-1])
                    ids.append(m["match_id"])
            return np.array(c_list), ids

        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform

        train_curves_np, train_ids = get_norm_curves(train_matches)
        
        print(f"Fitting DTW clustering on {len(train_matches)} train matches...")
        ds = dtw.distance_matrix_fast(train_curves_np)
        ds = (ds + ds.T) / 2
        np.fill_diagonal(ds, 0)
        
        condensed_ds = squareform(ds)
        linkage_matrix = linkage(condensed_ds, method="ward")
        cluster_assignments = fcluster(linkage_matrix, 4, criterion="maxclust")
        
        # Cluster labels and medoids
        assignments = [[] for _ in range(4)]
        for i, c_id in enumerate(cluster_assignments):
            assignments[int(c_id) - 1].append(i)
        
        medoids = {}
        labels = {}
        for c_id, idxs in enumerate(assignments):
            sub_matrix = ds[np.ix_(idxs, idxs)]
            medoid_local_idx = np.argmin(np.sum(sub_matrix, axis=0))
            medoid_global_idx = idxs[medoid_local_idx]
            medoid_curve = train_curves_np[medoid_global_idx]
            
            val_30, val_53 = medoid_curve[30], medoid_curve[53]
            if val_30 >= 0.35: label = "Early Surge"
            elif val_30 <= 0.25 and val_53 >= 0.60: label = "Late Surge"
            elif val_53 < 0.55: label = "Flat"
            else: label = "Consistent Gradual"
            
            if label in labels.values(): label = f"{label}_{c_id}"
            labels[c_id] = label
            medoids[label] = medoid_curve.tolist()

        match_results = {}
        # Apply labels to train
        for i, m_id in enumerate(train_ids):
            lbl = labels[cluster_assignments[i] - 1]
            match_results[m_id] = {
                "archetype": lbl,
                "archetype_deviation_T14": float(train_curves_np[i][46] - medoids[lbl][46]),
                "archetype_deviation_T7": float(train_curves_np[i][53] - medoids[lbl][53])
            }

        # FIX 6: Transform validation matches (predict archetype)
        print(f"Assigning archetypes to {len(val_matches)} validation matches...")
        val_curves_np, val_ids = get_norm_curves(val_matches)
        for i, m_id in enumerate(val_ids):
            v_curve = val_curves_np[i]
            # Find nearest medoid by DTW
            best_lbl, min_dist = None, float('inf')
            for lbl, m_curve in medoids.items():
                d = dtw.distance(v_curve, np.array(m_curve))
                if d < min_dist:
                    min_dist = d
                    best_lbl = lbl
            
            match_results[m_id] = {
                "archetype": best_lbl,
                "archetype_deviation_T14": float(v_curve[46] - medoids[best_lbl][46]),
                "archetype_deviation_T7": float(v_curve[53] - medoids[best_lbl][53])
            }

        self.archetype_results = {
            "n_clusters": 4,
            "medoid_curves": medoids,
            "match_archetype_assignments": match_results
        }
        
        with open(os.path.join(self.data_dir, "archetype_results.json"), "w") as f:
            json.dump(self.archetype_results, f, indent=2)


    async def _train_neural_prophet(self, matches):
        """Layer 1d: Neural Prophet velocity forecaster."""
        # Long format conversion
        rows = []
        opp_map = {"Elite": 2, "Competitive": 1, "Standard": 0}
        stakes_map = {"Final": 2, "Playoff": 1, "Group": 0}

        for idx, m in enumerate(matches):
            curve = np.array(m["booking_curve"])
            norm_curve = curve / curve[-1] if curve[-1] > 0 else curve
            
            # FIX 3: NeuralProphet Panel Leakage fix
            # Each match gets its own non-overlapping time sequence (anchored 100 days apart)
            # This turns ds into a pure selling-window indicator.
            anchor = datetime(2000, 1, 1) + timedelta(days=idx * 100)
            for t in range(61):
                ds = anchor + timedelta(days=t)
                rows.append({
                    "ds": ds,
                    "y": norm_curve[t],
                    "ID": m["match_id"]
                })
        
        np_df = pd.DataFrame(rows)
        np_df["ds"] = pd.to_datetime(np_df["ds"])
        np_df["y"] = np_df["y"].astype(float)
        
        # Training (Pure selling-window sequence model)
        m_np = NeuralProphet(
            n_forecasts=1,
            n_lags=7,
            yearly_seasonality=False,
            weekly_seasonality=False, 
            daily_seasonality=False,
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
                # FIX 7: Zone velocities
                "velocity_T14_VIP": row.get("velocity_T14_Courtside VIP", 0),
                "velocity_T7_VIP": row.get("velocity_T7_Courtside VIP", 0),
                "velocity_T14_LB": row.get("velocity_T14_Lower Bowl / Club Seats", 0),
                "velocity_T7_LB": row.get("velocity_T7_Lower Bowl / Club Seats", 0),
                "velocity_T14_US": row.get("velocity_T14_Upper Standard", 0),
                "velocity_T7_US": row.get("velocity_T7_Upper Standard", 0),
                "velocity_T14_ST": row.get("velocity_T14_Standing", 0),
                "velocity_T7_ST": row.get("velocity_T7_Standing", 0),
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

    def _train_quantile_models(self, X, y, df_all):
        """Layer 2b: Quantile Regression Forest for uncertainty (train only)."""
        train_idx = df_all[df_all["season"] != "2023-24"].index
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        for alpha in [0.1, 0.5, 0.9]:
            model = GradientBoostingRegressor(
                loss="quantile",
                alpha=alpha,
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05
            )
            model.fit(X_train, y_train)
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
            # FIX 2: Clipping to ensure coherence and physical possibility
            rec_tickets = min(float(capacities[z]), zone_tickets[z] * scale)
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
        # FIX 8: STL/SARIMA/Archetype Lookup for inference
        # Load latest temporal components for the club
        latest_path = os.path.join(self.data_dir, "latest_temporal_features.json")
        if os.path.exists(latest_path):
            with open(latest_path, "r") as f:
                lt = json.load(f)
            club_id = match_features.get("home_club_id")
            if club_id in lt:
                match_features["stl_trend_value"] = lt[club_id]["latest_stl_trend"]
                match_features["stl_seasonal_value"] = lt[club_id]["latest_stl_seasonal"]
                match_features["sarima_residual"] = lt[club_id]["sarima_residual_default"]

        # Convert dict to DataFrame row and align with training features
        X_row = pd.DataFrame([match_features])
        if self.feature_cols:
            # Drop extra columns and maintain order. fill_value=0 is now a safe fallback.
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
