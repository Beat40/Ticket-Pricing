import json
import os
import asyncio
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from concurrent.futures import ThreadPoolExecutor
import pymc as pm
import arviz as az

class ConjointEngine:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.survey_path = os.path.join(data_dir, "survey_responses.json")
        self.ground_truth_path = os.path.join(data_dir, "ground_truth_utilities.json")
        self.feature_names = [
            "opponent_competitive", "opponent_elite",
            "zone_upper_standard", "zone_lower_bowl", "zone_courtside_vip",
            "stakes_playoff", "stakes_final",
            "price",
            "bundle_sbb", "bundle_sbb_food",
            "star_uncertain", "star_confirmed",
            "kickoff_sat_afternoon", "kickoff_sat_evening"
        ]
        self.scaler = StandardScaler()

    async def run(self) -> dict:
        print("Starting Conjoint Analysis Engine...")
        
        # Step 1: Load and Encode
        encoded = self._load_and_encode()
        
        # Step 2: Population-level MNL
        mnl_results = self._run_mnl(encoded)
        
        # Step 3: HB-MNL
        individual_betas, hb_diagnostics, used_fallback = await self._run_hb_mnl(encoded, mnl_results)
        
        # Step 4: WTP and LP Bounds
        wtp_results = self._compute_wtp(individual_betas, mnl_results)
        
        # Step 5: Segmentation
        segmentation = self._segment_fans(individual_betas, encoded["respondent_ids"])
        
        # Step 6: Validation
        validation = self._validate(mnl_results, segmentation)
        
        # Save HB results
        with open(os.path.join(self.data_dir, "hb_diagnostics.json"), "w") as f:
            json.dump(hb_diagnostics, f, indent=2)
            
        method_str = "Simulated HB (fallback)" if used_fallback else "HB-MNL (PyMC NUTS)"
        
        return {
            "status": "complete",
            "method": method_str,
            "n_respondents_used": encoded["n_respondents"],
            "convergence": hb_diagnostics,
            "mnl_log_likelihood": mnl_results["log_likelihood"],
            "zone_price_bounds": wtp_results["zone_price_bounds"],
            "segment_summary": segmentation["segment_summary"],
            "validation": validation
        }

    def _load_and_encode(self) -> dict:
        with open(self.survey_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # Filter by consistency
        valid_respondents = [r for r in data if r.get("consistency_flag", False)]
        print(f"Loaded {len(data)} respondents. {len(valid_respondents)} remained after consistency filtering.")
        
        respondent_ids = []
        observations = []
        n_obs_per_respondent = []
        
        for r_idx, r in enumerate(valid_respondents):
            r_id = r["respondent_id"]
            respondent_ids.append(r_id)
            respondent_obs_count = 0
            
            for task in r["responses"]:
                idx = task["task_index"]
                # Skip holdouts and "neither"
                if idx in [13, 14] or task["option_chosen"] == "neither":
                    continue
                
                res = self._encode_task(task)
                if res:
                    observations.append({"respondent_idx": r_idx, **res})
                    respondent_obs_count += 1
            
            n_obs_per_respondent.append(respondent_obs_count)
            
        total_obs = len(observations)
        respondent_idx = np.array([o["respondent_idx"] for o in observations])
        x_chosen = np.zeros((total_obs, 14))
        x_rejected = np.zeros((total_obs, 14))
        
        for i, o in enumerate(observations):
            x_chosen[i] = o["chosen"]
            x_rejected[i] = o["rejected"]
            
        # Scaling non-price features (0-6 and 8-13)
        non_price_indices = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]
        
        # Combine chosen and rejected for scaling
        all_x = np.vstack([x_chosen[:, non_price_indices], x_rejected[:, non_price_indices]])
        self.scaler.fit(all_x)
        
        x_chosen_scaled = x_chosen.copy()
        x_rejected_scaled = x_rejected.copy()
        
        x_chosen_scaled[:, non_price_indices] = self.scaler.transform(x_chosen[:, non_price_indices])
        x_rejected_scaled[:, non_price_indices] = self.scaler.transform(x_rejected[:, non_price_indices])
        
        return {
            "respondent_idx": respondent_idx,
            "x_chosen": x_chosen_scaled,
            "x_rejected": x_rejected_scaled,
            "n_respondents": len(valid_respondents),
            "respondent_ids": respondent_ids,
            "n_obs_per_respondent": n_obs_per_respondent
        }

    def _encode_task(self, task):
        chosen_key = "option_a" if task["option_chosen"] == "A" else "option_b"
        rejected_key = "option_b" if task["option_chosen"] == "A" else "option_a"
        
        return {
            "chosen": self._encode_attributes(task[chosen_key]),
            "rejected": self._encode_attributes(task[rejected_key])
        }

    def _encode_attributes(self, attrs):
        # Feature order must be consistent
        vec = []
        
        # Opponent
        vec.append(1 if attrs["opponent"] == "Competitive" else 0)
        vec.append(1 if attrs["opponent"] == "Elite" else 0)
        
        # Seat Zone
        vec.append(1 if attrs["seat_zone"] == "Upper Standard" else 0)
        vec.append(1 if attrs["seat_zone"] == "Lower Bowl / Club Seats" else 0)
        vec.append(1 if attrs["seat_zone"] == "Courtside VIP" else 0)
        
        # Stakes
        vec.append(1 if attrs["stakes"] == "League Playoff / Knockout" else 0)
        vec.append(1 if attrs["stakes"] == "EHF EURO / Cup Final" else 0)
        
        # Price (index 7)
        vec.append(float(attrs["price"]))
        
        # Bundle
        vec.append(1 if attrs["bundle"] == "Ticket + SBB Travel" else 0)
        vec.append(1 if attrs["bundle"] == "Ticket + SBB Travel + Food & Drink" else 0)
        
        # Star Player
        vec.append(1 if attrs["star_player"].startswith("Uncertain") else 0)
        vec.append(1 if attrs["star_player"].startswith("Yes") else 0)
        
        # Kickoff
        vec.append(1 if attrs["kickoff"] == "Saturday afternoon 15:00" else 0)
        vec.append(1 if attrs["kickoff"] == "Saturday evening 19:30" else 0)
        
        return np.array(vec)

    def _run_mnl(self, encoded) -> dict:
        x_c = encoded["x_chosen"]
        x_r = encoded["x_rejected"]
        
        def neg_log_likelihood(beta):
            v_c = np.dot(x_c, beta)
            v_r = np.dot(x_r, beta)
            # MNL for binary choice: exp(v_c) / (exp(v_c) + exp(v_r))
            ll = np.sum(v_c - np.log(np.exp(v_c) + np.exp(v_r)))
            return -ll

        res = minimize(neg_log_likelihood, np.zeros(14), method="BFGS")
        betas = res.x
        
        if betas[7] >= 0:
            raise ValueError(f"MNL price coefficient ({betas[7]:.4f}) is positive — data has no valid price signal. Stopping.")
            
        results = {
            "coefficients": dict(zip(self.feature_names, betas.tolist())),
            "log_likelihood": float(-res.fun),
            "n_observations": len(x_c),
            "n_respondents": encoded["n_respondents"]
        }
        
        with open(os.path.join(self.data_dir, "mnl_results.json"), "w") as f:
            json.dump(results, f, indent=2)
            
        return results

    async def _run_hb_mnl(self, encoded, mnl_results) -> tuple:
        # HB-MNL using PyMC NUTS sampler.
        # POC settings: draws=1000, tune=500, chains=2, cores=1
        # Production settings: draws=2000, tune=1000, chains=4, cores=4
        # Expected runtime: 5-15 minutes on laptop
        
        n_resp = encoded["n_respondents"]
        resp_idx = encoded["respondent_idx"]
        x_c = encoded["x_chosen"]
        x_r = encoded["x_rejected"]
        
        def _sample_model():
            with pm.Model() as hb_mnl_model:
                # HYPERPRIORS (mu and sigma)
                # Non-price pop means (13 coeffs)
                mu_raw = pm.Normal("mu_raw", 0, 1, shape=(13,))
                # Price pop mean (informative)
                mu_price = pm.Normal("mu_price", -0.05, 0.02)
                
                # Pop SDs for all 14 coeffs
                sigma = pm.HalfNormal("sigma", 0.5, shape=(14,))
                
                # NON-CENTERED PARAMETRIZATION
                # This formulation (offset * sigma + mu) is much easier for NUTS to sample
                beta_offset = pm.Normal("beta_offset", 0, 1, shape=(n_resp, 13))
                beta_raw = pm.Deterministic("beta_raw", mu_raw + beta_offset * sigma[[0,1,2,3,4,5,6,8,9,10,11,12,13]])
                
                price_offset = pm.Normal("price_offset", 0, 1, shape=(n_resp,))
                raw_price = pm.Deterministic("raw_price", mu_price + price_offset * sigma[7])
                beta_price = pm.Deterministic("beta_price", -pm.math.exp(raw_price))
                
                # Assemble full beta matrix per observation
                betas = pm.math.concatenate([
                    beta_raw[:, :7], 
                    beta_price[:, None], 
                    beta_raw[:, 7:]
                ], axis=1)
                
                # Map respondents to observations
                beta_obs = betas[resp_idx]
                
                # Likelihood
                v_diff = pm.math.sum(beta_obs * (x_c - x_r), axis=1)
                pm.Bernoulli("choice", logit_p=v_diff, observed=np.ones(len(x_c)))
                
                return pm.sample(
                    draws=2000, 
                    tune=1500, 
                    chains=4, 
                    cores=4, 
                    target_accept=0.95,
                    return_inferencedata=True, 
                    progressbar=True,
                    random_seed=42
                )

        try:
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=1) as executor:
                trace = await loop.run_in_executor(executor, _sample_model)
                
            # Extract means across chains and draws
            beta_raw_mean = trace.posterior["beta_raw"].mean(dim=("chain", "draw")).values
            beta_price_mean = trace.posterior["beta_price"].mean(dim=("chain", "draw")).values
            
            individual_betas = np.concatenate([
                beta_raw_mean[:, :7],
                beta_price_mean[:, None],
                beta_raw_mean[:, 7:]
            ], axis=1)
            
            # Diagnostics using ArviZ summary for robust R-hat and ESS
            summary = az.summary(trace, var_names=["mu_raw", "mu_price", "sigma"])
            max_rhat = float(summary["r_hat"].max())
            min_ess = float(summary["ess_bulk"].min())
            
            diagnostics = {
                "converged": bool(max_rhat < 1.1 and min_ess > 100),
                "max_rhat": max_rhat,
                "min_ess": min_ess,
                "warnings": []
            }
            if max_rhat >= 1.1:
                diagnostics["warnings"].append(f"High R-hat ({max_rhat:.2f}) - model may not have converged.")
            
            return individual_betas, diagnostics, False
            
        except Exception as e:
            print(f"HB-MNL Sampling failed: {str(e)}. Falling back to simulated individual betas.")
            # Fallback: simulate individual betas around MNL results
            mnl_beta = np.array(list(mnl_results["coefficients"].values()))
            individual_betas = []
            for _ in range(n_resp):
                # beta_i = beta_mnl + N(0, 0.30 * |beta_mnl|)
                noise = np.random.normal(0, 0.30 * np.abs(mnl_beta))
                beta_i = mnl_beta + noise
                if beta_i[7] >= 0:
                    beta_i[7] = -0.0001 # Fix if positive
                individual_betas.append(beta_i)
            
            diagnostics = {
                "converged": False,
                "max_rhat": 0.0,
                "min_ess": 0.0,
                "warnings": [f"Sampling failed: {str(e)}", "Using simulated fallback"]
            }
            return np.array(individual_betas), diagnostics, True

    def _compute_wtp(self, betas, mnl) -> dict:
        # betas shape: (n_resp, 14)
        n_resp = betas.shape[0]
        
        results = {"zone_price_bounds": {}, "attribute_wtp": {}}
        
        # Zone indices: 2, 3, 4 (Standard index 18 baseline)
        zones = {
            "Standing": None,
            "Upper Standard": 2,
            "Lower Bowl / Club Seats": 3,
            "Courtside VIP": 4
        }
        
        for zone_name, idx in zones.items():
            if zone_name == "Standing":
                results["zone_price_bounds"][zone_name] = {
                    "floor": 12,
                    "median": 22,
                    "ceiling": 32,
                    "p10": 12.0,
                    "p25": 17.0,
                    "p50": 22.0,
                    "p75": 27.0,
                    "p90": 32.0,
                    "governance_override_applied": True
                }
                continue
            else:
                wtp_premiums = betas[:, idx] / np.abs(betas[:, 7])
                wtp_array = wtp_premiums + 18.0
                
            p10, p25, p50, p75, p90 = np.percentile(wtp_array, [10, 25, 50, 75, 90])
            
            results["zone_price_bounds"][zone_name] = {
                "floor": int(round(p10)),
                "median": int(round(p50)),
                "ceiling": int(round(p90)),
                "p10": float(p10),
                "p25": float(p25),
                "p50": float(p50),
                "p75": float(p75),
                "p90": float(p90),
                "governance_override_applied": False
            }

        # Attributes WTP
        attr_map = {
            "opponent_competitive": 0, "opponent_elite": 1,
            "stakes_playoff": 5, "stakes_final": 6,
            "bundle_sbb": 8, "bundle_sbb_food": 9,
            "star_uncertain": 10, "star_confirmed": 11,
            "kickoff_sat_afternoon": 12, "kickoff_sat_evening": 13
        }
        
        for attr_name, idx in attr_map.items():
            wtp_vals = betas[:, idx] / np.abs(betas[:, 7])
            stats = np.percentile(wtp_vals, [10, 25, 50, 75, 90])
            results["attribute_wtp"][attr_name] = {
                "mean": float(np.mean(wtp_vals)),
                "p10": float(stats[0]),
                "p25": float(stats[1]),
                "p50": float(stats[2]),
                "p75": float(stats[3]),
                "p90": float(stats[4])
            }
            
        results["n_respondents_used"] = n_resp
        results["mnl_log_likelihood"] = mnl["log_likelihood"]
        
        with open(os.path.join(self.data_dir, "wtp_results.json"), "w") as f:
            json.dump(results, f, indent=2)
            
        return results

    def _segment_fans(self, betas, respondent_ids) -> dict:
        # Features for clustering: WTP courtside, lower_bowl, upper_standard, bundle_sbb_food, stakes_final, star_confirmed
        # Indices: 4, 3, 2, 9, 6, 11
        indices = [4, 3, 2, 9, 6, 11]
        raw_wtp = betas[:, indices] / np.abs(betas[:, 7][:, None])
        
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(raw_wtp)
        
        kmeans = KMeans(n_clusters=4, random_state=42)
        clusters = kmeans.fit_predict(x_scaled)
        
        # Labeling clusters
        cluster_labels = {}
        for i in range(4):
            mask = (clusters == i)
            mean_wtp = np.mean(raw_wtp[mask], axis=0)
            cluster_labels[i] = {
                "id": i,
                "means": mean_wtp.tolist(),
                "n": int(np.sum(mask))
            }

        print("\nMean WTP Values per Cluster (Features: courtside, lower, upper, food, final, star):")
        for i in range(4):
            print(f"Cluster {i}: {[round(x, 2) for x in cluster_labels[i]['means']]}")

        # Labeling clusters by priority order
        # 1. Premium Seeker: cluster with highest mean wtp_courtside_vip
        premium_id = max(cluster_labels, key=lambda i: cluster_labels[i]["means"][0])
        
        # 2. Occasional Neutral: cluster with lowest mean wtp_courtside_vip
        remaining = [i for i in range(4) if i != premium_id]
        neutral_id = min(remaining, key=lambda i: cluster_labels[i]["means"][0])
        
        # 3. Value Loyalist: higher mean wtp_bundle_sbb_food between remaining two
        remaining = [i for i in remaining if i != neutral_id]
        value_id = max(remaining, key=lambda i: cluster_labels[i]["means"][3])
        
        # 4. Atmosphere Seeker: remaining
        atmo_id = [i for i in remaining if i != value_id][0]
        
        id_to_name = {
            premium_id: "Premium Seeker", 
            value_id: "Value Loyalist", 
            atmo_id: "Atmosphere Seeker", 
            neutral_id: "Occasional Neutral"
        }
        
        assignments = {respondent_ids[i]: id_to_name[clusters[i]] for i in range(len(respondent_ids))}
        
        summary = {}
        for c_id, name in id_to_name.items():
            c_data = cluster_labels[c_id]
            summary[name] = {
                "n": c_data["n"],
                "pct": round(c_data["n"] / len(respondent_ids) * 100, 1),
                "mean_wtp_courtside": round(c_data["means"][0], 2),
                "mean_wtp_lower_bowl": round(c_data["means"][1], 2),
                "mean_wtp_upper_standard": round(c_data["means"][2], 2),
                "mean_wtp_bundle_sbb_food": round(c_data["means"][3], 2),
                "mean_wtp_stakes_final": round(c_data["means"][4], 2),
                "mean_wtp_star_confirmed": round(c_data["means"][5], 2)
            }
            
        final_segments = {"segment_assignments": assignments, "segment_summary": summary}
        
        with open(os.path.join(self.data_dir, "fan_segments.json"), "w") as f:
            json.dump(final_segments, f, indent=2)
            
        return final_segments

    def _validate(self, mnl_results, segmentation) -> dict:
        with open(self.ground_truth_path, "r") as f:
            gt_data = json.load(f)
            
        total_n = sum(s["n"] for s in gt_data.values())
        
        validation = {"mnl_vs_ground_truth": {}, "segmentation_size_comparison": {}}
        
        # Coefficients comparison
        keys_to_compare = {
            "opponent_elite": "opponent_elite",
            "zone_courtside_vip": "zone_courtside_vip",
            "stakes_final": "stakes_final",
            "bundle_sbb_food": "bundle_sbb_food",
            "star_confirmed": "star_confirmed",
            "price": "price_coef"
        }
        
        for label, gt_key in keys_to_compare.items():
            weighted_avg = sum(s["n"] * s["utilities"][gt_key] for s in gt_data.values()) / total_n
            recovered = mnl_results["coefficients"][label]
            
            deviation = abs(recovered - weighted_avg) / abs(weighted_avg) if weighted_avg != 0 else 0
            
            validation["mnl_vs_ground_truth"][label] = {
                "mnl_recovered": round(recovered, 4),
                "ground_truth_weighted_avg": round(weighted_avg, 4),
                "deviation_pct": round(deviation * 100, 1),
                "status": "OK" if deviation < 0.25 else "WARN"
            }
            
        # Segment size comparison
        with open(self.survey_path, "r") as f:
            survey_data = json.load(f)
        
        # Load truth from all responses (including inconsistent ones) - but segmentation only has valid ones.
        # Actually task says "Compare against K-Means recovered segment sizes"
        # We need true N for all respondents
        true_counts = {}
        for r in survey_data:
            s_true = r["segment_true"]
            true_counts[s_true] = true_counts.get(s_true, 0) + 1
            
        for name in gt_data.keys():
            validation["segmentation_size_comparison"][name] = {
                "true_n": true_counts.get(name, 0),
                "recovered_n": segmentation["segment_summary"].get(name, {}).get("n", 0)
            }
            
        with open(os.path.join(self.data_dir, "validation_report.json"), "w") as f:
            json.dump(validation, f, indent=2)
            
        return validation

if __name__ == "__main__":
    import asyncio
    engine = ConjointEngine(data_dir="data")
    asyncio.run(engine.run())
