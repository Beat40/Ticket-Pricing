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

class EPLConjointEngine:
    def __init__(self, tier_name, data_dir="EPL/data"):
        self.tier_name = tier_name
        self.data_dir = data_dir
        self.survey_path = os.path.join(data_dir, f"epl_survey_{tier_name}.json")
        
        # Attribute names per tier
        if tier_name == "big_six":
            self.feature_names = [
                "opponent_mid", "opponent_big_six",
                "zone_lower", "zone_upper", "zone_hospitality",
                "stakes_top4", "stakes_title",
                "price",
                "derby", "hospitality", "tv_prime"
            ]
        elif tier_name == "mid":
            self.feature_names = [
                "opponent_mid", "opponent_big_six",
                "zone_lower", "zone_upper", "zone_hospitality",
                "stakes_top4", "stakes_relegation", "stakes_euro",
                "price",
                "derby", "tv_prime"
            ]
        else: # small
            self.feature_names = [
                "opponent_mid", "opponent_big_six",
                "zone_lower", "zone_upper", "zone_hospitality",
                "stakes_top4", "stakes_relegation", "stakes_euro",
                "price",
                "derby", "tv_prime"
            ]
            
        self.scaler = StandardScaler()

    async def run(self) -> dict:
        print(f"Starting EPL Conjoint Analysis Engine for tier: {self.tier_name}...")
        
        # Step 1: Load and Encode
        encoded = self._load_and_encode()
        
        # Step 2: Population-level MNL
        mnl_results = self._run_mnl(encoded)
        
        # Step 3: HB-MNL
        individual_betas, hb_diagnostics, used_fallback = await self._run_hb_mnl(encoded, mnl_results)
        
        # Step 4: WTP and LP Bounds
        wtp_results = self._compute_wtp(individual_betas, mnl_results)
        
        # Save results
        with open(os.path.join(self.data_dir, f"epl_wtp_{self.tier_name}.json"), "w") as f:
            json.dump(wtp_results, f, indent=2)
            
        # Save estimated individual utilities
        est_utils = []
        for i, res_id in enumerate(encoded["respondent_ids"]):
            est_utils.append({
                "respondent_id": res_id,
                "utilities": dict(zip(self.feature_names, individual_betas[i].tolist()))
            })
        with open(os.path.join(self.data_dir, f"epl_estimated_individual_utilities_{self.tier_name}.json"), "w") as f:
            json.dump(est_utils, f, indent=2)
            
        return {
            "tier": self.tier_name,
            "method": "HB-MNL (PyMC NUTS)" if not used_fallback else "Fallback",
            "n_respondents": encoded["n_respondents"],
            "wtp_results": wtp_results
        }

    def _load_and_encode(self) -> dict:
        with open(self.survey_path, "r") as f:
            data = json.load(f)
            
        valid_respondents = [r for r in data if r.get("consistency_flag", False)]
        
        respondent_ids = []
        observations = []
        respondent_idx = []
        
        for r_idx, r in enumerate(valid_respondents):
            respondent_ids.append(r["respondent_id"])
            for task in r["responses"]:
                if task["task_index"] in [13, 14] or task["option_chosen"] == "neither":
                    continue
                
                obs = self._encode_task(task)
                observations.append(obs)
                respondent_idx.append(r_idx)
        
        n_features = len(self.feature_names)
        x_chosen = np.zeros((len(observations), n_features))
        x_rejected = np.zeros((len(observations), n_features))
        
        for i, o in enumerate(observations):
            x_chosen[i] = o["chosen"]
            x_rejected[i] = o["rejected"]
            
        return {
            "respondent_idx": np.array(respondent_idx),
            "x_chosen": x_chosen,
            "x_rejected": x_rejected,
            "n_respondents": len(valid_respondents),
            "respondent_ids": respondent_ids
        }

    def _encode_task(self, task):
        chosen_key = "option_a" if task["option_chosen"] == "A" else "option_b"
        rejected_key = "option_b" if task["option_chosen"] == "A" else "option_a"
        
        return {
            "chosen": self._encode_attributes(task[chosen_key]),
            "rejected": self._encode_attributes(task[rejected_key])
        }

    def _encode_attributes(self, attrs):
        vec = []
        
        # Opponent
        vec.append(1 if attrs["opponent"] == "Mid" else 0)
        vec.append(1 if attrs["opponent"] == "Big Six" else 0)
        
        # Zone
        vec.append(1 if attrs["seat_zone"] == "Lower Tier" else 0)
        vec.append(1 if attrs["seat_zone"] == "Upper Tier" else 0)
        vec.append(1 if attrs["seat_zone"] == "Hospitality" else 0)
        
        # Stakes
        if self.tier_name == "big_six":
            vec.append(1 if attrs["stakes"] == "Top 4" else 0)
            vec.append(1 if attrs["stakes"] == "Title Decider" else 0)
        else:
            vec.append(1 if attrs["stakes"] == "Top 4" else 0)
            vec.append(1 if attrs["stakes"] == "Relegation" else 0)
            vec.append(1 if attrs["stakes"] == "Euro" else 0)
            
        # Price (index changes depending on stakes count)
        price_idx = len(vec)
        vec.append(float(attrs["price"]))
        
        # Derby
        vec.append(1 if attrs["derby"] == "Yes" else 0)
        
        # Tier-specific extras
        if self.tier_name == "big_six":
            vec.append(1 if attrs["hospitality_inclusion"] == "Yes" else 0)
            vec.append(1 if attrs["tv_prime"] == "Yes" else 0)
        else:
            vec.append(1 if attrs["tv_prime"] == "Yes" else 0)
            
        return np.array(vec)

    def _run_mnl(self, encoded) -> dict:
        x_c = encoded["x_chosen"]
        x_r = encoded["x_rejected"]
        n_features = x_c.shape[1]
        
        # Price index is constant relative to end? 
        # Big Six: 11 features, price is index 7
        # Mid/Small: 11 features, price is index 8
        price_idx = 7 if self.tier_name == "big_six" else 8
        
        def neg_log_likelihood(beta):
            v_c = np.dot(x_c, beta)
            v_r = np.dot(x_r, beta)
            ll = np.sum(v_c - np.log(np.exp(v_c) + np.exp(v_r)))
            return -ll

        res = minimize(neg_log_likelihood, np.zeros(n_features), method="BFGS")
        betas = res.x
        
        return {
            "coefficients": dict(zip(self.feature_names, betas.tolist())),
            "log_likelihood": float(-res.fun),
            "price_idx": price_idx
        }

    async def _run_hb_mnl(self, encoded, mnl_results) -> tuple:
        n_resp = encoded["n_respondents"]
        resp_idx = encoded["respondent_idx"]
        x_c = encoded["x_chosen"]
        x_r = encoded["x_rejected"]
        n_features = x_c.shape[1]
        price_idx = mnl_results["price_idx"]
        
        other_indices = [i for i in range(n_features) if i != price_idx]
        
        def _sample_model():
            with pm.Model() as hb_mnl_model:
                mu_raw = pm.Normal("mu_raw", 0, 2, shape=(n_features - 1,))
                mu_price = pm.Normal("mu_price", -3.0, 1.0) # More flexible, centered around exp(-3) ~ 0.05
                sigma = pm.HalfNormal("sigma", 1.0, shape=(n_features,))
                
                beta_offset = pm.Normal("beta_offset", 0, 1, shape=(n_resp, n_features - 1))
                beta_raw = pm.Deterministic("beta_raw", mu_raw + beta_offset * sigma[other_indices])
                
                price_offset = pm.Normal("price_offset", 0, 1, shape=(n_resp,))
                raw_price = pm.Deterministic("raw_price", mu_price + price_offset * sigma[price_idx])
                beta_price = pm.Deterministic("beta_price", -pm.math.exp(raw_price))
                
                # Reconstruct full beta matrix
                betas_list = []
                other_ptr = 0
                for i in range(n_features):
                    if i == price_idx:
                        betas_list.append(beta_price[:, None])
                    else:
                        betas_list.append(beta_raw[:, other_ptr:other_ptr+1])
                        other_ptr += 1
                
                betas = pm.math.concatenate(betas_list, axis=1)
                beta_obs = betas[resp_idx]
                
                v_diff = pm.math.sum(beta_obs * (x_c - x_r), axis=1)
                pm.Bernoulli("choice", logit_p=v_diff, observed=np.ones(len(x_c)))
                
                return pm.sample(draws=1000, tune=500, chains=2, cores=1, progressbar=False)

        try:
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=1) as executor:
                trace = await loop.run_in_executor(executor, _sample_model)
            
            beta_raw_mean = trace.posterior["beta_raw"].mean(dim=("chain", "draw")).values
            beta_price_mean = trace.posterior["beta_price"].mean(dim=("chain", "draw")).values
            
            individual_betas = np.zeros((n_resp, n_features))
            other_ptr = 0
            for i in range(n_features):
                if i == price_idx:
                    individual_betas[:, i] = beta_price_mean
                else:
                    individual_betas[:, i] = beta_raw_mean[:, other_ptr]
                    other_ptr += 1
            
            return individual_betas, {}, False
        except Exception as e:
            print(f"HB-MNL Failed for {self.tier_name}: {e}")
            mnl_beta = np.array(list(mnl_results["coefficients"].values()))
            individual_betas = np.tile(mnl_beta, (n_resp, 1))
            return individual_betas, {}, True

    def _compute_wtp(self, betas, mnl) -> dict:
        price_idx = mnl["price_idx"]
        price_betas = np.abs(betas[:, price_idx])
        
        results = {"zone_price_bounds": {}, "attribute_wtp": {}}
        
        # Zone indices: Lower=2, Upper=3, Hosp=4
        zones = {"Lower Tier": 2, "Upper Tier": 3, "Hospitality": 4}
        
        # Baseline for zones (Away End is £30)
        base_p = 30.0
        
        for zone_name, idx in zones.items():
            # Utility difference relative to Away End (0)
            # WTP premium = (U_zone - U_away) / abs(beta_price)
            # Since U_away is 0 (reference), it's just betas[:, idx] / price_betas
            wtp_premiums = betas[:, idx] / price_betas
            wtp_array = wtp_premiums + base_p
            
            # Ensure floor is at least base_p for Lower/Upper
            p10, p50, p90 = np.percentile(wtp_array, [10, 50, 90])
            
            results["zone_price_bounds"][zone_name] = {
                "p10": float(p10), "p50": float(p50), "p90": float(p90),
                "floor": int(max(base_p, round(p10))), 
                "median": int(max(base_p, round(p50))), 
                "ceiling": int(max(base_p + 10, round(p90)))
            }

        for i, feat in enumerate(self.feature_names):
            if i == price_idx: continue
            wtp_vals = betas[:, i] / price_betas
            results["attribute_wtp"][feat] = {
                "mean": float(np.mean(wtp_vals)),
                "p50": float(np.median(wtp_vals))
            }
            
        return results

async def main():
    for tier in ["big_six", "mid", "small"]:
        engine = EPLConjointEngine(tier)
        res = await engine.run()
        print(f"Tier {tier} complete. Respondents: {res['n_respondents']}")

if __name__ == "__main__":
    asyncio.run(main())
