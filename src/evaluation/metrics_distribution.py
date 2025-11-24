# src/evaluation/metrics_distribution.py
import json
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

def numeric_distribution_metrics(real: pd.DataFrame, synth: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Any]:
    results = {}
    for c in numeric_cols:
        r = real[c].dropna().values
        s = synth[c].dropna().values
        if len(r) == 0 or len(s) == 0:
            results[c] = {"ks": None, "wasserstein": None}
            continue
        try:
            ks = ks_2samp(r, s).statistic
        except Exception:
            ks = float("nan")
        try:
            w = wasserstein_distance(r, s)
        except Exception:
            w = float("nan")
        # quantile differences (median absolute)
        try:
            qd = np.abs(np.quantile(r, [0.1,0.25,0.5,0.75,0.9]) - np.quantile(s, [0.1,0.25,0.5,0.75,0.9])).tolist()
        except Exception:
            qd = []
        results[c] = {"ks": float(ks), "wasserstein": float(w), "quantile_abs_diff": qd}
    return results

def categorical_distribution_metrics(real: pd.DataFrame, synth: pd.DataFrame, cat_cols: List[str]) -> Dict[str, Any]:
    results = {}
    for c in cat_cols:
        r = real[c].astype(str).fillna("##NA##")
        s = synth[c].astype(str).fillna("##NA##")
        freq_r = r.value_counts(normalize=True)
        freq_s = s.value_counts(normalize=True)
        # align indices
        all_idx = set(freq_r.index).union(set(freq_s.index))
        diffs = {}
        for k in all_idx:
            diffs[k] = float(abs(freq_r.get(k, 0.0) - freq_s.get(k, 0.0)))
        # L1 frequency error
        l1 = float(sum(diffs.values()))
        results[c] = {"l1_freq_error": l1, "per_category_abs_diff": diffs}
    return results

def distribution_summary(real: pd.DataFrame, synth: pd.DataFrame, numeric_cols: List[str], cat_cols: List[str]) -> Dict[str, Any]:
    return {
        "numeric": numeric_distribution_metrics(real, synth, numeric_cols),
        "categorical": categorical_distribution_metrics(real, synth, cat_cols)
    }
