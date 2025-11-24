# src/evaluation/metrics_structure.py
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mutual_info_score
from typing import List, Dict, Any

def correlation_distance(real: pd.DataFrame, synth: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Any]:
    if len(numeric_cols) == 0:
        return {"corr_real": None, "corr_synth": None, "frobenius_diff": None}
    cr = real[numeric_cols].corr().fillna(0).values
    cs = synth[numeric_cols].corr().fillna(0).values
    frob = float(np.linalg.norm(cr - cs, ord='fro'))
    return {"corr_real": cr.tolist(), "corr_synth": cs.tolist(), "frobenius_diff": frob}

def mutual_information_matrix(real: pd.DataFrame, synth: pd.DataFrame, numeric_cols: List[str], cat_cols: List[str]) -> Dict[str, Any]:
    # numeric-numeric: use mutual_info_regression pairwise (slow for many features)
    nm = {}
    # numeric->numeric MI (approx via discretization)
    for i, a in enumerate(numeric_cols):
        for j, b in enumerate(numeric_cols):
            if j <= i: continue
            try:
                # discretize into 20 bins for MI
                x = pd.qcut(real[a].fillna(0), q=20, duplicates='drop').astype(str)
                y = pd.qcut(real[b].fillna(0), q=20, duplicates='drop').astype(str)
                mi_real = mutual_info_score(x, y)
            except Exception:
                mi_real = 0.0
            try:
                x = pd.qcut(synth[a].fillna(0), q=20, duplicates='drop').astype(str)
                y = pd.qcut(synth[b].fillna(0), q=20, duplicates='drop').astype(str)
                mi_synth = mutual_info_score(x, y)
            except Exception:
                mi_synth = 0.0
            nm[f"{a}__{b}"] = {"mi_real": float(mi_real), "mi_synth": float(mi_synth)}
    # numeric-categorical: use mutual_info_regression on real and synth separately
    nc = {}
    for a in numeric_cols:
        for b in cat_cols:
            try:
                y = real[b].astype(str)
                x = real[a].fillna(0).values.reshape(-1,1)
                mi_r = mutual_info_regression(x, y.factorize()[0], discrete_features=True, random_state=0)
                mi_r = float(mi_r[0])
            except Exception:
                mi_r = 0.0
            try:
                y2 = synth[b].astype(str)
                x2 = synth[a].fillna(0).values.reshape(-1,1)
                mi_s = mutual_info_regression(x2, y2.factorize()[0], discrete_features=True, random_state=0)
                mi_s = float(mi_s[0])
            except Exception:
                mi_s = 0.0
            nc[f"{a}__{b}"] = {"mi_real": mi_r, "mi_synth": mi_s}
    return {"numeric_pairwise_mi": nm, "numeric_categorical_mi": nc}

def coverage_diversity(real: pd.DataFrame, synth: pd.DataFrame, numeric_cols: List[str], k: int = 1) -> Dict[str, Any]:
    """
    Compute nearest-neighbor distance of each synthetic sample to real dataset in numeric feature space.
    Return mean & percentile distances and coverage ratio (fraction of synth within real radius quantile).
    """
    if len(numeric_cols) == 0:
        return {"mean_nn_dist": None}
    X_real = real[numeric_cols].fillna(0).values
    X_synth = synth[numeric_cols].fillna(0).values
    dists = pairwise_distances(X_synth, X_real, metric='euclidean')
    nn = dists.min(axis=1)
    mean = float(nn.mean())
    p90 = float(np.percentile(nn, 90))
    p50 = float(np.percentile(nn, 50))
    # coverage: fraction of synth that have nn <= median real->real NN
    real_self = pairwise_distances(X_real, X_real, metric='euclidean')
    real_self[np.eye(real_self.shape[0],dtype=bool)] = np.inf
    real_nn = real_self.min(axis=1)
    radius = float(np.median(real_nn))
    coverage = float((nn <= radius).mean())
    return {"mean_nn_dist": mean, "p50_nn": p50, "p90_nn": p90, "coverage_fraction": coverage}
