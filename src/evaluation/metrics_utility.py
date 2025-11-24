# src/evaluation/metrics_utility.py
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def _fit_and_eval_clf(clf, X_train, y_train, X_test, y_test) -> Dict[str, float]:
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    result = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds, average='binary' if len(np.unique(y_test))==2 else 'macro'))
    }
    # attempt AUC for binary
    if len(np.unique(y_test)) == 2:
        try:
            proba = clf.predict_proba(X_test)[:,1]
            result["roc_auc"] = float(roc_auc_score(y_test, proba))
        except Exception:
            result["roc_auc"] = None
    else:
        result["roc_auc"] = None
    return result

def utility_evaluation(real: pd.DataFrame, synth: pd.DataFrame, features: List[str], target_col: str, test_size=0.2, random_state=42) -> Dict[str, Any]:
    """
    Trains LogisticRegression and RandomForest on (real -> test on synth) and (synth -> test on real).
    Returns metrics for both directions.
    """
    out = {}
    # prepare data: encode target if needed
    le = None
    y_real = real[target_col]
    if y_real.dtype == object or y_real.dtype == str:
        le = LabelEncoder()
        y_real = le.fit_transform(y_real.astype(str))
        y_synth = le.transform(synth[target_col].astype(str)) if target_col in synth.columns else None
    else:
        y_real = y_real.values
        y_synth = synth[target_col].values if target_col in synth.columns else None

    X_real = real[features].fillna(0).values
    X_synth = synth[features].fillna(0).values

    # If synth has different length, ok
    # Real-> synth (train on real, test on synth)
    lr = LogisticRegression(max_iter=500)
    rf = RandomForestClassifier(n_estimators=100)
    dirA = {}
    try:
        dirA["logistic"] = _fit_and_eval_clf(lr, X_real, y_real, X_synth, y_synth)
    except Exception as e:
        dirA["logistic"] = {"error": str(e)}
    try:
        dirA["random_forest"] = _fit_and_eval_clf(rf, X_real, y_real, X_synth, y_synth)
    except Exception as e:
        dirA["random_forest"] = {"error": str(e)}
    out["train_real_test_synth"] = dirA

    # synth -> real (train on synth, test on real)
    dirB = {}
    try:
        dirB["logistic"] = _fit_and_eval_clf(lr, X_synth, y_synth, X_real, y_real)
    except Exception as e:
        dirB["logistic"] = {"error": str(e)}
    try:
        dirB["random_forest"] = _fit_and_eval_clf(rf, X_synth, y_synth, X_real, y_real)
    except Exception as e:
        dirB["random_forest"] = {"error": str(e)}
    out["train_synth_test_real"] = dirB

    return out
