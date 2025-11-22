# src/data/preprocess.py
import os
import json
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, LabelEncoder
from sklearn.model_selection import train_test_split
from joblib import dump, load

NUMERIC_KEY = "numeric"
CATEGORICAL_KEY = "categorical"

class TabularPreprocessor:
    def __init__(self,
                 numeric_transform_strategy: str = "quantile",  # currently only quantile supported
                 n_quantiles: int = 1000,
                 random_state: int = 42):
        self.numeric_transform_strategy = numeric_transform_strategy
        self.n_quantiles = n_quantiles
        self.random_state = random_state

        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.quantile_transformers: Dict[str, QuantileTransformer] = {}
        self.label_encoders: Dict[str, LabelEncoder] = {}

    def _detect_column_types(self, df: pd.DataFrame, cat_threshold: int = 30):
        # Heuristic: object dtype or low-cardinality ints -> categorical
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        other_cols = [c for c in df.columns if c not in numeric_cols]
        categorical = other_cols[:]
        # Among numerics, small cardinality -> categorical (e.g., encoded categories)
        for c in numeric_cols:
            nunique = df[c].nunique(dropna=True)
            if nunique <= cat_threshold:
                categorical.append(c)
            else:
                self.numeric_cols.append(c)
        # filter categorical duplicates
        categorical = list(dict.fromkeys(categorical))
        # keep categorical that exist
        categorical = [c for c in categorical if c in df.columns]
        self.categorical_cols = categorical

    def fit(self, df: pd.DataFrame, exclude: Optional[List[str]] = None):
        if exclude is None:
            exclude = []
        df = df.copy()
        for col in exclude:
            if col in df.columns:
                df = df.drop(columns=[col])
        self._detect_column_types(df)
        # Fit quantile transformers for numeric
        if self.numeric_transform_strategy == "quantile":
            for c in self.numeric_cols:
                qt = QuantileTransformer(n_quantiles=min(self.n_quantiles, max(10, int(len(df) * 0.01))),
                                        output_distribution="normal",
                                        random_state=self.random_state,
                                        copy=True)
                vals = df[[c]].fillna(df[c].median()).values
                qt.fit(vals)
                self.quantile_transformers[c] = qt
        # Fit label encoders for categorical
        for c in self.categorical_cols:
            le = LabelEncoder()
            vals = df[c].fillna("##NA##").astype(str).values
            le.fit(vals)
            self.label_encoders[c] = le

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Numeric transform
        for c, qt in self.quantile_transformers.items():
            if c in df.columns:
                vals = df[[c]].fillna(df[c].median()).values
                df[c] = qt.transform(vals).astype(float).reshape(-1)
        # Categorical encode (as ints)
        for c, le in self.label_encoders.items():
            if c in df.columns:
                vals = df[c].fillna("##NA##").astype(str).values
                df[c] = le.transform(vals).astype(int)
        return df

    def inverse_transform_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for c, qt in self.quantile_transformers.items():
            if c in df.columns:
                vals = df[[c]].values
                df[c] = qt.inverse_transform(vals).reshape(-1)
        return df

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        # Save metadata
        meta = {
            NUMERIC_KEY: self.numeric_cols,
            CATEGORICAL_KEY: self.categorical_cols
        }
        json.dump(meta, open(os.path.join(path, "metadata.json"), "w"))
        # Save transformers
        for c, qt in self.quantile_transformers.items():
            dump(qt, os.path.join(path, f"quantile_{c}.joblib"))
        for c, le in self.label_encoders.items():
            dump(le, os.path.join(path, f"labelenc_{c}.joblib"))

    def load(self, path: str):
        meta = json.load(open(os.path.join(path, "metadata.json"), "r"))
        self.numeric_cols = meta.get(NUMERIC_KEY, [])
        self.categorical_cols = meta.get(CATEGORICAL_KEY, [])
        for c in self.numeric_cols:
            p = os.path.join(path, f"quantile_{c}.joblib")
            if os.path.exists(p):
                self.quantile_transformers[c] = load(p)
        for c in self.categorical_cols:
            p = os.path.join(path, f"labelenc_{c}.joblib")
            if os.path.exists(p):
                self.label_encoders[c] = load(p)

def simple_train_val_test_split(df: pd.DataFrame, test_size=0.2, val_size=0.1, random_state=42):
    train_val, test = train_test_split(df, test_size=test_size, random_state=random_state, shuffle=True)
    val_relative = val_size / (1 - test_size)
    train, val = train_test_split(train_val, test_size=val_relative, random_state=random_state, shuffle=True)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

def preprocess_and_save(df: pd.DataFrame,
                        out_dir: str,
                        exclude: Optional[List[str]] = None,
                        target_col: Optional[str] = None,
                        test_size=0.2,
                        val_size=0.1,
                        random_state=42):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pre = TabularPreprocessor(random_state=random_state)
    pre.fit(df.drop(columns=[target_col]) if target_col and target_col in df.columns else df)
    df_trans = pre.transform(df)
    train, val, test = simple_train_val_test_split(df_trans, test_size=test_size, val_size=val_size, random_state=random_state)
    # Save CSVs and preprocessor
    train.to_csv(out_dir / "train.csv", index=False)
    val.to_csv(out_dir / "val.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)
    pre.save(str(out_dir / "preprocessor"))
    return {"train": str(out_dir / "train.csv"),
            "val": str(out_dir / "val.csv"),
            "test": str(out_dir / "test.csv"),
            "preprocessor": str(out_dir / "preprocessor/metadata.json")}
