# src/data/loaders.py
import json
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# -----------------------------------------------------------------------------
# Helper: read preprocessor metadata created by preprocess.py
# -----------------------------------------------------------------------------
def load_preprocessor_metadata(preprocessor_meta_path: str) -> Dict:
    p = Path(preprocessor_meta_path)
    if not p.exists():
        raise FileNotFoundError(f"Preprocessor metadata not found: {preprocessor_meta_path}")
    meta = json.load(open(p, "r"))
    numeric_cols = meta.get("numeric", [])
    categorical_cols = meta.get("categorical", [])
    return {"numeric_cols": numeric_cols, "categorical_cols": categorical_cols}

# -----------------------------------------------------------------------------
# Dataset: TabularDataset
# -----------------------------------------------------------------------------
class TabularDataset(Dataset):
    """
    TabularDataset loads a preprocessed CSV and supplies:
      - numeric tensor (float32) shaped (num_numeric,)
      - categorical tensor (int64) shaped (num_categorical,)
      - optional target tensor (float or long)
    Expects preprocess to have encoded categorical columns as integers and
    numeric columns already transformed (e.g., quantile -> normal).
    """
    def __init__(self,
                 csv_path: str,
                 preprocessor_meta_path: str,
                 target_col: Optional[str] = None,
                 device: Optional[torch.device] = None):
        self.csv_path = Path(csv_path)
        self.preprocessor_meta_path = Path(preprocessor_meta_path)
        self.device = device or torch.device("cpu")
        self.target_col = target_col

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")
        meta = load_preprocessor_metadata(str(self.preprocessor_meta_path))
        self.numeric_cols: List[str] = meta["numeric_cols"]
        self.categorical_cols: List[str] = meta["categorical_cols"]

        self.df = pd.read_csv(self.csv_path)
        # Validate columns presence
        missing = [c for c in (self.numeric_cols + self.categorical_cols) if c not in self.df.columns]
        if missing:
            raise ValueError(f"Columns missing from CSV: {missing}. Check preprocess metadata and CSV.")

        # Optional: convert dtypes for speed
        for c in self.numeric_cols:
            self.df[c] = pd.to_numeric(self.df[c], errors="coerce").fillna(0.0).astype(np.float32)
        for c in self.categorical_cols:
            # categorical encoded as integers by preprocessor
            self.df[c] = pd.to_numeric(self.df[c], errors="coerce").fillna(0).astype(np.int64)

        if self.target_col is not None:
            if self.target_col not in self.df.columns:
                raise ValueError(f"target_col {self.target_col} not found in CSV")
            # keep as is (could be string), convert to numeric if possible
            try:
                self.df[self.target_col] = pd.to_numeric(self.df[self.target_col], errors="coerce").fillna(0)
            except Exception:
                pass

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        numeric_vals = row[self.numeric_cols].to_numpy(dtype=np.float32) if self.numeric_cols else np.zeros((0,), dtype=np.float32)
        cat_vals = row[self.categorical_cols].to_numpy(dtype=np.int64) if self.categorical_cols else np.zeros((0,), dtype=np.int64)
        sample = {
            "numeric": torch.from_numpy(numeric_vals).to(self.device),
            "categorical": torch.from_numpy(cat_vals).to(self.device),
        }
        if self.target_col:
            target = row[self.target_col]
            # choose dtype based on whether target is integer-like
            if float(target).is_integer():
                sample["target"] = torch.tensor(int(target), dtype=torch.long, device=self.device)
            else:
                sample["target"] = torch.tensor(float(target), dtype=torch.float32, device=self.device)
        return sample

# -----------------------------------------------------------------------------
# Collate function
# -----------------------------------------------------------------------------
def tabular_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate list of samples into batched tensors.
    Returns dict with keys: numeric (B x N), categorical (B x C), optional target.
    """
    if len(batch) == 0:
        return {}

    numeric_list = [b["numeric"] for b in batch]
    cat_list = [b["categorical"] for b in batch]
    numeric = torch.stack(numeric_list, dim=0) if numeric_list and numeric_list[0].numel() > 0 else torch.empty((len(batch), 0))
    categorical = torch.stack(cat_list, dim=0) if cat_list and cat_list[0].numel() > 0 else torch.empty((len(batch), 0), dtype=torch.long)

    batch_out = {"numeric": numeric, "categorical": categorical}

    if "target" in batch[0]:
        targets = []
        for b in batch:
            t = b["target"]
            # ensure consistent dtype
            targets.append(t)
        target_tensor = torch.stack(targets, dim=0)
        batch_out["target"] = target_tensor

    return batch_out

# -----------------------------------------------------------------------------
# Utility: create DataLoader
# -----------------------------------------------------------------------------
def make_dataloader(csv_path: str,
                    preprocessor_meta_path: str,
                    batch_size: int = 128,
                    shuffle: bool = True,
                    num_workers: int = 0,
                    pin_memory: bool = True,
                    target_col: Optional[str] = None,
                    device: Optional[torch.device] = None) -> Tuple[DataLoader, TabularDataset]:
    ds = TabularDataset(csv_path=csv_path, preprocessor_meta_path=preprocessor_meta_path, target_col=target_col, device=device)
    dl = DataLoader(ds,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    collate_fn=tabular_collate_fn)
    return dl, ds
