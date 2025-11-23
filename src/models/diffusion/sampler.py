# src/models/diffusion/sampler.py
"""
Sampler utilities for EfficientTabDDPM
Supports:
 - ddpm ancestral sampling (stochastic)
 - ddim deterministic / semi-deterministic sampling
Also includes utilities to:
 - load checkpoint (model + embeddings)
 - decode generated vectors back to tabular (inverse quantile + label decoding)
"""

import os
import math
from typing import List, Tuple, Optional
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.models.diffusion.forward import DiffusionScheduler, ddpm_sample_step, ddim_sample_step
from src.models.diffusion.denoiser import EfficientDenoiser

# -------------------------
# Checkpoint loader
# -------------------------
def load_checkpoint(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    return ckpt

def build_model_from_checkpoint(ckpt: dict, input_dim: int, device: torch.device, model_kwargs: dict):
    model = EfficientDenoiser(input_dim=input_dim, **model_kwargs).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

def build_embeddings_from_checkpoint(ckpt: dict, categorical_cardinalities: List[int], embed_dim: int, device: torch.device):
    """
    Build ModuleList of embeddings and load state if available in checkpoint.
    Return (embeddings_module, embedding_dim_total)
    """
    embeddings = nn.ModuleList([nn.Embedding(n, embed_dim) for n in categorical_cardinalities])
    # load states if present
    if ckpt.get("embeddings_state_dict", None) is not None:
        try:
            embeddings_state = ckpt["embeddings_state_dict"]
            embeddings.load_state_dict(embeddings_state)
        except Exception:
            # sometimes embeddings saved differently; try best-effort
            pass
    embeddings.to(device)
    total_dim = embed_dim * len(categorical_cardinalities)
    return embeddings, total_dim

# -------------------------
# Decoding helpers
# -------------------------
def decode_categorical_from_embeddings(embeddings_module: nn.ModuleList, generated_cat_emb: torch.Tensor) -> np.ndarray:
    """
    Given generated_cat_emb (B, total_cat_emb_dim), and embeddings_module (list of embedding layers),
    compute the nearest neighbor in embedding weight space for each categorical column.
    Returns numpy array of shape (B, num_cat)
    """
    device = generated_cat_emb.device
    B = generated_cat_emb.shape[0]
    cat_values = []
    offset = 0
    for i, emb in enumerate(embeddings_module):
        emb_weight = emb.weight.detach()  # (V, E)
        E = emb_weight.shape[1]
        slice_emb = generated_cat_emb[:, offset:offset+E]  # (B, E)
        # compute distances to all category vectors (V)
        # use broadcasting: (B, 1, E) - (1, V, E) -> (B, V, E)
        # but do it efficiently:
        # dist^2 = ||x||^2 + ||v||^2 - 2 x.v
        x2 = (slice_emb**2).sum(dim=1, keepdim=True)  # (B,1)
        v2 = (emb_weight**2).sum(dim=1).unsqueeze(0)   # (1,V)
        xv = slice_emb @ emb_weight.t()                # (B, V)
        d2 = x2 + v2 - 2.0 * xv
        idx = torch.argmin(d2, dim=1).cpu().numpy()   # (B,)
        cat_values.append(idx)
        offset += E
    # stack as (B, num_cat)
    cat_arr = np.stack(cat_values, axis=1) if len(cat_values) > 0 else np.zeros((B,0), dtype=np.int64)
    return cat_arr

def decode_to_dataframe(numeric_arr: np.ndarray,
                        categorical_arr: np.ndarray,
                        preprocessor_dir: str,
                        numeric_cols: List[str],
                        categorical_cols: List[str]) -> pd.DataFrame:
    """
    numeric_arr: (B, num_numeric) in normalized space (quantile->normal)
    categorical_arr: (B, num_cat) ints (label indices)
    preprocessor_dir: path to directory where 'quantile_<col>.joblib' and 'labelenc_<col>.joblib' live
    """
    out = {}
    # numeric inverse transform
    for i, col in enumerate(numeric_cols):
        qt_path = Path(preprocessor_dir) / f"quantile_{col}.joblib"
        if qt_path.exists():
            qt = joblib.load(str(qt_path))
            inv = qt.inverse_transform(numeric_arr[:, i].reshape(-1,1)).reshape(-1)
            out[col] = inv
        else:
            # fallback: raw values
            out[col] = numeric_arr[:, i]
    # categorical inverse mapping
    for j, col in enumerate(categorical_cols):
        le_path = Path(preprocessor_dir) / f"labelenc_{col}.joblib"
        if le_path.exists():
            le = joblib.load(str(le_path))
            vals = le.inverse_transform(categorical_arr[:, j].astype(int))
            out[col] = vals
        else:
            out[col] = categorical_arr[:, j]
    df = pd.DataFrame(out)
    return df

# -------------------------
# Sampling routines
# -------------------------
@torch.no_grad()
def sample_ddpm(model: nn.Module,
                embeddings_module: Optional[nn.ModuleList],
                scheduler: DiffusionScheduler,
                num_samples: int,
                input_dim: int,
                numeric_dim: int,
                categorical_dim_total: int,
                device: torch.device,
                noise_scale: float = 1.0) -> torch.Tensor:
    """
    Standard DDPM ancestral sampling loop.
    Returns tensor of shape (num_samples, input_dim) in normalized space.
    Partition of input_dim: [numeric_dim | categorical_embs_total]
    """
    model.eval()
    B = num_samples
    x = torch.randn(B, input_dim, device=device)
    for t_int in reversed(range(scheduler.timesteps)):
        t = torch.full((B,), t_int, device=device, dtype=torch.long)
        x, x0_pred, eps = ddpm_sample_step(scheduler, model, x, t, noise_scale=noise_scale)
    return x.cpu()

@torch.no_grad()
def sample_ddim(model: nn.Module,
                embeddings_module: Optional[nn.ModuleList],
                scheduler: DiffusionScheduler,
                num_samples: int,
                input_dim: int,
                numeric_dim: int,
                categorical_dim_total: int,
                device: torch.device,
                ddim_steps: Optional[List[int]] = None,
                eta: float = 0.0) -> torch.Tensor:
    """
    DDIM sampling from t = T-1 to 0 using DDIM deterministic/semi-stochastic transitions.
    ddim_steps: list of timesteps to step through (e.g., np.linspace(T-1,0,K, dtype=int))
    If ddim_steps is None, we use full schedule (every timestep).
    Returns (B, input_dim)
    """
    model.eval()
    B = num_samples
    x = torch.randn(B, input_dim, device=device)
    T = scheduler.timesteps
    if ddim_steps is None:
        seq = list(reversed(range(T)))
    else:
        seq = list(reversed(list(ddim_steps)))
    # we need to step from seq[0]=T-1 down to last-> maybe 0
    for i in range(len(seq)-1):
        t_cur = int(seq[i])
        t_next = int(seq[i+1])
        x, x0_pred, eps = ddim_sample_step(scheduler, model, x, t_cur, t_next, eta=eta)
    # final step to t=0 if not included
    if seq[-1] != 0:
        x, x0_pred, eps = ddim_sample_step(scheduler, model, x, seq[-1], 0, eta=eta)
    return x.cpu()

# -------------------------
# High-level sampling + decode
# -------------------------
def sample_and_decode(checkpoint_path: str,
                      preprocessor_dir: str,
                      output_csv: str,
                      num_samples: int = 100,
                      method: str = "ddim",
                      ddim_steps: int = 50,
                      eta: float = 0.0,
                      device: Optional[torch.device] = None,
                      embed_dim: int = 8,
                      model_kwargs: dict = None,
                      scheduler_kwargs: dict = None):
    """
    checkpoint_path: path to saved checkpoint (torch .pth)
    preprocessor_dir: directory where metadata.json and quantile_/labelenc_*.joblib live
                     e.g., data/processed/adult/preprocessor
    output_csv: path to save decoded CSV
    method: 'ddpm' or 'ddim'
    ddim_steps: number of steps to use if method == 'ddim'
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    ckpt = load_checkpoint(checkpoint_path, device=device)

    # read metadata
    meta = joblib.load(open(os.path.join(preprocessor_dir, "metadata.json"), "rb")) if os.path.exists(os.path.join(preprocessor_dir, "metadata.json")) else None
    # fallback to json
    import json
    meta = json.load(open(os.path.join(preprocessor_dir, "metadata.json"), "r"))
    numeric_cols = meta.get("numeric", [])
    categorical_cols = meta.get("categorical", [])
    # infer categorical cardinalities from saved label encoders (joblib files)
    categorical_cardinalities = []
    for col in categorical_cols:
        p = Path(preprocessor_dir) / f"labelenc_{col}.joblib"
        if p.exists():
            le = joblib.load(str(p))
            # label encoder has classes_
            cardinality = len(le.classes_)
        else:
            # fallback: try to read max value from ckpt info? default 10
            cardinality = 10
        categorical_cardinalities.append(cardinality)

    numeric_dim = len(numeric_cols)
    total_cat_emb_dim = embed_dim * len(categorical_cardinalities)
    input_dim = numeric_dim + total_cat_emb_dim

    # build model and embeddings
    model_kwargs = model_kwargs or {}
    model = EfficientDenoiser(input_dim=input_dim, **model_kwargs).to(device)
    # load model state if available
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    embeddings_module, _ = build_embeddings_from_checkpoint(ckpt, categorical_cardinalities, embed_dim, device)
    # ensure embeddings loaded
    try:
        if ckpt.get("embeddings_state_dict", None) is not None:
            embeddings_module.load_state_dict(ckpt["embeddings_state_dict"])
    except Exception:
        pass
    embeddings_module.to(device)
    # scheduler
    scheduler_kwargs = scheduler_kwargs or {}
    timesteps = ckpt.get("scheduler_cfg", {}).get("timesteps", scheduler_kwargs.get("timesteps", 1000))
    schedule = ckpt.get("scheduler_cfg", {}).get("schedule", scheduler_kwargs.get("schedule", "linear"))
    scheduler = DiffusionScheduler(timesteps=timesteps, schedule=schedule, device=device).to(device)

    # sampling
    if method.lower() == "ddpm":
        generated = sample_ddpm(model=model,
                                embeddings_module=embeddings_module,
                                scheduler=scheduler,
                                num_samples=num_samples,
                                input_dim=input_dim,
                                numeric_dim=numeric_dim,
                                categorical_dim_total=total_cat_emb_dim,
                                device=device)
    elif method.lower() == "ddim":
        # build reduced timestep list
        T = scheduler.timesteps
        if ddim_steps >= T:
            ddim_ts = list(range(T-1, -1, -1))
        else:
            # uniform spaced timesteps including T-1 and 0
            ddim_ts = list(np.linspace(T-1, 0, ddim_steps, dtype=int))
        generated = sample_ddim(model=model,
                                embeddings_module=embeddings_module,
                                scheduler=scheduler,
                                num_samples=num_samples,
                                input_dim=input_dim,
                                numeric_dim=numeric_dim,
                                categorical_dim_total=total_cat_emb_dim,
                                device=device,
                                ddim_steps=ddim_ts,
                                eta=eta)
    else:
        raise ValueError("Unknown sampling method: choose 'ddpm' or 'ddim'")

    # generated: (B, input_dim) tensor on CPU
    gen = generated.numpy()
    # split numeric vs embeddings
    if total_cat_emb_dim > 0:
        numeric_part = gen[:, :numeric_dim]
        cat_emb_part = gen[:, numeric_dim:]
        # decode embeddings to categories
        cat_arr = decode_categorical_from_embeddings(embeddings_module, torch.from_numpy(cat_emb_part).to(device))
    else:
        numeric_part = gen
        cat_arr = np.zeros((gen.shape[0], 0), dtype=np.int64)

    # inverse-transform numeric & categorical to original values
    decoded_df = decode_to_dataframe(numeric_part, cat_arr, preprocessor_dir, numeric_cols, categorical_cols)
    decoded_df.to_csv(output_csv, index=False)
    print(f"Saved {num_samples} generated rows to {output_csv}")
    return decoded_df

