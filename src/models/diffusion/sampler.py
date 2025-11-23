# src/models/diffusion/sampler.py
"""
Clean, fixed sampler for EfficientTabDDPM (DDPM + DDIM).
"""

import os
import json
from typing import List, Optional
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.models.diffusion.forward import (
    DiffusionScheduler,
    ddpm_sample_step,
    ddim_sample_step,
)
from src.models.diffusion.denoiser import EfficientDenoiser


# ---------------------------------------------------------------------
# 1. Load checkpoint cleanly
# ---------------------------------------------------------------------
def load_checkpoint(path: str, device: torch.device):
    return torch.load(path, map_location=device)


# ---------------------------------------------------------------------
# 2. Build denoiser model from checkpoint
# ---------------------------------------------------------------------
def build_model_from_checkpoint(ckpt, input_dim, device, model_kwargs):
    model = EfficientDenoiser(input_dim=input_dim, **model_kwargs).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# ---------------------------------------------------------------------
# 3. Build categorical embeddings from checkpoint
# ---------------------------------------------------------------------
def build_embeddings_from_checkpoint(ckpt, categorical_cardinalities, embed_dim, device):
    embeddings = nn.ModuleList([
        nn.Embedding(n, embed_dim) for n in categorical_cardinalities
    ])

    # Load embeddings if saved
    if "embeddings_state_dict" in ckpt and ckpt["embeddings_state_dict"] is not None:
        try:
            embeddings.load_state_dict(ckpt["embeddings_state_dict"])
        except:
            pass

    embeddings.to(device)
    return embeddings, embed_dim * len(categorical_cardinalities)


# ---------------------------------------------------------------------
# 4. Decode categorical embeddings by nearest neighbor
# ---------------------------------------------------------------------
def decode_categorical_from_embeddings(embeddings_module, generated_cat_emb):
    device = generated_cat_emb.device
    B = generated_cat_emb.shape[0]

    cat_values = []
    offset = 0

    for emb in embeddings_module:
        weight = emb.weight.detach()  # (V, E)
        E = weight.shape[1]
        slice_emb = generated_cat_emb[:, offset:offset+E]

        # distances via ||x||^2 + ||v||^2 - 2 x.v
        x2 = (slice_emb**2).sum(dim=1, keepdim=True)
        v2 = (weight**2).sum(dim=1).unsqueeze(0)
        xv = slice_emb @ weight.t()

        d2 = x2 + v2 - 2 * xv
        idx = torch.argmin(d2, dim=1).cpu().numpy()

        cat_values.append(idx)
        offset += E

    if len(cat_values) == 0:
        return np.zeros((B, 0), dtype=np.int64)

    return np.stack(cat_values, axis=1)


# ---------------------------------------------------------------------
# 5. Decode numeric + categorical back to dataframe
# ---------------------------------------------------------------------
def decode_to_dataframe(numeric_arr, categorical_arr, preprocessor_dir, numeric_cols, categorical_cols):
    out = {}

    # numeric inverse quantile
    for i, col in enumerate(numeric_cols):
        qt_path = Path(preprocessor_dir) / f"quantile_{col}.joblib"
        if qt_path.exists():
            qt = joblib.load(str(qt_path))
            out[col] = qt.inverse_transform(numeric_arr[:, i:i+1]).reshape(-1)
        else:
            out[col] = numeric_arr[:, i]

    # categorical inverse transform
    for j, col in enumerate(categorical_cols):
        le_path = Path(preprocessor_dir) / f"labelenc_{col}.joblib"
        if le_path.exists():
            le = joblib.load(str(le_path))
            out[col] = le.inverse_transform(categorical_arr[:, j].astype(int))
        else:
            out[col] = categorical_arr[:, j]

    return pd.DataFrame(out)


# ---------------------------------------------------------------------
# 6. DDPM sampling loop
# ---------------------------------------------------------------------
@torch.no_grad()
def sample_ddpm(model, scheduler, num_samples, input_dim, device, noise_scale=1.0):
    x = torch.randn(num_samples, input_dim, device=device)

    for t_int in reversed(range(scheduler.timesteps)):
        t = torch.full((num_samples,), t_int, device=device, dtype=torch.long)
        x, _, _ = ddpm_sample_step(scheduler, model, x, t, noise_scale=noise_scale)

    return x.cpu()


# ---------------------------------------------------------------------
# 7. DDIM sampling loop
# ---------------------------------------------------------------------
@torch.no_grad()
def sample_ddim(model, scheduler, num_samples, input_dim, device, ddim_steps=50, eta=0.0):
    x = torch.randn(num_samples, input_dim, device=device)
    T = scheduler.timesteps

    # build reduced steps
    if ddim_steps >= T:
        steps = list(range(T - 1, -1, -1))
    else:
        steps = list(np.linspace(T - 1, 0, ddim_steps, dtype=int))

    # iterate
    for i in range(len(steps) - 1):
        t_cur = steps[i]
        t_next = steps[i + 1]
        x, _, _ = ddim_sample_step(scheduler, model, x, t_cur, t_next, eta=eta)

    # final step to t=0
    if steps[-1] != 0:
        x, _, _ = ddim_sample_step(scheduler, model, x, steps[-1], 0, eta=eta)

    return x.cpu()


# ---------------------------------------------------------------------
# 8. High-level: sample + decode
# ---------------------------------------------------------------------
def sample_and_decode(
    checkpoint_path: str,
    preprocessor_dir: str,
    output_csv: str,
    num_samples: int = 100,
    method: str = "ddim",
    ddim_steps: int = 50,
    eta: float = 0.0,
    device: Optional[torch.device] = None,
    embed_dim: int = 8,
    model_kwargs: dict = None
):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    # ---- load checkpoint ----
    ckpt = load_checkpoint(checkpoint_path, device)

    # ---- load metadata.json correctly ----
    meta_path = os.path.join(preprocessor_dir, "metadata.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    numeric_cols = meta["numeric"]
    categorical_cols = meta["categorical"]

    numeric_dim = len(numeric_cols)

    # ---- find cardinalities ----
    categorical_cardinalities = []
    for col in categorical_cols:
        le_path = Path(preprocessor_dir) / f"labelenc_{col}.joblib"
        if le_path.exists():
            le = joblib.load(str(le_path))
            categorical_cardinalities.append(len(le.classes_))
        else:
            categorical_cardinalities.append(10)

    cat_emb_total = embed_dim * len(categorical_cardinalities)
    input_dim = numeric_dim + cat_emb_total

    # ---- build model ----
    model_kwargs = model_kwargs or {}
    model = EfficientDenoiser(input_dim=input_dim, **model_kwargs).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ---- build embeddings ----
    embeddings_module, _ = build_embeddings_from_checkpoint(
        ckpt, categorical_cardinalities, embed_dim, device
    )

    # ---- scheduler ----
    sched_cfg = ckpt.get("scheduler_cfg", {})
    timesteps = sched_cfg.get("timesteps", 1000)
    schedule = sched_cfg.get("schedule", "linear")

    scheduler = DiffusionScheduler(timesteps, schedule=schedule, device=device).to(device)

    # ---- sampling ----
    if method == "ddpm":
        gen = sample_ddpm(model, scheduler, num_samples, input_dim, device)
    else:
        gen = sample_ddim(model, scheduler, num_samples, input_dim, device, ddim_steps, eta)

    gen = gen.numpy()

    # ---- split numeric + embeddings ----
    if len(categorical_cols) > 0:
        numeric_arr = gen[:, :numeric_dim]
        cat_emb_arr = gen[:, numeric_dim:]

        # nearest-neighbor decode
        cat_arr = decode_categorical_from_embeddings(
            embeddings_module, torch.from_numpy(cat_emb_arr).to(device)
        )
    else:
        numeric_arr = gen
        cat_arr = np.zeros((gen.shape[0], 0), dtype=np.int64)

    # ---- inverse transforms ----
    decoded_df = decode_to_dataframe(
        numeric_arr,
        cat_arr,
        preprocessor_dir,
        numeric_cols,
        categorical_cols,
    )

    decoded_df.to_csv(output_csv, index=False)
    print(f"Saved {num_samples} samples to {output_csv}")

    return decoded_df
