# src/models/diffusion/sampler.py
"""
Robust DDPM + DDIM sampler for EfficientTabDDPM.

This file is a fully working, defensive implementation intended to be dropped
into src/models/diffusion/sampler.py. It:
 - loads checkpoints robustly (partial-loading tolerated)
 - infers model architecture where possible
 - supports DDPM ancestral sampling and DDIM deterministic/semi-stochastic sampling
 - decodes numeric features via saved QuantileTransformer joblib files
 - decodes categorical features by nearest-neighbor in embedding space using saved LabelEncoders

Notes:
 - Make sure `--embed_dim` used here matches what you used during training (default 8).
 - The sampler will try to reuse embeddings saved in checkpoint under
   "embeddings_state_dict" if present. Otherwise it builds new embedding modules
   using label encoder cardinalities discovered in preprocessor directory.

"""

import os
import json
from typing import List, Optional, Dict, Any, Tuple
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

# Example runtime-uploaded resource paths (present in some runtimes)
RESOURCE_FILE_1 = "/mnt/data/EE782 2025 A1 Image Captioning.pdf"
RESOURCE_FILE_2 = "/mnt/data/EE782_25_L01_Intro_to_AML.pdf"


# -------------------------
# Checkpoint load helpers
# -------------------------
def load_checkpoint(checkpoint_path: str, device: torch.device) -> Dict[str, Any]:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    return ckpt


def safe_load_model_state(model: torch.nn.Module, ckpt_state: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
    """
    Copy matching parameters from ckpt_state into model.state_dict(). Tolerates:
     - missing keys
     - extra keys in checkpoint
     - keys with shape mismatch (skipped)
    Returns a report dict with lists of loaded / skipped keys and shape mismatches.
    """
    model_state = model.state_dict()
    loaded_keys = []
    skipped_keys = []
    shape_mismatch = []

    for k, v in ckpt_state.items():
        if k in model_state:
            try:
                if isinstance(v, torch.Tensor):
                    if v.shape == model_state[k].shape:
                        model_state[k] = v.clone().to(model_state[k].device)
                        loaded_keys.append(k)
                    else:
                        shape_mismatch.append((k, tuple(v.shape), tuple(model_state[k].shape)))
                else:
                    vt = torch.as_tensor(v)
                    if vt.shape == model_state[k].shape:
                        model_state[k] = vt.to(model_state[k].device)
                        loaded_keys.append(k)
                    else:
                        shape_mismatch.append((k, tuple(vt.shape), tuple(model_state[k].shape)))
            except Exception:
                skipped_keys.append(k)
        else:
            skipped_keys.append(k)

    # Load with strict=False to allow for missing keys in either direction
    model.load_state_dict(model_state, strict=False)

    report = {
        "loaded": loaded_keys,
        "skipped": skipped_keys,
        "shape_mismatch": shape_mismatch,
    }
    if verbose:
        print(f"safe_load_model_state: loaded {len(loaded_keys)} keys; skipped {len(skipped_keys)} keys; shape mismatches {len(shape_mismatch)}")
        if shape_mismatch:
            print("Shape mismatches (first 20 shown):")
            for item in shape_mismatch[:20]:
                print("  ", item)
        if skipped_keys:
            print("Some checkpoint keys were not present in the model (first 20 shown):", skipped_keys[:20])
    return report


def infer_model_kwargs_from_state(ckpt_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Try to infer common model kwargs (hidden_dim, num_layers, time_emb_dim) from a checkpoint state dict.
    Returns a dict with any discovered values.
    """
    inferred = {}
    try:
        # hidden_dim: try to infer from input_proj.weight (shape hidden_dim x input_dim)
        if "input_proj.weight" in ckpt_state:
            inferred["hidden_dim"] = int(ckpt_state["input_proj.weight"].shape[0])

        # num_layers: find max index in keys like layers.{i}.
        layer_idxs = set()
        for k in ckpt_state.keys():
            if k.startswith("layers."):
                parts = k.split(".")
                if len(parts) > 1 and parts[1].isdigit():
                    layer_idxs.add(int(parts[1]))
        if layer_idxs:
            inferred["num_layers"] = int(max(layer_idxs) + 1)

        # time_emb_dim: inspect time_emb.proj.0.weight shape if present
        # In our TimeEmbedding, proj is Linear(dim, dim*4) then Linear(dim*4, dim)
        if "time_emb.proj.0.weight" in ckpt_state:
            w = ckpt_state["time_emb.proj.0.weight"]
            # weight shape: (dim*4, dim) -> second dimension is time_emb_dim
            if hasattr(w, "shape") and len(w.shape) >= 2:
                inferred["time_emb_dim"] = int(w.shape[1])
    except Exception:
        pass
    return inferred


# -------------------------
# Embeddings building / loading
# -------------------------
def build_embeddings_from_cardinalities(cardinalities: List[int], embed_dim: int, device: torch.device) -> Tuple[nn.ModuleList, int]:
    """
    Build nn.ModuleList of embeddings and return (modulelist, total_dim)
    """
    embeddings = nn.ModuleList([nn.Embedding(n, embed_dim) for n in cardinalities])
    embeddings.to(device)
    total = embed_dim * len(cardinalities)
    return embeddings, total


def try_load_embeddings_state(embeddings_module: nn.ModuleList, ckpt: Dict[str, Any], verbose: bool = True) -> None:
    """
    Attempt to load embeddings_state_dict from checkpoint into embeddings_module.
    """
    if "embeddings_state_dict" in ckpt and ckpt["embeddings_state_dict"] is not None:
        try:
            embeddings_module.load_state_dict(ckpt["embeddings_state_dict"])
            if verbose:
                print("Loaded embeddings_state_dict from checkpoint.")
        except Exception as e:
            if verbose:
                print("Could not load embeddings_state_dict from checkpoint (ignored). Err:", e)


# -------------------------
# Decoding helpers
# -------------------------
def decode_categorical_from_embeddings(embeddings_module: nn.ModuleList, generated_cat_emb: torch.Tensor) -> np.ndarray:
    """
    Given generated_cat_emb (B, total_cat_emb_dim) and embeddings_module (list of nn.Embedding),
    decode to integer labels by nearest-neighbor in embedding space.
    Returns array shape (B, num_cat).
    """
    device = generated_cat_emb.device
    B = generated_cat_emb.shape[0]
    cat_values = []
    offset = 0
    for emb in embeddings_module:
        weight = emb.weight.detach()  # (V, E)
        E = weight.shape[1]
        slice_emb = generated_cat_emb[:, offset:offset + E]  # (B, E)
        # pairwise squared distances: ||x||^2 + ||v||^2 - 2 x.v
        x2 = (slice_emb ** 2).sum(dim=1, keepdim=True)  # (B,1)
        v2 = (weight ** 2).sum(dim=1).unsqueeze(0)  # (1,V)
        xv = slice_emb @ weight.t()  # (B,V)
        d2 = x2 + v2 - 2.0 * xv
        idx = torch.argmin(d2, dim=1).cpu().numpy()  # (B,)
        cat_values.append(idx)
        offset += E
    if len(cat_values) == 0:
        return np.zeros((B, 0), dtype=np.int64)
    return np.stack(cat_values, axis=1)


def decode_to_dataframe(numeric_arr: np.ndarray,
                        categorical_arr: np.ndarray,
                        preprocessor_dir: str,
                        numeric_cols: List[str],
                        categorical_cols: List[str]) -> pd.DataFrame:
    """
    numeric_arr: (B, num_numeric) normalized space (quantile->normal)
    categorical_arr: (B, num_cat) integer labels
    preprocessor_dir: directory containing quantile_<col>.joblib and labelenc_<col>.joblib
    """
    out = {}
    pre = Path(preprocessor_dir)
    # numeric
    for i, col in enumerate(numeric_cols):
        qt_path = pre / f"quantile_{col}.joblib"
        if qt_path.exists():
            qt = joblib.load(str(qt_path))
            inv = qt.inverse_transform(numeric_arr[:, i:i + 1]).reshape(-1)
            out[col] = inv
        else:
            out[col] = numeric_arr[:, i]
    # categorical
    for j, col in enumerate(categorical_cols):
        le_path = pre / f"labelenc_{col}.joblib"
        if le_path.exists():
            le = joblib.load(str(le_path))
            vals = le.inverse_transform(categorical_arr[:, j].astype(int))
            out[col] = vals
        else:
            out[col] = categorical_arr[:, j]
    return pd.DataFrame(out)


# -------------------------
# Sampling loops
# -------------------------
@torch.no_grad()
def sample_ddpm(model: nn.Module, scheduler: DiffusionScheduler, num_samples: int, input_dim: int, device: torch.device, noise_scale: float = 1.0) -> torch.Tensor:
    """
    Standard DDPM ancestral sampling (stochastic).
    Returns Tensor (num_samples, input_dim) on CPU.
    """
    model.eval()
    x = torch.randn(num_samples, input_dim, device=device)
    for t_int in reversed(range(scheduler.timesteps)):
        t = torch.full((num_samples,), t_int, device=device, dtype=torch.long)
        x, _, _ = ddpm_sample_step(scheduler, model, x, t, noise_scale=noise_scale)
    return x.cpu()


@torch.no_grad()
def sample_ddim(model: nn.Module, scheduler: DiffusionScheduler, num_samples: int, input_dim: int, device: torch.device, ddim_steps: int = 50, eta: float = 0.0) -> torch.Tensor:
    """
    DDIM sampling (deterministic if eta==0).
    Returns Tensor (num_samples, input_dim) on CPU.
    """
    model.eval()
    x = torch.randn(num_samples, input_dim, device=device)
    T = scheduler.timesteps
    if ddim_steps >= T:
        steps = list(range(T - 1, -1, -1))
    else:
        steps = list(np.linspace(T - 1, 0, ddim_steps, dtype=int))
    for i in range(len(steps) - 1):
        t_cur = int(steps[i])
        t_next = int(steps[i + 1])
        x, _, _ = ddim_sample_step(scheduler, model, x, t_cur, t_next, eta=eta)
    if steps[-1] != 0:
        x, _, _ = ddim_sample_step(scheduler, model, x, steps[-1], 0, eta=eta)
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
                      hidden_dim: int = 128, # Added argument to function signature
                      num_layers: int = 4,   # Added argument to function signature
                      model_kwargs: Optional[Dict[str, Any]] = None,
                      scheduler_kwargs: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Load checkpoint, sample with DDIM or DDPM, and decode to original tabular CSV.
    - checkpoint_path: path to the .pth checkpoint saved by training script
    - preprocessor_dir: directory containing metadata.json and joblib files
    - output_csv: where to save generated CSV
    - method: 'ddim' or 'ddpm'
    - embed_dim: embedding dimension used during training
    - hidden_dim: Hidden dimension for the denoiser model
    - num_layers: Number of layers for the denoiser model
    - model_kwargs: optional extra kwargs for EfficientDenoiser
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    ckpt = load_checkpoint(checkpoint_path, device)

    # load metadata.json safely
    meta_path = Path(preprocessor_dir) / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {preprocessor_dir}")
    meta = json.load(open(str(meta_path), "r"))
    numeric_cols = meta.get("numeric", [])
    categorical_cols = meta.get("categorical", [])

    # infer cardinalities from labelenc files
    categorical_cardinalities: List[int] = []
    for col in categorical_cols:
        p = Path(preprocessor_dir) / f"labelenc_{col}.joblib"
        if p.exists():
            le = joblib.load(str(p))
            if hasattr(le, "classes_"):
                categorical_cardinalities.append(int(len(le.classes_)))
            else:
                # fallback
                categorical_cardinalities.append(10)
        else:
            categorical_cardinalities.append(10)

    numeric_dim = len(numeric_cols)
    total_cat_emb_dim = embed_dim * len(categorical_cardinalities)
    input_dim = numeric_dim + total_cat_emb_dim

    # try to infer model kwargs from checkpoint, and merge with provided model_kwargs
    ckpt_state = ckpt.get("model_state_dict", {}) if isinstance(ckpt, dict) else {}
    inferred = infer_model_kwargs_from_state(ckpt_state)
    model_kwargs_final = {
        "hidden_dim": hidden_dim, # Pass to model_kwargs_final
        "num_layers": num_layers, # Pass to model_kwargs_final
    }
    if model_kwargs:
        model_kwargs_final.update(model_kwargs)
    model_kwargs_final.update(inferred)

    # ensure some defaults for robustness
    # The explicit passing above handles this, but keep for robustness if `hidden_dim`/`num_layers` were not passed explicitly
    if "hidden_dim" not in model_kwargs_final:
        model_kwargs_final["hidden_dim"] = model_kwargs_final.get("hidden_dim", 128)
    if "num_layers" not in model_kwargs_final:
        model_kwargs_final["num_layers"] = model_kwargs_final.get("num_layers", 6)
    if "time_emb_dim" not in model_kwargs_final:
        model_kwargs_final["time_emb_dim"] = model_kwargs_final.get("time_emb_dim", 128)

    print("Sampler: building model with kwargs:", model_kwargs_final)
    model = EfficientDenoiser(input_dim=input_dim, **model_kwargs_final).to(device)

    # load model weights safely (partial loads allowed)
    if "model_state_dict" in ckpt:
        print("Sampler: loading checkpoint state dict with safe loader...")
        safe_load_model_state(model, ckpt["model_state_dict"], verbose=True)
    else:
        # If entire checkpoint is a state_dict directly
        safe_load_model_state(model, ckpt if isinstance(ckpt, dict) else {}, verbose=True)

    model.eval()

    # build embeddings and try to load checkpointed embeddings if available
    embeddings_module, _ = build_embeddings_from_cardinalities(categorical_cardinalities, embed_dim, device)
    try_load_embeddings_state(embeddings_module, ckpt, verbose=True)

    # scheduler
    sched_cfg = {}
    if isinstance(ckpt, dict) and "scheduler_cfg" in ckpt:
        sched_cfg = ckpt["scheduler_cfg"]
    if scheduler_kwargs:
        sched_cfg.update(scheduler_kwargs)
    timesteps = int(sched_cfg.get("timesteps", 1000))
    schedule = str(sched_cfg.get("schedule", "linear"))
    scheduler = DiffusionScheduler(timesteps=timesteps, schedule=schedule, device=device).to(device)

    # sampling
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    if method.lower() == "ddpm":
        generated = sample_ddpm(model, scheduler, num_samples, input_dim, device, noise_scale=1.0)
    elif method.lower() == "ddim":
        generated = sample_ddim(model, scheduler, num_samples, input_dim, device, ddim_steps=ddim_steps, eta=eta)
    else:
        raise ValueError("Unknown sampling method: choose 'ddim' or 'ddpm'")

    gen = generated.numpy()

    # split numeric and categorical-embeddings
    if len(categorical_cardinalities) > 0:
        numeric_part = gen[:, :numeric_dim]
        cat_emb_part = gen[:, numeric_dim:]
        # decode embeddings to categories
        cat_arr = decode_categorical_from_embeddings(embeddings_module, torch.from_numpy(cat_emb_part).to(device))
    else:
        numeric_part = gen
        cat_arr = np.zeros((gen.shape[0], 0), dtype=np.int64)

    # inverse transform to original values
    decoded_df = decode_to_dataframe(numeric_part, cat_arr, preprocessor_dir, numeric_cols, categorical_cols)

    decoded_df.to_csv(output_csv, index=False)
    print(f"Saved {len(decoded_df)} generated rows to {output_csv}")

    return decoded_df


# If invoked directly (for quick debugging), allow a tiny CLI
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--preprocessor_dir", type=str, required=True)
    parser.add_argument("--out_csv", type=str, default="results/samples/generated.csv")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--method", type=str, default="ddim", choices=["ddim", "ddpm"])
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--embed_dim", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=128) # Added argument to internal CLI
    parser.add_argument("--num_layers", type=int, default=4)   # Added argument to internal CLI
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    sample_and_decode(checkpoint_path=args.checkpoint,
                      preprocessor_dir=args.preprocessor_dir,
                      output_csv=args.out_csv,
                      num_samples=args.num_samples,
                      method=args.method,
                      ddim_steps=args.ddim_steps,
                      eta=args.eta,
                      device=device,
                      embed_dim=args.embed_dim,
                      hidden_dim=args.hidden_dim, # Pass argument to sample_and_decode
                      num_layers=args.num_layers) # Pass argument to sample_and_decode
