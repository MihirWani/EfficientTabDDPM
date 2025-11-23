# src/training/train_diffusion.py
"""
Train script for EfficientTabDDPM (lightweight tabular diffusion).
Usage (from repo root in Colab):
    python -m src.training.train_diffusion --config configs/adult.yaml
Or run with CLI args (see below).
"""

import argparse
import os
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Import project modules (ensure repo root is on PYTHONPATH)
from src.data.loaders import make_dataloader, load_preprocessor_metadata
from src.models.diffusion.denoiser import EfficientDenoiser
from src.models.diffusion.forward import DiffusionScheduler, q_sample

# --------------------------
# Simple EMA helper
# --------------------------
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        # initialize
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    def update(self, model: nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert name in self.shadow
                self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    def apply_shadow(self, model: nn.Module):
        self._backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self._backup[name] = p.detach().clone()
                p.data.copy_(self.shadow[name].data)

    def restore(self, model: nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self._backup[name].data)
        self._backup = {}

# --------------------------
# Utility: build embeddings for categorical columns
# --------------------------
def build_categorical_embeddings(dataset, categorical_cols: List[str], embed_dim: int = 8, device=None) -> nn.Module:
    """
    For each categorical column, estimate cardinality from dataset and create an nn.ModuleList of embeddings.
    dataset: TabularDataset object (from src.data.loaders)
    categorical_cols: list of categorical column names (order must match preprocessor metadata)
    Returns: nn.Module (nn.ModuleList) and a helper function embed_cat(cat_tensor) -> (B, total_cat_emb_dim)
    """
    # infer max categories per column from dataset dataframe
    # dataset.df exists on TabularDataset
    max_vals = []
    for c in categorical_cols:
        max_v = int(dataset.df[c].max()) if c in dataset.df.columns else 1
        max_vals.append(max(2, max_v + 1))
    embeddings = nn.ModuleList([nn.Embedding(n, embed_dim) for n in max_vals])
    # device placement done later by moving embeddings.to(device)
    def embed_fn(cat_tensor: torch.Tensor):
        # cat_tensor: (B, num_cat)
        parts = []
        for i in range(cat_tensor.shape[1]):
            parts.append(embeddings[i](cat_tensor[:, i]))
        if parts:
            return torch.cat(parts, dim=1)
        else:
            # return empty tensor if no categorical columns
            return torch.zeros((cat_tensor.shape[0], 0), device=cat_tensor.device)
    return embeddings, embed_fn

# --------------------------
# Training step
# --------------------------
def train_one_epoch(model: nn.Module,
                    embeddings_module: nn.Module,
                    embed_fn,
                    dataloader: DataLoader,
                    scheduler: DiffusionScheduler,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    ema: EMA = None,
                    clip_grad: float = 1.0,
                    log_every: int = 100) -> dict:
    model.train()
    total_loss = 0.0
    n_samples = 0
    pbar = tqdm(dataloader, desc="train")
    for step, batch in enumerate(pbar):
        numeric = batch["numeric"].to(device)  # (B, N_num)
        categorical = batch["categorical"].to(device)  # (B, N_cat) ints

        # build input vector x0: numeric + categorical embeddings
        if categorical.numel() > 0:
            # ensure embeddings module on device
            embeddings_module.to(device)
            cat_embs = []
            for i in range(categorical.shape[1]):
                emb_layer = embeddings_module[i]
                cat_embs.append(emb_layer(categorical[:, i]))
            cat_emb = torch.cat(cat_embs, dim=1)
            x0 = torch.cat([numeric, cat_emb], dim=1)
        else:
            x0 = numeric

        B = x0.shape[0]
        # sample random timesteps t uniformly in [0, T-1]
        t = torch.randint(low=0, high=scheduler.timesteps, size=(B,), device=device)

        # sample noise eps and get x_t
        eps = torch.randn_like(x0, device=device)
        x_t = q_sample(scheduler, x0, t, noise=eps)

        # forward model: predict eps
        eps_pred = model(x_t, t)

        # loss: mean squared error between eps and eps_pred
        loss = torch.mean((eps - eps_pred).pow(2))
        optimizer.zero_grad()
        loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        # ema update
        if ema is not None:
            ema.update(model)

        total_loss += loss.item() * B
        n_samples += B

        if step % log_every == 0:
            pbar.set_postfix({"mse": loss.item()})
    avg_loss = total_loss / max(1, n_samples)
    return {"train_mse": avg_loss}

# --------------------------
# Validation step (simple)
# --------------------------
def validate_one_epoch(model: nn.Module,
                       embeddings_module: nn.Module,
                       dataloader: DataLoader,
                       scheduler: DiffusionScheduler,
                       device: torch.device,
                       num_batches: int = 20) -> dict:
    model.eval()
    total_loss = 0.0
    n_samples = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            numeric = batch["numeric"].to(device)
            categorical = batch["categorical"].to(device)
            if categorical.numel() > 0:
                cat_embs = []
                for j in range(categorical.shape[1]):
                    cat_embs.append(embeddings_module[j](categorical[:, j].to(device)))
                cat_emb = torch.cat(cat_embs, dim=1)
                x0 = torch.cat([numeric, cat_emb], dim=1)
            else:
                x0 = numeric
            B = x0.shape[0]
            t = torch.randint(low=0, high=scheduler.timesteps, size=(B,), device=device)
            eps = torch.randn_like(x0, device=device)
            x_t = q_sample(scheduler, x0, t, noise=eps)
            eps_pred = model(x_t, t)
            loss = torch.mean((eps - eps_pred).pow(2))
            total_loss += loss.item() * B
            n_samples += B
    avg_loss = total_loss / max(1, n_samples)
    return {"val_mse": avg_loss}

# --------------------------
# Save & Load helpers
# --------------------------
def save_checkpoint(out_dir: str, model: nn.Module, embeddings: nn.Module, optimizer: torch.optim.Optimizer, scheduler_cfg: dict, epoch: int, ema: EMA = None):
    os.makedirs(out_dir, exist_ok=True)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "embeddings_state_dict": embeddings.state_dict() if embeddings is not None else None,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_cfg": scheduler_cfg,
        "epoch": epoch
    }
    if ema is not None:
        ckpt["ema_shadow"] = ema.shadow
    torch.save(ckpt, os.path.join(out_dir, f"checkpoint_epoch{epoch}.pth"))
    print(f"Saved checkpoint to {out_dir}")

# --------------------------
# CLI / main
# --------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", type=str, default="data/processed/adult/train.csv")
    p.add_argument("--val_csv", type=str, default="data/processed/adult/val.csv")
    p.add_argument("--preproc_meta", type=str, default="data/processed/adult/preprocessor/metadata.json")
    p.add_argument("--out_dir", type=str, default="results/checkpoints")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--schedule", type=str, default="linear")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--embed_dim", type=int, default=8)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--low_rank", type=int, default=0)
    p.add_argument("--ema_decay", type=float, default=0.995)
    p.add_argument("--max_val_batches", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    return args

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)

    # Load metadata to get numeric/categorical order (we need categorical column names to build embeddings)
    meta = json.load(open(args.preproc_meta, "r"))
    numeric_cols = meta.get("numeric", [])
    categorical_cols = meta.get("categorical", [])

    # Build dataloaders
    train_dl, train_ds = make_dataloader(csv_path=args.train_csv,
                                         preprocessor_meta_path=args.preproc_meta,
                                         batch_size=args.batch_size,
                                         shuffle=True,
                                         num_workers=2,
                                         pin_memory=True if device.type == "cuda" else False,
                                         target_col=None,
                                         device=None)
    val_dl, val_ds = make_dataloader(csv_path=args.val_csv,
                                     preprocessor_meta_path=args.preproc_meta,
                                     batch_size=args.batch_size,
                                     shuffle=False,
                                     num_workers=2,
                                     pin_memory=True if device.type == "cuda" else False,
                                     target_col=None,
                                     device=None)

    # Build categorical embeddings
    embeddings_module, embed_fn = build_categorical_embeddings(train_ds, categorical_cols, embed_dim=args.embed_dim)
    # Move embeddings module to device
    embeddings_module.to(device)

    # derive input_dim: numeric_dim + num_cat * embed_dim
    numeric_dim = len(numeric_cols)
    num_cat = len(categorical_cols)
    input_dim = numeric_dim + num_cat * args.embed_dim

    # instantiate denoiser model
    model = EfficientDenoiser(input_dim=input_dim,
                              hidden_dim=args.hidden_dim,
                              num_layers=args.num_layers,
                              time_emb_dim=128,
                              low_rank=args.low_rank,
                              dropout=0.0).to(device)

    # scheduler
    scheduler = DiffusionScheduler(timesteps=args.timesteps, schedule=args.schedule, device=device).to(device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # EMA
    ema = EMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None

    # training loop
    best_val = float("inf")
    os.makedirs(args.out_dir, exist_ok=True)
    scheduler_cfg = {"timesteps": args.timesteps, "schedule": args.schedule}

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model=model,
                                        embeddings_module=embeddings_module,
                                        embed_fn=embed_fn,
                                        dataloader=train_dl,
                                        scheduler=scheduler,
                                        optimizer=optimizer,
                                        device=device,
                                        ema=ema,
                                        clip_grad=1.0,
                                        log_every=200)
        val_metrics = validate_one_epoch(model=model,
                                         embeddings_module=embeddings_module,
                                         dataloader=val_dl,
                                         scheduler=scheduler,
                                         device=device,
                                         num_batches=args.max_val_batches)
        print(f"Epoch {epoch} | train_mse: {train_metrics['train_mse']:.6f} | val_mse: {val_metrics['val_mse']:.6f}")

        # checkpoint & best save
        save_checkpoint(args.out_dir, model, embeddings_module, optimizer, scheduler_cfg, epoch, ema=ema)

    print("Training finished. Checkpoints saved to:", args.out_dir)

if __name__ == "__main__":
   
    main()
