# src/models/diffusion/forward.py
"""
Forward diffusion & scheduler utilities for Tabular DDPM.
Provides:
- beta schedules (linear, cosine)
- precomputation of alphas, alpha_bars and sqrt terms
- q_sample: closed-form sampling of x_t from x_0
- predict_x0_from_eps: invert closed-form
- posterior mean & variance helpers for DDPM sampling
- one-step DDPM ancestral sampling and DDIM deterministic step
"""

from typing import Tuple, Optional
import math
import torch
import torch.nn as nn


# -------------------------
# Beta schedules
# -------------------------
def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    """
    Linear schedule from beta_start to beta_end across timesteps.
    Returns betas shape (T,)
    """
    return torch.linspace(beta_start, beta_end, steps=timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine schedule as in Nichol & Dhariwal (and used in many implementations).
    Returns betas as torch.float64 for numerical stability.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps=steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas_cumprod = alphas_cumprod[1:]  # drop t=0
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clamp(betas, 0.000001, 0.999999).to(dtype=torch.float64)
    # We return length timesteps
    return torch.cat([betas.new_tensor([betas[0]]), betas])[:timesteps]


# -------------------------
# Scheduler / Precomputation container
# -------------------------
class DiffusionScheduler:
    def __init__(self, timesteps: int = 1000, schedule: str = "linear", beta_start: float = 1e-4, beta_end: float = 2e-2, device: Optional[torch.device]=None):
        self.timesteps = timesteps
        self.device = device or torch.device("cpu")
        if schedule == "linear":
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        elif schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError("Unknown schedule: choose 'linear' or 'cosine'")

        # store as float32 on device for runtime efficiency (use float64 for precompute if desired)
        self.betas = betas.to(dtype=torch.float32, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=self.device, dtype=self.alphas_cumprod.dtype), self.alphas_cumprod[:-1]], dim=0)

        # precompute useful terms
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # posterior variance (for p(x_{t-1}|x_t, x0) closed form)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        # clip small values
        self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min=1e-20))

    def to(self, device: torch.device):
        self.device = device
        self.betas = self.betas.to(device=device)
        self.alphas = self.alphas.to(device=device)
        self.alphas_cumprod = self.alphas_cumprod.to(device=device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device=device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device=device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device=device)
        self.log_one_minus_alphas_cumprod = self.log_one_minus_alphas_cumprod.to(device=device)
        self.sqrt_recip_alphas_cumprod = self.sqrt_recip_alphas_cumprod.to(device=device)
        self.sqrt_recipm1_alphas_cumprod = self.sqrt_recipm1_alphas_cumprod.to(device=device)
        self.posterior_variance = self.posterior_variance.to(device=device)
        self.posterior_log_variance_clipped = self.posterior_log_variance_clipped.to(device=device)
        return self


# -------------------------
# q(x_t | x_0) closed-form sampling
# -------------------------
def q_sample(scheduler: DiffusionScheduler, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Generate x_t from x_0 in closed-form:
    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
    - scheduler: precomputed DiffusionScheduler (on correct device)
    - x_start: (B, D)
    - t: (B,) integer tensor of timesteps in [0, T-1]
    - noise: optional noise (B, D), otherwise drawn N(0,I)
    returns x_t on scheduler.device dtype matching x_start
    """
    if noise is None:
        noise = torch.randn_like(x_start)
    # gather factors
    # make sure t is long tensor
    if not torch.is_tensor(t):
        t = torch.tensor(t, device=x_start.device)
    t = t.to(dtype=torch.long)
    sqrt_alpha_bar_t = scheduler.sqrt_alphas_cumprod[t].view(-1, *([1] * (x_start.ndim - 1))).to(x_start.device, x_start.dtype)
    sqrt_one_minus_alpha_bar_t = scheduler.sqrt_one_minus_alphas_cumprod[t].view(-1, *([1] * (x_start.ndim - 1))).to(x_start.device, x_start.dtype)
    return sqrt_alpha_bar_t * x_start + sqrt_one_minus_alpha_bar_t * noise


# -------------------------
# Inversion: predict x0 from xt + eps
# -------------------------
def predict_x0_from_eps(scheduler: DiffusionScheduler, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """
    Reconstruct x0 estimate from x_t and predicted eps:
    x0 = (x_t - sqrt(1-alpha_bar_t) * eps) / sqrt(alpha_bar_t)
    """
    t = t.to(dtype=torch.long)
    sqrt_alpha_bar_t = scheduler.sqrt_alphas_cumprod[t].view(-1, *([1] * (x_t.ndim - 1))).to(x_t.device, x_t.dtype)
    sqrt_one_minus_alpha_bar_t = scheduler.sqrt_one_minus_alphas_cumprod[t].view(-1, *([1] * (x_t.ndim - 1))).to(x_t.device, x_t.dtype)
    return (x_t - sqrt_one_minus_alpha_bar_t * eps) / sqrt_alpha_bar_t


# -------------------------
# Posterior (p(x_{t-1} | x_t, x0)) mean and variance used in ancestral sampling
# -------------------------
def q_posterior(scheduler: DiffusionScheduler, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute posterior mean and variance for q(x_{t-1} | x_t, x0).
    Returns (mean, variance)
    mean shape: same as x_start
    variance shape: (B,1,...)
    """
    t = t.to(dtype=torch.long)
    coef1 = scheduler.betas[t].to(x_t.device) * torch.sqrt(scheduler.alphas_cumprod_prev[t].to(x_t.device)) / (1.0 - scheduler.alphas_cumprod[t].to(x_t.device))
    coef2 = (1.0 - scheduler.alphas_cumprod_prev[t].to(x_t.device)) * torch.sqrt(scheduler.alphas[t].to(x_t.device)) / (1.0 - scheduler.alphas_cumprod[t].to(x_t.device))
    coef1 = coef1.view(-1, *([1] * (x_t.ndim - 1)))
    coef2 = coef2.view(-1, *([1] * (x_t.ndim - 1)))
    posterior_mean = coef1 * x_start + coef2 * x_t
    posterior_variance = scheduler.posterior_variance[t].view(-1, *([1] * (x_t.ndim - 1))).to(x_t.device)
    posterior_log_variance_clipped = scheduler.posterior_log_variance_clipped[t].view(-1, *([1] * (x_t.ndim - 1))).to(x_t.device)
    return posterior_mean, posterior_variance, posterior_log_variance_clipped


# -------------------------
# Prediction of p_mean_variance from model's eps prediction
# -------------------------
def p_mean_variance_from_eps(scheduler: DiffusionScheduler, model, x_t: torch.Tensor, t: torch.Tensor, clip_denoised: bool = True):
    """
    Given a model that predicts eps (noise), compute the p mean & variance for sampling:
    - derive x0 from eps
    - compute posterior mean/var
    """
    eps = model(x_t, t)
    x0_pred = predict_x0_from_eps(scheduler, x_t, t, eps)
    if clip_denoised:
        # Optionally clip x0_pred to plausible range. For normalized numeric features we may clip to [-5,5]
        x0_pred = torch.clamp(x0_pred, -10.0, 10.0)
    posterior_mean, posterior_variance, posterior_log_variance_clipped = q_posterior(scheduler, x0_pred, x_t, t)
    return posterior_mean, posterior_variance, posterior_log_variance_clipped, x0_pred, eps


# -------------------------
# One ancestral DDPM sampling step (stochastic)
# -------------------------
def ddpm_sample_step(scheduler: DiffusionScheduler, model, x_t: torch.Tensor, t: torch.Tensor, noise_scale: float = 1.0):
    """
    Take one step t -> t-1 using DDPM ancestral sampling.
    - noise_scale: multiply the posterior std by this factor (1.0 is standard)
    """
    device = x_t.device
    posterior_mean, posterior_variance, posterior_log_variance_clipped, x0_pred, eps = p_mean_variance_from_eps(scheduler, model, x_t, t)
    # sample z for all but t==0
    noise = torch.randn_like(x_t, device=device)
    nonzero_mask = (t != 0).to(device).view(-1, *([1] * (x_t.ndim - 1)))
    std = torch.sqrt(posterior_variance) * noise_scale
    x_prev = posterior_mean + nonzero_mask * std * noise
    return x_prev, x0_pred, eps


# -------------------------
# DDIM deterministic step (one step) - useful for fast deterministic sampling
# -------------------------
def ddim_sample_step(scheduler: DiffusionScheduler, model, x_t: torch.Tensor, t: int, next_t: int, eta: float = 0.0):
    """
    Deterministic / semi-deterministic DDIM step from time t -> next_t (next_t < t)
    - eta controls stochasticity (eta=0 -> deterministic)
    - t and next_t are integers
    """
    # ensure tensors
    device = x_t.device
    with torch.no_grad():
        eps = model(x_t, torch.full((x_t.shape[0],), t, dtype=torch.long, device=device))
        alpha_t = scheduler.alphas_cumprod[t].to(device)
        alpha_s = scheduler.alphas_cumprod[next_t].to(device)

        sqrt_alpha_t = math.sqrt(alpha_t)
        sqrt_alpha_s = math.sqrt(alpha_s)

        sqrt_one_alpha_t = math.sqrt(1 - alpha_t)
        # estimate x0
        x0_pred = (x_t - sqrt_one_alpha_t * eps) / sqrt_alpha_t

        # compute direction pointing to eps (same as paper)
        dir_xt = math.sqrt(1.0 - alpha_s) * eps

        x_s = sqrt_alpha_s * x0_pred + dir_xt

        if eta > 0:
            # add noise scaled by eta (to reintroduce stochasticity)
            sigma = eta * math.sqrt((1 - alpha_s) / (1 - alpha_t)) * math.sqrt(1 - alpha_t / alpha_s)
            noise = torch.randn_like(x_t) * sigma
            x_s = x_s + noise
    return x_s, x0_pred, eps
