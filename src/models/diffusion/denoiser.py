# src/models/diffusion/denoiser.py
"""
Lightweight residual MLP denoiser for tabular diffusion (EfficientTabDDPM).
- Input: x (B, D) : concatenated numeric + categorical-embeddings (floats)
- Input: t (B,) or scalar : timestep(s) (int or float)
- Output: eps_pred (B, D) : predicted noise (same shape as input)
Design choices:
- small width (cfg.hidden_dim) and more depth (cfg.num_layers)
- residual MLP blocks with LayerNorm and SiLU
- sinusoidal timestep embedding projected to hidden dimension and added via FiLM-like modulation
- optional low-rank linear layers to reduce parameters
"""

from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Utilities
# -------------------------
class SiLU(nn.Module):
    def forward(self, x):
        return F.silu(x)


def get_sinusoidal_embedding(n_timesteps: int, dim: int, device=None):
    """Create sinusoidal embedding table (like transformer/time embedding)"""
    half = dim // 2
    emb = torch.arange(n_timesteps, dtype=torch.float32, device=device).unsqueeze(1)  # (T,1)
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=device) / float(half - 1))
    table = emb * freqs.unsqueeze(0)  # (T, half)
    sin = torch.sin(table)
    cos = torch.cos(table)
    emb_table = torch.cat([sin, cos], dim=1)
    if dim % 2 == 1:
        emb_table = F.pad(emb_table, (0, 1), value=0)
    return emb_table  # shape (n_timesteps, dim)


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t):
        """
        t: (B,) long or float tensor or scalar (B may be 1)
        returns: (B, dim)
        """
        if not torch.is_tensor(t):
            t = torch.tensor([t], dtype=torch.long, device=next(self.proj.parameters()).device)
        # If integer timesteps use sin/cos embedding
        if t.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
            # build sinusoidal on the fly (small cost)
            device = t.device
            max_t = int(t.max().item()) + 1
            emb_table = get_sinusoidal_embedding(max_t, self.dim, device=device)
            emb = emb_table[t]
        else:
            # treat as float: project to embedding via small MLP
            emb = t.unsqueeze(-1).to(dtype=torch.float32)
            emb = emb.repeat(1, self.dim)
        emb = self.proj(emb)
        return emb


# -------------------------
# Low-rank Linear (optional)
# -------------------------
class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        if rank is None or rank <= 0 or rank >= min(in_features, out_features):
            # use full linear
            self.use_low_rank = False
            self.linear = nn.Linear(in_features, out_features)
        else:
            self.use_low_rank = True
            self.U = nn.Parameter(torch.randn(in_features, rank) * (2.0 / math.sqrt(in_features + rank)))
            self.V = nn.Parameter(torch.randn(rank, out_features) * (2.0 / math.sqrt(rank + out_features)))
            self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        if self.use_low_rank:
            # x: (B, in_features) -> (B, rank) -> (B, out_features)
            return (x @ self.U) @ self.V + self.bias
        else:
            return self.linear(x)


# -------------------------
# Residual MLP Block
# -------------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim, hidden_dim=None, time_dim=None, low_rank: Optional[int] = 0, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.norm1 = nn.LayerNorm(dim)
        self.fc1 = LowRankLinear(dim, hidden_dim, rank=low_rank)
        self.act = SiLU()
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = LowRankLinear(hidden_dim, dim, rank=low_rank)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # optional time conditioning: project time embedding and add (FiLM-like)
        if time_dim is not None:
            self.time_proj = nn.Sequential(nn.Linear(time_dim, hidden_dim), SiLU())
        else:
            self.time_proj = None

    def forward(self, x, t_emb=None):
        h = self.norm1(x)
        h = self.fc1(h)
        if self.time_proj is not None and t_emb is not None:
            # add time conditioning into hidden layer
            h = h + self.time_proj(t_emb)
        h = self.act(h)
        h = self.norm2(h)
        h = self.dropout(h)
        h = self.fc2(h)
        return x + h  # residual


# -------------------------
# Efficient Denoiser (main)
# -------------------------
class EfficientDenoiser(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 6,
                 time_emb_dim: int = 128,
                 low_rank: int = 0,
                 dropout: float = 0.0,
                 use_final_tanh: bool = False):
        """
        input_dim: dimensionality of concatenated input (numeric + categorical embeddings)
        hidden_dim: width inside blocks (controls parameters)
        num_layers: number of residual blocks
        time_emb_dim: dimension of time embedding
        low_rank: rank for low-rank linearization (0 -> full linear)
        """
        super().__init__()
        self.input_dim = input_dim
        self.time_emb = TimeEmbedding(time_emb_dim)
        # initial projection into hidden dim if needed
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.layers = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim, time_dim=time_emb_dim, low_rank=low_rank, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.final_fc = nn.Linear(hidden_dim, input_dim)
        self.use_final_tanh = use_final_tanh

        # initialize weights carefully (small std)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, LowRankLinear):
                # linear layers are initialized in LowRankLinear; for Linear do xavier
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x, t):
        """
        x: (B, input_dim) float tensor (noisy inputs)
        t: (B,) or scalar (timesteps, int)
        returns: eps_pred (B, input_dim)
        """
        # time embedding
        if isinstance(t, int) or (torch.is_tensor(t) and t.ndim == 0):
            t = torch.full((x.shape[0],), int(t), dtype=torch.long, device=x.device)
        elif torch.is_tensor(t) and t.ndim == 1 and t.shape[0] == x.shape[0]:
            pass
        else:
            # broadcast scalar float
            t = torch.tensor([float(t)] * x.shape[0], device=x.device)

        t_emb = self.time_emb(t)  # (B, time_emb_dim)
        h = self.input_proj(x)  # (B, hidden_dim)
        for layer in self.layers:
            h = layer(h, t_emb=t_emb)
        h = self.final_norm(h)
        h = self.final_fc(h)
        if self.use_final_tanh:
            h = torch.tanh(h)
        # output shape matches input_dim
        return h
