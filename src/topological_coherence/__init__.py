"""
Topological Coherence - Toroidal attention constraints for reducing LLM hallucination.

Paper: Cormier (2026) "Topological Constraints for Coherent Language Models"
DOI: 10.5281/zenodo.18187835
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Dict

__version__ = "0.1.2"
__author__ = "Sylvain Cormier"

__all__ = [
    "create_tonnetz_distance_matrix",
    "create_tonnetz_mask",
    "create_random_graph_mask",
    "sinkhorn_knopp",
    "BaselineAttention",
    "ToroidalAttention",
    "RandomGraphAttention",
    "TinyTransformer",
    "compute_drift_rate",
    "compute_coherence_variance",
]


# =============================================================================
# TONNETZ TOPOLOGY
# =============================================================================

def create_tonnetz_distance_matrix(n_tokens: int, grid_size: int = 12) -> torch.Tensor:
    """Create distance matrix based on Tonnetz (toroidal) topology."""
    coords = []
    for i in range(n_tokens):
        x = i % grid_size
        y = (i // grid_size) % grid_size
        coords.append((x, y))

    dist = torch.zeros(n_tokens, n_tokens)
    for i in range(n_tokens):
        for j in range(n_tokens):
            dx = min(abs(coords[i][0] - coords[j][0]),
                     grid_size - abs(coords[i][0] - coords[j][0]))
            dy = min(abs(coords[i][1] - coords[j][1]),
                     grid_size - abs(coords[i][1] - coords[j][1]))
            dist[i, j] = dx + dy
    return dist


def create_tonnetz_mask(seq_len: int, radius: float = 2.0, alpha: float = 1.0) -> torch.Tensor:
    """Create attention mask based on Tonnetz topology with exponential decay."""
    dist = create_tonnetz_distance_matrix(seq_len)
    mask = torch.where(dist <= radius,
                       torch.ones_like(dist),
                       torch.exp(-alpha * dist))
    return mask


def create_random_graph_mask(seq_len: int, density: float = 0.3, seed: int = 123) -> torch.Tensor:
    """Create random sparse attention mask (negative control)."""
    torch.manual_seed(seed)
    mask = torch.rand(seq_len, seq_len)
    mask = (mask < density).float()
    mask = (mask + mask.T) / 2
    mask = torch.clamp(mask, 0, 1)
    mask.fill_diagonal_(1.0)
    return mask


# =============================================================================
# SINKHORN-KNOPP
# =============================================================================

def sinkhorn_knopp(matrix: torch.Tensor, n_iters: int = 20) -> torch.Tensor:
    """Apply Sinkhorn-Knopp normalization to make matrix doubly stochastic."""
    M = torch.exp(matrix)
    for _ in range(n_iters):
        M = M / (M.sum(dim=-1, keepdim=True) + 1e-8)
        M = M / (M.sum(dim=-2, keepdim=True) + 1e-8)
    return M


# =============================================================================
# ATTENTION MECHANISMS
# =============================================================================

class BaselineAttention(nn.Module):
    """Standard multi-head self-attention."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(out)


class ToroidalAttention(nn.Module):
    """Multi-head attention with Tonnetz (toroidal) topology constraints."""

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 64):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.register_buffer('tonnetz_mask', create_tonnetz_mask(max_seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        mask = self.tonnetz_mask[:T, :T].unsqueeze(0).unsqueeze(0)
        attn = attn * mask
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(out)


class RandomGraphAttention(nn.Module):
    """Multi-head attention with random sparse mask (negative control)."""

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 64):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.register_buffer('random_mask', create_random_graph_mask(max_seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        mask = self.random_mask[:T, :T].unsqueeze(0).unsqueeze(0)
        attn = attn * mask
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(out)


# =============================================================================
# TRANSFORMER MODEL
# =============================================================================

class TinyTransformer(nn.Module):
    """Small transformer for demonstrating topological attention effects."""

    def __init__(self, vocab_size: int, d_model: int, n_heads: int,
                 attention_type: str = "baseline", max_seq_len: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        if attention_type == "baseline":
            self.attn1 = BaselineAttention(d_model, n_heads)
            self.attn2 = BaselineAttention(d_model, n_heads)
        elif attention_type == "toroidal":
            self.attn1 = ToroidalAttention(d_model, n_heads, max_seq_len)
            self.attn2 = ToroidalAttention(d_model, n_heads, max_seq_len)
        elif attention_type == "random":
            self.attn1 = RandomGraphAttention(d_model, n_heads, max_seq_len)
            self.attn2 = RandomGraphAttention(d_model, n_heads, max_seq_len)

        self.ff1 = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model))
        self.ff2 = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model))
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.ln4 = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos_embed(pos)
        h = h + self.attn1(self.ln1(h))
        h = h + self.ff1(self.ln2(h))
        h = h + self.attn2(self.ln3(h))
        h = h + self.ff2(self.ln4(h))
        logits = self.head(h)
        return logits, h


# =============================================================================
# METRICS
# =============================================================================

def compute_drift_rate(hidden_states: torch.Tensor, vocab_size: int) -> float:
    """Compute semantic drift rate - fraction of predictions violating topology."""
    grid_size = int(np.sqrt(vocab_size))
    dist_matrix = create_tonnetz_distance_matrix(vocab_size, grid_size)
    predictions = hidden_states.argmax(dim=-1)
    drift_count = 0
    total = 0
    for b in range(predictions.shape[0]):
        for t in range(predictions.shape[1] - 1):
            current = predictions[b, t].item()
            next_pred = predictions[b, t + 1].item()
            if current < vocab_size and next_pred < vocab_size:
                d = dist_matrix[current, next_pred].item()
                if d > 2:
                    drift_count += 1
                total += 1
    return drift_count / max(total, 1)


def compute_coherence_variance(hidden_states: torch.Tensor) -> float:
    """Compute variance of hidden state norms (lower = more stable)."""
    norms = torch.norm(hidden_states, dim=-1)
    return norms.var().item()
