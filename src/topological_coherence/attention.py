"""
Attention mechanisms with topological constraints.

Three variants for comparison:
- BaselineAttention: Standard multi-head self-attention (control)
- ToroidalAttention: Tonnetz topology constraint (experimental)
- RandomGraphAttention: Random sparse mask (negative control)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .masks import ToroidalMask


def create_random_graph_mask(seq_len: int, density: float = 0.3, seed: int = 123) -> torch.Tensor:
    """Create random sparse attention mask (negative control)."""
    torch.manual_seed(seed)
    mask = torch.rand(seq_len, seq_len)
    mask = (mask < density).float()
    mask = (mask + mask.T) / 2
    mask = torch.clamp(mask, 0, 1)
    mask.fill_diagonal_(1.0)
    return mask


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
        mask = ToroidalMask.hybrid(max_seq_len)
        self.register_buffer('tonnetz_mask', mask.to_tensor())

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
