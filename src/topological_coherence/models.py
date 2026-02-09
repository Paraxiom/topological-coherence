"""
Demo transformer model for topological attention experiments.
"""

import torch
import torch.nn as nn
from typing import Tuple

from .attention import BaselineAttention, ToroidalAttention, RandomGraphAttention


class TinyTransformer(nn.Module):
    """Small transformer for demonstrating topological attention effects.

    A 2-layer transformer that supports swapping attention mechanisms
    to compare baseline, toroidal, and random-graph constraints.

    Args:
        vocab_size: Number of tokens in vocabulary.
        d_model: Hidden dimension.
        n_heads: Number of attention heads.
        attention_type: One of "baseline", "toroidal", "random".
        max_seq_len: Maximum sequence length.
    """

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
