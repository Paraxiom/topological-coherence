"""
Topological Attention Masks for Language Models
================================================

Four conditions:
1. baseline     - Standard causal attention
2. local_window - Local window mask (locality without topology)
3. random       - Random sparse mask (negative control)
4. toroidal     - Toroidal/Tonnetz mask (treatment)

GPU-optimized: precomputed and cached per sequence length.
"""

import torch
import torch.nn as nn
import math
from functools import lru_cache
from typing import Literal, Optional

MaskType = Literal["baseline", "local_window", "random", "toroidal", "hybrid"]


def compute_attention_entropy(attn_weights: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy of attention distribution per position.

    Higher entropy = more uniform attention (attending broadly)
    Lower entropy = more focused attention (attending narrowly)

    Args:
        attn_weights: (batch, heads, seq_len, seq_len) attention probabilities

    Returns:
        Tensor of shape (batch, heads, seq_len) with entropy values
    """
    # Clamp to avoid log(0)
    attn_weights = attn_weights.clamp(min=1e-10)
    entropy = -torch.sum(attn_weights * torch.log(attn_weights), dim=-1)
    return entropy


def compute_cycle_lengths(mask: torch.Tensor, grid_size: int) -> dict:
    """
    Compute statistics about attention cycles in toroidal topology.

    For toroidal masks, measures the distribution of "wrap-around" distances.

    Returns:
        dict with cycle statistics:
        - mean_cycle_length: average path length before wraparound
        - cycle_length_std: variance in cycle lengths
        - wraparound_fraction: fraction of attention going through wraparound
    """
    seq_len = mask.shape[0]

    # Positions on torus
    i = torch.arange(seq_len, device=mask.device)
    xi = (i // grid_size) % grid_size
    yi = i % grid_size

    # Compute which attention paths use wraparound
    xi_diff = xi.unsqueeze(1) - xi.unsqueeze(0)
    yi_diff = yi.unsqueeze(1) - yi.unsqueeze(0)

    # Direct distance
    direct_dx = torch.abs(xi_diff).float()
    direct_dy = torch.abs(yi_diff).float()
    direct_dist = torch.sqrt(direct_dx**2 + direct_dy**2)

    # Wrapped distance
    wrap_dx = grid_size - direct_dx
    wrap_dy = grid_size - direct_dy

    # Check if wraparound is shorter in either dimension
    uses_x_wrap = wrap_dx < direct_dx
    uses_y_wrap = wrap_dy < direct_dy
    uses_wraparound = uses_x_wrap | uses_y_wrap

    # Weighted by attention values
    attn_sum = mask.sum()
    wraparound_attn = (mask * uses_wraparound.float()).sum()
    wraparound_fraction = (wraparound_attn / attn_sum).item() if attn_sum > 0 else 0

    # Effective cycle lengths (geodesic distances)
    min_dx = torch.minimum(direct_dx, wrap_dx)
    min_dy = torch.minimum(direct_dy, wrap_dy)
    geodesic_dist = torch.sqrt(min_dx**2 + min_dy**2)

    # Weighted mean and std of geodesic distances
    weighted_dist = (mask * geodesic_dist).sum() / attn_sum if attn_sum > 0 else 0
    weighted_var = (mask * (geodesic_dist - weighted_dist)**2).sum() / attn_sum if attn_sum > 0 else 0

    return {
        "mean_cycle_length": weighted_dist.item() if torch.is_tensor(weighted_dist) else weighted_dist,
        "cycle_length_std": torch.sqrt(weighted_var).item() if torch.is_tensor(weighted_var) else math.sqrt(weighted_var),
        "wraparound_fraction": wraparound_fraction,
        "grid_size": grid_size,
    }


class TopologicalAttentionMask:
    """
    GPU-optimized attention mask generator with caching.

    Key insight from paper: We constrain attention to geometrically
    coherent neighborhoods. Hallucinations occur when attention "jumps"
    to unrelated concepts - geometric distance prevents these jumps.
    """

    def __init__(
        self,
        grid_size: int = 12,
        decay: float = 0.3,
        window_size: int = 64,
        device: str = "cuda"
    ):
        self.grid_size = grid_size
        self.decay = decay
        self.window_size = window_size
        self.device = device
        self._cache = {}

    def get_mask(
        self,
        seq_len: int,
        mask_type: MaskType = "toroidal",
        causal: bool = True
    ) -> torch.Tensor:
        """
        Get attention mask for given sequence length.

        Returns:
            Tensor of shape (seq_len, seq_len) with values in [0, 1]
            where 1 = full attention, 0 = no attention
        """
        cache_key = (seq_len, mask_type, causal)
        if cache_key in self._cache:
            return self._cache[cache_key]

        if mask_type == "baseline":
            mask = self._baseline_mask(seq_len, causal)
        elif mask_type == "local_window":
            mask = self._local_window_mask(seq_len, causal)
        elif mask_type == "random":
            mask = self._random_mask(seq_len, causal)
        elif mask_type == "toroidal":
            mask = self._toroidal_mask(seq_len, causal)
        elif mask_type == "hybrid":
            mask = self._hybrid_mask(seq_len, causal)
        else:
            raise ValueError(f"Unknown mask type: {mask_type}")

        self._cache[cache_key] = mask
        return mask

    def _baseline_mask(self, seq_len: int, causal: bool) -> torch.Tensor:
        """Standard causal attention - attend to all previous tokens equally."""
        if causal:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device))
        else:
            mask = torch.ones(seq_len, seq_len, device=self.device)
        return mask

    def _local_window_mask(self, seq_len: int, causal: bool) -> torch.Tensor:
        """
        Local window mask - same effective radius as toroidal, but no wraparound.
        This is the critical control: tests if it's locality or topology that matters.
        """
        # Create position indices
        i = torch.arange(seq_len, device=self.device).unsqueeze(1)
        j = torch.arange(seq_len, device=self.device).unsqueeze(0)

        # Linear distance (no wraparound)
        distance = torch.abs(i - j).float()

        # Same decay as toroidal for fair comparison
        mask = torch.exp(-self.decay * distance)

        if causal:
            mask = mask * torch.tril(torch.ones(seq_len, seq_len, device=self.device))

        return mask

    def _random_mask(self, seq_len: int, causal: bool) -> torch.Tensor:
        """
        Random sparse mask - negative control.
        Same sparsity pattern as toroidal but random structure.
        """
        # Match the average sparsity of toroidal mask
        toroidal = self._toroidal_mask(seq_len, causal)
        sparsity = (toroidal > 0.1).float().mean()

        # Generate random mask with same density
        mask = torch.rand(seq_len, seq_len, device=self.device)
        threshold = 1.0 - sparsity
        mask = (mask > threshold).float()

        if causal:
            mask = mask * torch.tril(torch.ones(seq_len, seq_len, device=self.device))

        # Ensure diagonal is always 1 (self-attention)
        mask = mask + torch.eye(seq_len, device=self.device)
        mask = mask.clamp(0, 1)

        return mask

    def _toroidal_mask(self, seq_len: int, causal: bool) -> torch.Tensor:
        """
        Toroidal/Tonnetz attention mask - the treatment condition.

        Maps token positions to a 2D torus and computes geodesic distances.
        Attention decays exponentially with toroidal distance.

        This is the key innovation: wraparound creates harmonic relationships
        that mirror how concepts relate in semantic space.
        """
        # Map linear positions to 2D torus coordinates
        i = torch.arange(seq_len, device=self.device)

        # 2D coordinates on the torus
        xi = (i // self.grid_size) % self.grid_size
        yi = i % self.grid_size

        # Compute pairwise toroidal distances (GPU-vectorized)
        xi_diff = xi.unsqueeze(1) - xi.unsqueeze(0)
        yi_diff = yi.unsqueeze(1) - yi.unsqueeze(0)

        # Wraparound distance (min of direct and wrapped)
        dx = torch.minimum(
            torch.abs(xi_diff),
            self.grid_size - torch.abs(xi_diff)
        ).float()
        dy = torch.minimum(
            torch.abs(yi_diff),
            self.grid_size - torch.abs(yi_diff)
        ).float()

        # Euclidean distance on torus
        distance = torch.sqrt(dx**2 + dy**2)

        # Exponential decay
        mask = torch.exp(-self.decay * distance)

        if causal:
            mask = mask * torch.tril(torch.ones(seq_len, seq_len, device=self.device))

        return mask

    def _hybrid_mask(self, seq_len: int, causal: bool) -> torch.Tensor:
        """
        Hybrid mask: local_window + low-rank toroidal wrap.

        Combines:
        1. Strong local attention (window_size neighborhood)
        2. Low-rank toroidal connections (sparse long-range via wrap)

        This preserves strong local attention while adding structured
        long-range connections through the toroidal topology.
        """
        # Get local window mask (strong local attention)
        local = self._local_window_mask(seq_len, causal)

        # Get toroidal mask
        toroidal = self._toroidal_mask(seq_len, causal)

        # Extract only the "wrap-around" connections from toroidal
        # These are positions where toroidal distance < linear distance
        i = torch.arange(seq_len, device=self.device)
        xi = (i // self.grid_size) % self.grid_size
        yi = i % self.grid_size

        xi_diff = xi.unsqueeze(1) - xi.unsqueeze(0)
        yi_diff = yi.unsqueeze(1) - yi.unsqueeze(0)

        direct_dx = torch.abs(xi_diff).float()
        direct_dy = torch.abs(yi_diff).float()
        wrap_dx = self.grid_size - direct_dx
        wrap_dy = self.grid_size - direct_dy

        # Wraparound is beneficial in either dimension
        uses_x_wrap = wrap_dx < direct_dx
        uses_y_wrap = wrap_dy < direct_dy
        wraparound_mask = (uses_x_wrap | uses_y_wrap).float()

        # Scale down the toroidal wrap contribution (low-rank)
        # Only 30% weight for wrap connections vs full local
        wrap_weight = 0.3

        # Combine: local (full strength) + toroidal wrap (reduced)
        mask = local + wrap_weight * toroidal * wraparound_mask

        # Normalize to [0, 1] range
        mask = mask / mask.max()

        if causal:
            mask = mask * torch.tril(torch.ones(seq_len, seq_len, device=self.device))

        return mask

    def clear_cache(self):
        """Clear the mask cache (useful when changing parameters)."""
        self._cache = {}


class TopologicalAttention(nn.Module):
    """
    Drop-in replacement for standard attention with topological constraints.

    Can be injected into any HuggingFace transformer model.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mask_type: MaskType = "toroidal",
        grid_size: int = 12,
        decay: float = 0.3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.mask_type = mask_type

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.mask_generator = TopologicalAttentionMask(
            grid_size=grid_size,
            decay=decay
        )

        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Apply topological mask
        topo_mask = self.mask_generator.get_mask(
            seq_len,
            self.mask_type,
            causal=True
        )

        # Convert to attention bias (log space)
        # Where mask is 0, attention should be -inf
        # Where mask is 1, attention should be 0 (no modification)
        topo_bias = torch.log(topo_mask + 1e-10)
        attn_scores = attn_scores + topo_bias.unsqueeze(0).unsqueeze(0)

        # Apply any additional attention mask (e.g., padding)
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        # Softmax and dropout
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Compute output
        output = torch.matmul(attn_probs, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.out_proj(output)

        return output


def inject_topological_attention(model, mask_type: MaskType = "toroidal", **kwargs):
    """
    Inject topological attention into an existing HuggingFace model.

    This modifies the model in-place, replacing standard attention
    with topological attention.

    Args:
        model: HuggingFace transformer model
        mask_type: One of "baseline", "local_window", "random", "toroidal"
        **kwargs: Additional arguments for TopologicalAttentionMask
    """
    mask_gen = TopologicalAttentionMask(**kwargs)

    # Store original forward methods and create wrappers
    for name, module in model.named_modules():
        if "attention" in name.lower() and hasattr(module, "forward"):
            original_forward = module.forward

            def make_wrapper(orig_fwd, mask_generator, mtype):
                def wrapper(hidden_states, attention_mask=None, **kw):
                    seq_len = hidden_states.shape[1]

                    # Get topological mask
                    topo_mask = mask_generator.get_mask(seq_len, mtype, causal=True)
                    topo_bias = torch.log(topo_mask + 1e-10)

                    # Combine with existing mask
                    if attention_mask is None:
                        attention_mask = topo_bias.unsqueeze(0).unsqueeze(0)
                    else:
                        attention_mask = attention_mask + topo_bias.unsqueeze(0).unsqueeze(0)

                    return orig_fwd(hidden_states, attention_mask=attention_mask, **kw)
                return wrapper

            module.forward = make_wrapper(original_forward, mask_gen, mask_type)

    return model


# Quick test
if __name__ == "__main__":
    print("Testing TopologicalAttentionMask...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mask_gen = TopologicalAttentionMask(device=device)

    seq_len = 128

    for mask_type in ["baseline", "local_window", "random", "toroidal", "hybrid"]:
        mask = mask_gen.get_mask(seq_len, mask_type)
        sparsity = (mask < 0.1).float().mean().item()
        print(f"{mask_type:15} - shape: {mask.shape}, sparsity: {sparsity:.2%}")

        # Compute cycle stats for toroidal masks
        if mask_type in ["toroidal", "hybrid"]:
            cycle_stats = compute_cycle_lengths(mask, mask_gen.grid_size)
            print(f"                  cycle_len: {cycle_stats['mean_cycle_length']:.2f} ± {cycle_stats['cycle_length_std']:.2f}, wrap_frac: {cycle_stats['wraparound_fraction']:.2%}")

    print("\nMask generation test passed!")
