"""
Topological Coherence - Toroidal attention constraints for reducing LLM hallucination.

Paper: Cormier (2026) "Topological Constraints for Coherent Language Models"
DOI: 10.5281/zenodo.18187835

Toroidal Logit Bias: DOI 10.5281/zenodo.18516477
"""

__version__ = "0.2.1"
__author__ = "Sylvain Cormier"

# Core geometry
from .tonnetz import Tonnetz, distance_matrix

# Mask variants
from .masks import ToroidalMask, SparseMask, sinkhorn_knopp, is_doubly_stochastic

# Logit processor
from .logit_bias import ToroidalLogitProcessor

# Drift measurement
from .drift import DriftMeter, compute_drift_rate, compute_coherence_variance

# Attention mechanisms
from .attention import (
    BaselineAttention,
    ToroidalAttention,
    RandomGraphAttention,
    create_random_graph_mask,
)

# Demo model
from .models import TinyTransformer

# ---------------------------------------------------------------------------
# Backward-compatible aliases from v0.1.2
# ---------------------------------------------------------------------------

def create_tonnetz_distance_matrix(n_tokens: int, grid_size: int = 12):
    """Create distance matrix based on Tonnetz (toroidal) topology.

    Backward-compatible wrapper around tonnetz.distance_matrix.
    Returns a torch.Tensor (v0.1.2 behavior).
    """
    import torch
    return torch.from_numpy(distance_matrix(n_tokens, grid_size).astype("float32"))


def create_tonnetz_mask(seq_len: int, radius: float = 2.0, alpha: float = 1.0):
    """Create attention mask based on Tonnetz topology with exponential decay.

    Backward-compatible wrapper around masks.ToroidalMask.hybrid.
    Returns a torch.Tensor (v0.1.2 behavior).
    """
    mask = ToroidalMask.hybrid(seq_len, radius=radius, alpha=alpha)
    return mask.to_tensor()


__all__ = [
    # v0.2.0 classes
    "Tonnetz",
    "ToroidalMask",
    "SparseMask",
    "ToroidalLogitProcessor",
    "DriftMeter",
    # Attention
    "BaselineAttention",
    "ToroidalAttention",
    "RandomGraphAttention",
    # Model
    "TinyTransformer",
    # Functions
    "distance_matrix",
    "sinkhorn_knopp",
    "is_doubly_stochastic",
    "compute_drift_rate",
    "compute_coherence_variance",
    # v0.1.2 backward compat
    "create_tonnetz_distance_matrix",
    "create_tonnetz_mask",
    "create_random_graph_mask",
]
