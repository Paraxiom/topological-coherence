"""
Drift measurement utilities.

Mirrors the Rust `DriftMeter` struct:
- Record prediction/target pairs
- Compute drift rate (fraction violating distance threshold)

Also provides backward-compatible functions from v0.1.2.
"""

import torch
import numpy as np
from typing import Optional

from .tonnetz import Tonnetz, distance_matrix


class DriftMeter:
    """Drift rate tracker.

    Drift rate = fraction of transitions where d_T(pred, target) > threshold.

    Args:
        threshold: Distance threshold for "drift" classification (default 2).
        grid_size: Side length N of the torus (default 12).
    """

    def __init__(self, threshold: int = 2, grid_size: int = 12):
        self.threshold = threshold
        self.grid_size = grid_size
        self._tonnetz = Tonnetz(grid_size)
        self.count: int = 0
        self.drifts: int = 0

    def record(self, pred: int, target: int) -> None:
        """Record a transition from predicted to target position.

        Args:
            pred: Linear index of predicted token position.
            target: Linear index of target token position.
        """
        d = self._tonnetz.distance(pred, target)
        self.count += 1
        if d > self.threshold:
            self.drifts += 1

    def rate(self) -> float:
        """Current drift rate (0.0 to 1.0)."""
        if self.count == 0:
            return 0.0
        return self.drifts / self.count

    def reset(self) -> None:
        """Reset all measurements."""
        self.count = 0
        self.drifts = 0


# ---------------------------------------------------------------------------
# Backward-compatible functions from v0.1.2
# ---------------------------------------------------------------------------

def compute_drift_rate(hidden_states: torch.Tensor, vocab_size: int) -> float:
    """Compute semantic drift rate - fraction of predictions violating topology.

    This is the original v0.1.2 function, preserved for backward compatibility.

    Args:
        hidden_states: Tensor of shape (batch, seq_len, hidden_dim) or logits.
        vocab_size: Vocabulary size for grid computation.

    Returns:
        Drift rate between 0.0 and 1.0.
    """
    grid_size = int(np.sqrt(vocab_size))
    dist = distance_matrix(vocab_size, grid_size)
    dist_t = torch.from_numpy(dist).float()
    predictions = hidden_states.argmax(dim=-1)

    drift_count = 0
    total = 0
    for b in range(predictions.shape[0]):
        for t in range(predictions.shape[1] - 1):
            current = predictions[b, t].item()
            next_pred = predictions[b, t + 1].item()
            if current < vocab_size and next_pred < vocab_size:
                d = dist_t[current, next_pred].item()
                if d > 2:
                    drift_count += 1
                total += 1
    return drift_count / max(total, 1)


def compute_coherence_variance(hidden_states: torch.Tensor) -> float:
    """Compute variance of hidden state norms (lower = more stable).

    Original v0.1.2 function, preserved for backward compatibility.

    Args:
        hidden_states: Tensor of shape (batch, seq_len, hidden_dim).

    Returns:
        Scalar variance of hidden state norms.
    """
    norms = torch.norm(hidden_states, dim=-1)
    return norms.var().item()
