"""
Toroidal attention masks with three variants.

Mirrors the Rust `ToroidalMask` and `MaskType` enum:
- HardCutoff: M(i,j) = 1 if d <= r, else 0
- SoftExponential: M(i,j) = exp(-alpha * d)
- Hybrid: M(i,j) = 1 if d <= r, else exp(-alpha * (d - r))

Also provides Sinkhorn-Knopp projection to doubly-stochastic matrices.
"""

import numpy as np
import torch
from typing import Optional

from .tonnetz import distance_matrix


class ToroidalMask:
    """Toroidal attention mask generator.

    Implements Eq. 17 from the paper:
        M_Tonnetz(i, j) = 1                              if d_T(i,j) <= r
                          exp(-alpha * (d_T(i,j) - r))    otherwise

    Args:
        seq_len: Sequence length (mask is seq_len x seq_len).
        radius: Locality radius for hard cutoff region.
        alpha: Decay rate for soft falloff.
        grid_size: Side length N of the N x N torus.
        mask_type: One of 'hybrid', 'hard_cutoff', 'soft_exponential'.
    """

    def __init__(self, seq_len: int, radius: float = 2.0, alpha: float = 1.0,
                 grid_size: int = 12, mask_type: str = "hybrid"):
        self.seq_len = seq_len
        self.radius = radius
        self.alpha = alpha
        self.grid_size = grid_size
        if mask_type not in ("hybrid", "hard_cutoff", "soft_exponential"):
            raise ValueError(f"Unknown mask_type: {mask_type}")
        self.mask_type = mask_type

    @classmethod
    def hard_cutoff(cls, seq_len: int, radius: float, grid_size: int = 12) -> "ToroidalMask":
        """Create hard cutoff mask: 1 if d <= r, else 0."""
        return cls(seq_len, radius=radius, alpha=0.0, grid_size=grid_size,
                   mask_type="hard_cutoff")

    @classmethod
    def soft_exponential(cls, seq_len: int, alpha: float, grid_size: int = 12) -> "ToroidalMask":
        """Create soft exponential mask: exp(-alpha * d)."""
        return cls(seq_len, radius=0.0, alpha=alpha, grid_size=grid_size,
                   mask_type="soft_exponential")

    @classmethod
    def hybrid(cls, seq_len: int, radius: float = 2.0, alpha: float = 1.0,
               grid_size: int = 12) -> "ToroidalMask":
        """Create hybrid mask (default): 1 if d <= r, else exp(-alpha*(d-r))."""
        return cls(seq_len, radius=radius, alpha=alpha, grid_size=grid_size,
                   mask_type="hybrid")

    def generate(self) -> np.ndarray:
        """Generate full mask matrix as numpy array."""
        dist = distance_matrix(self.seq_len, self.grid_size).astype(np.float32)

        if self.mask_type == "hard_cutoff":
            return np.where(dist <= self.radius, 1.0, 0.0).astype(np.float32)
        elif self.mask_type == "soft_exponential":
            return np.exp(-self.alpha * dist).astype(np.float32)
        else:  # hybrid
            return np.where(
                dist <= self.radius,
                1.0,
                np.exp(-self.alpha * (dist - self.radius))
            ).astype(np.float32)

    def to_tensor(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Generate mask as a torch tensor."""
        mask_np = self.generate()
        t = torch.from_numpy(mask_np)
        if device is not None:
            t = t.to(device)
        return t


class SparseMask:
    """Sparse mask wrapper using torch sparse tensors.

    For large sequence lengths, stores only entries above a threshold.
    """

    def __init__(self, mask: ToroidalMask, threshold: float = 0.01):
        dense = mask.generate()
        rows, cols = np.nonzero(dense > threshold)
        values = dense[rows, cols]
        self.size = mask.seq_len
        self._indices = torch.stack([
            torch.from_numpy(rows.astype(np.int64)),
            torch.from_numpy(cols.astype(np.int64)),
        ])
        self._values = torch.from_numpy(values)
        self._sparse = torch.sparse_coo_tensor(
            self._indices, self._values, (self.size, self.size)
        )

    @property
    def nnz(self) -> int:
        return self._values.shape[0]

    @property
    def sparsity(self) -> float:
        total = self.size * self.size
        return 1.0 - (self.nnz / total) if total > 0 else 0.0

    def to_dense(self) -> torch.Tensor:
        return self._sparse.to_dense()


def sinkhorn_knopp(matrix: torch.Tensor, n_iters: int = 20) -> torch.Tensor:
    """Project matrix to doubly-stochastic via Sinkhorn-Knopp.

    Alternately normalizes rows and columns until convergence.
    A doubly-stochastic matrix has all rows and columns sum to 1.

    Args:
        matrix: Input non-negative matrix.
        n_iters: Number of normalization iterations.

    Returns:
        Approximately doubly-stochastic matrix.
    """
    M = torch.exp(matrix) if matrix.min() < 0 else matrix.clone().float()
    M = M + 1e-8  # avoid division by zero
    for _ in range(n_iters):
        M = M / M.sum(dim=-1, keepdim=True)
        M = M / M.sum(dim=-2, keepdim=True)
    return M


def is_doubly_stochastic(matrix: torch.Tensor, tol: float = 0.01) -> bool:
    """Check if matrix is approximately doubly-stochastic.

    Args:
        matrix: Square matrix to check.
        tol: Tolerance for row/column sum deviation from 1.0.

    Returns:
        True if all row sums and column sums are within tol of 1.0.
    """
    row_sums = matrix.sum(dim=-1)
    col_sums = matrix.sum(dim=-2)
    return (
        torch.all(torch.abs(row_sums - 1.0) < tol).item()
        and torch.all(torch.abs(col_sums - 1.0) < tol).item()
    )
