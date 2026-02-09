"""
Tonnetz topology on a 2D torus.

Mirrors the Rust `Tonnetz<N>` implementation:
- L1 (Manhattan) distance with wraparound
- Spectral gap of the torus Laplacian
- Vectorized distance matrix for attention integration

Pure numpy — no torch dependency for core geometry.
"""

import math
import numpy as np
from typing import Tuple


class Tonnetz:
    """Tonnetz topology on an N x N torus.

    The Tonnetz is a toroidal lattice originally from music theory
    (Euler, 1739). We use it as a constructive existence proof of a
    low-genus manifold with constant spectral gap.

    Args:
        grid_size: Side length N of the N x N torus (default 12).
    """

    def __init__(self, grid_size: int = 12):
        self.grid_size = grid_size

    @property
    def total_positions(self) -> int:
        """Total number of positions on the torus (N^2)."""
        return self.grid_size * self.grid_size

    def coordinates(self, index: int) -> Tuple[int, int]:
        """Convert linear index to 2D torus coordinates (row, col)."""
        n = self.grid_size
        return (index % n, (index // n) % n)

    def to_index(self, row: int, col: int) -> int:
        """Convert 2D torus coordinates to linear index."""
        n = self.grid_size
        return (row % n) + (col % n) * n

    def distance(self, i: int, j: int) -> int:
        """L1 toroidal distance between two linear indices.

        d_T(i, j) = min(|x_i - x_j|, N - |x_i - x_j|) + min(|y_i - y_j|, N - |y_i - y_j|)
        """
        n = self.grid_size
        xi, yi = self.coordinates(i)
        xj, yj = self.coordinates(j)
        dx = abs(xi - xj)
        dy = abs(yi - yj)
        return min(dx, n - dx) + min(dy, n - dy)

    def distance_coords(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """L1 toroidal distance between two coordinate pairs."""
        n = self.grid_size
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return min(dx, n - dx) + min(dy, n - dy)

    def spectral_gap(self) -> float:
        """First non-trivial eigenvalue of the torus Laplacian.

        For a 2D torus T^2_N:
            lambda_1 = 2 - 2*cos(2*pi/N) = Theta(1) for fixed N

        This constant spectral gap bounds drift decay.
        """
        return 2.0 - 2.0 * math.cos(2.0 * math.pi / self.grid_size)

    def decay_rate(self, t: float) -> float:
        """Spectral gap decay for non-resonant modes: e^(-lambda_1 * t)."""
        return math.exp(-self.spectral_gap() * t)


def distance_matrix(n_tokens: int, grid_size: int = 12) -> np.ndarray:
    """Vectorized toroidal distance matrix.

    Computes the full n_tokens x n_tokens L1 distance matrix on an
    N x N torus. O(n^2) but fully vectorized with numpy.

    Args:
        n_tokens: Number of tokens (rows/cols of output matrix).
        grid_size: Side length N of the torus.

    Returns:
        np.ndarray of shape (n_tokens, n_tokens) with integer distances.
    """
    idx = np.arange(n_tokens)
    x = idx % grid_size
    y = (idx // grid_size) % grid_size

    # Pairwise absolute differences
    dx = np.abs(x[:, None] - x[None, :])
    dy = np.abs(y[:, None] - y[None, :])

    # Wraparound
    dx = np.minimum(dx, grid_size - dx)
    dy = np.minimum(dy, grid_size - dy)

    return dx + dy
