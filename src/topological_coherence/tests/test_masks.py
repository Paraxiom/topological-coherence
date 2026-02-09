"""Tests for toroidal masks and Sinkhorn-Knopp."""

import numpy as np
import torch
import pytest

from topological_coherence.masks import (
    ToroidalMask,
    SparseMask,
    sinkhorn_knopp,
    is_doubly_stochastic,
)


class TestToroidalMask:

    def test_hybrid_shape(self):
        mask = ToroidalMask.hybrid(32)
        m = mask.generate()
        assert m.shape == (32, 32)

    def test_hard_cutoff_shape(self):
        mask = ToroidalMask.hard_cutoff(32, radius=2.0)
        m = mask.generate()
        assert m.shape == (32, 32)

    def test_soft_exponential_shape(self):
        mask = ToroidalMask.soft_exponential(32, alpha=1.0)
        m = mask.generate()
        assert m.shape == (32, 32)

    def test_hard_cutoff_values(self):
        mask = ToroidalMask.hard_cutoff(64, radius=2.0, grid_size=12)
        m = mask.generate()
        # Diagonal is always 1 (distance 0 <= any radius)
        np.testing.assert_array_equal(np.diag(m), np.ones(64))
        # All values are 0 or 1
        assert np.all((m == 0.0) | (m == 1.0))

    def test_soft_exponential_self(self):
        mask = ToroidalMask.soft_exponential(64, alpha=1.0, grid_size=12)
        m = mask.generate()
        # exp(-0) = 1.0 on diagonal
        np.testing.assert_allclose(np.diag(m), 1.0, atol=1e-6)

    def test_soft_exponential_decay(self):
        mask = ToroidalMask.soft_exponential(64, alpha=1.0, grid_size=12)
        m = mask.generate()
        # Adjacent position (0,1) has distance 1: exp(-1) ~ 0.368
        assert abs(m[0, 1] - np.exp(-1.0)) < 0.01

    def test_hybrid_within_radius(self):
        mask = ToroidalMask.hybrid(64, radius=2.0, alpha=1.0)
        m = mask.generate()
        # Diagonal: distance 0 <= 2.0 → 1.0
        np.testing.assert_allclose(np.diag(m), 1.0, atol=1e-6)

    def test_hybrid_beyond_radius(self):
        mask = ToroidalMask.hybrid(64, radius=1.0, alpha=1.0, grid_size=12)
        m = mask.generate()
        # Position (0, 2) on 12-grid has distance 2 > radius 1
        # Value = exp(-1 * (2-1)) = exp(-1) ~ 0.368
        assert m[0, 2] < 1.0
        assert m[0, 2] > 0.0

    def test_symmetric(self):
        for variant in ("hybrid", "hard_cutoff", "soft_exponential"):
            mask = ToroidalMask(32, mask_type=variant, alpha=1.0, radius=2.0)
            m = mask.generate()
            np.testing.assert_allclose(m, m.T, atol=1e-6,
                                       err_msg=f"{variant} mask not symmetric")

    def test_to_tensor(self):
        mask = ToroidalMask.hybrid(16)
        t = mask.to_tensor()
        assert isinstance(t, torch.Tensor)
        assert t.shape == (16, 16)

    def test_invalid_mask_type(self):
        with pytest.raises(ValueError):
            ToroidalMask(32, mask_type="invalid")


class TestSparseMask:

    def test_sparsity(self):
        mask = ToroidalMask.hard_cutoff(32, radius=1.0, grid_size=8)
        sparse = SparseMask(mask, threshold=0.5)
        assert sparse.sparsity > 0.0

    def test_to_dense_matches(self):
        mask = ToroidalMask.hard_cutoff(16, radius=1.0, grid_size=4)
        sparse = SparseMask(mask, threshold=0.5)
        dense = sparse.to_dense().numpy()
        original = mask.generate()
        # Non-zero entries should match
        nz = original > 0.5
        np.testing.assert_allclose(dense[nz], original[nz], atol=1e-5)


class TestSinkhornKnopp:

    def test_doubly_stochastic(self):
        mask = ToroidalMask.hybrid(16, radius=2.0, alpha=0.5)
        m = mask.to_tensor()
        ds = sinkhorn_knopp(m, n_iters=50)
        assert is_doubly_stochastic(ds, tol=0.01)

    def test_preserves_locality(self):
        """After Sinkhorn, nearby entries should still be larger on average."""
        mask = ToroidalMask.hybrid(16, radius=2.0, alpha=0.5)
        m = mask.to_tensor()
        ds = sinkhorn_knopp(m, n_iters=50)
        diag_avg = ds.diagonal().mean().item()
        total_avg = ds.mean().item()
        assert diag_avg > total_avg

    def test_is_doubly_stochastic_true(self):
        n = 8
        ds = torch.ones(n, n) / n
        assert is_doubly_stochastic(ds, tol=0.01)

    def test_is_doubly_stochastic_false(self):
        m = torch.ones(4, 4)
        assert not is_doubly_stochastic(m, tol=0.01)
