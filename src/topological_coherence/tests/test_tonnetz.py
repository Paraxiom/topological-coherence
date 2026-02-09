"""Tests for Tonnetz topology — mirrors Rust property tests."""

import numpy as np
import pytest

from topological_coherence.tonnetz import Tonnetz, distance_matrix


class TestTonnetzDistance:
    """Metric space axioms on the torus."""

    def test_distance_self(self):
        t = Tonnetz(12)
        assert t.distance(0, 0) == 0
        assert t.distance(5, 5) == 0
        assert t.distance(143, 143) == 0

    def test_distance_adjacent(self):
        t = Tonnetz(12)
        assert t.distance(0, 1) == 1

    def test_distance_wraparound(self):
        t = Tonnetz(12)
        # (0,0) to (11,0) on a 12-grid wraps to distance 1
        assert t.distance(0, 11) == 1

    def test_distance_diagonal_wrap(self):
        t = Tonnetz(12)
        # index 0 = (0,0), index 143 = (11,11) → wraps to (1,1)
        assert t.distance(0, 143) == 2

    def test_distance_symmetry(self):
        t = Tonnetz(12)
        for i in range(0, 144, 7):
            for j in range(0, 144, 7):
                assert t.distance(i, j) == t.distance(j, i)

    def test_distance_identity(self):
        t = Tonnetz(12)
        for i in range(144):
            assert t.distance(i, i) == 0

    def test_triangle_inequality(self):
        t = Tonnetz(12)
        points = [0, 17, 55, 100, 143]
        for a in points:
            for b in points:
                for c in points:
                    assert t.distance(a, c) <= t.distance(a, b) + t.distance(b, c)

    def test_max_distance_bounded(self):
        t = Tonnetz(12)
        max_d = max(t.distance(0, j) for j in range(144))
        assert max_d == 12  # 6 + 6

    def test_non_negative(self):
        t = Tonnetz(12)
        for i in range(0, 144, 5):
            for j in range(0, 144, 5):
                assert t.distance(i, j) >= 0


class TestTonnetzSpectralGap:

    def test_spectral_gap_positive(self):
        t = Tonnetz(12)
        gap = t.spectral_gap()
        assert gap > 0.0
        assert gap < 1.0  # For N=12, gap ~ 0.268

    def test_spectral_gap_scales(self):
        gap_6 = Tonnetz(6).spectral_gap()
        gap_12 = Tonnetz(12).spectral_gap()
        gap_24 = Tonnetz(24).spectral_gap()
        assert gap_6 > gap_12 > gap_24

    def test_decay_rate(self):
        t = Tonnetz(12)
        assert t.decay_rate(0) == pytest.approx(1.0)
        assert 0.0 < t.decay_rate(1.0) < 1.0


class TestTonnetzCoordinates:

    def test_roundtrip(self):
        t = Tonnetz(12)
        for idx in range(144):
            r, c = t.coordinates(idx)
            assert t.to_index(r, c) == idx

    def test_distance_coords_matches(self):
        t = Tonnetz(12)
        for i in range(0, 144, 11):
            for j in range(0, 144, 11):
                ci = t.coordinates(i)
                cj = t.coordinates(j)
                assert t.distance(i, j) == t.distance_coords(ci, cj)


class TestDistanceMatrix:

    def test_shape(self):
        dm = distance_matrix(64, 12)
        assert dm.shape == (64, 64)

    def test_diagonal_zero(self):
        dm = distance_matrix(64, 12)
        assert np.all(np.diag(dm) == 0)

    def test_symmetric(self):
        dm = distance_matrix(64, 12)
        np.testing.assert_array_equal(dm, dm.T)

    def test_non_negative(self):
        dm = distance_matrix(64, 12)
        assert np.all(dm >= 0)

    def test_matches_scalar(self):
        """Vectorized matrix matches scalar distance computation."""
        t = Tonnetz(12)
        dm = distance_matrix(32, 12)
        for i in range(32):
            for j in range(32):
                assert dm[i, j] == t.distance(i, j)
