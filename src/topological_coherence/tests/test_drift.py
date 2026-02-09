"""Tests for DriftMeter and backward-compatible drift functions."""

import torch
import pytest

from topological_coherence.drift import DriftMeter, compute_drift_rate, compute_coherence_variance


class TestDriftMeter:

    def test_initial_state(self):
        m = DriftMeter(threshold=2, grid_size=12)
        assert m.count == 0
        assert m.drifts == 0
        assert m.rate() == 0.0

    def test_record_no_drift(self):
        m = DriftMeter(threshold=2, grid_size=12)
        m.record(0, 1)  # distance 1 <= threshold 2
        assert m.count == 1
        assert m.drifts == 0
        assert m.rate() == 0.0

    def test_record_drift(self):
        m = DriftMeter(threshold=2, grid_size=12)
        m.record(0, 6)  # distance 6 > threshold 2
        assert m.count == 1
        assert m.drifts == 1
        assert m.rate() == 1.0

    def test_mixed_recording(self):
        m = DriftMeter(threshold=2, grid_size=12)
        m.record(0, 1)   # no drift (d=1)
        m.record(0, 6)   # drift (d=6)
        m.record(0, 0)   # no drift (d=0)
        assert m.count == 3
        assert m.drifts == 1
        assert abs(m.rate() - 1.0/3.0) < 0.01

    def test_reset(self):
        m = DriftMeter(threshold=2, grid_size=12)
        m.record(0, 6)
        m.record(0, 6)
        m.reset()
        assert m.count == 0
        assert m.drifts == 0
        assert m.rate() == 0.0

    def test_threshold_boundary(self):
        m = DriftMeter(threshold=2, grid_size=12)
        m.record(0, 2)   # distance 2, NOT drift (d > threshold, not d >= threshold)
        assert m.drifts == 0
        m.record(0, 3)   # distance 3, drift
        assert m.drifts == 1


class TestBackwardCompat:

    def test_compute_drift_rate_shape(self):
        """compute_drift_rate should work with logit-shaped tensors."""
        hidden = torch.randn(2, 8, 16)  # batch=2, seq=8, vocab=16
        rate = compute_drift_rate(hidden, vocab_size=16)
        assert isinstance(rate, float)
        assert 0.0 <= rate <= 1.0

    def test_compute_coherence_variance(self):
        hidden = torch.randn(2, 8, 32)
        var = compute_coherence_variance(hidden)
        assert isinstance(var, float)
        assert var >= 0.0

    def test_coherence_variance_constant_input(self):
        """Constant hidden states should have near-zero variance."""
        hidden = torch.ones(1, 4, 16)
        var = compute_coherence_variance(hidden)
        assert var < 1e-6
