"""Tests for distance-based ToroidalLogitProcessor."""

import torch
import pytest

from topological_coherence.logit_bias import ToroidalLogitProcessor


class TestToroidalLogitProcessor:

    def test_output_shape(self):
        proc = ToroidalLogitProcessor(grid_size=12)
        input_ids = torch.randint(0, 1000, (2, 16))
        scores = torch.randn(2, 1000)
        out = proc(input_ids, scores)
        assert out.shape == scores.shape

    def test_no_context_passthrough(self):
        """With no context tokens, scores should be unchanged."""
        proc = ToroidalLogitProcessor(grid_size=12)
        input_ids = torch.randint(0, 1000, (1, 0))
        scores = torch.randn(1, 500)
        out = proc(input_ids, scores)
        torch.testing.assert_close(out, scores)

    def test_bias_is_zero_centered(self):
        """Bias should be approximately zero-mean across vocab."""
        proc = ToroidalLogitProcessor(grid_size=6, bias_strength=2.0)
        input_ids = torch.tensor([[10, 20, 30, 40]])
        scores = torch.zeros(1, 200)
        out = proc(input_ids, scores)
        # The bias (which is the output since scores=0) should be ~zero mean
        assert abs(out.mean().item()) < 0.1

    def test_bias_sign(self):
        """Tokens near context should get positive bias, far ones negative."""
        proc = ToroidalLogitProcessor(grid_size=6, bias_strength=2.0, context_window=4)
        # All context tokens map to position 0 on the 6x6 torus
        input_ids = torch.tensor([[0, 36, 0, 36]])  # all mod 36 == 0
        scores = torch.zeros(1, 36)

        out = proc(input_ids, scores)
        bias = out[0]
        # Token 0 maps to position 0, should have highest bias
        # Token at max distance should have lowest bias
        assert bias[0] > bias.mean()

    def test_batch_independence(self):
        """Each batch element should be processed independently."""
        proc = ToroidalLogitProcessor(grid_size=6)
        input_ids = torch.tensor([[0, 1, 2], [10, 20, 30]])
        scores = torch.zeros(2, 100)
        out = proc(input_ids, scores)
        # Different contexts should produce different biases
        assert not torch.allclose(out[0], out[1])

    def test_context_window_respected(self):
        """Only last `context_window` tokens should matter."""
        proc = ToroidalLogitProcessor(grid_size=6, context_window=2)
        # Long context but only last 2 matter
        ids_long = torch.tensor([[99, 99, 99, 99, 0, 1]])
        ids_short = torch.tensor([[0, 1]])
        scores = torch.zeros(1, 100)
        out_long = proc(ids_long, scores)
        out_short = proc(ids_short, scores)
        torch.testing.assert_close(out_long, out_short)
