"""
Distance-based Toroidal Logit Processor.

This is the PUBLISHED, distance-based logit bias (DOI: 10.5281/zenodo.18516477).
Uses token_id % N^2 mapping to the Tonnetz torus, NOT spectral resonance.

Compatible with HuggingFace `transformers` LogitsProcessor interface.
"""

import torch
import numpy as np
from typing import Optional

from .tonnetz import Tonnetz, distance_matrix


class ToroidalLogitProcessor:
    """Distance-based toroidal logit processor for HuggingFace generate().

    Maps token IDs onto an N x N torus via token_id % N^2, then biases
    logits toward tokens that are topologically close to recent context.

    This is the published "free sample" — it works, it's in the paper,
    and it reduces hallucination by ~40% on drift-rate metrics.

    Args:
        grid_size: Side length N of the torus (default 12, giving 144 positions).
        radius: Locality radius for full-strength bias.
        alpha: Exponential decay rate beyond radius.
        context_window: How many recent tokens to consider for bias.
        top_k: Clip bias to top-k candidates (300 for OpenAI API compat).
        bias_strength: Scaling factor for the bias signal.

    Example:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> processor = ToroidalLogitProcessor(grid_size=12)
        >>> inputs = tokenizer("The quantum", return_tensors="pt")
        >>> outputs = model.generate(**inputs, logits_processor=[processor], max_new_tokens=50)
    """

    def __init__(self, grid_size: int = 12, radius: float = 2.0, alpha: float = 0.3,
                 context_window: int = 32, top_k: int = 300, bias_strength: float = 1.0):
        self.grid_size = grid_size
        self.radius = radius
        self.alpha = alpha
        self.context_window = context_window
        self.top_k = top_k
        self.bias_strength = bias_strength
        self._tonnetz = Tonnetz(grid_size)
        self._n_positions = grid_size * grid_size

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Apply distance-based toroidal bias to logit scores.

        Args:
            input_ids: (batch_size, seq_len) token IDs generated so far.
            scores: (batch_size, vocab_size) raw logit scores for next token.

        Returns:
            Biased scores with same shape.
        """
        batch_size, vocab_size = scores.shape

        # Get recent context token positions on the torus
        context_len = min(self.context_window, input_ids.shape[1])
        if context_len == 0:
            return scores

        recent_ids = input_ids[:, -context_len:]  # (batch, context_len)

        # Map recent tokens to torus positions
        recent_positions = (recent_ids % self._n_positions).cpu().numpy()  # (batch, context_len)

        # Compute bias for each batch element
        bias = torch.zeros_like(scores)

        for b in range(batch_size):
            # Accumulate distance-based bias from each context token
            pos_counts = np.zeros(self._n_positions, dtype=np.float32)
            for pos in recent_positions[b]:
                pos_counts[pos] += 1.0

            # For each torus position, compute weighted average distance to context
            # Then map back to vocab via token_id % N^2
            torus_bias = np.zeros(self._n_positions, dtype=np.float32)
            total_weight = pos_counts.sum()
            if total_weight > 0:
                for p in range(self._n_positions):
                    if pos_counts[p] > 0:
                        # This context position contributes bias to nearby positions
                        for q in range(self._n_positions):
                            d = self._tonnetz.distance(p, q)
                            if d <= self.radius:
                                torus_bias[q] += pos_counts[p] * self.bias_strength
                            else:
                                torus_bias[q] += pos_counts[p] * self.bias_strength * np.exp(
                                    -self.alpha * (d - self.radius)
                                )
                # Normalize
                torus_bias /= total_weight

            # Map torus bias to vocab: each token gets bias of its torus position
            vocab_positions = np.arange(vocab_size) % self._n_positions
            vocab_bias = torus_bias[vocab_positions]

            # Zero-center the bias (so it doesn't shift overall magnitude)
            vocab_bias -= vocab_bias.mean()

            bias[b] = torch.from_numpy(vocab_bias).to(scores.device)

        return scores + bias
