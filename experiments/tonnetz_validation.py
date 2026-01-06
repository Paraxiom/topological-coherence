"""
Minimal Validation Experiment: Topological Constraints for Coherent LLMs
=========================================================================
From: Cormier (2026) "Topological Constraints for Coherent Language Models"

Setup:
- 2-layer transformer, d_model=64, 4 heads
- Synthetic task with controlled semantic drift
- 3 conditions: Baseline, mHC (doubly-stochastic), Toroidal

Cost: <1 GPU-hour (runs on Colab free tier or CPU)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict
import math

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {DEVICE}")


# =============================================================================
# 1. TONNETZ TOPOLOGY
# =============================================================================

def create_tonnetz_distance_matrix(n_tokens: int, grid_size: int = 12) -> torch.Tensor:
    """
    Create toroidal distance matrix for token positions.
    Maps tokens to a 2D torus and computes graph distances.
    """
    # Map tokens to 2D torus coordinates
    coords = []
    for i in range(n_tokens):
        x = i % grid_size
        y = (i // grid_size) % grid_size
        coords.append((x, y))

    # Compute toroidal distances
    dist = torch.zeros(n_tokens, n_tokens)
    for i in range(n_tokens):
        for j in range(n_tokens):
            dx = min(abs(coords[i][0] - coords[j][0]),
                     grid_size - abs(coords[i][0] - coords[j][0]))
            dy = min(abs(coords[i][1] - coords[j][1]),
                     grid_size - abs(coords[i][1] - coords[j][1]))
            dist[i, j] = dx + dy  # Manhattan distance on torus

    return dist


def create_tonnetz_mask(seq_len: int, radius: float = 2.0, alpha: float = 1.0) -> torch.Tensor:
    """
    Create attention mask based on Tonnetz topology.
    Eq. from paper: M(i,j) = 1 if d <= r, else exp(-alpha * d)
    """
    dist = create_tonnetz_distance_matrix(seq_len)
    mask = torch.where(dist <= radius,
                       torch.ones_like(dist),
                       torch.exp(-alpha * dist))
    return mask


def create_random_graph_mask(seq_len: int, density: float = 0.3, seed: int = 123) -> torch.Tensor:
    """
    NEGATIVE CONTROL: Random graph mask (no topological structure).
    Same approximate sparsity as Tonnetz but random connectivity.
    """
    torch.manual_seed(seed)
    mask = torch.rand(seq_len, seq_len)
    mask = (mask < density).float()
    mask = (mask + mask.T) / 2  # Symmetrize
    mask = torch.clamp(mask, 0, 1)
    mask.fill_diagonal_(1.0)  # Self-attention always allowed
    return mask


# =============================================================================
# 2. SINKHORN-KNOPP (mHC doubly-stochastic projection)
# =============================================================================

def sinkhorn_knopp(matrix: torch.Tensor, n_iters: int = 20) -> torch.Tensor:
    """
    Project matrix onto Birkhoff polytope (doubly-stochastic).
    From mHC paper (Xie et al., 2026).
    """
    # Make positive
    M = torch.exp(matrix)

    for _ in range(n_iters):
        # Row normalization
        M = M / (M.sum(dim=-1, keepdim=True) + 1e-8)
        # Column normalization
        M = M / (M.sum(dim=-2, keepdim=True) + 1e-8)

    return M


# =============================================================================
# 3. TRANSFORMER VARIANTS
# =============================================================================

class BaselineAttention(nn.Module):
    """Standard unconstrained attention."""
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(out)


class ToroidalAttention(nn.Module):
    """Tonnetz-constrained attention (Eq. 17 in paper)."""
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 64):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Pre-compute Tonnetz mask
        self.register_buffer('tonnetz_mask', create_tonnetz_mask(max_seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply Tonnetz mask (element-wise multiply before softmax)
        mask = self.tonnetz_mask[:T, :T].unsqueeze(0).unsqueeze(0)
        attn = attn * mask

        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(out)


class mHCAttention(nn.Module):
    """mHC-style doubly-stochastic residual mixing."""
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.base_attn = BaselineAttention(d_model, n_heads)
        self.mix_weights = nn.Parameter(torch.randn(d_model, d_model) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.base_attn(x)

        # Apply doubly-stochastic mixing to residual
        mix = sinkhorn_knopp(self.mix_weights)
        mixed = F.linear(attn_out, mix)

        return mixed


class RandomGraphAttention(nn.Module):
    """NEGATIVE CONTROL: Random graph mask (no topological structure)."""
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 64):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Pre-compute random mask (same sparsity, no structure)
        self.register_buffer('random_mask', create_random_graph_mask(max_seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply random mask
        mask = self.random_mask[:T, :T].unsqueeze(0).unsqueeze(0)
        attn = attn * mask

        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(out)


class TinyTransformer(nn.Module):
    """2-layer transformer for validation experiment."""
    def __init__(self, vocab_size: int, d_model: int, n_heads: int,
                 attention_type: str = "baseline", max_seq_len: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        # Select attention type
        if attention_type == "baseline":
            self.attn1 = BaselineAttention(d_model, n_heads)
            self.attn2 = BaselineAttention(d_model, n_heads)
        elif attention_type == "mhc":
            self.attn1 = mHCAttention(d_model, n_heads)
            self.attn2 = mHCAttention(d_model, n_heads)
        elif attention_type == "toroidal":
            self.attn1 = ToroidalAttention(d_model, n_heads, max_seq_len)
            self.attn2 = ToroidalAttention(d_model, n_heads, max_seq_len)
        elif attention_type == "random":
            self.attn1 = RandomGraphAttention(d_model, n_heads, max_seq_len)
            self.attn2 = RandomGraphAttention(d_model, n_heads, max_seq_len)

        self.ff1 = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.ff2 = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.ln4 = nn.LayerNorm(d_model)

        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)

        h = self.embed(x) + self.pos_embed(pos)

        # Layer 1
        h = h + self.attn1(self.ln1(h))
        h = h + self.ff1(self.ln2(h))

        # Layer 2
        h = h + self.attn2(self.ln3(h))
        h = h + self.ff2(self.ln4(h))

        logits = self.head(h)
        return logits, h  # Return hidden states for coherence analysis


# =============================================================================
# 4. SYNTHETIC DATASET WITH CONTROLLED DRIFT
# =============================================================================

def generate_tonnetz_sequences(n_samples: int, seq_len: int, vocab_size: int) -> torch.Tensor:
    """
    Generate sequences where valid next tokens are Tonnetz-adjacent.
    Invalid continuations require "jumps" across the torus.
    """
    grid_size = int(np.sqrt(vocab_size))
    sequences = []

    for _ in range(n_samples):
        seq = [np.random.randint(0, vocab_size)]

        for _ in range(seq_len - 1):
            current = seq[-1]
            x, y = current % grid_size, (current // grid_size) % grid_size

            # Valid next tokens: Tonnetz-adjacent (distance <= 2)
            neighbors = []
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if abs(dx) + abs(dy) <= 2:
                        nx = (x + dx) % grid_size
                        ny = (y + dy) % grid_size
                        neighbors.append(ny * grid_size + nx)

            next_token = np.random.choice(neighbors)
            seq.append(next_token)

        sequences.append(seq)

    return torch.tensor(sequences, dtype=torch.long)


# =============================================================================
# 5. METRICS
# =============================================================================

def compute_drift_rate(hidden_states: torch.Tensor, vocab_size: int) -> float:
    """
    Measure how often predictions would require Tonnetz distance > 2.
    """
    grid_size = int(np.sqrt(vocab_size))
    dist_matrix = create_tonnetz_distance_matrix(vocab_size, grid_size)

    # Get predicted tokens
    predictions = hidden_states.argmax(dim=-1)  # B, T

    drift_count = 0
    total = 0

    for b in range(predictions.shape[0]):
        for t in range(predictions.shape[1] - 1):
            current = predictions[b, t].item()
            next_pred = predictions[b, t + 1].item()

            if current < vocab_size and next_pred < vocab_size:
                d = dist_matrix[current, next_pred].item()
                if d > 2:
                    drift_count += 1
                total += 1

    return drift_count / max(total, 1)


def compute_coherence_variance(hidden_states: torch.Tensor) -> float:
    """
    Measure variance in hidden state norms across sequence.
    Lower = more stable/coherent.
    """
    norms = torch.norm(hidden_states, dim=-1)  # B, T
    return norms.var().item()


def compute_gradient_norm(model: nn.Module) -> float:
    """Compute total gradient norm."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


# =============================================================================
# 6. MAIN EXPERIMENT
# =============================================================================

def run_experiment(attention_type: str, n_epochs: int = 100) -> Dict:
    """Run training and collect metrics."""

    # Hyperparameters (from paper)
    vocab_size = 144  # 12x12 grid
    d_model = 64
    n_heads = 4
    seq_len = 32
    batch_size = 32
    lr = 1e-3

    # Create model
    model = TinyTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        attention_type=attention_type,
        max_seq_len=seq_len
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Generate data
    train_data = generate_tonnetz_sequences(1000, seq_len, vocab_size).to(DEVICE)
    test_data = generate_tonnetz_sequences(100, seq_len, vocab_size).to(DEVICE)

    # Training metrics
    metrics = {
        "loss": [],
        "drift_rate": [],
        "coherence_var": [],
        "grad_norm": []
    }

    print(f"\n{'='*50}")
    print(f"Training: {attention_type.upper()}")
    print(f"{'='*50}")

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0

        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]

            optimizer.zero_grad()
            logits, hidden = model(batch[:, :-1])

            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                batch[:, 1:].contiguous().view(-1)
            )

            loss.backward()

            # Record gradient norm at step 1000 equivalent
            if epoch == n_epochs // 10:
                metrics["grad_norm"].append(compute_gradient_norm(model))

            optimizer.step()
            epoch_loss += loss.item()

        # Evaluation
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            model.eval()
            with torch.no_grad():
                logits, hidden = model(test_data[:, :-1])

                drift = compute_drift_rate(logits, vocab_size)
                coh_var = compute_coherence_variance(hidden)

                metrics["loss"].append(epoch_loss)
                metrics["drift_rate"].append(drift)
                metrics["coherence_var"].append(coh_var)

                print(f"Epoch {epoch:3d} | Loss: {epoch_loss:.4f} | "
                      f"Drift: {drift:.4f} | Coherence Var: {coh_var:.4f}")

    return metrics


def main():
    print("="*60)
    print("MINIMAL VALIDATION: Topological Constraints for Coherent LLMs")
    print("="*60)
    print(f"\nHypothesis: Toroidal attention shows lowest drift rate,")
    print(f"lowest coherence variance, and stable gradients.\n")

    results = {}

    for attn_type in ["baseline", "mhc", "toroidal", "random"]:
        results[attn_type] = run_experiment(attn_type, n_epochs=100)

    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    print(f"\n{'Condition':<12} | {'Final Drift':<12} | {'Final Coh.Var':<14} | {'Grad Norm':<12}")
    print("-" * 60)

    for attn_type in ["baseline", "mhc", "toroidal", "random"]:
        m = results[attn_type]
        drift = m["drift_rate"][-1] if m["drift_rate"] else 0
        coh = m["coherence_var"][-1] if m["coherence_var"] else 0
        grad = np.mean(m["grad_norm"]) if m["grad_norm"] else 0

        print(f"{attn_type:<12} | {drift:<12.4f} | {coh:<14.4f} | {grad:<12.4f}")

    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)

    baseline_drift = results["baseline"]["drift_rate"][-1]
    toroidal_drift = results["toroidal"]["drift_rate"][-1]

    if toroidal_drift < baseline_drift:
        print("\n✓ HYPOTHESIS SUPPORTED: Toroidal attention reduces drift rate")
        reduction = (1 - toroidal_drift/baseline_drift) * 100
        print(f"  Drift reduction: {reduction:.1f}%")
    else:
        print("\n✗ HYPOTHESIS NOT SUPPORTED in this run")
        print("  (May need hyperparameter tuning or longer training)")

    return results


if __name__ == "__main__":
    results = main()
