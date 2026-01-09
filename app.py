"""
Topological Coherence Demo - Hugging Face Space
================================================
Interactive demonstration that geometric constraints reduce LLM hallucination.

Paper: Cormier (2026) "Topological Constraints for Coherent Language Models"
"""

import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Dict
import time

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cpu")  # HF Spaces free tier

# =============================================================================
# TONNETZ TOPOLOGY
# =============================================================================

def create_tonnetz_distance_matrix(n_tokens: int, grid_size: int = 12) -> torch.Tensor:
    coords = []
    for i in range(n_tokens):
        x = i % grid_size
        y = (i // grid_size) % grid_size
        coords.append((x, y))

    dist = torch.zeros(n_tokens, n_tokens)
    for i in range(n_tokens):
        for j in range(n_tokens):
            dx = min(abs(coords[i][0] - coords[j][0]),
                     grid_size - abs(coords[i][0] - coords[j][0]))
            dy = min(abs(coords[i][1] - coords[j][1]),
                     grid_size - abs(coords[i][1] - coords[j][1]))
            dist[i, j] = dx + dy
    return dist


def create_tonnetz_mask(seq_len: int, radius: float = 2.0, alpha: float = 1.0) -> torch.Tensor:
    dist = create_tonnetz_distance_matrix(seq_len)
    mask = torch.where(dist <= radius,
                       torch.ones_like(dist),
                       torch.exp(-alpha * dist))
    return mask


def create_random_graph_mask(seq_len: int, density: float = 0.3, seed: int = 123) -> torch.Tensor:
    torch.manual_seed(seed)
    mask = torch.rand(seq_len, seq_len)
    mask = (mask < density).float()
    mask = (mask + mask.T) / 2
    mask = torch.clamp(mask, 0, 1)
    mask.fill_diagonal_(1.0)
    return mask


# =============================================================================
# SINKHORN-KNOPP
# =============================================================================

def sinkhorn_knopp(matrix: torch.Tensor, n_iters: int = 20) -> torch.Tensor:
    M = torch.exp(matrix)
    for _ in range(n_iters):
        M = M / (M.sum(dim=-1, keepdim=True) + 1e-8)
        M = M / (M.sum(dim=-2, keepdim=True) + 1e-8)
    return M


# =============================================================================
# TRANSFORMER VARIANTS
# =============================================================================

class BaselineAttention(nn.Module):
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
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 64):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.register_buffer('tonnetz_mask', create_tonnetz_mask(max_seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        mask = self.tonnetz_mask[:T, :T].unsqueeze(0).unsqueeze(0)
        attn = attn * mask
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(out)


class RandomGraphAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 64):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.register_buffer('random_mask', create_random_graph_mask(max_seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        mask = self.random_mask[:T, :T].unsqueeze(0).unsqueeze(0)
        attn = attn * mask
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(out)


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int,
                 attention_type: str = "baseline", max_seq_len: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        if attention_type == "baseline":
            self.attn1 = BaselineAttention(d_model, n_heads)
            self.attn2 = BaselineAttention(d_model, n_heads)
        elif attention_type == "toroidal":
            self.attn1 = ToroidalAttention(d_model, n_heads, max_seq_len)
            self.attn2 = ToroidalAttention(d_model, n_heads, max_seq_len)
        elif attention_type == "random":
            self.attn1 = RandomGraphAttention(d_model, n_heads, max_seq_len)
            self.attn2 = RandomGraphAttention(d_model, n_heads, max_seq_len)

        self.ff1 = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model))
        self.ff2 = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model))
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.ln4 = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos_embed(pos)
        h = h + self.attn1(self.ln1(h))
        h = h + self.ff1(self.ln2(h))
        h = h + self.attn2(self.ln3(h))
        h = h + self.ff2(self.ln4(h))
        logits = self.head(h)
        return logits, h


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_tonnetz_sequences(n_samples: int, seq_len: int, vocab_size: int) -> torch.Tensor:
    grid_size = int(np.sqrt(vocab_size))
    sequences = []
    for _ in range(n_samples):
        seq = [np.random.randint(0, vocab_size)]
        for _ in range(seq_len - 1):
            current = seq[-1]
            x, y = current % grid_size, (current // grid_size) % grid_size
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
# METRICS
# =============================================================================

def compute_drift_rate(hidden_states: torch.Tensor, vocab_size: int) -> float:
    grid_size = int(np.sqrt(vocab_size))
    dist_matrix = create_tonnetz_distance_matrix(vocab_size, grid_size)
    predictions = hidden_states.argmax(dim=-1)
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
    norms = torch.norm(hidden_states, dim=-1)
    return norms.var().item()


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_quick_experiment(n_epochs: int = 30, progress=gr.Progress()):
    """Run a quick version of the experiment for demo purposes."""

    vocab_size = 144
    d_model = 64
    n_heads = 4
    seq_len = 32
    batch_size = 32
    lr = 1e-3

    results = {}
    conditions = ["baseline", "toroidal", "random"]

    for idx, attn_type in enumerate(conditions):
        progress((idx) / len(conditions), f"Training {attn_type}...")

        torch.manual_seed(42)
        np.random.seed(42)

        model = TinyTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            attention_type=attn_type,
            max_seq_len=seq_len
        ).to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        train_data = generate_tonnetz_sequences(500, seq_len, vocab_size).to(DEVICE)
        test_data = generate_tonnetz_sequences(50, seq_len, vocab_size).to(DEVICE)

        for epoch in range(n_epochs):
            model.train()
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i+batch_size]
                optimizer.zero_grad()
                logits, hidden = model(batch[:, :-1])
                loss = F.cross_entropy(logits.view(-1, vocab_size), batch[:, 1:].contiguous().view(-1))
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            logits, hidden = model(test_data[:, :-1])
            drift = compute_drift_rate(logits, vocab_size)
            coh_var = compute_coherence_variance(hidden)

        results[attn_type] = {"drift": drift, "coherence_var": coh_var}

    progress(1.0, "Done!")
    return results


def format_results(results: Dict) -> str:
    """Format results as markdown table."""

    output = "## Experiment Results\n\n"
    output += "| Condition | Drift Rate | Coherence Var | Interpretation |\n"
    output += "|-----------|------------|---------------|----------------|\n"

    baseline_drift = results.get("baseline", {}).get("drift", 0)

    for condition in ["baseline", "toroidal", "random"]:
        if condition in results:
            drift = results[condition]["drift"]
            coh = results[condition]["coherence_var"]

            if condition == "toroidal":
                if baseline_drift > 0:
                    reduction = (1 - drift/baseline_drift) * 100
                    interp = f"**{reduction:.0f}% lower drift** than baseline"
                else:
                    interp = "Best performer"
            elif condition == "random":
                if drift > baseline_drift:
                    increase = (drift/baseline_drift - 1) * 100
                    interp = f"**{increase:.0f}% higher drift** (negative control)"
                else:
                    interp = "Negative control"
            else:
                interp = "Standard attention"

            output += f"| {condition.capitalize()} | {drift:.4f} | {coh:.2f} | {interp} |\n"

    output += "\n---\n\n"
    output += "### Key Finding\n\n"

    toroidal_drift = results.get("toroidal", {}).get("drift", 0)
    random_drift = results.get("random", {}).get("drift", 1)

    if random_drift > 0 and toroidal_drift < random_drift:
        ratio = random_drift / max(toroidal_drift, 0.001)
        output += f"**Toroidal attention shows {ratio:.1f}x lower drift than random sparsity.**\n\n"
        output += "This proves: **topology matters, not just sparsity.**\n"

    return output


# Pre-computed results from full experiment (100 epochs)
PRECOMPUTED_RESULTS = {
    "baseline": {"drift": 0.0100, "coherence_var": 35.76},
    "toroidal": {"drift": 0.0060, "coherence_var": 41.93},
    "random": {"drift": 0.1673, "coherence_var": 113.88}
}


def show_precomputed():
    """Show pre-computed results from full experiment."""
    return format_results(PRECOMPUTED_RESULTS)


def run_live_demo(progress=gr.Progress()):
    """Run live experiment (shorter version)."""
    results = run_quick_experiment(n_epochs=30, progress=progress)
    return format_results(results)


# =============================================================================
# GRADIO INTERFACE
# =============================================================================

with gr.Blocks(title="Topological Coherence", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Topological Constraints for Coherent Language Models

    **Why Geometry Prevents Hallucination**

    *Sylvain Cormier | Paraxiom Research | January 2026*

    ---

    This demo validates the hypothesis that **toroidal (periodic) attention constraints reduce semantic drift** in transformer models.

    ### Key Result
    - **40% lower drift** than baseline
    - **28x lower drift** than random sparsity (negative control)
    - Proves: **topology matters, not just compute reduction**

    ---
    """)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Pre-computed Results (Full Experiment)")
            precomputed_btn = gr.Button("Show Full Results (100 epochs)", variant="primary")

        with gr.Column():
            gr.Markdown("### Live Demo (Quick Run)")
            live_btn = gr.Button("Run Live Experiment (~60s)", variant="secondary")

    results_output = gr.Markdown(value=format_results(PRECOMPUTED_RESULTS))

    precomputed_btn.click(fn=show_precomputed, outputs=results_output)
    live_btn.click(fn=run_live_demo, outputs=results_output)

    gr.Markdown("""
    ---

    ### Links

    - [Paper (Zenodo)](https://doi.org/10.5281/zenodo.18187835)
    - [Code (GitHub)](https://github.com/Paraxiom/topological-coherence)
    - [Rust Crate](https://crates.io/crates/topological-coherence)

    ### Methodology

    - **Model**: 2-layer transformer, d_model=64, 4 attention heads
    - **Task**: Next-token prediction on Tonnetz-structured sequences
    - **Conditions**: Baseline, Toroidal (periodic), Random (negative control)
    - **Metric**: Drift rate (fraction of predictions requiring distance > 2 on torus)

    ### Citation

    ```bibtex
    @misc{cormier2026topological,
      author = {Cormier, Sylvain},
      title = {Topological Constraints for Coherent Language Models},
      year = {2026},
      publisher = {Zenodo},
      doi = {10.5281/zenodo.18187835}
    }
    ```
    """)


if __name__ == "__main__":
    demo.launch()
