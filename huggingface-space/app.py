"""
Toroidal Logit Bias — Interactive Demo
=======================================
Compares baseline vs toroidal-biased generation on factual prompts.
Paper: https://doi.org/10.5281/zenodo.18516477
"""

import gradio as gr
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Toroidal logit bias (from paper, Eq. 1-3)
# ---------------------------------------------------------------------------

def toroidal_distance(i, j, grid_size=12):
    """Wraparound Manhattan distance on a grid_size x grid_size torus."""
    xi, yi = i % grid_size, (i // grid_size) % grid_size
    xj, yj = j % grid_size, (j // grid_size) % grid_size
    dx = min(abs(xi - xj), grid_size - abs(xi - xj))
    dy = min(abs(yi - yj), grid_size - abs(yi - yj))
    return dx + dy


def compute_toroidal_bias(vocab_size, recent_tokens, alpha, radius, max_tokens, grid_size=12):
    """
    Compute logit bias vector from recent tokens using toroidal proximity.

    Only biases the first `max_tokens` of the vocabulary — the critical
    finding from the paper. Full-vocabulary bias causes harm.
    """
    bias = torch.zeros(vocab_size)
    k = min(len(recent_tokens), 5)

    for offset in range(1, k + 1):
        token_id = recent_tokens[-offset]
        weight = 1.0 / offset  # recency weighting

        for v in range(min(max_tokens, vocab_size)):
            d = toroidal_distance(token_id, v, grid_size)
            if d <= radius:
                bias[v] += weight * alpha * (radius - d + 1)
            elif d <= 2 * radius:
                bias[v] += weight * alpha * 0.5

    return bias


def generate(model, tokenizer, prompt, max_new_tokens=60, use_toroidal=False,
             alpha=0.3, radius=2.0, max_tokens=1440):
    """Generate text, optionally with toroidal logit bias."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    generated = input_ids[0].tolist()

    start = time.time()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            inputs = torch.tensor([generated], device=model.device)
            logits = model(inputs).logits[0, -1, :]

            if use_toroidal:
                bias = compute_toroidal_bias(
                    vocab_size=logits.shape[0],
                    recent_tokens=generated,
                    alpha=alpha,
                    radius=radius,
                    max_tokens=max_tokens,
                )
                logits = logits + bias.to(logits.device)

            next_token = logits.argmax().item()
            generated.append(next_token)

            if next_token == tokenizer.eos_token_id:
                break

    elapsed = time.time() - start
    text = tokenizer.decode(generated[input_ids.shape[1]:], skip_special_tokens=True)
    return text.strip(), elapsed


# ---------------------------------------------------------------------------
# Load model (small enough for free HF Space T4 GPU / CPU fallback)
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

print(f"Loading {MODEL_ID}...")
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=dtype, trust_remote_code=True
).to(device).eval()
print(f"Model loaded on {device}")


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------

EXAMPLE_PROMPTS = [
    "The capital of France is",
    "The chemical symbol for gold is",
    "World War II ended in",
    "The speed of light is approximately",
    "The Mona Lisa was painted by",
    "The largest planet in our solar system is",
    "Shakespeare wrote",
    "Einstein developed the theory of",
    "A byte contains how many bits:",
    "The Great Wall of China is located in",
    "Newton discovered",
    "The boiling point of water at sea level is",
]


def compare(prompt, alpha, radius, max_tokens):
    """Run baseline and toroidal generation, return both."""
    baseline_text, baseline_time = generate(
        model, tokenizer, prompt, use_toroidal=False
    )
    toroidal_text, toroidal_time = generate(
        model, tokenizer, prompt, use_toroidal=True,
        alpha=alpha, radius=radius, max_tokens=int(max_tokens)
    )

    overhead = ((toroidal_time - baseline_time) / baseline_time * 100) if baseline_time > 0 else 0

    stats = (
        f"Baseline: {baseline_time:.2f}s  |  "
        f"Toroidal: {toroidal_time:.2f}s  |  "
        f"Overhead: {overhead:+.1f}%"
    )

    return baseline_text, toroidal_text, stats


with gr.Blocks(
    title="Toroidal Logit Bias Demo",
    theme=gr.themes.Base(primary_hue="green", neutral_hue="gray"),
) as demo:
    gr.Markdown("""
# Toroidal Logit Bias for Hallucination Reduction

Map tokens to a 12x12 torus. Bias logits toward nearby tokens. No fine-tuning.

**Paper**: [DOI: 10.5281/zenodo.18516477](https://doi.org/10.5281/zenodo.18516477)
 |  **Code**: [github.com/Paraxiom/topological-coherence](https://github.com/Paraxiom/topological-coherence)
 |  **Cormier, 2026 — Paraxiom Research**
    """)

    with gr.Row():
        prompt = gr.Textbox(
            label="Prompt",
            placeholder="Enter a factual prompt...",
            value="The capital of France is",
            lines=1,
        )

    with gr.Row():
        alpha = gr.Slider(0.05, 0.5, value=0.3, step=0.05, label="Bias strength (alpha)")
        radius = gr.Slider(1.0, 5.0, value=2.0, step=0.5, label="Neighborhood radius (r)")
        max_tokens = gr.Slider(500, 5000, value=1440, step=100, label="Tokens to bias (N)")

    btn = gr.Button("Compare", variant="primary")

    with gr.Row():
        baseline_out = gr.Textbox(label="Baseline (no bias)", lines=4)
        toroidal_out = gr.Textbox(label="Toroidal Logit Bias", lines=4)

    stats_out = gr.Textbox(label="Timing", lines=1)

    btn.click(compare, inputs=[prompt, alpha, radius, max_tokens],
              outputs=[baseline_out, toroidal_out, stats_out])

    gr.Examples(
        examples=[[p, 0.3, 2.0, 1440] for p in EXAMPLE_PROMPTS],
        inputs=[prompt, alpha, radius, max_tokens],
    )

    gr.Markdown("""
---
**How it works**: Each token ID is mapped to a position on a 12x12 torus via modular
arithmetic. At each generation step, tokens "near" the recently generated tokens (in
toroidal distance) receive a small logit boost. Only the first N high-frequency tokens
are biased — full-vocabulary bias destroys performance.

**Key finding**: Structure matters, sparsity doesn't. Validated on 4 models (Qwen 0.5B/1.5B/7B,
Mistral 7B) — improvement scales with model capacity: +2.8pp on Mistral-7B, +2.1pp on Qwen-7B
(817 TruthfulQA samples, LLM-judged).

*Using Qwen 2.5-0.5B-Instruct for this demo. Full results on 7B models in the paper.*
    """)

demo.launch()
