#!/usr/bin/env python3
"""
TOROIDAL COHERENCE - FIXED VERSION
===================================
Properly applies toroidal bias to ALL vocab tokens.
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from datetime import datetime
import os

# ============================================================================
# TOROIDAL FUNCTIONS - FIXED
# ============================================================================

def toroidal_distance(i, j, grid_size=12):
    """Manhattan distance on 2D torus"""
    xi, yi = i % grid_size, (i // grid_size) % grid_size
    xj, yj = j % grid_size, (j // grid_size) % grid_size
    dx = min(abs(xi - xj), grid_size - abs(xi - xj))
    dy = min(abs(yi - yj), grid_size - abs(yi - yj))
    return dx + dy

# Pre-compute distance matrix for efficiency
_DIST_MATRIX = {}
def get_distance_matrix(grid_size=12):
    if grid_size not in _DIST_MATRIX:
        n = grid_size * grid_size
        matrix = torch.zeros(n, n)
        for i in range(n):
            for j in range(n):
                matrix[i, j] = toroidal_distance(i, j, grid_size)
        _DIST_MATRIX[grid_size] = matrix
    return _DIST_MATRIX[grid_size]

def get_toroidal_logit_bias_v2(vocab_size, recent_tokens, grid_size=12, radius=2.0, alpha=1.0, device='cuda'):
    """
    FIXED: Apply toroidal bias to ALL vocab tokens.

    Method: Each vocab token maps to a torus position.
    Tokens "near" recent tokens on torus get boosted.
    """
    bias = torch.zeros(vocab_size, device=device, dtype=torch.float16)

    if len(recent_tokens) < 2:
        return bias

    grid_cells = grid_size * grid_size  # 144 for 12x12

    # Get positions of recent tokens on torus
    recent_positions = [t % grid_cells for t in recent_tokens[-5:]]

    # For EVERY vocab token, compute its torus position and distance to recent tokens
    for vocab_id in range(vocab_size):
        vocab_pos = vocab_id % grid_cells

        # Average distance to recent tokens
        total_boost = 0.0
        for i, recent_pos in enumerate(recent_positions):
            dist = toroidal_distance(vocab_pos, recent_pos, grid_size)
            weight = 1.0 / (i + 1)  # More recent = more weight

            if dist <= radius:
                # Boost nearby tokens
                total_boost += alpha * (radius - dist + 1) * weight
            elif dist <= radius * 2:
                # Smaller boost for medium distance
                total_boost += alpha * 0.3 * weight
            # Far tokens get no boost (bias stays 0)

        bias[vocab_id] = total_boost

    return bias

# Vectorized version for speed
def get_toroidal_logit_bias_fast(vocab_size, recent_tokens, grid_size=12, radius=2.0, alpha=1.0, device='cuda'):
    """Vectorized version - much faster"""
    if len(recent_tokens) < 2:
        return torch.zeros(vocab_size, device=device, dtype=torch.float16)

    grid_cells = grid_size * grid_size

    # All vocab positions on torus
    vocab_positions = torch.arange(vocab_size, device=device) % grid_cells

    # Compute bias
    bias = torch.zeros(vocab_size, device=device, dtype=torch.float16)

    for i, token in enumerate(recent_tokens[-5:]):
        token_pos = token % grid_cells

        # Compute torus distance for all vocab tokens at once
        # Map to 2D coordinates
        vx = vocab_positions % grid_size
        vy = vocab_positions // grid_size
        tx = token_pos % grid_size
        ty = token_pos // grid_size

        # Manhattan distance with wraparound
        dx = torch.minimum(torch.abs(vx - tx), grid_size - torch.abs(vx - tx))
        dy = torch.minimum(torch.abs(vy - ty), grid_size - torch.abs(vy - ty))
        dist = dx + dy

        weight = 1.0 / (i + 1)

        # Apply boost based on distance
        near_mask = dist <= radius
        mid_mask = (dist > radius) & (dist <= radius * 2)

        bias[near_mask] += alpha * (radius - dist[near_mask].float() + 1) * weight
        bias[mid_mask] += alpha * 0.3 * weight

    return bias

# ============================================================================
# GENERATION
# ============================================================================

def generate_baseline(model, tokenizer, prompt, max_tokens=30):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                            pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0], skip_special_tokens=True)

def generate_toroidal(model, tokenizer, prompt, max_tokens=30,
                      grid_size=12, radius=2.0, alpha=1.0):
    """Generation with FIXED toroidal bias"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generated = inputs['input_ids'][0].tolist()
    vocab_size = model.config.vocab_size

    for step in range(max_tokens):
        with torch.no_grad():
            outputs = model(torch.tensor([generated], device=model.device))
            logits = outputs.logits[0, -1, :]

            # Apply FIXED toroidal bias
            topo_bias = get_toroidal_logit_bias_fast(
                vocab_size, generated, grid_size, radius, alpha, model.device
            )
            logits = logits + topo_bias

            next_token = logits.argmax().item()

        generated.append(next_token)
        if next_token == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated, skip_special_tokens=True)

# ============================================================================
# TEST
# ============================================================================

TEST_PROMPTS = [
    ("The capital of France is", ["Paris"]),
    ("Water freezes at", ["0", "32", "zero"]),
    ("The largest planet is", ["Jupiter"]),
    ("Einstein developed the theory of", ["relativity"]),
    ("The chemical symbol for gold is", ["Au"]),
    ("World War II ended in", ["1945"]),
    ("DNA stands for", ["deoxyribonucleic"]),
    ("The Mona Lisa was painted by", ["Leonardo", "Vinci"]),
    ("Shakespeare wrote", ["Hamlet", "Romeo", "Macbeth"]),
    ("Mount Everest is in", ["Nepal", "Himalaya"]),
    ("The atomic number of hydrogen is", ["1", "one"]),
    ("Photosynthesis converts", ["energy", "glucose", "sugar"]),
    ("The currency of Japan is", ["yen"]),
    ("Newton discovered", ["gravity", "motion"]),
    ("The Amazon River is in", ["South America", "Brazil"]),
    ("The Great Wall is in", ["China"]),
    ("Oxygen is about what percent of air", ["21", "20"]),
    ("Pi equals approximately", ["3.14"]),
    ("The speed of light is", ["300", "299", "186"]),
    ("The human heart has", ["four", "4"]),
]

def run_test(model_name="allenai/OLMo-1.7-7B-hf", num_samples=20, alpha=1.0):
    print("=" * 60)
    print("TOROIDAL COHERENCE - FIXED VERSION")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Samples: {num_samples}")
    print(f"Alpha (bias strength): {alpha}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    print(f"Loaded. GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB")
    print(f"Vocab size: {model.config.vocab_size}")

    prompts = (TEST_PROMPTS * ((num_samples // len(TEST_PROMPTS)) + 1))[:num_samples]

    results = {
        "model": model_name,
        "alpha": alpha,
        "baseline": {"correct": 0, "total": 0},
        "toroidal": {"correct": 0, "total": 0},
        "comparisons": []
    }

    print("\nRunning tests...")

    for i, (prompt, expected) in enumerate(prompts):
        # Baseline
        resp_b = generate_baseline(model, tokenizer, prompt)
        ok_b = any(e.lower() in resp_b.lower() for e in expected)
        results["baseline"]["total"] += 1
        if ok_b:
            results["baseline"]["correct"] += 1

        # Toroidal with stronger bias
        resp_t = generate_toroidal(model, tokenizer, prompt, alpha=alpha)
        ok_t = any(e.lower() in resp_t.lower() for e in expected)
        results["toroidal"]["total"] += 1
        if ok_t:
            results["toroidal"]["correct"] += 1

        results["comparisons"].append({
            "prompt": prompt,
            "baseline": resp_b[:120],
            "toroidal": resp_t[:120],
            "b_ok": ok_b, "t_ok": ok_t
        })

        m_b = "✓" if ok_b else "✗"
        m_t = "✓" if ok_t else "✗"
        diff = "SAME" if ok_b == ok_t else ("TORO+" if ok_t else "BASE+")
        print(f"[{i+1:2d}] B:{m_b} T:{m_t} {diff:5s} | {prompt[:35]}...")

    # Results
    b_acc = results["baseline"]["correct"] / results["baseline"]["total"]
    t_acc = results["toroidal"]["correct"] / results["toroidal"]["total"]
    b_err, t_err = 1 - b_acc, 1 - t_acc
    reduction = ((b_err - t_err) / b_err * 100) if b_err > 0 else (0 if t_err == 0 else -100)

    results["summary"] = {"baseline": b_acc, "toroidal": t_acc, "reduction": reduction}

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Baseline accuracy: {b_acc:.1%}")
    print(f"Toroidal accuracy: {t_acc:.1%}")
    print(f"Error reduction:   {reduction:+.1f}%")

    os.makedirs("./results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = f"./results/fixed_{ts}.json"
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {outfile}")

    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="allenai/OLMo-1.7-7B-hf")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=1.0, help="Bias strength (try 1.0, 2.0, 5.0)")
    args = parser.parse_args()
    run_test(args.model, args.samples, args.alpha)
