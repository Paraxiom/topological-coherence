#!/usr/bin/env python3
"""
MANUAL TOROIDAL GENERATION
==========================
Custom generation loop that applies toroidal bias to logits.
This is a proxy for attention modification that works reliably.
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from datetime import datetime
import os

# ============================================================================
# TOROIDAL FUNCTIONS
# ============================================================================

def toroidal_distance(i, j, grid_size=12):
    xi, yi = i % grid_size, (i // grid_size) % grid_size
    xj, yj = j % grid_size, (j // grid_size) % grid_size
    dx = min(abs(xi - xj), grid_size - abs(xi - xj))
    dy = min(abs(yi - yj), grid_size - abs(yi - yj))
    return dx + dy

def get_toroidal_logit_bias(vocab_size, current_pos, recent_tokens, grid_size=12, radius=2.0, alpha=0.3, device='cuda'):
    """
    Create logit bias based on toroidal distance from recent tokens.
    Tokens that are 'nearby' on the torus get a boost.
    """
    bias = torch.zeros(vocab_size, device=device)

    if len(recent_tokens) == 0:
        return bias

    # For each recent token, compute its position on torus
    for offset, token_id in enumerate(recent_tokens[-5:]):  # Last 5 tokens
        token_pos = token_id % (grid_size * grid_size)

        # Boost tokens that are nearby on the torus
        for vocab_id in range(min(vocab_size, grid_size * grid_size * 10)):
            target_pos = vocab_id % (grid_size * grid_size)
            dist = toroidal_distance(token_pos, target_pos, grid_size)

            if dist <= radius:
                bias[vocab_id] += alpha * (radius - dist + 1) / (offset + 1)
            elif dist <= radius * 2:
                bias[vocab_id] += alpha * 0.5 / (offset + 1)

    return bias

# ============================================================================
# GENERATION
# ============================================================================

def generate_baseline(model, tokenizer, prompt, max_tokens=30):
    """Standard greedy generation"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)

def generate_toroidal(model, tokenizer, prompt, max_tokens=30, grid_size=12, radius=2.0, alpha=0.3):
    """Generation with toroidal logit bias"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs['input_ids']
    generated = input_ids[0].tolist()

    for _ in range(max_tokens):
        with torch.no_grad():
            outputs = model(torch.tensor([generated], device=model.device))
            logits = outputs.logits[0, -1, :]

            # Apply toroidal bias
            topo_bias = get_toroidal_logit_bias(
                logits.shape[0],
                len(generated),
                generated,
                grid_size=grid_size,
                radius=radius,
                alpha=alpha,
                device=model.device
            )
            logits = logits + topo_bias

            # Greedy selection
            next_token = logits.argmax().item()

        generated.append(next_token)

        # Stop at EOS
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

def run_test(model_name="mistralai/Mistral-7B-v0.1", num_samples=20):
    print("=" * 60)
    print("TOROIDAL COHERENCE - LOGIT BIAS METHOD")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Samples: {num_samples}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print(f"Loaded. GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    prompts = (TEST_PROMPTS * 2)[:num_samples]

    results = {
        "model": model_name,
        "method": "logit_bias",
        "config": {"grid_size": 12, "radius": 2.0, "alpha": 0.3},
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

        # Toroidal
        resp_t = generate_toroidal(model, tokenizer, prompt)
        ok_t = any(e.lower() in resp_t.lower() for e in expected)
        results["toroidal"]["total"] += 1
        if ok_t:
            results["toroidal"]["correct"] += 1

        results["comparisons"].append({
            "prompt": prompt,
            "baseline": resp_b[:120],
            "toroidal": resp_t[:120],
            "b_ok": ok_b,
            "t_ok": ok_t,
        })

        m_b = "✓" if ok_b else "✗"
        m_t = "✓" if ok_t else "✗"
        diff = "SAME" if ok_b == ok_t else ("TORO+" if ok_t else "BASE+")
        print(f"[{i+1:2d}] B:{m_b} T:{m_t} {diff:5s} | {prompt[:35]}...")

    # Results
    b_acc = results["baseline"]["correct"] / results["baseline"]["total"]
    t_acc = results["toroidal"]["correct"] / results["toroidal"]["total"]
    b_err = 1 - b_acc
    t_err = 1 - t_acc

    if b_err > 0:
        reduction = ((b_err - t_err) / b_err) * 100
    else:
        reduction = 0 if t_err == 0 else -100

    results["summary"] = {
        "baseline_accuracy": b_acc,
        "toroidal_accuracy": t_acc,
        "error_reduction": reduction
    }

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Baseline accuracy: {b_acc:.1%}")
    print(f"Toroidal accuracy: {t_acc:.1%}")
    print(f"Error reduction:   {reduction:+.1f}%")

    os.makedirs("./results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = f"./results/manual_{ts}.json"
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {outfile}")

    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()
    run_test(args.model, args.samples)
