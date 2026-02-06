#!/usr/bin/env python3
"""
SMART BIAS - Skip byte tokens, only bias real words
====================================================
OLMo's first ~256 tokens are byte-level tokens that rarely appear.
We should skip those and bias tokens 256-1696 instead.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from datetime import datetime
import os
import gc
import argparse

def toroidal_distance(i, j, grid_size=12):
    xi, yi = i % grid_size, (i // grid_size) % grid_size
    xj, yj = j % grid_size, (j // grid_size) % grid_size
    dx = min(abs(xi - xj), grid_size - abs(xi - xj))
    dy = min(abs(yi - yj), grid_size - abs(yi - yj))
    return dx + dy

def get_bias_SMART(vocab_size, recent_tokens, grid_size=12, radius=2.0, alpha=0.3,
                   start_token=256, num_tokens=1440, device='cuda'):
    """
    SMART: Skip byte tokens (0-255), bias tokens starting from 256.
    This targets actual word tokens, not byte-level encoding artifacts.
    """
    bias = torch.zeros(vocab_size, device=device, dtype=torch.float16)

    if len(recent_tokens) == 0:
        return bias

    end_token = min(vocab_size, start_token + num_tokens)

    for offset, token_id in enumerate(recent_tokens[-5:]):
        # Map token position relative to start_token
        token_pos = (token_id - start_token) % (grid_size * grid_size) if token_id >= start_token else token_id % (grid_size * grid_size)

        for vocab_id in range(start_token, end_token):
            target_pos = (vocab_id - start_token) % (grid_size * grid_size)
            dist = toroidal_distance(token_pos, target_pos, grid_size)

            if dist <= radius:
                bias[vocab_id] += alpha * (radius - dist + 1) / (offset + 1)
            elif dist <= radius * 2:
                bias[vocab_id] += alpha * 0.5 / (offset + 1)

    return bias

def get_bias_LIMITED(vocab_size, recent_tokens, grid_size=12, radius=2.0, alpha=0.3, device='cuda'):
    """Original: bias tokens 0-1439"""
    bias = torch.zeros(vocab_size, device=device, dtype=torch.float16)
    if len(recent_tokens) == 0:
        return bias
    max_tokens = grid_size * grid_size * 10
    for offset, token_id in enumerate(recent_tokens[-5:]):
        token_pos = token_id % (grid_size * grid_size)
        for vocab_id in range(min(vocab_size, max_tokens)):
            target_pos = vocab_id % (grid_size * grid_size)
            dist = toroidal_distance(token_pos, target_pos, grid_size)
            if dist <= radius:
                bias[vocab_id] += alpha * (radius - dist + 1) / (offset + 1)
            elif dist <= radius * 2:
                bias[vocab_id] += alpha * 0.5 / (offset + 1)
    return bias

def get_bias_NEGATIVE(vocab_size, recent_tokens, grid_size=12, radius=2.0, alpha=0.3, device='cuda'):
    """
    NEGATIVE: Penalize distant tokens instead of boosting near ones.
    Theory: reduce probability of incoherent jumps.
    """
    bias = torch.zeros(vocab_size, device=device, dtype=torch.float16)
    if len(recent_tokens) == 0:
        return bias

    for offset, token_id in enumerate(recent_tokens[-5:]):
        token_pos = token_id % (grid_size * grid_size)

        for vocab_id in range(min(vocab_size, 5000)):  # Check first 5000 tokens
            target_pos = vocab_id % (grid_size * grid_size)
            dist = toroidal_distance(token_pos, target_pos, grid_size)

            weight = 1.0 / (offset + 1)

            if dist > radius * 3:
                # Penalize very distant tokens
                bias[vocab_id] -= alpha * 0.5 * weight
            elif dist > radius * 2:
                # Small penalty for medium-distant
                bias[vocab_id] -= alpha * 0.2 * weight
            # Near tokens get no change (not boosted, just not penalized)

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

def generate_with_bias(model, tokenizer, prompt, bias_fn, bias_args, max_tokens=30):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generated = inputs['input_ids'][0].tolist()
    vocab_size = model.config.vocab_size

    for _ in range(max_tokens):
        with torch.no_grad():
            outputs = model(torch.tensor([generated], device=model.device))
            logits = outputs.logits[0, -1, :]
            topo_bias = bias_fn(vocab_size, generated, **bias_args, device=model.device)
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
    ("The capital of Japan is", ["Tokyo"]),
    ("The capital of Australia is", ["Canberra"]),
    ("The capital of Brazil is", ["Brasilia"]),
    ("The capital of Canada is", ["Ottawa"]),
    ("Mount Everest is in", ["Nepal", "Himalaya"]),
    ("The Amazon River is in", ["South America", "Brazil"]),
    ("The Great Wall is in", ["China"]),
    ("The Nile River flows through", ["Egypt", "Africa"]),
    ("The Sahara Desert is in", ["Africa"]),
    ("The largest ocean is the", ["Pacific"]),
    ("The longest river in the world is the", ["Nile", "Amazon"]),
    ("Australia is a", ["continent", "country"]),
    ("The Alps are in", ["Europe"]),
    ("The Dead Sea is between", ["Israel", "Jordan"]),
    ("Venice is famous for its", ["canals", "water"]),
    ("The Eiffel Tower is in", ["Paris", "France"]),
    ("The Great Barrier Reef is near", ["Australia"]),
    ("Tokyo is the capital of", ["Japan"]),
    ("The Statue of Liberty is in", ["New York", "USA"]),
    ("Water freezes at", ["0", "32", "zero"]),
    ("The largest planet is", ["Jupiter"]),
    ("Einstein developed the theory of", ["relativity"]),
    ("The chemical symbol for gold is", ["Au"]),
    ("DNA stands for", ["deoxyribonucleic"]),
    ("The atomic number of hydrogen is", ["1", "one"]),
    ("Photosynthesis converts", ["energy", "glucose", "sugar", "sunlight"]),
    ("Oxygen is about what percent of air", ["21", "20"]),
    ("Pi equals approximately", ["3.14"]),
    ("The speed of light is", ["300", "299", "186"]),
    ("The human heart has", ["four", "4"]),
    ("Newton discovered", ["gravity", "motion"]),
    ("The chemical symbol for water is", ["H2O"]),
    ("The boiling point of water is", ["100", "212"]),
    ("Electrons have a", ["negative"]),
    ("The sun is a", ["star"]),
    ("Diamonds are made of", ["carbon"]),
    ("The human body has how many bones", ["206"]),
    ("Sound travels faster in", ["water", "solid"]),
    ("The earth revolves around the", ["sun"]),
    ("Gravity was discovered by", ["Newton"]),
    ("The smallest unit of life is a", ["cell"]),
    ("Mitochondria are the", ["powerhouse"]),
    ("The pH of pure water is", ["7", "seven", "neutral"]),
    ("The chemical symbol for iron is", ["Fe"]),
    ("World War II ended in", ["1945"]),
    ("The Declaration of Independence was signed in", ["1776"]),
    ("The first human on the moon was", ["Armstrong", "Neil"]),
    ("The Berlin Wall fell in", ["1989"]),
    ("World War I started in", ["1914"]),
]

def run_test(model_name, num_samples=50, alpha=0.1):
    print("=" * 70)
    print("SMART BIAS TEST - Skip byte tokens + Negative bias")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Samples: {num_samples}")
    print(f"Alpha: {alpha}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Show what first tokens look like
    print("\nFirst 10 tokens (byte-level):")
    for i in range(10):
        print(f"  {i}: {repr(tokenizer.decode([i]))}")
    print("Tokens 256-266 (real words):")
    for i in range(256, 266):
        print(f"  {i}: {repr(tokenizer.decode([i]))}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    print(f"\nLoaded. GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    prompts = TEST_PROMPTS[:num_samples]

    results = {
        "baseline": {"correct": 0, "total": 0},
        "limited": {"correct": 0, "total": 0},
        "smart": {"correct": 0, "total": 0},
        "negative": {"correct": 0, "total": 0},
    }

    print(f"\nRunning {num_samples} tests (4 conditions)...")
    print("-" * 70)

    for i, (prompt, expected) in enumerate(prompts):
        # Baseline
        resp_b = generate_baseline(model, tokenizer, prompt)
        ok_b = any(e.lower() in resp_b.lower() for e in expected)
        results["baseline"]["total"] += 1
        if ok_b: results["baseline"]["correct"] += 1

        # Limited (original)
        resp_l = generate_with_bias(model, tokenizer, prompt, get_bias_LIMITED,
                                    {"grid_size": 12, "radius": 2.0, "alpha": alpha})
        ok_l = any(e.lower() in resp_l.lower() for e in expected)
        results["limited"]["total"] += 1
        if ok_l: results["limited"]["correct"] += 1

        # Smart (skip byte tokens)
        resp_s = generate_with_bias(model, tokenizer, prompt, get_bias_SMART,
                                    {"grid_size": 12, "radius": 2.0, "alpha": alpha,
                                     "start_token": 256, "num_tokens": 1440})
        ok_s = any(e.lower() in resp_s.lower() for e in expected)
        results["smart"]["total"] += 1
        if ok_s: results["smart"]["correct"] += 1

        # Negative (penalize distant)
        resp_n = generate_with_bias(model, tokenizer, prompt, get_bias_NEGATIVE,
                                    {"grid_size": 12, "radius": 2.0, "alpha": alpha})
        ok_n = any(e.lower() in resp_n.lower() for e in expected)
        results["negative"]["total"] += 1
        if ok_n: results["negative"]["correct"] += 1

        b = "Y" if ok_b else "X"
        l = "Y" if ok_l else "X"
        s = "Y" if ok_s else "X"
        n = "Y" if ok_n else "X"
        print(f"[{i+1:2d}] B:{b} L:{l} S:{s} N:{n} | {prompt[:40]}...")

    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    b_acc = results["baseline"]["correct"] / results["baseline"]["total"]
    b_err = 1 - b_acc

    for method in ["limited", "smart", "negative"]:
        acc = results[method]["correct"] / results[method]["total"]
        err_red = ((b_err - (1-acc)) / b_err * 100) if b_err > 0 else 0
        print(f"{method:10s}: {acc:.1%} ({results[method]['correct']}/{num_samples}) | Error reduction: {err_red:+.1f}%")

    print(f"{'baseline':10s}: {b_acc:.1%} ({results['baseline']['correct']}/{num_samples})")

    # Save
    os.makedirs("./results", exist_ok=True)
    model_short = model_name.split("/")[-1].replace("-", "_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"./results/smart_bias_{model_short}_{ts}.json", "w") as f:
        json.dump(results, f, indent=2)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="allenai/OLMo-1.7-7B-hf")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()
    run_test(args.model, args.samples, args.alpha)
