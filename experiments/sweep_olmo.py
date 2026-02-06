#!/usr/bin/env python3
"""
PARAMETER SWEEP FOR OLMo
========================
Find optimal alpha, radius, and token count for OLMo.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from datetime import datetime
import os
import gc

# ============================================================================
# BIAS FUNCTION WITH CONFIGURABLE TOKEN COUNT
# ============================================================================

def toroidal_distance(i, j, grid_size=12):
    xi, yi = i % grid_size, (i // grid_size) % grid_size
    xj, yj = j % grid_size, (j // grid_size) % grid_size
    dx = min(abs(xi - xj), grid_size - abs(xi - xj))
    dy = min(abs(yi - yj), grid_size - abs(yi - yj))
    return dx + dy

def get_bias(vocab_size, recent_tokens, grid_size=12, radius=2.0, alpha=0.3,
             max_tokens=1440, device='cuda'):
    bias = torch.zeros(vocab_size, device=device, dtype=torch.float16)

    if len(recent_tokens) == 0:
        return bias

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

# ============================================================================
# GENERATION
# ============================================================================

def generate_baseline(model, tokenizer, prompt, max_tokens=30):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                            pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0], skip_special_tokens=True)

def generate_with_bias(model, tokenizer, prompt, grid_size, radius, alpha, max_tokens_bias, max_gen=30):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generated = inputs['input_ids'][0].tolist()
    vocab_size = model.config.vocab_size

    for _ in range(max_gen):
        with torch.no_grad():
            outputs = model(torch.tensor([generated], device=model.device))
            logits = outputs.logits[0, -1, :]

            topo_bias = get_bias(vocab_size, generated, grid_size, radius, alpha,
                                max_tokens_bias, model.device)
            logits = logits + topo_bias

            next_token = logits.argmax().item()

        generated.append(next_token)
        if next_token == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated, skip_special_tokens=True)

# ============================================================================
# TEST PROMPTS - Focus on ones OLMo gets wrong
# ============================================================================

TEST_PROMPTS = [
    ("The Alps are in", ["Europe"]),
    ("The Great Barrier Reef is near", ["Australia"]),
    ("DNA stands for", ["deoxyribonucleic"]),
    ("Pi equals approximately", ["3.14"]),
    ("The speed of light is", ["300", "299", "186"]),
    ("Newton discovered", ["gravity", "motion"]),
    ("Sound travels faster in", ["water", "solid"]),
    ("Martin Luther King Jr gave his famous speech in", ["1963"]),
    ("Shakespeare wrote", ["Hamlet", "Romeo", "Macbeth"]),
    ("The first computer programmer was", ["Ada", "Lovelace"]),
    ("A decade is how many years", ["10", "ten"]),
    ("The largest mammal is the", ["whale", "blue"]),
    ("The Great Pyramid was built in", ["Egypt", "Giza"]),
    # Add some that baseline gets right to check for regressions
    ("The capital of France is", ["Paris"]),
    ("Water freezes at", ["0", "32", "zero"]),
    ("Einstein developed the theory of", ["relativity"]),
    ("World War II ended in", ["1945"]),
    ("The Mona Lisa was painted by", ["Leonardo", "Vinci"]),
    ("The currency of Japan is", ["yen"]),
    ("The sun is a", ["star"]),
]

def run_sweep():
    print("=" * 70)
    print("OLMo PARAMETER SWEEP")
    print("=" * 70)

    model_name = "allenai/OLMo-1.7-7B-hf"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    print(f"Loaded. GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    # First, get baseline results
    print("\nBaseline results:")
    baseline_correct = 0
    for prompt, expected in TEST_PROMPTS:
        resp = generate_baseline(model, tokenizer, prompt)
        ok = any(e.lower() in resp.lower() for e in expected)
        baseline_correct += ok
        status = "✓" if ok else "✗"
        print(f"  {status} {prompt[:40]}...")

    baseline_acc = baseline_correct / len(TEST_PROMPTS)
    print(f"\nBaseline: {baseline_acc:.1%} ({baseline_correct}/{len(TEST_PROMPTS)})")

    # Parameter grid
    alphas = [0.05, 0.1, 0.15, 0.2, 0.25]
    radii = [1.5, 2.0, 2.5, 3.0]
    max_tokens_list = [500, 1000, 1440, 2000, 3000]

    results = []
    best_result = {"acc": baseline_acc, "params": "baseline"}

    print("\n" + "=" * 70)
    print("SWEEPING PARAMETERS")
    print("=" * 70)

    for alpha in alphas:
        for radius in radii:
            for max_tokens in max_tokens_list:
                correct = 0
                for prompt, expected in TEST_PROMPTS:
                    resp = generate_with_bias(model, tokenizer, prompt,
                                             12, radius, alpha, max_tokens)
                    ok = any(e.lower() in resp.lower() for e in expected)
                    correct += ok

                acc = correct / len(TEST_PROMPTS)
                err_red = ((1-baseline_acc) - (1-acc)) / (1-baseline_acc) * 100 if baseline_acc < 1 else 0

                result = {
                    "alpha": alpha, "radius": radius, "max_tokens": max_tokens,
                    "accuracy": acc, "correct": correct, "error_reduction": err_red
                }
                results.append(result)

                if acc > best_result["acc"]:
                    best_result = {"acc": acc, "params": f"α={alpha}, r={radius}, n={max_tokens}"}

                marker = "**" if acc > baseline_acc else "  "
                print(f"{marker}α={alpha:.2f} r={radius:.1f} n={max_tokens:4d} → {acc:.1%} ({correct}/{len(TEST_PROMPTS)}) err_red={err_red:+.1f}%")

    print("\n" + "=" * 70)
    print("BEST RESULT")
    print("=" * 70)
    print(f"Baseline: {baseline_acc:.1%}")
    print(f"Best: {best_result['acc']:.1%} with {best_result['params']}")

    # Save results
    os.makedirs("./results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"./results/sweep_olmo_{ts}.json", "w") as f:
        json.dump({"baseline": baseline_acc, "best": best_result, "all": results}, f, indent=2)

    return results

if __name__ == "__main__":
    run_sweep()
