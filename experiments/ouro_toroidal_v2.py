"""
Ouro-1.4B + Toroidal Attention: Direct Attention Weight Masking (v2)

v1 used output blending (70/30) which had zero effect.
v2 hooks directly into OuroAttention.forward to inject a toroidal
additive mask into the attention scores BEFORE softmax — matching
the mechanism in Eq. 18 of the paper (TopoAttention).

Author: Sylvain Cormier / Paraxiom Research
Date: 2026-02-14
"""

import sys
import os
import json
import time
import math
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
RESULTS_FILE = RESULTS_DIR / f"ouro_toroidal_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"


# --- Toroidal mask (additive, log-space) ---

def toroidal_distance_matrix(n_tokens, grid_size=12):
    idx = np.arange(n_tokens)
    x = idx % grid_size
    y = (idx // grid_size) % grid_size
    dx = np.abs(x[:, None] - x[None, :])
    dy = np.abs(y[:, None] - y[None, :])
    dx = np.minimum(dx, grid_size - dx)
    dy = np.minimum(dy, grid_size - dy)
    return dx + dy


def make_toroidal_additive_mask(seq_len, radius=2.0, alpha=1.0, grid_size=12):
    """Create additive toroidal mask for pre-softmax injection.

    Returns log-space values: 0 where d<=r (no change), negative elsewhere.
    This is added to attention scores before softmax, suppressing
    tokens that are toroidally distant.
    """
    dist = toroidal_distance_matrix(seq_len, grid_size).astype(np.float32)
    # Hybrid mask in multiplicative space
    mult_mask = np.where(dist <= radius, 1.0, np.exp(-alpha * (dist - radius)))
    # Convert to additive (log space) for pre-softmax injection
    additive = np.log(np.clip(mult_mask, 1e-8, 1.0))
    return torch.from_numpy(additive.astype(np.float32))


# --- Direct attention hooking ---

def hook_attention_direct(model, toroidal_mask, layer_fraction=0.67, strength=1.0):
    """Hook into OuroAttention to inject toroidal mask into attention scores.

    This modifies the attention_mask argument passed to eager_attention_forward,
    adding the toroidal additive mask directly to the causal mask.
    """
    handles = []

    # Find all OuroAttention modules
    attn_modules = []
    for name, module in model.named_modules():
        if module.__class__.__name__ == 'OuroAttention':
            attn_modules.append((name, module))

    n_layers = len(attn_modules)
    start_idx = int(n_layers * layer_fraction)
    hooked = 0

    for i, (name, module) in enumerate(attn_modules):
        if i >= start_idx:
            original_forward = module.forward

            def make_hooked_forward(orig_fwd, layer_name):
                def hooked_forward(
                    hidden_states,
                    position_embeddings,
                    attention_mask=None,
                    past_key_value=None,
                    cache_position=None,
                    current_ut=0,
                    **kwargs
                ):
                    # Inject toroidal bias into attention_mask
                    if attention_mask is not None:
                        seq_len = attention_mask.shape[-1]
                        toro = toroidal_mask[:seq_len, :seq_len].to(
                            attention_mask.device, attention_mask.dtype
                        )
                        # Reshape for broadcasting: [1, 1, S, S]
                        toro_4d = toro.unsqueeze(0).unsqueeze(0) * strength
                        attention_mask = attention_mask + toro_4d

                    return orig_fwd(
                        hidden_states,
                        position_embeddings,
                        attention_mask=attention_mask,
                        past_key_value=past_key_value,
                        cache_position=cache_position,
                        current_ut=current_ut,
                        **kwargs
                    )
                return hooked_forward

            module.forward = make_hooked_forward(original_forward, name)
            hooked += 1

    print(f"  Hooked {hooked}/{n_layers} attention layers (last {n_layers - start_idx})")
    return handles


# --- Evaluation ---

EVAL_PROMPTS = [
    "The capital of France is",
    "Water boils at a temperature of",
    "The speed of light is approximately",
    "Newton discovered the law of",
    "The chemical formula for water is",
    "DNA stands for",
    "The largest planet in our solar system is",
    "Shakespeare wrote the play",
    "The theory of relativity was proposed by",
    "Photosynthesis converts sunlight into",
    "The human body has approximately how many bones:",
    "Mount Everest is located in",
    "The periodic table was created by",
    "An electron has a charge of",
    "The Great Wall of China was built to",
    "Pi is approximately equal to",
    "The mitochondria is known as the",
    "The French Revolution began in the year",
    "Gravity on Earth accelerates objects at approximately",
    "The Pythagorean theorem states that",
]

EXPECTED_KEYWORDS = [
    ["paris"],
    ["100", "212", "celsius", "fahrenheit"],
    ["300", "3", "km", "186"],
    ["gravity", "gravitation", "motion"],
    ["h2o"],
    ["deoxyribonucleic"],
    ["jupiter"],
    ["hamlet", "romeo", "macbeth", "othello", "lear"],
    ["einstein", "albert"],
    ["energy", "glucose", "sugar", "chemical"],
    ["206"],
    ["nepal", "himalaya", "tibet"],
    ["mendeleev"],
    ["negative", "-1", "1.6"],
    ["protect", "defend", "invad", "mongol", "border"],
    ["3.14"],
    ["powerhouse"],
    ["1789"],
    ["9.8", "10", "m/s"],
    ["a²", "a^2", "square", "hypotenuse", "right triangle"],
]


def evaluate_model(model, tokenizer, device, max_new_tokens=30):
    correct = 0
    results = []
    for prompt, keywords in zip(EVAL_PROMPTS, EXPECTED_KEYWORDS):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, temperature=1.0, use_cache=False,
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response[len(prompt):].strip().lower()
        hit = any(kw.lower() in answer for kw in keywords)
        if hit:
            correct += 1
        results.append({
            "prompt": prompt, "answer": answer[:200],
            "keywords": keywords, "correct": hit,
        })
    return correct / len(EVAL_PROMPTS), results


def compute_perplexity(model, tokenizer, device):
    texts = [
        "The theory of general relativity describes gravity as the curvature of spacetime caused by mass and energy.",
        "Quantum mechanics is a fundamental theory in physics that describes nature at the smallest scales of energy levels of atoms and subatomic particles.",
        "The human genome contains approximately 3 billion base pairs of DNA organized into 23 pairs of chromosomes.",
        "Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy.",
        "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of squares of the other two sides.",
    ]
    total_loss = 0.0
    total_tokens = 0
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"], use_cache=False)
            loss = outputs.loss.item()
            n_tokens = inputs["input_ids"].shape[1]
            total_loss += loss * n_tokens
            total_tokens += n_tokens
    return math.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')


# --- Main ---

def run_experiment():
    print("=" * 70)
    print("OURO-1.4B + TOROIDAL: DIRECT ATTENTION MASKING (v2)")
    print("=" * 70)
    print(f"Start: {datetime.now().isoformat()}")
    print(flush=True)

    device = torch.device("cpu")

    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    model_name = "ByteDance/Ouro-1.4B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_seq = 512
    toro_mask = make_toroidal_additive_mask(max_seq, radius=2.0, alpha=1.0, grid_size=12)
    print(f"Toroidal additive mask: {max_seq}x{max_seq}, min={toro_mask.min():.3f}, max={toro_mask.max():.3f}")

    loop_counts = [1, 2, 3, 4, 6, 8]
    # Test multiple strengths to find the sweet spot
    strengths = [0.5, 1.0, 2.0]

    all_results = {
        "experiment": "ouro_toroidal_v2_direct_masking",
        "model": model_name,
        "device": str(device),
        "date": datetime.now().isoformat(),
        "loop_counts": loop_counts,
        "strengths": strengths,
        "method": "additive toroidal mask injected into attention_mask before softmax",
        "baseline": {},
        "toroidal": {},
    }

    # --- BASELINE (4 and 8 loops only — we already have full sweep) ---
    print("\n[1/3] Baseline (4 and 8 loops)...", flush=True)

    for n_loops in [4, 8]:
        print(f"\n  --- Baseline: {n_loops} loops ---", flush=True)
        t1 = time.time()
        try:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            config.total_ut_steps = n_loops
            config.early_exit_threshold = 1.0

            model = AutoModelForCausalLM.from_pretrained(
                model_name, config=config, trust_remote_code=True,
                torch_dtype=torch.float32,
            ).to(device)
            model.eval()
            print(f"  Loaded in {time.time()-t1:.1f}s", flush=True)

            ppl = compute_perplexity(model, tokenizer, device)
            print(f"  PPL: {ppl:.2f}", flush=True)
            acc, details = evaluate_model(model, tokenizer, device)
            print(f"  Acc: {acc:.0%} ({int(acc*20)}/20)", flush=True)

            all_results["baseline"][str(n_loops)] = {
                "perplexity": ppl, "accuracy": acc,
                "correct_count": int(acc * 20), "total_prompts": 20,
                "details": details,
            }
            del model
            import gc; gc.collect()
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            all_results["baseline"][str(n_loops)] = {"error": str(e)}

        with open(RESULTS_FILE, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

    # --- TOROIDAL: sweep strengths × loop counts ---
    print("\n[2/3] Toroidal runs (direct attention masking)...", flush=True)

    for strength in strengths:
        strength_key = f"strength_{strength}"
        all_results["toroidal"][strength_key] = {}

        for n_loops in loop_counts:
            print(f"\n  --- Toroidal: {n_loops} loops, strength={strength} ---", flush=True)
            t1 = time.time()
            try:
                config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
                config.total_ut_steps = n_loops
                config.early_exit_threshold = 1.0

                model = AutoModelForCausalLM.from_pretrained(
                    model_name, config=config, trust_remote_code=True,
                    torch_dtype=torch.float32,
                ).to(device)
                model.eval()
                print(f"  Loaded in {time.time()-t1:.1f}s", flush=True)

                hook_attention_direct(model, toro_mask, layer_fraction=0.67, strength=strength)

                ppl = compute_perplexity(model, tokenizer, device)
                print(f"  PPL: {ppl:.2f}", flush=True)
                acc, details = evaluate_model(model, tokenizer, device)
                print(f"  Acc: {acc:.0%} ({int(acc*20)}/20)", flush=True)

                # Show which answers changed vs baseline
                baseline_4 = all_results["baseline"].get("4", {}).get("details", [])
                if baseline_4 and n_loops == 4:
                    changes = []
                    for bd, td in zip(baseline_4, details):
                        if bd["correct"] != td["correct"]:
                            direction = "FIXED" if td["correct"] else "BROKE"
                            changes.append(f"    {direction}: {td['prompt']}")
                    if changes:
                        print(f"  Changes vs baseline@4:", flush=True)
                        for c in changes:
                            print(c, flush=True)

                all_results["toroidal"][strength_key][str(n_loops)] = {
                    "perplexity": ppl, "accuracy": acc,
                    "correct_count": int(acc * 20), "total_prompts": 20,
                    "details": details,
                }
                del model
                import gc; gc.collect()
            except Exception as e:
                print(f"  ERROR: {e}", flush=True)
                all_results["toroidal"][strength_key][str(n_loops)] = {"error": str(e)}

            with open(RESULTS_FILE, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)

    # --- SUMMARY ---
    print("\n" + "=" * 70)
    print("[3/3] SUMMARY")
    print("=" * 70, flush=True)

    # Get baseline reference
    b4 = all_results["baseline"].get("4", {})
    b8 = all_results["baseline"].get("8", {})

    print(f"\nBaseline@4: PPL={b4.get('perplexity','?'):.2f}  Acc={b4.get('accuracy','?'):.0%}")
    print(f"Baseline@8: PPL={b8.get('perplexity','?'):.2f}  Acc={b8.get('accuracy','?'):.0%}")

    for strength in strengths:
        sk = f"strength_{strength}"
        print(f"\nToroidal strength={strength}:")
        print(f"  {'Loops':<8} {'PPL':<10} {'Acc':<10} {'Δ vs B@4':<12} {'Δ vs B@8':<12}")
        print(f"  {'-'*52}")
        for n in loop_counts:
            t = all_results["toroidal"].get(sk, {}).get(str(n), {})
            tp = t.get("perplexity", float('nan'))
            ta = t.get("accuracy", float('nan'))
            d4 = ta - b4.get("accuracy", 0) if not math.isnan(ta) else float('nan')
            d8 = ta - b8.get("accuracy", 0) if not math.isnan(ta) else float('nan')
            print(f"  {n:<8} {tp:<10.2f} {ta:<10.1%} {d4:+.1%}{'':<7} {d8:+.1%}")

    all_results["end_time"] = datetime.now().isoformat()
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults: {RESULTS_FILE}")
    print(f"End: {datetime.now().isoformat()}")
    print("=" * 70, flush=True)


if __name__ == "__main__":
    run_experiment()
