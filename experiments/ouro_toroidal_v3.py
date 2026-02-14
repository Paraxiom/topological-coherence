"""
Ouro-1.4B + Toroidal Attention: Light Strength Sweep (v3)

v2 showed strength=0.5 was too aggressive — PPL jumped 4x, acc dropped 15pp.
v3 uses much lighter strengths (0.01, 0.05, 0.1, 0.2) to find the sweet spot
where toroidal bias gently guides attention without destroying UT convergence.

Also tests hooking fewer layers (last 33% vs last 67%) and a wider radius.

GPU-ready: auto-detects CUDA and uses it when available.
RunPod one-liner:
  pip install torch transformers numpy && python ouro_toroidal_v3.py

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

# --- Device selection ---
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DTYPE = torch.float16  # faster on GPU
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
else:
    DEVICE = torch.device("cpu")
    DTYPE = torch.float32
    print("No GPU — running on CPU (will be slow)")

RESULTS_DIR = Path(__file__).parent.parent / "results"
if not RESULTS_DIR.exists():
    RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)
RESULTS_FILE = RESULTS_DIR / f"ouro_toroidal_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"


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
    """Create additive toroidal mask for pre-softmax injection."""
    dist = toroidal_distance_matrix(seq_len, grid_size).astype(np.float32)
    mult_mask = np.where(dist <= radius, 1.0, np.exp(-alpha * (dist - radius)))
    additive = np.log(np.clip(mult_mask, 1e-8, 1.0))
    return torch.from_numpy(additive.astype(np.float32))


# --- Direct attention hooking ---

def hook_attention_direct(model, toroidal_mask, layer_fraction=0.67, strength=1.0):
    """Hook into OuroAttention to inject toroidal mask into attention scores."""
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
                    if attention_mask is not None:
                        seq_len = attention_mask.shape[-1]
                        toro = toroidal_mask[:seq_len, :seq_len].to(
                            attention_mask.device, attention_mask.dtype
                        )
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

    print(f"  Hooked {hooked}/{n_layers} layers (from idx {start_idx})")
    return hooked


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
    print("OURO-1.4B + TOROIDAL: LIGHT STRENGTH SWEEP (v3)")
    print("=" * 70)
    print(f"Device: {DEVICE} ({DTYPE})")
    print(f"Start: {datetime.now().isoformat()}")
    print(flush=True)

    device = DEVICE

    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    model_name = "ByteDance/Ouro-1.4B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_seq = 512

    # Focus on 4 loops (default) and 8 loops (best baseline)
    loop_counts = [4, 8]

    # Light strengths — v2 showed 0.5 was way too much
    strengths = [0.01, 0.05, 0.1, 0.2]

    # Test two mask configs: narrow radius (r=2) and wide radius (r=4)
    mask_configs = [
        {"name": "r2_a1", "radius": 2.0, "alpha": 1.0, "grid_size": 12},
        {"name": "r4_a0.5", "radius": 4.0, "alpha": 0.5, "grid_size": 12},
    ]

    # Also test hooking fewer layers
    layer_fractions = [0.67, 0.85]  # last 33% vs last 15%

    all_results = {
        "experiment": "ouro_toroidal_v3_light_strengths",
        "model": model_name,
        "device": str(device),
        "date": datetime.now().isoformat(),
        "loop_counts": loop_counts,
        "strengths": strengths,
        "mask_configs": mask_configs,
        "layer_fractions": layer_fractions,
        "method": "additive toroidal mask, light strengths, variable radius/layers",
        "baseline": {},
        "toroidal": {},
    }

    # --- BASELINE ---
    print("\n[1/2] Baselines...", flush=True)

    for n_loops in loop_counts:
        print(f"\n  --- Baseline: {n_loops} loops ---", flush=True)
        t1 = time.time()
        try:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            config.total_ut_steps = n_loops
            config.early_exit_threshold = 1.0

            model = AutoModelForCausalLM.from_pretrained(
                model_name, config=config, trust_remote_code=True,
                torch_dtype=DTYPE,
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

    # --- TOROIDAL SWEEP ---
    # Total configs: 2 masks × 4 strengths × 2 loop_counts × 2 layer_fracs = 32
    # But we'll do the most promising combos first: default mask, both layer fracs
    # Then wide mask only at best strength
    print("\n[2/2] Toroidal sweep...", flush=True)

    run_idx = 0
    total_runs = len(mask_configs) * len(strengths) * len(loop_counts) * len(layer_fractions)
    print(f"  Total configs to test: {total_runs}", flush=True)

    for mc in mask_configs:
        mask_name = mc["name"]
        toro_mask = make_toroidal_additive_mask(
            max_seq, radius=mc["radius"], alpha=mc["alpha"], grid_size=mc["grid_size"]
        )
        print(f"\n  Mask '{mask_name}': min={toro_mask.min():.3f}, max={toro_mask.max():.3f}")

        for lf in layer_fractions:
            lf_name = f"last_{int((1-lf)*100)}pct"

            for strength in strengths:
                for n_loops in loop_counts:
                    run_idx += 1
                    config_key = f"{mask_name}__{lf_name}__s{strength}"

                    if config_key not in all_results["toroidal"]:
                        all_results["toroidal"][config_key] = {}

                    print(f"\n  [{run_idx}/{total_runs}] {config_key}, {n_loops} loops", flush=True)
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

                        n_hooked = hook_attention_direct(model, toro_mask, layer_fraction=lf, strength=strength)

                        ppl = compute_perplexity(model, tokenizer, device)
                        acc, details = evaluate_model(model, tokenizer, device)

                        elapsed = time.time() - t1
                        print(f"  PPL: {ppl:.2f}  Acc: {acc:.0%} ({int(acc*20)}/20)  [{elapsed:.0f}s]", flush=True)

                        # Show changes vs matching baseline
                        bl = all_results["baseline"].get(str(n_loops), {})
                        bl_acc = bl.get("accuracy", 0)
                        delta = acc - bl_acc
                        if abs(delta) > 0.001:
                            print(f"  Δ vs baseline@{n_loops}: {delta:+.0%}", flush=True)

                        # Highlight any fixes
                        bl_details = bl.get("details", [])
                        if bl_details:
                            fixes = []
                            breaks = []
                            for bd, td in zip(bl_details, details):
                                if not bd["correct"] and td["correct"]:
                                    fixes.append(td["prompt"])
                                elif bd["correct"] and not td["correct"]:
                                    breaks.append(td["prompt"])
                            if fixes:
                                print(f"  FIXED: {fixes}", flush=True)
                            if breaks:
                                print(f"  BROKE: {breaks}", flush=True)

                        all_results["toroidal"][config_key][str(n_loops)] = {
                            "perplexity": ppl, "accuracy": acc,
                            "correct_count": int(acc * 20), "total_prompts": 20,
                            "n_hooked_layers": n_hooked,
                            "details": details,
                        }

                        del model
                        import gc; gc.collect()

                    except Exception as e:
                        print(f"  ERROR: {e}", flush=True)
                        all_results["toroidal"][config_key][str(n_loops)] = {"error": str(e)}

                    with open(RESULTS_FILE, 'w') as f:
                        json.dump(all_results, f, indent=2, default=str)

    # --- SUMMARY ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70, flush=True)

    for n_loops in loop_counts:
        bl = all_results["baseline"].get(str(n_loops), {})
        bl_ppl = bl.get("perplexity", float('nan'))
        bl_acc = bl.get("accuracy", float('nan'))
        print(f"\nBaseline@{n_loops}: PPL={bl_ppl:.2f}  Acc={bl_acc:.0%}")

        print(f"  {'Config':<40} {'PPL':<8} {'Acc':<8} {'Δ Acc':<8}")
        print(f"  {'-'*64}")

        for ck, cv in sorted(all_results["toroidal"].items()):
            r = cv.get(str(n_loops), {})
            if "error" in r:
                continue
            rp = r.get("perplexity", float('nan'))
            ra = r.get("accuracy", float('nan'))
            da = ra - bl_acc if not math.isnan(ra) else float('nan')
            marker = " ***" if da > 0.001 else ""
            print(f"  {ck:<40} {rp:<8.2f} {ra:<8.0%} {da:+.0%}{marker}")

    all_results["end_time"] = datetime.now().isoformat()
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults: {RESULTS_FILE}")
    print(f"End: {datetime.now().isoformat()}")
    print("=" * 70, flush=True)


if __name__ == "__main__":
    run_experiment()
