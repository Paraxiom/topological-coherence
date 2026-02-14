"""
Ouro-1.4B + Toroidal Logit Bias (TLB) — Fine-Grained Sweep

Previous coarse sweep found:
  - Best config: alpha=0.5, radius=3.0 -> 90% acc at 4 loops (+10% over baseline)
  - alpha=0.3, radius=3.0 -> 85% (+5%)
  - alpha=1.0, radius=3.0 -> 75% (-5%, destructive)
  - All radius=2.0 configs had zero effect
  - Baseline@4: 80%, Baseline@8: 85%

This script zooms into the productive alpha=[0.3..0.7] and radius=[2.5..4.0]
range to find the optimal configuration with finer granularity.

Author: Sylvain Cormier / Paraxiom Research
Date: 2026-02-14
"""

import sys
import os
import json
import time
import math
import traceback
import gc
import numpy as np
import torch
from pathlib import Path
from datetime import datetime


# --- Device selection ---
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DTYPE = torch.float16
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory
    print(f"GPU detected: {gpu_name}")
    print(f"VRAM: {gpu_mem / 1e9:.1f} GB")
else:
    DEVICE = torch.device("cpu")
    DTYPE = torch.float32
    print("No GPU -- running on CPU (will be slow)")

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_FILE = RESULTS_DIR / f"ouro_tlb_fine_{TIMESTAMP}.json"


# --- Toroidal Logit Bias (inline, no external deps) ---

def toroidal_distance(i, j, grid_size=12):
    xi, yi = i % grid_size, (i // grid_size) % grid_size
    xj, yj = j % grid_size, (j // grid_size) % grid_size
    dx = min(abs(xi - xj), grid_size - abs(xi - xj))
    dy = min(abs(yi - yj), grid_size - abs(yi - yj))
    return dx + dy


def get_toroidal_bias(vocab_size, recent_tokens, alpha, radius, max_tokens,
                      grid_size=12, device='cuda'):
    """Compute toroidal logit bias from recent context tokens."""
    bias = torch.zeros(vocab_size, device=device, dtype=torch.float32)
    if len(recent_tokens) == 0:
        return bias

    n_positions = grid_size * grid_size
    context = recent_tokens[-5:]  # last 5 tokens

    for offset, token_id in enumerate(context):
        token_pos = token_id % n_positions
        for vocab_id in range(min(vocab_size, max_tokens)):
            target_pos = vocab_id % n_positions
            dist = toroidal_distance(token_pos, target_pos, grid_size)
            if dist <= radius:
                bias[vocab_id] += alpha * (radius - dist + 1) / (offset + 1)
            elif dist <= radius * 2:
                bias[vocab_id] += alpha * 0.5 / (offset + 1)

    # Zero-center
    if bias.abs().sum() > 0:
        bias -= bias.mean()

    return bias


class ToroidalLogitProcessor:
    """HuggingFace-compatible logits processor for TLB."""

    def __init__(self, alpha=0.3, radius=2.0, max_tokens=1440, grid_size=12):
        self.alpha = alpha
        self.radius = radius
        self.max_tokens = max_tokens
        self.grid_size = grid_size

    def __call__(self, input_ids, scores):
        batch_size = scores.shape[0]
        for b in range(batch_size):
            recent = input_ids[b].tolist()
            bias = get_toroidal_bias(
                vocab_size=scores.shape[1],
                recent_tokens=recent,
                alpha=self.alpha,
                radius=self.radius,
                max_tokens=self.max_tokens,
                grid_size=self.grid_size,
                device=scores.device,
            )
            scores[b] = scores[b] + bias.to(scores.dtype)
        return scores


# --- Evaluation prompts (same 20 as coarse sweep) ---

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
    ["a\u00b2", "a^2", "square", "hypotenuse", "right triangle"],
]


def evaluate_model(model, tokenizer, device, logits_processor=None, max_new_tokens=30):
    """Run 20-prompt factual eval. Returns (accuracy, detail_list)."""
    correct = 0
    results = []
    processors = [logits_processor] if logits_processor else []

    for prompt, keywords in zip(EVAL_PROMPTS, EXPECTED_KEYWORDS):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, temperature=1.0, use_cache=False,
                logits_processor=processors,
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
    """Compute perplexity on 5 reference texts."""
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


def cleanup_model(model):
    """Delete model and free GPU memory."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_results(all_results):
    """Incremental save."""
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)


# --- Main ---

def run_experiment():
    print("=" * 70)
    print("OURO-1.4B + TOROIDAL LOGIT BIAS -- FINE-GRAINED SWEEP")
    print("=" * 70)
    print(f"Device: {DEVICE} ({DTYPE})")
    print(f"Results: {RESULTS_FILE}")
    print(f"Start: {datetime.now().isoformat()}")
    print(flush=True)

    device = DEVICE

    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    model_name = "ByteDance/Ouro-1.4B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Fine sweep parameters ---
    alphas = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]
    radii = [2.5, 3.0, 3.5, 4.0]
    max_tokens = 1440
    loop_counts = [4, 8]

    # Build TLB config list
    tlb_configs = []
    for alpha in alphas:
        for radius in radii:
            name = f"a{alpha:.2f}_r{radius:.1f}"
            tlb_configs.append({
                "name": name,
                "alpha": alpha,
                "radius": radius,
                "max_tokens": max_tokens,
            })

    total_runs = len(tlb_configs) * len(loop_counts)
    print(f"Sweep: {len(alphas)} alphas x {len(radii)} radii x {len(loop_counts)} loops = {total_runs} configs")
    print(f"Alphas: {alphas}")
    print(f"Radii: {radii}")
    print(f"max_tokens: {max_tokens}")
    print(f"loop_counts: {loop_counts}")
    print(flush=True)

    all_results = {
        "experiment": "ouro_tlb_fine_sweep",
        "model": model_name,
        "device": str(device),
        "dtype": str(DTYPE),
        "date": datetime.now().isoformat(),
        "sweep_params": {
            "alphas": alphas,
            "radii": radii,
            "max_tokens": max_tokens,
            "loop_counts": loop_counts,
        },
        "method": "Toroidal Logit Bias on output logits -- fine-grained sweep around optimal zone",
        "prior_results": {
            "best": "alpha=0.5, radius=3.0 -> 90% at 4 loops (+10%)",
            "baseline_4": "80%",
            "baseline_8": "85%",
        },
        "baseline": {},
        "tlb": {},
    }

    # =====================================================================
    # BASELINES
    # =====================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: BASELINES")
    print("=" * 70, flush=True)

    for n_loops in loop_counts:
        print(f"\n  --- Baseline: {n_loops} UT loops ---", flush=True)
        t1 = time.time()
        try:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            config.total_ut_steps = n_loops
            config.early_exit_threshold = 1.0
            config.pad_token_id = tokenizer.eos_token_id

            model = AutoModelForCausalLM.from_pretrained(
                model_name, config=config, trust_remote_code=True,
                torch_dtype=DTYPE,
            ).to(device)
            model.eval()
            print(f"  Loaded in {time.time()-t1:.1f}s", flush=True)

            ppl = compute_perplexity(model, tokenizer, device)
            print(f"  PPL: {ppl:.2f}", flush=True)
            acc, details = evaluate_model(model, tokenizer, device)
            n_correct = int(acc * 20)
            print(f"  Acc: {acc:.0%} ({n_correct}/20)", flush=True)

            all_results["baseline"][str(n_loops)] = {
                "perplexity": ppl, "accuracy": acc,
                "correct_count": n_correct, "total_prompts": 20,
                "details": details,
                "elapsed_s": time.time() - t1,
            }
            cleanup_model(model)

        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            traceback.print_exc()
            all_results["baseline"][str(n_loops)] = {"error": str(e)}

        save_results(all_results)

    # =====================================================================
    # TLB FINE SWEEP
    # =====================================================================
    print("\n" + "=" * 70)
    print(f"PHASE 2: TLB FINE SWEEP ({total_runs} configs)")
    print("=" * 70, flush=True)

    run_idx = 0
    for tlb_cfg in tlb_configs:
        tlb_name = tlb_cfg["name"]

        for n_loops in loop_counts:
            run_idx += 1
            alpha_val = tlb_cfg["alpha"]
            radius_val = tlb_cfg["radius"]
            print(f"\n  [{run_idx}/{total_runs}] alpha={alpha_val:.2f}  radius={radius_val:.1f}  loops={n_loops}",
                  flush=True)
            t1 = time.time()

            try:
                config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
                config.total_ut_steps = n_loops
                config.early_exit_threshold = 1.0
                config.pad_token_id = tokenizer.eos_token_id

                model = AutoModelForCausalLM.from_pretrained(
                    model_name, config=config, trust_remote_code=True,
                    torch_dtype=DTYPE,
                ).to(device)
                model.eval()

                processor = ToroidalLogitProcessor(
                    alpha=tlb_cfg["alpha"],
                    radius=tlb_cfg["radius"],
                    max_tokens=tlb_cfg["max_tokens"],
                )

                ppl = compute_perplexity(model, tokenizer, device)
                acc, details = evaluate_model(model, tokenizer, device,
                                              logits_processor=processor)
                n_correct = int(acc * 20)
                elapsed = time.time() - t1

                print(f"  PPL: {ppl:.2f}  Acc: {acc:.0%} ({n_correct}/20)  [{elapsed:.0f}s]", flush=True)

                # Compare vs matching baseline
                bl = all_results["baseline"].get(str(n_loops), {})
                bl_acc = bl.get("accuracy", 0)
                bl_details = bl.get("details", [])
                delta_acc = acc - bl_acc

                # Use a variable for the delta symbol (no backslash in f-string)
                delta_sym = "\u0394"
                if abs(delta_acc) > 0.001:
                    print(f"  {delta_sym} vs baseline@{n_loops}: {delta_acc:+.0%}", flush=True)

                # Per-prompt FIXED/BROKE
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

                # Store
                if tlb_name not in all_results["tlb"]:
                    all_results["tlb"][tlb_name] = {}

                all_results["tlb"][tlb_name][str(n_loops)] = {
                    "perplexity": ppl,
                    "accuracy": acc,
                    "correct_count": n_correct,
                    "total_prompts": 20,
                    "tlb_alpha": tlb_cfg["alpha"],
                    "tlb_radius": tlb_cfg["radius"],
                    "tlb_max_tokens": tlb_cfg["max_tokens"],
                    "delta_vs_baseline": delta_acc,
                    "elapsed_s": elapsed,
                    "details": details,
                }

                cleanup_model(model)

            except Exception as e:
                print(f"  ERROR: {e}", flush=True)
                traceback.print_exc()
                if tlb_name not in all_results["tlb"]:
                    all_results["tlb"][tlb_name] = {}
                all_results["tlb"][tlb_name][str(n_loops)] = {"error": str(e)}

            save_results(all_results)

    # =====================================================================
    # SUMMARY TABLE
    # =====================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: OURO-1.4B + TLB FINE-GRAINED SWEEP")
    print("=" * 70, flush=True)

    delta_sym = "\u0394"

    for n_loops in loop_counts:
        bl = all_results["baseline"].get(str(n_loops), {})
        bl_ppl = bl.get("perplexity", float('nan'))
        bl_acc = bl.get("accuracy", float('nan'))
        print(f"\n--- {n_loops} UT loops  |  Baseline: PPL={bl_ppl:.2f}  Acc={bl_acc:.0%} ---")
        header = f"  {'alpha':<8} {'radius':<8} {'PPL':<10} {'Acc':<8} {delta_sym + ' Acc':<10} {'Note'}"
        print(header)
        print(f"  {'-' * 60}")

        # Collect results for this loop count to find best
        loop_results = []

        for tlb_cfg in tlb_configs:
            tn = tlb_cfg["name"]
            r = all_results["tlb"].get(tn, {}).get(str(n_loops), {})
            if "error" in r:
                print(f"  {tlb_cfg['alpha']:<8.2f} {tlb_cfg['radius']:<8.1f} {'ERROR':<10}")
                continue
            rp = r.get("perplexity", float('nan'))
            ra = r.get("accuracy", float('nan'))
            da = ra - bl_acc if not math.isnan(ra) and not math.isnan(bl_acc) else float('nan')

            note = ""
            if not math.isnan(da):
                if da > 0.05:
                    note = "*** STRONG"
                elif da > 0.001:
                    note = "* improved"
                elif da < -0.05:
                    note = "!!! destructive"
                elif da < -0.001:
                    note = "- degraded"
                else:
                    note = "= same"

            da_str = f"{da:+.0%}" if not math.isnan(da) else "N/A"
            ra_str = f"{ra:.0%}" if not math.isnan(ra) else "N/A"
            rp_str = f"{rp:.2f}" if not math.isnan(rp) else "N/A"

            print(f"  {tlb_cfg['alpha']:<8.2f} {tlb_cfg['radius']:<8.1f} {rp_str:<10} {ra_str:<8} {da_str:<10} {note}")

            if not math.isnan(da):
                loop_results.append((da, ra, tlb_cfg['alpha'], tlb_cfg['radius'], rp))

        # Report best config for this loop count
        if loop_results:
            loop_results.sort(key=lambda x: (-x[0], x[4]))  # best delta, then lowest PPL
            best = loop_results[0]
            print(f"\n  BEST @ {n_loops} loops: alpha={best[2]:.2f}  radius={best[3]:.1f}  "
                  f"Acc={best[1]:.0%}  {delta_sym}={best[0]:+.0%}  PPL={best[4]:.2f}")

    # Overall best
    print("\n" + "=" * 70)
    print("OVERALL BEST CONFIGS")
    print("=" * 70)

    all_ranked = []
    for tlb_cfg in tlb_configs:
        tn = tlb_cfg["name"]
        for n_loops in loop_counts:
            r = all_results["tlb"].get(tn, {}).get(str(n_loops), {})
            if "error" in r or "accuracy" not in r:
                continue
            bl = all_results["baseline"].get(str(n_loops), {})
            bl_acc = bl.get("accuracy", 0)
            da = r["accuracy"] - bl_acc
            all_ranked.append({
                "alpha": tlb_cfg["alpha"],
                "radius": tlb_cfg["radius"],
                "loops": n_loops,
                "accuracy": r["accuracy"],
                "delta": da,
                "ppl": r.get("perplexity", float('nan')),
            })

    all_ranked.sort(key=lambda x: (-x["delta"], -x["accuracy"], x["ppl"]))
    for i, entry in enumerate(all_ranked[:10]):
        rank = i + 1
        delta_sym_local = "\u0394"
        print(f"  #{rank}: alpha={entry['alpha']:.2f}  radius={entry['radius']:.1f}  "
              f"loops={entry['loops']}  Acc={entry['accuracy']:.0%}  "
              f"{delta_sym_local}={entry['delta']:+.0%}  PPL={entry['ppl']:.2f}")

    all_results["end_time"] = datetime.now().isoformat()
    all_results["overall_ranking"] = all_ranked[:10]
    save_results(all_results)

    print(f"\nResults saved: {RESULTS_FILE}")
    print(f"End: {datetime.now().isoformat()}")
    print("=" * 70, flush=True)


if __name__ == "__main__":
    run_experiment()
