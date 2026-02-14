#!/usr/bin/env python3
"""
Ouro-1.4B TruthfulQA Evaluation with Toroidal Logit Bias (TLB)
===============================================================

Runs the standard TruthfulQA MC1 benchmark on ByteDance/Ouro-1.4B with and
without Toroidal Logit Bias. Ouro uses Universal Transformer (UT) loops,
so we also sweep loop counts [4, 8] to see UT x TLB interaction.

TLB configs tested (from prior experiments):
  - Best:    alpha=0.5, radius=3.0, max_tokens=1440
  - Alt 1:   alpha=0.3, radius=3.0, max_tokens=1440
  - Alt 2:   alpha=0.5, radius=3.5, max_tokens=1440

Evaluation method: MC1 (multiple-choice, single correct answer).
For each question, score each choice via mean of last-token logits,
optionally with toroidal bias applied. Highest-scored choice = prediction.

Statistical test: McNemar's test on paired (baseline, TLB) predictions.

Author: Sylvain Cormier / Paraxiom Research
Date: 2026-02-14
"""

import sys
import os
import gc
import json
import time
import math
import traceback
import torch
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DTYPE = torch.float16
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory
    print("GPU detected: " + gpu_name)
    print("VRAM: " + str(round(gpu_mem / 1e9, 1)) + " GB")
else:
    DEVICE = torch.device("cpu")
    DTYPE = torch.float32
    print("No GPU -- running on CPU (will be slow)")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).parent.parent / "results"
if not RESULTS_DIR.exists():
    RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_FILE = RESULTS_DIR / ("ouro_tlb_truthfulqa_" + TIMESTAMP + ".json")

MODEL_NAME = "ByteDance/Ouro-1.4B"
N_SAMPLES = 200  # Quick run; full dataset is 817

LOOP_COUNTS = [4, 8]

TLB_CONFIGS = [
    {"name": "best_a0.5_r3.0", "alpha": 0.5, "radius": 3.0, "max_tokens": 1440},
    {"name": "alt1_a0.3_r3.0", "alpha": 0.3, "radius": 3.0, "max_tokens": 1440},
    {"name": "alt2_a0.5_r3.5", "alpha": 0.5, "radius": 3.5, "max_tokens": 1440},
]


# ---------------------------------------------------------------------------
# Toroidal Logit Bias (inline, no external deps)
# ---------------------------------------------------------------------------

def toroidal_distance(i, j, grid_size=12):
    """Manhattan distance on a torus (grid_size x grid_size)."""
    xi, yi = i % grid_size, (i // grid_size) % grid_size
    xj, yj = j % grid_size, (j // grid_size) % grid_size
    dx = min(abs(xi - xj), grid_size - abs(xi - xj))
    dy = min(abs(yi - yj), grid_size - abs(yi - yj))
    return dx + dy


def get_toroidal_bias(vocab_size, recent_tokens, alpha, radius, max_tokens,
                      grid_size=12, device="cuda"):
    """Compute toroidal logit bias vector from recent context tokens."""
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

    # Zero-center so bias does not shift overall logit scale
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


# ---------------------------------------------------------------------------
# McNemar's test
# ---------------------------------------------------------------------------

def mcnemar_test(b, c):
    """
    McNemar's test for paired nominal data.
    b = baseline wrong, TLB right  (improvements)
    c = baseline right, TLB wrong  (regressions)
    Returns (chi2, p_value).
    """
    if b + c == 0:
        return 0.0, 1.0

    chi2 = (abs(b - c) - 1) ** 2 / (b + c)

    from math import exp, sqrt

    def chi2_sf(x, df=1):
        """Survival function for chi-squared (1 df) -- approximate."""
        if x <= 0:
            return 1.0
        z = sqrt(x)
        t = 1 / (1 + 0.2316419 * z)
        d = 0.3989423 * exp(-z * z / 2)
        p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))))
        return 2 * p  # Two-tailed

    return chi2, chi2_sf(chi2)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_choice(model, tokenizer, question, choice, device, logits_processor=None):
    """Score a single MC choice. Higher score = model thinks more likely."""
    prompt = "Question: " + question + "\nAnswer: " + choice
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        if logits_processor:
            logits = outputs.logits[0, -1, :]
            logits = logits.unsqueeze(0)  # (1, vocab)
            input_ids_for_bias = inputs["input_ids"]
            logits = logits_processor(input_ids_for_bias, logits)
            score = logits[0].mean().item()
        else:
            score = outputs.logits[0, -1].mean().item()
    return score


# ---------------------------------------------------------------------------
# Paired evaluation on TruthfulQA MC1
# ---------------------------------------------------------------------------

def evaluate_truthfulqa_paired(model, tokenizer, device, tlb_processor,
                               samples, n_samples):
    """
    Paired baseline vs TLB evaluation on the same prompts.
    Returns dict with accuracies, discordant counts, details.
    """
    from tqdm import tqdm

    baseline_correct_count = 0
    tlb_correct_count = 0
    # Discordant pairs
    b_improvements = 0  # baseline wrong, TLB right
    c_regressions = 0   # baseline right, TLB wrong
    both_correct = 0
    both_wrong = 0

    details = []

    for example in tqdm(samples[:n_samples], desc="TruthfulQA MC1 (paired)"):
        question = example["question"]
        choices = example["mc1_targets"]["choices"]
        labels = example["mc1_targets"]["labels"]
        correct_idx = labels.index(1)

        # --- Baseline ---
        baseline_scores = []
        for ch in choices:
            s = score_choice(model, tokenizer, question, ch, device,
                             logits_processor=None)
            baseline_scores.append(s)
        baseline_pred = baseline_scores.index(max(baseline_scores))
        baseline_ok = (baseline_pred == correct_idx)

        # --- TLB ---
        tlb_scores = []
        for ch in choices:
            s = score_choice(model, tokenizer, question, ch, device,
                             logits_processor=tlb_processor)
            tlb_scores.append(s)
        tlb_pred = tlb_scores.index(max(tlb_scores))
        tlb_ok = (tlb_pred == correct_idx)

        # Count
        if baseline_ok:
            baseline_correct_count += 1
        if tlb_ok:
            tlb_correct_count += 1

        if baseline_ok and tlb_ok:
            both_correct += 1
            pair_type = "both_correct"
        elif not baseline_ok and not tlb_ok:
            both_wrong += 1
            pair_type = "both_wrong"
        elif not baseline_ok and tlb_ok:
            b_improvements += 1
            pair_type = "improvement"
        else:
            c_regressions += 1
            pair_type = "regression"

        # Truncate question for storage
        q_short = question[:100]
        details.append({
            "question": q_short,
            "baseline_correct": baseline_ok,
            "tlb_correct": tlb_ok,
            "pair_type": pair_type,
            "n_choices": len(choices),
            "correct_idx": correct_idx,
            "baseline_pred": baseline_pred,
            "tlb_pred": tlb_pred,
        })

    return {
        "n_samples": n_samples,
        "baseline_correct": baseline_correct_count,
        "tlb_correct": tlb_correct_count,
        "baseline_accuracy": baseline_correct_count / n_samples,
        "tlb_accuracy": tlb_correct_count / n_samples,
        "discordant": {
            "b_improvements": b_improvements,
            "c_regressions": c_regressions,
        },
        "concordant": {
            "both_correct": both_correct,
            "both_wrong": both_wrong,
        },
        "details": details,
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment():
    print("=" * 70)
    print("OURO-1.4B TRUTHFULQA MC1 + TOROIDAL LOGIT BIAS")
    print("=" * 70)
    print("Device:    " + str(DEVICE) + " (" + str(DTYPE) + ")")
    print("Model:     " + MODEL_NAME)
    print("Samples:   " + str(N_SAMPLES) + " (of 817)")
    print("Loops:     " + str(LOOP_COUNTS))
    print("TLB cfgs:  " + str([c["name"] for c in TLB_CONFIGS]))
    print("Results -> " + str(RESULTS_FILE))
    print("Start:     " + datetime.now().isoformat())
    print("=" * 70, flush=True)

    device = DEVICE

    # ------------------------------------------------------------------
    # Load tokenizer (shared across all runs)
    # ------------------------------------------------------------------
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    print("\nLoading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    # Load TruthfulQA dataset
    # ------------------------------------------------------------------
    print("Loading TruthfulQA dataset...", flush=True)
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets library...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
        from datasets import load_dataset

    dataset = load_dataset("truthful_qa", "multiple_choice", split="validation")
    total_available = len(dataset)
    print("Dataset loaded: " + str(total_available) + " questions")

    # Sample uniformly
    if N_SAMPLES < total_available:
        step = total_available // N_SAMPLES
        indices = list(range(0, total_available, step))[:N_SAMPLES]
        samples = [dataset[i] for i in indices]
    else:
        samples = list(dataset)

    actual_n = len(samples)
    print("Using " + str(actual_n) + " samples", flush=True)

    # ------------------------------------------------------------------
    # Results container
    # ------------------------------------------------------------------
    all_results = {
        "experiment": "ouro_tlb_truthfulqa",
        "model": MODEL_NAME,
        "device": str(device),
        "dtype": str(DTYPE),
        "date": datetime.now().isoformat(),
        "n_samples": actual_n,
        "loop_counts": LOOP_COUNTS,
        "tlb_configs": [c.copy() for c in TLB_CONFIGS],
        "runs": [],
    }

    # ------------------------------------------------------------------
    # Run all combos: loop_count x (baseline + each TLB config)
    # ------------------------------------------------------------------
    total_combos = len(LOOP_COUNTS) * (1 + len(TLB_CONFIGS))
    combo_idx = 0

    for n_loops in LOOP_COUNTS:
        print("\n" + "=" * 70)
        print("UT LOOPS: " + str(n_loops))
        print("=" * 70, flush=True)

        # Load model with this loop count
        print("Loading model (loops=" + str(n_loops) + ")...", flush=True)
        t_load = time.time()
        try:
            config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
            config.total_ut_steps = n_loops
            config.early_exit_threshold = 1.0
            config.pad_token_id = tokenizer.eos_token_id

            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                config=config,
                trust_remote_code=True,
                torch_dtype=DTYPE,
            ).to(device)
            model.eval()
            load_secs = time.time() - t_load
            print("Model loaded in " + str(round(load_secs, 1)) + "s", flush=True)
        except Exception as e:
            print("ERROR loading model: " + str(e))
            traceback.print_exc()
            # Record error for all combos at this loop count
            for tlb_cfg in [None] + TLB_CONFIGS:
                combo_idx += 1
                cfg_name = "baseline" if tlb_cfg is None else tlb_cfg["name"]
                all_results["runs"].append({
                    "n_loops": n_loops,
                    "tlb_config": cfg_name,
                    "error": str(e),
                })
            continue

        # --- Baseline (no TLB) ---
        combo_idx += 1
        print("\n  [" + str(combo_idx) + "/" + str(total_combos) + "] Baseline, loops=" + str(n_loops), flush=True)
        t_eval = time.time()
        baseline_details = None
        try:
            results = evaluate_truthfulqa_paired(
                model, tokenizer, device,
                tlb_processor=None,
                samples=samples,
                n_samples=actual_n,
            )
            # When tlb_processor=None, both baseline and "TLB" paths produce
            # identical scores, so baseline_accuracy == tlb_accuracy.
            baseline_acc = results["baseline_accuracy"]
            baseline_correct = results["baseline_correct"]
            elapsed = time.time() - t_eval
            print("  Baseline accuracy: " + str(round(baseline_acc * 100, 2)) + "%" +
                  " (" + str(baseline_correct) + "/" + str(actual_n) + ")" +
                  " [" + str(round(elapsed, 0)) + "s]", flush=True)

            run_record = {
                "n_loops": n_loops,
                "tlb_config": "baseline",
                "accuracy": baseline_acc,
                "correct": baseline_correct,
                "n_samples": actual_n,
                "elapsed_s": round(elapsed, 1),
            }
            all_results["runs"].append(run_record)

            # Store baseline details for paired comparison
            baseline_details = results["details"]

        except Exception as e:
            print("  ERROR: " + str(e))
            traceback.print_exc()
            all_results["runs"].append({
                "n_loops": n_loops,
                "tlb_config": "baseline",
                "error": str(e),
            })

        # Save intermediate
        with open(RESULTS_FILE, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        # --- TLB configs ---
        for tlb_cfg in TLB_CONFIGS:
            combo_idx += 1
            cfg_name = tlb_cfg["name"]
            alpha_val = tlb_cfg["alpha"]
            radius_val = tlb_cfg["radius"]
            max_tok = tlb_cfg["max_tokens"]

            print("\n  [" + str(combo_idx) + "/" + str(total_combos) + "] " + cfg_name +
                  ", loops=" + str(n_loops), flush=True)
            print("    alpha=" + str(alpha_val) + " radius=" + str(radius_val) +
                  " max_tokens=" + str(max_tok), flush=True)

            processor = ToroidalLogitProcessor(
                alpha=alpha_val,
                radius=radius_val,
                max_tokens=max_tok,
            )

            t_eval = time.time()
            try:
                results = evaluate_truthfulqa_paired(
                    model, tokenizer, device,
                    tlb_processor=processor,
                    samples=samples,
                    n_samples=actual_n,
                )
                elapsed = time.time() - t_eval

                bl_acc = results["baseline_accuracy"]
                tlb_acc = results["tlb_accuracy"]
                bl_err = 1 - bl_acc
                tlb_err = 1 - tlb_acc
                err_reduction = ((bl_err - tlb_err) / bl_err * 100) if bl_err > 0 else 0.0
                delta_pp = (tlb_acc - bl_acc) * 100

                b_imp = results["discordant"]["b_improvements"]
                c_reg = results["discordant"]["c_regressions"]
                chi2, p_value = mcnemar_test(b_imp, c_reg)

                print("    Baseline:  " + str(round(bl_acc * 100, 2)) + "%  (" +
                      str(results["baseline_correct"]) + "/" + str(actual_n) + ")", flush=True)
                print("    TLB:       " + str(round(tlb_acc * 100, 2)) + "%  (" +
                      str(results["tlb_correct"]) + "/" + str(actual_n) + ")", flush=True)
                delta_sign = "+" if delta_pp >= 0 else ""
                print("    Delta:     " + delta_sign + str(round(delta_pp, 2)) + "pp", flush=True)
                print("    Err reduc: " + str(round(err_reduction, 1)) + "%", flush=True)
                print("    McNemar:   chi2=" + str(round(chi2, 3)) +
                      " p=" + str(round(p_value, 4)), flush=True)
                print("    Pairs:     improvements=" + str(b_imp) +
                      " regressions=" + str(c_reg) +
                      " net=" + str(b_imp - c_reg), flush=True)

                sig_label = ""
                if p_value < 0.05:
                    if b_imp > c_reg:
                        sig_label = "SIGNIFICANT IMPROVEMENT"
                    else:
                        sig_label = "SIGNIFICANT REGRESSION"
                    print("    >>> " + sig_label + " (p < 0.05)", flush=True)

                run_record = {
                    "n_loops": n_loops,
                    "tlb_config": cfg_name,
                    "tlb_alpha": alpha_val,
                    "tlb_radius": radius_val,
                    "tlb_max_tokens": max_tok,
                    "baseline_accuracy": bl_acc,
                    "tlb_accuracy": tlb_acc,
                    "delta_pp": round(delta_pp, 2),
                    "error_reduction_pct": round(err_reduction, 1),
                    "mcnemar_chi2": round(chi2, 4),
                    "mcnemar_p_value": round(p_value, 6),
                    "improvements": b_imp,
                    "regressions": c_reg,
                    "net_improvement": b_imp - c_reg,
                    "both_correct": results["concordant"]["both_correct"],
                    "both_wrong": results["concordant"]["both_wrong"],
                    "n_samples": actual_n,
                    "elapsed_s": round(elapsed, 1),
                    "significant": sig_label if sig_label else None,
                    "details": results["details"],
                }
                all_results["runs"].append(run_record)

            except Exception as e:
                print("    ERROR: " + str(e))
                traceback.print_exc()
                all_results["runs"].append({
                    "n_loops": n_loops,
                    "tlb_config": cfg_name,
                    "error": str(e),
                })

            # Save after each config
            with open(RESULTS_FILE, "w") as f:
                json.dump(all_results, f, indent=2, default=str)

        # Free model before loading next loop count
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY: OURO-1.4B TRUTHFULQA MC1 + TLB")
    print("=" * 70, flush=True)

    # Collect baseline accuracies per loop count
    baseline_accs = {}
    for run in all_results["runs"]:
        if run.get("tlb_config") == "baseline" and "accuracy" in run:
            baseline_accs[run["n_loops"]] = run["accuracy"]

    for n_loops in LOOP_COUNTS:
        bl_acc = baseline_accs.get(n_loops, None)
        if bl_acc is None:
            print("\nLoops=" + str(n_loops) + ": baseline failed")
            continue

        print("\nLoops=" + str(n_loops) + " | Baseline: " + str(round(bl_acc * 100, 2)) + "%")
        header = "  {:<25} {:>10} {:>10} {:>10} {:>8} {:>8} {:>10}".format(
            "TLB Config", "BL Acc", "TLB Acc", "Delta", "Improv", "Regr", "p-value"
        )
        print(header)
        print("  " + "-" * 81)

        for run in all_results["runs"]:
            if run.get("n_loops") != n_loops:
                continue
            if run.get("tlb_config") == "baseline":
                continue
            if "error" in run:
                line = "  {:<25} ERROR: {}".format(
                    run.get("tlb_config", "?"), run["error"]
                )
                print(line)
                continue

            cfg_name = run["tlb_config"]
            bl_a = run["baseline_accuracy"]
            tlb_a = run["tlb_accuracy"]
            delta = run["delta_pp"]
            imp = run["improvements"]
            reg = run["regressions"]
            pv = run["mcnemar_p_value"]

            delta_sign = "+" if delta >= 0 else ""
            delta_str = delta_sign + str(delta) + "pp"
            sig_marker = ""
            if pv < 0.05 and imp > reg:
                sig_marker = " ***"
            elif pv < 0.05 and reg > imp:
                sig_marker = " !!!"

            row = "  {:<25} {:>9}% {:>9}% {:>10} {:>8} {:>8} {:>10}{}".format(
                cfg_name,
                round(bl_a * 100, 2),
                round(tlb_a * 100, 2),
                delta_str,
                imp,
                reg,
                round(pv, 4),
                sig_marker,
            )
            print(row)

    # Cross-loop best config
    print("\n" + "-" * 70)
    print("BEST CONFIGS (by delta_pp):")
    tlb_runs = [r for r in all_results["runs"]
                if r.get("tlb_config") != "baseline" and "delta_pp" in r]
    if tlb_runs:
        tlb_runs_sorted = sorted(tlb_runs, key=lambda r: r["delta_pp"], reverse=True)
        for i, run in enumerate(tlb_runs_sorted[:5]):
            print("  #" + str(i + 1) + ": " + run["tlb_config"] +
                  " loops=" + str(run["n_loops"]) +
                  " delta=" + str(run["delta_pp"]) + "pp" +
                  " p=" + str(round(run["mcnemar_p_value"], 4)))

    # Reference: prior results
    print("\n" + "-" * 70)
    print("REFERENCE (published Toroidal Logit Bias results):")
    print("  Phi-2 baseline:       14.44%")
    print("  Phi-2 + TLB:          17.26%  (+2.82pp, 19.5% error reduction)")
    print("  Qwen-2.5-7B baseline: see run_truthfulqa_eval.py")
    print("-" * 70)

    all_results["end_time"] = datetime.now().isoformat()
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\nResults saved to: " + str(RESULTS_FILE))
    print("End: " + datetime.now().isoformat())
    print("=" * 70, flush=True)


if __name__ == "__main__":
    try:
        run_experiment()
    except Exception as e:
        print("FATAL ERROR: " + str(e))
        traceback.print_exc()
        sys.exit(1)
