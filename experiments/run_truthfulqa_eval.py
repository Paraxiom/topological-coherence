#!/usr/bin/env python3
"""
TruthfulQA Evaluation for Toroidal Logit Bias
==============================================
Run standard TruthfulQA benchmark on Qwen and OLMo with/without toroidal bias.
Uses paired evaluation (same prompts) with McNemar's test for proper statistics.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import json
from datetime import datetime
import os
import argparse
import math

# ============================================================================
# TOROIDAL BIAS (same as validated experiments)
# ============================================================================

def toroidal_distance(i, j, grid_size=12):
    xi, yi = i % grid_size, (i // grid_size) % grid_size
    xj, yj = j % grid_size, (j // grid_size) % grid_size
    dx = min(abs(xi - xj), grid_size - abs(xi - xj))
    dy = min(abs(yi - yj), grid_size - abs(yi - yj))
    return dx + dy

def get_toroidal_bias(vocab_size, recent_tokens, alpha, radius, max_tokens, grid_size=12, device='cuda'):
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
# MCNEMAR'S TEST
# ============================================================================

def mcnemar_test(b, c):
    """
    McNemar's test for paired nominal data.
    b = baseline wrong, toroidal right (improvements)
    c = baseline right, toroidal wrong (regressions)

    Returns chi-squared statistic and p-value.
    """
    if b + c == 0:
        return 0.0, 1.0

    # McNemar's chi-squared (with continuity correction)
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)

    # Approximate p-value from chi-squared distribution (1 df)
    # Using simple approximation
    from math import exp, sqrt, pi

    def chi2_sf(x, df=1):
        """Survival function for chi-squared (1 df) - approximate."""
        if x <= 0:
            return 1.0
        # For df=1, chi2 is square of standard normal
        z = sqrt(x)
        # Standard normal CDF approximation
        t = 1 / (1 + 0.2316419 * z)
        d = 0.3989423 * exp(-z * z / 2)
        p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))))
        return 2 * p  # Two-tailed

    p_value = chi2_sf(chi2)
    return chi2, p_value

# ============================================================================
# MODEL CONFIGS
# ============================================================================

MODEL_CONFIGS = {
    "qwen": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "alpha": 0.3,
        "radius": 2.0,
        "max_tokens": 1440
    },
    "olmo": {
        "name": "allenai/OLMo-1.7-7B-hf",
        "alpha": 0.2,
        "radius": 3.0,
        "max_tokens": 3000
    }
}

# ============================================================================
# TRUTHFULQA EVALUATION (PAIRED)
# ============================================================================

def score_choice(model, tokenizer, question, choice, use_toroidal=False, config=None):
    """Score a single choice using mean logit of the answer tokens."""
    prompt = f"Question: {question}\nAnswer: {choice}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)

    with torch.no_grad():
        if use_toroidal and config:
            input_ids = inputs['input_ids'][0].tolist()
            outputs = model(inputs['input_ids'])
            logits = outputs.logits[0, -1, :]

            bias = get_toroidal_bias(
                vocab_size=logits.shape[0],
                recent_tokens=input_ids,
                alpha=config["alpha"],
                radius=config["radius"],
                max_tokens=config["max_tokens"],
                device=model.device
            )
            logits = logits + bias
            score = logits.mean().item()
        else:
            outputs = model(**inputs)
            score = outputs.logits[0, -1].mean().item()

    return score

def evaluate_single_prompt(model, tokenizer, config, question, choices, correct_idx):
    """Evaluate a single prompt with both baseline and toroidal."""

    # Baseline
    baseline_scores = []
    for choice in choices:
        score = score_choice(model, tokenizer, question, choice, use_toroidal=False)
        baseline_scores.append(score)
    baseline_pred = baseline_scores.index(max(baseline_scores))
    baseline_correct = (baseline_pred == correct_idx)

    # Toroidal
    toroidal_scores = []
    for choice in choices:
        score = score_choice(model, tokenizer, question, choice, use_toroidal=True, config=config)
        toroidal_scores.append(score)
    toroidal_pred = toroidal_scores.index(max(toroidal_scores))
    toroidal_correct = (toroidal_pred == correct_idx)

    return baseline_correct, toroidal_correct

def evaluate_truthfulqa_paired(model, tokenizer, config, n_samples=None):
    """
    Paired evaluation on TruthfulQA.
    Returns accuracy for both conditions plus discordant pair counts.
    """
    print(f"\nLoading TruthfulQA dataset...")
    dataset = load_dataset("truthful_qa", "multiple_choice", split="validation")

    if n_samples and n_samples < len(dataset):
        step = len(dataset) // n_samples
        indices = list(range(0, len(dataset), step))[:n_samples]
        samples = [dataset[i] for i in indices]
    else:
        samples = list(dataset)
        n_samples = len(samples)

    print(f"Evaluating {n_samples} samples (paired: baseline + toroidal per prompt)...")

    # Counters
    baseline_correct_count = 0
    toroidal_correct_count = 0

    # Discordant pairs (for McNemar's test)
    b = 0  # baseline wrong, toroidal right (IMPROVEMENTS)
    c = 0  # baseline right, toroidal wrong (REGRESSIONS)

    # Concordant pairs (for reference)
    both_correct = 0
    both_wrong = 0

    details = []

    for example in tqdm(samples, desc="TruthfulQA (paired)"):
        question = example["question"]
        choices = example["mc1_targets"]["choices"]
        labels = example["mc1_targets"]["labels"]
        correct_idx = labels.index(1)

        baseline_ok, toroidal_ok = evaluate_single_prompt(
            model, tokenizer, config, question, choices, correct_idx
        )

        if baseline_ok:
            baseline_correct_count += 1
        if toroidal_ok:
            toroidal_correct_count += 1

        # Classify pair
        if baseline_ok and toroidal_ok:
            both_correct += 1
        elif not baseline_ok and not toroidal_ok:
            both_wrong += 1
        elif not baseline_ok and toroidal_ok:
            b += 1  # Improvement
        else:  # baseline_ok and not toroidal_ok
            c += 1  # Regression

        details.append({
            "question": question[:80],
            "baseline_correct": baseline_ok,
            "toroidal_correct": toroidal_ok,
            "pair_type": "both_correct" if (baseline_ok and toroidal_ok) else
                        "both_wrong" if (not baseline_ok and not toroidal_ok) else
                        "improvement" if (not baseline_ok and toroidal_ok) else "regression"
        })

    return {
        "n_samples": n_samples,
        "baseline_correct": baseline_correct_count,
        "toroidal_correct": toroidal_correct_count,
        "baseline_accuracy": baseline_correct_count / n_samples,
        "toroidal_accuracy": toroidal_correct_count / n_samples,
        "discordant": {
            "b_improvements": b,  # baseline wrong → toroidal right
            "c_regressions": c,   # baseline right → toroidal wrong
        },
        "concordant": {
            "both_correct": both_correct,
            "both_wrong": both_wrong,
        },
        "details": details
    }

# ============================================================================
# MAIN
# ============================================================================

def run_evaluation(model_key, n_samples=200):
    """Run paired TruthfulQA evaluation for a model."""
    config = MODEL_CONFIGS[model_key]
    model_name = config["name"]

    print("=" * 70)
    print(f"TRUTHFULQA PAIRED EVALUATION: {model_name}")
    print("=" * 70)

    # Load model
    print(f"\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    print(f"Loaded. GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    # Paired evaluation
    results = evaluate_truthfulqa_paired(model, tokenizer, config, n_samples=n_samples)

    b_acc = results['baseline_accuracy']
    t_acc = results['toroidal_accuracy']
    b_err = 1 - b_acc
    t_err = 1 - t_acc
    err_reduction = ((b_err - t_err) / b_err * 100) if b_err > 0 else 0

    b = results['discordant']['b_improvements']
    c = results['discordant']['c_regressions']

    # McNemar's test
    chi2, p_value = mcnemar_test(b, c)

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Samples: {n_samples}")
    print(f"")
    print(f"Baseline accuracy:  {b_acc:.2%} ({results['baseline_correct']}/{n_samples})")
    print(f"Toroidal accuracy:  {t_acc:.2%} ({results['toroidal_correct']}/{n_samples})")
    print(f"Error reduction:    {err_reduction:+.1f}%")

    print("\n" + "-" * 70)
    print("PAIRED ANALYSIS (McNemar's Test)")
    print("-" * 70)
    print(f"b (baseline wrong → toroidal right): {b} improvements")
    print(f"c (baseline right → toroidal wrong): {c} regressions")
    print(f"Net improvement: {b - c} prompts")
    print(f"")
    print(f"Both correct:   {results['concordant']['both_correct']}")
    print(f"Both wrong:     {results['concordant']['both_wrong']}")
    print(f"")
    print(f"McNemar's χ²:   {chi2:.3f}")
    print(f"p-value:        {p_value:.4f}")

    if p_value < 0.05:
        if b > c:
            print(f"\n>>> SIGNIFICANT IMPROVEMENT (p < 0.05)")
        else:
            print(f"\n>>> SIGNIFICANT REGRESSION (p < 0.05)")
    else:
        print(f"\n>>> Not statistically significant (p ≥ 0.05)")

    # Compare to Phi-2
    print("\n" + "-" * 70)
    print("COMPARISON TO PHI-2 (January 2026)")
    print("-" * 70)
    print(f"Phi-2 baseline:     14.44%")
    print(f"Phi-2 local_window: 17.26% (+19.5% error reduction)")
    print(f"{model_key.upper()} baseline:     {b_acc:.2%}")
    print(f"{model_key.upper()} toroidal:     {t_acc:.2%} ({err_reduction:+.1f}% error reduction)")

    # Save results
    os.makedirs("./results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"./results/truthfulqa_{model_key}_{timestamp}.json"

    output = {
        "model": model_name,
        "model_key": model_key,
        "n_samples": n_samples,
        "timestamp": timestamp,
        "config": {
            "alpha": config["alpha"],
            "radius": config["radius"],
            "max_tokens": config["max_tokens"]
        },
        "baseline_accuracy": b_acc,
        "toroidal_accuracy": t_acc,
        "error_reduction_pct": err_reduction,
        "paired_analysis": {
            "b_improvements": b,
            "c_regressions": c,
            "net_improvement": b - c,
            "both_correct": results['concordant']['both_correct'],
            "both_wrong": results['concordant']['both_wrong'],
            "mcnemar_chi2": chi2,
            "mcnemar_p_value": p_value
        }
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    return output

def main():
    parser = argparse.ArgumentParser(description="TruthfulQA paired evaluation with toroidal bias")
    parser.add_argument("--model", choices=["qwen", "olmo", "both"], default="both",
                       help="Which model to evaluate")
    parser.add_argument("--samples", type=int, default=200,
                       help="Number of TruthfulQA samples (200 for quick, 817 for full)")
    args = parser.parse_args()

    all_results = {}

    if args.model in ["qwen", "both"]:
        all_results["qwen"] = run_evaluation("qwen", args.samples)
        torch.cuda.empty_cache()

    if args.model in ["olmo", "both"]:
        all_results["olmo"] = run_evaluation("olmo", args.samples)

    # Final summary
    if len(all_results) > 1:
        print("\n" + "=" * 70)
        print("FINAL SUMMARY - ALL MODELS")
        print("=" * 70)
        print(f"{'Model':<12} {'Baseline':>10} {'Toroidal':>10} {'Err.Red':>10} {'b':>6} {'c':>6} {'p-val':>10}")
        print("-" * 70)
        for key, res in all_results.items():
            print(f"{key:<12} {res['baseline_accuracy']:>9.2%} {res['toroidal_accuracy']:>9.2%} "
                  f"{res['error_reduction_pct']:>+9.1f}% {res['paired_analysis']['b_improvements']:>6} "
                  f"{res['paired_analysis']['c_regressions']:>6} {res['paired_analysis']['mcnemar_p_value']:>10.4f}")
        print("-" * 70)
        print(f"{'Phi-2':<12} {'14.44%':>10} {'17.26%':>10} {'+19.5%':>10}")
        print("")
        print("b = improvements (baseline wrong → toroidal right)")
        print("c = regressions (baseline right → toroidal wrong)")

if __name__ == "__main__":
    main()
