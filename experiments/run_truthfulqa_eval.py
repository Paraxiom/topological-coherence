#!/usr/bin/env python3
"""
TruthfulQA Evaluation for Toroidal Logit Bias
==============================================
Run standard TruthfulQA benchmark on Qwen and OLMo with/without toroidal bias.
This allows direct comparison with Phi-2 results.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import json
from datetime import datetime
import os
import argparse

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
# TRUTHFULQA EVALUATION
# ============================================================================

def score_choice(model, tokenizer, question, choice, use_toroidal=False, config=None):
    """Score a single choice using mean logit of the answer tokens."""
    prompt = f"Question: {question}\nAnswer: {choice}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)

    with torch.no_grad():
        if use_toroidal and config:
            # Generate token by token with bias to get final representation
            input_ids = inputs['input_ids'][0].tolist()
            outputs = model(inputs['input_ids'])
            logits = outputs.logits[0, -1, :]

            # Apply toroidal bias
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

def evaluate_truthfulqa(model, tokenizer, config, use_toroidal=False, n_samples=None):
    """
    Evaluate on TruthfulQA multiple choice.

    Uses mc1_targets (single correct answer) for cleaner evaluation.
    """
    print(f"\nLoading TruthfulQA dataset...")
    dataset = load_dataset("truthful_qa", "multiple_choice", split="validation")

    if n_samples and n_samples < len(dataset):
        # Evenly sample
        step = len(dataset) // n_samples
        indices = list(range(0, len(dataset), step))[:n_samples]
        samples = [dataset[i] for i in indices]
    else:
        samples = list(dataset)
        n_samples = len(samples)

    print(f"Evaluating {n_samples} samples (toroidal={use_toroidal})...")

    correct = 0
    total = 0
    results_detail = []

    for example in tqdm(samples, desc="TruthfulQA"):
        question = example["question"]
        choices = example["mc1_targets"]["choices"]
        labels = example["mc1_targets"]["labels"]

        correct_idx = labels.index(1)

        scores = []
        for choice in choices:
            score = score_choice(model, tokenizer, question, choice,
                               use_toroidal=use_toroidal, config=config)
            scores.append(score)

        predicted_idx = scores.index(max(scores))
        is_correct = (predicted_idx == correct_idx)

        if is_correct:
            correct += 1
        total += 1

        results_detail.append({
            "question": question[:100],
            "correct_idx": correct_idx,
            "predicted_idx": predicted_idx,
            "is_correct": is_correct
        })

    accuracy = correct / total
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "details": results_detail
    }

# ============================================================================
# MAIN
# ============================================================================

def run_evaluation(model_key, n_samples=200):
    """Run TruthfulQA evaluation for a model."""
    config = MODEL_CONFIGS[model_key]
    model_name = config["name"]

    print("=" * 70)
    print(f"TRUTHFULQA EVALUATION: {model_name}")
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

    # Baseline evaluation
    print("\n" + "-" * 70)
    print("BASELINE (no toroidal bias)")
    print("-" * 70)
    baseline_results = evaluate_truthfulqa(model, tokenizer, config,
                                           use_toroidal=False, n_samples=n_samples)
    print(f"Baseline accuracy: {baseline_results['accuracy']:.2%} ({baseline_results['correct']}/{baseline_results['total']})")

    # Toroidal evaluation
    print("\n" + "-" * 70)
    print(f"TOROIDAL (α={config['alpha']}, r={config['radius']}, n={config['max_tokens']})")
    print("-" * 70)
    toroidal_results = evaluate_truthfulqa(model, tokenizer, config,
                                           use_toroidal=True, n_samples=n_samples)
    print(f"Toroidal accuracy: {toroidal_results['accuracy']:.2%} ({toroidal_results['correct']}/{toroidal_results['total']})")

    # Calculate improvement
    b_acc = baseline_results['accuracy']
    t_acc = toroidal_results['accuracy']
    b_err = 1 - b_acc
    t_err = 1 - t_acc
    err_reduction = ((b_err - t_err) / b_err * 100) if b_err > 0 else 0

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Samples: {n_samples}")
    print(f"Baseline:  {b_acc:.2%}")
    print(f"Toroidal:  {t_acc:.2%}")
    print(f"Error reduction: {err_reduction:+.1f}%")

    # Compare to Phi-2 benchmark
    print("\n" + "-" * 70)
    print("COMPARISON TO PHI-2 (January 2026)")
    print("-" * 70)
    print(f"Phi-2 baseline TruthfulQA:     14.44%")
    print(f"Phi-2 local_window TruthfulQA: 17.26% (+19.5% error reduction)")
    print(f"{model_key.upper()} baseline TruthfulQA:     {b_acc:.2%}")
    print(f"{model_key.upper()} toroidal TruthfulQA:     {t_acc:.2%} ({err_reduction:+.1f}% error reduction)")

    # Save results
    os.makedirs("./results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"./results/truthfulqa_{model_key}_{timestamp}.json"

    results = {
        "model": model_name,
        "model_key": model_key,
        "n_samples": n_samples,
        "timestamp": timestamp,
        "config": {
            "alpha": config["alpha"],
            "radius": config["radius"],
            "max_tokens": config["max_tokens"]
        },
        "baseline": {
            "accuracy": b_acc,
            "correct": baseline_results['correct'],
            "total": baseline_results['total']
        },
        "toroidal": {
            "accuracy": t_acc,
            "correct": toroidal_results['correct'],
            "total": toroidal_results['total']
        },
        "error_reduction": err_reduction
    }

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    return results

def main():
    parser = argparse.ArgumentParser(description="TruthfulQA evaluation with toroidal bias")
    parser.add_argument("--model", choices=["qwen", "olmo", "both"], default="both",
                       help="Which model to evaluate")
    parser.add_argument("--samples", type=int, default=200,
                       help="Number of TruthfulQA samples (200 for quick, 817 for full)")
    args = parser.parse_args()

    all_results = {}

    if args.model in ["qwen", "both"]:
        all_results["qwen"] = run_evaluation("qwen", args.samples)
        # Clear GPU memory
        torch.cuda.empty_cache()

    if args.model in ["olmo", "both"]:
        all_results["olmo"] = run_evaluation("olmo", args.samples)

    # Final summary
    if len(all_results) > 1:
        print("\n" + "=" * 70)
        print("FINAL SUMMARY - ALL MODELS")
        print("=" * 70)
        print(f"{'Model':<20} {'Baseline':>12} {'Toroidal':>12} {'Error Red.':>12}")
        print("-" * 70)
        for key, res in all_results.items():
            print(f"{key:<20} {res['baseline']['accuracy']:>11.2%} {res['toroidal']['accuracy']:>11.2%} {res['error_reduction']:>+11.1f}%")
        print("-" * 70)
        print(f"{'Phi-2 (ref)':<20} {'14.44%':>12} {'17.26%':>12} {'+19.5%':>12}")

if __name__ == "__main__":
    main()
