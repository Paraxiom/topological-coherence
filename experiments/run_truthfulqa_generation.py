#!/usr/bin/env python3
"""
TruthfulQA GENERATION-BASED Evaluation
=======================================
Instead of scoring pre-written answers, GENERATE answers and check if correct.
This aligns with how our custom benchmark works (where toroidal bias helped).
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
# TOROIDAL BIAS
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
# GENERATION FUNCTIONS
# ============================================================================

def generate_baseline(model, tokenizer, prompt, max_new_tokens=50):
    """Standard generation without bias."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
    # Return only the generated part
    generated = outputs[0][inputs['input_ids'].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)

def generate_with_toroidal(model, tokenizer, prompt, config, max_new_tokens=50):
    """Generation with toroidal logit bias at each step."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generated = inputs['input_ids'][0].tolist()
    vocab_size = model.config.vocab_size
    prompt_len = len(generated)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(torch.tensor([generated], device=model.device))
            logits = outputs.logits[0, -1, :]

            # Apply toroidal bias
            bias = get_toroidal_bias(
                vocab_size=vocab_size,
                recent_tokens=generated,
                alpha=config["alpha"],
                radius=config["radius"],
                max_tokens=config["max_tokens"],
                device=model.device
            )
            logits = logits + bias

            next_token = logits.argmax().item()

        generated.append(next_token)
        if next_token == tokenizer.eos_token_id:
            break

    # Return only the generated part
    return tokenizer.decode(generated[prompt_len:], skip_special_tokens=True)

def check_answer_match(generated_text, correct_answers):
    """Check if any correct answer appears in the generated text."""
    generated_lower = generated_text.lower().strip()
    for answer in correct_answers:
        if answer.lower() in generated_lower:
            return True
    return False

# ============================================================================
# TRUTHFULQA GENERATION EVALUATION
# ============================================================================

def evaluate_truthfulqa_generation(model, tokenizer, config, n_samples=None):
    """
    Generation-based TruthfulQA evaluation.
    Generate answer, then check if it matches correct choice.
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

    print(f"Evaluating {n_samples} samples (generation-based, paired)...")

    # Counters
    baseline_correct = 0
    toroidal_correct = 0
    b = 0  # improvements
    c = 0  # regressions
    both_correct = 0
    both_wrong = 0

    details = []

    for example in tqdm(samples, desc="TruthfulQA (generation)"):
        question = example["question"]
        choices = example["mc1_targets"]["choices"]
        labels = example["mc1_targets"]["labels"]
        correct_idx = labels.index(1)
        correct_answer = choices[correct_idx]

        # Also accept other correct answers if they exist
        correct_answers = [choices[i] for i, l in enumerate(labels) if l == 1]

        prompt = f"Question: {question}\nAnswer:"

        # Baseline generation
        baseline_gen = generate_baseline(model, tokenizer, prompt)
        baseline_ok = check_answer_match(baseline_gen, correct_answers)

        # Toroidal generation
        toroidal_gen = generate_with_toroidal(model, tokenizer, prompt, config)
        toroidal_ok = check_answer_match(toroidal_gen, correct_answers)

        if baseline_ok:
            baseline_correct += 1
        if toroidal_ok:
            toroidal_correct += 1

        # Classify pair
        if baseline_ok and toroidal_ok:
            both_correct += 1
        elif not baseline_ok and not toroidal_ok:
            both_wrong += 1
        elif not baseline_ok and toroidal_ok:
            b += 1
        else:
            c += 1

        details.append({
            "question": question[:60],
            "correct_answer": correct_answer[:30],
            "baseline_gen": baseline_gen[:50],
            "toroidal_gen": toroidal_gen[:50],
            "baseline_ok": baseline_ok,
            "toroidal_ok": toroidal_ok
        })

    return {
        "n_samples": n_samples,
        "baseline_correct": baseline_correct,
        "toroidal_correct": toroidal_correct,
        "baseline_accuracy": baseline_correct / n_samples,
        "toroidal_accuracy": toroidal_correct / n_samples,
        "b_improvements": b,
        "c_regressions": c,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "details": details
    }

# ============================================================================
# MAIN
# ============================================================================

def run_evaluation(model_key, n_samples=200):
    config = MODEL_CONFIGS[model_key]
    model_name = config["name"]

    print("=" * 70)
    print(f"TRUTHFULQA GENERATION EVALUATION: {model_name}")
    print("=" * 70)

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

    results = evaluate_truthfulqa_generation(model, tokenizer, config, n_samples)

    b_acc = results['baseline_accuracy']
    t_acc = results['toroidal_accuracy']
    b_err = 1 - b_acc
    err_reduction = ((b_err - (1 - t_acc)) / b_err * 100) if b_err > 0 else 0

    b = results['b_improvements']
    c = results['c_regressions']

    # McNemar
    if b + c > 0:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        # Approximate p-value
        import math
        z = math.sqrt(chi2) if chi2 > 0 else 0
        t = 1 / (1 + 0.2316419 * z) if z > 0 else 1
        d = 0.3989423 * math.exp(-z * z / 2)
        p_value = 2 * d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))))
    else:
        chi2 = 0
        p_value = 1.0

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
    print(f"Both correct:   {results['both_correct']}")
    print(f"Both wrong:     {results['both_wrong']}")
    print(f"")
    print(f"McNemar's χ²:   {chi2:.3f}")
    print(f"p-value:        {p_value:.4f}")

    if b > c:
        print(f"\n>>> b > c: TOROIDAL HELPS")
    elif c > b:
        print(f"\n>>> c > b: TOROIDAL HURTS")
    else:
        print(f"\n>>> b = c: NO EFFECT")

    # Show some examples
    print("\n" + "-" * 70)
    print("SAMPLE OUTPUTS (first 5)")
    print("-" * 70)
    for d in results['details'][:5]:
        status = "✓" if d['toroidal_ok'] else "✗"
        print(f"{status} Q: {d['question']}")
        print(f"  Correct: {d['correct_answer']}")
        print(f"  Baseline: {d['baseline_gen'][:40]}... ({'✓' if d['baseline_ok'] else '✗'})")
        print(f"  Toroidal: {d['toroidal_gen'][:40]}... ({'✓' if d['toroidal_ok'] else '✗'})")
        print()

    # Save
    os.makedirs("./results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"./results/truthfulqa_gen_{model_key}_{timestamp}.json"

    output = {
        "model": model_name,
        "model_key": model_key,
        "n_samples": n_samples,
        "timestamp": timestamp,
        "config": config,
        "baseline_accuracy": b_acc,
        "toroidal_accuracy": t_acc,
        "error_reduction_pct": err_reduction,
        "b_improvements": b,
        "c_regressions": c,
        "mcnemar_chi2": chi2,
        "mcnemar_p_value": p_value
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    return output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen", "olmo", "both"], default="both")
    parser.add_argument("--samples", type=int, default=200)
    args = parser.parse_args()

    all_results = {}

    if args.model in ["qwen", "both"]:
        all_results["qwen"] = run_evaluation("qwen", args.samples)
        torch.cuda.empty_cache()

    if args.model in ["olmo", "both"]:
        all_results["olmo"] = run_evaluation("olmo", args.samples)

    if len(all_results) > 1:
        print("\n" + "=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)
        print(f"{'Model':<12} {'Baseline':>10} {'Toroidal':>10} {'b':>6} {'c':>6} {'Net':>6}")
        print("-" * 70)
        for key, res in all_results.items():
            net = res['b_improvements'] - res['c_regressions']
            print(f"{key:<12} {res['baseline_accuracy']:>9.2%} {res['toroidal_accuracy']:>9.2%} "
                  f"{res['b_improvements']:>6} {res['c_regressions']:>6} {net:>+6}")

if __name__ == "__main__":
    main()
