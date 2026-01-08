#!/usr/bin/env python3
"""
Checkpoint Evaluation Script
============================

Run in parallel terminal to evaluate a training checkpoint without interrupting training.

Usage:
    python eval_checkpoint.py ./results_full/toroidal/checkpoint-7500
    python eval_checkpoint.py --checkpoint ./results_full/toroidal/checkpoint-7500 --mask_type toroidal
"""

import argparse
import json
import os
import sys
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm

# Import from local modules
from topological_attention import TopologicalAttentionMask


def load_checkpoint(checkpoint_path: str, base_model_name: str = "microsoft/phi-2"):
    """Load model from checkpoint."""
    print(f"Loading base model: {base_model_name}")

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        base_model.config.pad_token_id = tokenizer.eos_token_id

    print(f"Loading LoRA weights from: {checkpoint_path}")
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()

    return model, tokenizer


def evaluate_truthfulqa_quick(model, tokenizer, n_samples: int = 200):
    """
    Quick TruthfulQA evaluation on subset for checkpoint decisions.

    Args:
        n_samples: Number of samples to evaluate (200 for quick, 817 for full)
    """
    print(f"\nEvaluating TruthfulQA (n={n_samples})...")

    dataset = load_dataset("truthful_qa", "multiple_choice", split="validation")

    # Subsample for speed
    if n_samples < len(dataset):
        indices = list(range(0, len(dataset), len(dataset) // n_samples))[:n_samples]
        samples = [dataset[i] for i in indices]
    else:
        samples = list(dataset)

    correct = 0
    total = 0

    with torch.no_grad():
        for example in tqdm(samples, desc="TruthfulQA"):
            question = example["question"]
            choices = example["mc1_targets"]["choices"]
            labels = example["mc1_targets"]["labels"]

            correct_idx = labels.index(1)

            scores = []
            for choice in choices:
                prompt = f"Question: {question}\nAnswer: {choice}"
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

                outputs = model(**inputs)
                score = outputs.logits[0, -1].mean().item()
                scores.append(score)

            predicted_idx = scores.index(max(scores))
            if predicted_idx == correct_idx:
                correct += 1
            total += 1

    accuracy = correct / total
    return {"accuracy": accuracy, "correct": correct, "total": total}


def evaluate_halueval_quick(model, tokenizer, n_samples: int = 200):
    """
    Quick HaluEval evaluation on subset for checkpoint decisions.
    """
    print(f"\nEvaluating HaluEval (n={n_samples})...")

    try:
        dataset = load_dataset("pminervini/HaluEval", "qa_samples", split="data")
    except Exception as e:
        print(f"Could not load HaluEval: {e}")
        return {"accuracy": None}

    samples = list(dataset)[:n_samples]

    correct = 0
    total = 0

    with torch.no_grad():
        for example in tqdm(samples, desc="HaluEval"):
            question = example.get("question", "")
            knowledge = example.get("knowledge", "")
            answer = example.get("answer", "")
            hallucination = example.get("hallucination", "")

            factual_prompt = f"Context: {knowledge}\nQuestion: {question}\nAnswer: {answer}"
            halluc_prompt = f"Context: {knowledge}\nQuestion: {question}\nAnswer: {hallucination}"

            factual_inputs = tokenizer(factual_prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
            halluc_inputs = tokenizer(halluc_prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)

            factual_outputs = model(**factual_inputs)
            halluc_outputs = model(**halluc_inputs)

            factual_score = factual_outputs.logits[0, -1].mean().item()
            halluc_score = halluc_outputs.logits[0, -1].mean().item()

            if factual_score > halluc_score:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0
    return {"accuracy": accuracy, "correct": correct, "total": total}


def make_decision(tqa_acc: float, halu_acc: float) -> str:
    """
    Apply decision matrix to determine next action.

    Benchmarks:
        local_window TruthfulQA: 17.26%
        local_window HaluEval: 53.00%
    """

    if tqa_acc >= 0.1726 and halu_acc <= 0.53:
        return "STRONG_CONTINUE", "Beating local_window on both metrics"
    elif tqa_acc >= 0.1726:
        return "CONTINUE", "TruthfulQA competitive, HaluEval may improve"
    elif tqa_acc >= 0.155 and halu_acc <= 0.54:
        return "CONTINUE", "Competitive trajectory on both metrics"
    elif tqa_acc >= 0.155:
        return "CONTINUE_CAUTIOUS", "TruthfulQA ok, watch HaluEval"
    elif tqa_acc >= 0.14 and halu_acc <= 0.55:
        return "CONTINUE_CAUTIOUS", "May plateau, but not failing"
    elif tqa_acc < 0.14:
        return "STOP", "Below baseline - pivot to hybrid"
    else:
        return "STOP", "Both metrics underperforming - pivot to hybrid"


def main():
    parser = argparse.ArgumentParser(description="Evaluate training checkpoint")
    parser.add_argument("checkpoint", nargs="?", help="Path to checkpoint directory")
    parser.add_argument("--checkpoint", "-c", dest="checkpoint_flag", help="Path to checkpoint directory")
    parser.add_argument("--mask_type", "-m", default="toroidal", help="Mask type for logging")
    parser.add_argument("--quick", action="store_true", default=True, help="Quick eval (200 samples)")
    parser.add_argument("--full", action="store_true", help="Full eval (all samples)")
    parser.add_argument("--output", "-o", help="Output JSON file")

    args = parser.parse_args()

    checkpoint_path = args.checkpoint or args.checkpoint_flag
    if not checkpoint_path:
        # Try to find latest checkpoint
        toroidal_dir = "./results_full/toroidal"
        if os.path.exists(toroidal_dir):
            checkpoints = [d for d in os.listdir(toroidal_dir) if d.startswith("checkpoint-")]
            if checkpoints:
                latest = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
                checkpoint_path = os.path.join(toroidal_dir, latest)
                print(f"Auto-detected checkpoint: {checkpoint_path}")
            else:
                print("No checkpoints found. Please specify path.")
                sys.exit(1)
        else:
            print("No checkpoint specified and ./results_full/toroidal not found")
            sys.exit(1)

    n_samples = 200 if args.quick and not args.full else None

    # Load model
    model, tokenizer = load_checkpoint(checkpoint_path)

    # Run evaluations
    print("\n" + "=" * 60)
    print(f"CHECKPOINT EVALUATION: {args.mask_type}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Samples: {n_samples or 'full'}")
    print("=" * 60)

    tqa_results = evaluate_truthfulqa_quick(model, tokenizer, n_samples or 817)
    halu_results = evaluate_halueval_quick(model, tokenizer, n_samples or 500)

    tqa_acc = tqa_results["accuracy"]
    halu_acc = halu_results["accuracy"] if halu_results["accuracy"] else 0.5

    # Decision
    decision, reason = make_decision(tqa_acc, halu_acc)

    # Output
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"TruthfulQA: {tqa_acc:.2%} ({tqa_results['correct']}/{tqa_results['total']})")
    print(f"HaluEval:   {halu_acc:.2%} ({halu_results['correct']}/{halu_results['total']})")

    print("\n" + "-" * 60)
    print("BENCHMARKS (local_window @ 3 epochs)")
    print("-" * 60)
    print(f"TruthfulQA: 17.26%")
    print(f"HaluEval:   53.00%")

    print("\n" + "=" * 60)
    print(f"DECISION: {decision}")
    print(f"Reason: {reason}")
    print("=" * 60)

    # Color-coded recommendation
    if decision.startswith("STRONG"):
        print("\n>>> Let training complete. Strong signal detected.")
    elif decision.startswith("CONTINUE"):
        print("\n>>> Let training continue. Monitor for improvement.")
    elif decision == "CONTINUE_CAUTIOUS":
        print("\n>>> Continue but prepare hybrid fallback.")
    else:
        print("\n>>> STOP training. Pivot to hybrid mask experiment.")

    # Save results
    results = {
        "checkpoint": checkpoint_path,
        "mask_type": args.mask_type,
        "timestamp": datetime.now().isoformat(),
        "n_samples": n_samples,
        "truthfulqa": tqa_results,
        "halueval": halu_results,
        "decision": decision,
        "reason": reason,
    }

    output_path = args.output or f"checkpoint_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return decision


if __name__ == "__main__":
    decision = main()
    # Exit code for scripting: 0 = continue, 1 = stop
    sys.exit(0 if "CONTINUE" in decision else 1)
