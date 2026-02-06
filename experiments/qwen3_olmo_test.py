#!/usr/bin/env python3
"""
Toroidal Coherence Validation on Modern LLMs
Tests: Qwen3, OLMo 7B/32B
Author: Sylvain Cormier / Paraxiom Research
For RunPod A100/H100

Based on: experiments/llm_toroidal_attention.py
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict
import json
import time
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class MaskConfig:
    """Toroidal mask parameters - same as original Tonnetz"""
    grid_size: int = 12          # 12x12 torus (Tonnetz pitch classes)
    radius: float = 2.0          # Full weight within radius
    alpha: float = 1.0           # Decay rate outside radius

@dataclass
class ExperimentConfig:
    """Experiment settings for RunPod"""
    models: List[str] = None
    num_samples: int = 100       # Increase from 50 for statistical power
    max_length: int = 256
    batch_size: int = 4          # Adjust based on GPU memory
    use_4bit: bool = True        # Quantization for larger models
    save_results: bool = True
    output_dir: str = "./results"

    def __post_init__(self):
        if self.models is None:
            self.models = [
                # Qwen3 models
                "Qwen/Qwen2.5-7B",           # Latest Qwen
                "Qwen/Qwen2.5-14B",          # If memory allows
                "Qwen/Qwen2.5-32B",          # H100 only
                # OLMo models (fully open - training data transparent)
                "allenai/OLMo-7B",
                "allenai/OLMo-1.7-7B-hf",    # Latest OLMo
                # "allenai/OLMo-32B",        # When available
            ]

# ============================================================================
# TOROIDAL MASK (Tonnetz topology)
# ============================================================================

def toroidal_distance(i: int, j: int, grid_size: int = 12) -> int:
    """
    Compute Manhattan distance on a 2D torus.
    Maps token positions to grid coordinates with wraparound.
    """
    xi, yi = i % grid_size, (i // grid_size) % grid_size
    xj, yj = j % grid_size, (j // grid_size) % grid_size

    dx = min(abs(xi - xj), grid_size - abs(xi - xj))
    dy = min(abs(yi - yj), grid_size - abs(yi - yj))

    return dx + dy

def create_toroidal_mask(seq_len: int, config: MaskConfig) -> torch.Tensor:
    """
    Create attention mask based on toroidal (Tonnetz) distance.

    Returns log-space bias to add before softmax:
    - Full weight (0.0) within radius
    - Exponential decay outside radius
    """
    mask = torch.zeros(seq_len, seq_len)

    for i in range(seq_len):
        for j in range(seq_len):
            dist = toroidal_distance(i, j, config.grid_size)
            if dist <= config.radius:
                mask[i, j] = 0.0  # log(1) = 0, full weight
            else:
                mask[i, j] = -config.alpha * dist  # log decay

    return mask

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model_and_tokenizer(model_name: str, use_4bit: bool = True):
    """Load model with optional quantization for memory efficiency."""

    print(f"Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()
    return model, tokenizer

# ============================================================================
# ATTENTION HOOK INJECTION
# ============================================================================

class ToroidalAttentionHook:
    """
    Hook to inject toroidal bias into attention scores.
    Architecture-agnostic: works by modifying attention weights post-computation.
    """

    def __init__(self, mask_config: MaskConfig):
        self.mask_config = mask_config
        self.masks = {}  # Cache masks by sequence length

    def get_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if seq_len not in self.masks:
            self.masks[seq_len] = create_toroidal_mask(seq_len, self.mask_config).to(device)
        return self.masks[seq_len]

    def __call__(self, module, input, output):
        """Hook function to modify attention output."""
        # This is a simplified version - full implementation needs
        # architecture-specific attention score interception
        return output

def apply_toroidal_constraint(model, tokenizer, input_ids, mask_config: MaskConfig):
    """
    Apply toroidal constraint during generation.

    Method: Modify attention scores in the forward pass by adding
    toroidal bias before softmax.
    """
    seq_len = input_ids.shape[1]
    device = input_ids.device

    # Create toroidal mask
    topo_mask = create_toroidal_mask(seq_len, mask_config).to(device)

    # For now, return the mask - actual injection is model-specific
    # TODO: Implement per-architecture hooks for Qwen3/OLMo
    return topo_mask

# ============================================================================
# EVALUATION METRICS
# ============================================================================

def evaluate_hallucination(model, tokenizer, prompts: List[str],
                          use_toroidal: bool = False,
                          mask_config: Optional[MaskConfig] = None) -> Dict:
    """
    Evaluate hallucination rate using TruthfulQA-style prompts.

    Returns:
        - hallucination_rate: % of responses with factual errors
        - coherence_score: semantic consistency measure
        - drift_metrics: embedding stability
    """
    results = {
        "total": len(prompts),
        "hallucinations": 0,
        "coherence_scores": [],
        "generation_times": [],
    }

    for prompt in prompts:
        start_time = time.time()

        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            if use_toroidal and mask_config:
                # Apply toroidal constraint
                topo_mask = apply_toroidal_constraint(
                    model, tokenizer, inputs["input_ids"], mask_config
                )
                # TODO: Inject mask into forward pass

            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
            )

        generation_time = time.time() - start_time
        results["generation_times"].append(generation_time)

        # Decode output
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # TODO: Implement proper hallucination detection
        # For now, placeholder metrics

    results["avg_generation_time"] = np.mean(results["generation_times"])
    results["hallucination_rate"] = results["hallucinations"] / results["total"]

    return results

def run_halueval(model, tokenizer, num_samples: int = 100,
                use_toroidal: bool = False,
                mask_config: Optional[MaskConfig] = None) -> Dict:
    """
    Run HaluEval benchmark subset.
    """
    # Load HaluEval dataset
    try:
        dataset = load_dataset("pminervini/HaluEval", "qa_samples", split="data")
        samples = list(dataset.select(range(min(num_samples, len(dataset)))))
    except Exception as e:
        print(f"Could not load HaluEval: {e}")
        # Fallback to synthetic prompts
        samples = [{"question": f"What is {i}+{i}?"} for i in range(num_samples)]

    prompts = [s.get("question", s.get("prompt", str(s))) for s in samples]

    return evaluate_hallucination(
        model, tokenizer, prompts,
        use_toroidal=use_toroidal,
        mask_config=mask_config
    )

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment(config: ExperimentConfig):
    """
    Run full experiment across all models.

    For each model:
    1. Baseline (no constraint)
    2. Toroidal (Tonnetz constraint)

    Compare hallucination rates.
    """
    mask_config = MaskConfig()
    all_results = {}

    print("=" * 60)
    print("TOROIDAL COHERENCE VALIDATION")
    print(f"Models: {config.models}")
    print(f"Samples per condition: {config.num_samples}")
    print("=" * 60)

    for model_name in config.models:
        print(f"\n{'='*60}")
        print(f"Testing: {model_name}")
        print("=" * 60)

        try:
            model, tokenizer = load_model_and_tokenizer(model_name, config.use_4bit)

            # Baseline
            print("\n[1/2] Running BASELINE...")
            baseline_results = run_halueval(
                model, tokenizer,
                num_samples=config.num_samples,
                use_toroidal=False
            )

            # Toroidal
            print("\n[2/2] Running TOROIDAL...")
            toroidal_results = run_halueval(
                model, tokenizer,
                num_samples=config.num_samples,
                use_toroidal=True,
                mask_config=mask_config
            )

            # Calculate improvement
            baseline_rate = baseline_results["hallucination_rate"]
            toroidal_rate = toroidal_results["hallucination_rate"]

            if baseline_rate > 0:
                improvement = ((baseline_rate - toroidal_rate) / baseline_rate) * 100
            else:
                improvement = 0

            all_results[model_name] = {
                "baseline": baseline_results,
                "toroidal": toroidal_results,
                "improvement_percent": improvement,
            }

            print(f"\n--- Results for {model_name} ---")
            print(f"Baseline hallucination rate: {baseline_rate:.2%}")
            print(f"Toroidal hallucination rate: {toroidal_rate:.2%}")
            print(f"Improvement: {improvement:+.1f}%")

            # Free memory
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"ERROR with {model_name}: {e}")
            all_results[model_name] = {"error": str(e)}

    # Save results
    if config.save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{config.output_dir}/results_{timestamp}.json"

        import os
        os.makedirs(config.output_dir, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        print(f"\nResults saved to: {output_file}")

    return all_results

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Toroidal Coherence Validation")
    parser.add_argument("--models", nargs="+", default=None,
                       help="Models to test (default: Qwen + OLMo)")
    parser.add_argument("--samples", type=int, default=100,
                       help="Number of samples per condition")
    parser.add_argument("--no-quantize", action="store_true",
                       help="Disable 4-bit quantization")
    parser.add_argument("--output", type=str, default="./results",
                       help="Output directory")

    args = parser.parse_args()

    config = ExperimentConfig(
        models=args.models,
        num_samples=args.samples,
        use_4bit=not args.no_quantize,
        output_dir=args.output,
    )

    results = run_experiment(config)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for model, result in results.items():
        if "error" in result:
            print(f"{model}: ERROR - {result['error']}")
        else:
            imp = result.get("improvement_percent", 0)
            sign = "+" if imp > 0 else ""
            print(f"{model}: {sign}{imp:.1f}% hallucination change")
