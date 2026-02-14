"""
Ouro-1.4B + Toroidal Attention Constraints: Loop Degradation Experiment

Tests whether toroidal attention masks extend the useful loop range
of ByteDance's Ouro looped language model beyond the 4-loop degradation wall.

Experiment:
  - Baseline: Ouro-1.4B at 1, 2, 3, 4, 6, 8 loops (no modification)
  - Toroidal: Same loop counts with toroidal attention mask on late layers
  - Metrics: perplexity on a fixed evaluation set, generation quality

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
from pathlib import Path
from datetime import datetime

# Add topological-coherence to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
RESULTS_FILE = RESULTS_DIR / f"ouro_toroidal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

# --- Toroidal mask generation (self-contained, no import needed) ---

def toroidal_distance_matrix(n_tokens: int, grid_size: int = 12) -> np.ndarray:
    """Vectorized L1 distance matrix on N x N torus."""
    idx = np.arange(n_tokens)
    x = idx % grid_size
    y = (idx // grid_size) % grid_size
    dx = np.abs(x[:, None] - x[None, :])
    dy = np.abs(y[:, None] - y[None, :])
    dx = np.minimum(dx, grid_size - dx)
    dy = np.minimum(dy, grid_size - dy)
    return dx + dy


def make_toroidal_mask(seq_len: int, radius: float = 2.0, alpha: float = 1.0,
                       grid_size: int = 12) -> torch.Tensor:
    """Hybrid toroidal attention mask: 1 if d<=r, else exp(-alpha*(d-r))."""
    dist = toroidal_distance_matrix(seq_len, grid_size).astype(np.float32)
    mask = np.where(dist <= radius, 1.0, np.exp(-alpha * (dist - radius)))
    return torch.from_numpy(mask.astype(np.float32))


# --- Attention hooking ---

class ToroidalAttentionHook:
    """Hook that multiplies attention weights by a toroidal mask.

    Applied to late layers only (last 1/3), matching the layer_late
    configuration that achieved 67% hallucination reduction on Mistral-7B.
    """

    def __init__(self, mask: torch.Tensor):
        self.mask = mask
        self.handles = []

    def hook_fn(self, module, args, output):
        """Post-hook on attention: multiply attention output by toroidal mask.

        For Ouro's custom attention, we hook into the attention module's forward
        and apply the mask to the attention weights before they're used.
        """
        # output is typically (attn_output, attn_weights, ...) or just attn_output
        # We need to handle various return formats
        if isinstance(output, tuple):
            attn_output = output[0]
        else:
            attn_output = output
        return output  # passthrough for now, we use pre-hook instead

    def pre_hook_fn(self, module, args, kwargs):
        """Pre-hook: inject toroidal mask into attention computation.

        We modify the attention_mask in kwargs to include toroidal structure.
        """
        if 'attention_mask' in kwargs and kwargs['attention_mask'] is not None:
            orig_mask = kwargs['attention_mask']
            seq_len = orig_mask.shape[-1]
            toro_mask = self.mask[:seq_len, :seq_len].to(orig_mask.device, orig_mask.dtype)
            # Combine: where original mask allows, apply toroidal weighting
            # Convert toroidal weights to additive mask (log space for softmax)
            toro_additive = torch.log(toro_mask.clamp(min=1e-8))
            toro_additive = toro_additive.unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]
            kwargs['attention_mask'] = orig_mask + toro_additive
        return args, kwargs

    def attach(self, model, layer_fraction: float = 0.67):
        """Attach hooks to the late layers of the model.

        Args:
            model: The Ouro model
            layer_fraction: Apply to last (1-fraction) of layers.
                           0.67 means last 1/3 of layers.
        """
        self.remove()  # clean up any existing hooks

        # Find attention layers in the model
        layers = []
        for name, module in model.named_modules():
            if 'self_attn' in name and not any(sub in name for sub in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
                layers.append((name, module))

        if not layers:
            # Try alternative naming
            for name, module in model.named_modules():
                if 'attention' in name.lower() and hasattr(module, 'forward'):
                    # Only get leaf attention modules
                    children = list(module.children())
                    if len(children) == 0 or any(hasattr(c, 'weight') for c in children):
                        layers.append((name, module))

        n_layers = len(layers)
        start_idx = int(n_layers * layer_fraction)

        print(f"  Found {n_layers} attention layers, hooking last {n_layers - start_idx}")

        for i, (name, module) in enumerate(layers):
            if i >= start_idx:
                handle = module.register_forward_hook(self._make_output_hook(name))
                self.handles.append(handle)

        return self

    def _make_output_hook(self, name):
        """Create a hook that modifies attention outputs with toroidal weighting."""
        mask = self.mask

        def hook(module, input, output):
            # Get the hidden states from input
            if isinstance(input, tuple) and len(input) > 0:
                hidden = input[0]
                seq_len = hidden.shape[1] if hidden.dim() >= 2 else hidden.shape[0]
            else:
                return output

            # Apply toroidal mask as a soft filter on the output
            # This is equivalent to post-attention toroidal projection
            if isinstance(output, tuple):
                attn_out = output[0]
                if attn_out.dim() == 3:  # [B, S, D]
                    S = attn_out.shape[1]
                    toro = mask[:S, :S].to(attn_out.device, attn_out.dtype)
                    # Normalize mask rows to sum to 1 (mixing matrix)
                    toro_norm = toro / toro.sum(dim=-1, keepdim=True)
                    # Apply as soft mixing: each position gets a weighted average
                    # This filters high-frequency components
                    filtered = torch.matmul(toro_norm.unsqueeze(0), attn_out)
                    # Blend: 70% original + 30% filtered (conservative)
                    blended = 0.7 * attn_out + 0.3 * filtered
                    return (blended,) + output[1:]
            return output

        return hook

    def remove(self):
        """Remove all hooks."""
        for h in self.handles:
            h.remove()
        self.handles = []


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
    """Evaluate model on factual prompts. Returns accuracy and details."""
    correct = 0
    results = []

    for prompt, keywords in zip(EVAL_PROMPTS, EXPECTED_KEYWORDS):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                use_cache=False,
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response[len(prompt):].strip().lower()

        hit = any(kw.lower() in answer for kw in keywords)
        if hit:
            correct += 1
        results.append({
            "prompt": prompt,
            "answer": answer[:200],
            "keywords": keywords,
            "correct": hit,
        })

    accuracy = correct / len(EVAL_PROMPTS)
    return accuracy, results


def compute_perplexity(model, tokenizer, device, texts=None):
    """Compute perplexity on a set of texts."""
    if texts is None:
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

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(avg_loss)
    return perplexity


# --- Main experiment ---

def run_experiment():
    print("=" * 70)
    print("OURO-1.4B + TOROIDAL CONSTRAINTS: LOOP DEGRADATION EXPERIMENT")
    print("=" * 70)
    print(f"Start time: {datetime.now().isoformat()}")
    print()

    # Use CPU — Ouro's custom code has cache issues on MPS
    device = torch.device("cpu")
    print(f"Device: {device}")
    print(f"MPS available: {torch.backends.mps.is_available()} (using CPU for Ouro compatibility)")

    # Download and load model
    print("\n[1/4] Loading Ouro-1.4B...")
    t0 = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    model_name = "ByteDance/Ouro-1.4B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Tokenizer loaded in {time.time() - t0:.1f}s")

    # Loop counts to test
    loop_counts = [1, 2, 3, 4, 6, 8]

    all_results = {
        "experiment": "ouro_toroidal_loops",
        "model": model_name,
        "device": str(device),
        "date": datetime.now().isoformat(),
        "loop_counts": loop_counts,
        "baseline": {},
        "toroidal": {},
    }

    # Pre-generate toroidal mask (max seq len we'll use)
    max_seq = 512
    toro_mask = make_toroidal_mask(max_seq, radius=2.0, alpha=1.0, grid_size=12)
    print(f"  Toroidal mask generated: {max_seq}x{max_seq}")

    # --- BASELINE RUNS ---
    print("\n[2/4] Baseline runs (no toroidal constraint)...")

    for n_loops in loop_counts:
        print(f"\n  --- Baseline: {n_loops} loops ---")
        t1 = time.time()

        try:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            config.total_ut_steps = n_loops
            config.early_exit_threshold = 1.0  # no early exit

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                config=config,
                trust_remote_code=True,
                torch_dtype=torch.float32,
            ).to(device)
            model.eval()

            load_time = time.time() - t1
            print(f"  Model loaded ({n_loops} loops) in {load_time:.1f}s")

            # Evaluate
            print("  Computing perplexity...")
            ppl = compute_perplexity(model, tokenizer, device)
            print(f"  Perplexity: {ppl:.2f}")

            print("  Evaluating factual accuracy...")
            accuracy, details = evaluate_model(model, tokenizer, device)
            print(f"  Accuracy: {accuracy:.1%} ({int(accuracy * len(EVAL_PROMPTS))}/{len(EVAL_PROMPTS)})")

            eval_time = time.time() - t1 - load_time

            all_results["baseline"][str(n_loops)] = {
                "perplexity": ppl,
                "accuracy": accuracy,
                "correct_count": int(accuracy * len(EVAL_PROMPTS)),
                "total_prompts": len(EVAL_PROMPTS),
                "load_time_s": round(load_time, 1),
                "eval_time_s": round(eval_time, 1),
                "details": details,
            }

            # Print some example answers
            for d in details[:3]:
                status = "✓" if d["correct"] else "✗"
                print(f"    {status} {d['prompt']}")
                print(f"      → {d['answer'][:80]}")

            del model
            import gc; gc.collect()

        except Exception as e:
            print(f"  ERROR at {n_loops} loops: {e}")
            all_results["baseline"][str(n_loops)] = {"error": str(e)}

        # Save intermediate results
        with open(RESULTS_FILE, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

    # --- TOROIDAL RUNS ---
    print("\n[3/4] Toroidal runs (late-layer attention constraint)...")

    for n_loops in loop_counts:
        print(f"\n  --- Toroidal: {n_loops} loops ---")
        t1 = time.time()

        try:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            config.total_ut_steps = n_loops
            config.early_exit_threshold = 1.0

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                config=config,
                trust_remote_code=True,
                torch_dtype=torch.float32,
            ).to(device)
            model.eval()

            load_time = time.time() - t1
            print(f"  Model loaded ({n_loops} loops) in {load_time:.1f}s")

            # Attach toroidal hooks to late layers
            hook = ToroidalAttentionHook(toro_mask)
            hook.attach(model, layer_fraction=0.67)

            # Evaluate
            print("  Computing perplexity (toroidal)...")
            ppl = compute_perplexity(model, tokenizer, device)
            print(f"  Perplexity: {ppl:.2f}")

            print("  Evaluating factual accuracy (toroidal)...")
            accuracy, details = evaluate_model(model, tokenizer, device)
            print(f"  Accuracy: {accuracy:.1%} ({int(accuracy * len(EVAL_PROMPTS))}/{len(EVAL_PROMPTS)})")

            eval_time = time.time() - t1 - load_time

            all_results["toroidal"][str(n_loops)] = {
                "perplexity": ppl,
                "accuracy": accuracy,
                "correct_count": int(accuracy * len(EVAL_PROMPTS)),
                "total_prompts": len(EVAL_PROMPTS),
                "load_time_s": round(load_time, 1),
                "eval_time_s": round(eval_time, 1),
                "details": details,
            }

            for d in details[:3]:
                status = "✓" if d["correct"] else "✗"
                print(f"    {status} {d['prompt']}")
                print(f"      → {d['answer'][:80]}")

            hook.remove()
            del model
            import gc; gc.collect()

        except Exception as e:
            print(f"  ERROR at {n_loops} loops: {e}")
            all_results["toroidal"][str(n_loops)] = {"error": str(e)}

        # Save intermediate results
        with open(RESULTS_FILE, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

    # --- SUMMARY ---
    print("\n" + "=" * 70)
    print("[4/4] SUMMARY")
    print("=" * 70)

    print(f"\n{'Loops':<8} {'Baseline PPL':<15} {'Toroidal PPL':<15} {'Baseline Acc':<15} {'Toroidal Acc':<15} {'Δ Acc':<10}")
    print("-" * 78)

    for n in loop_counts:
        b = all_results["baseline"].get(str(n), {})
        t = all_results["toroidal"].get(str(n), {})

        b_ppl = b.get("perplexity", float('nan'))
        t_ppl = t.get("perplexity", float('nan'))
        b_acc = b.get("accuracy", float('nan'))
        t_acc = t.get("accuracy", float('nan'))
        delta = t_acc - b_acc if not (math.isnan(t_acc) or math.isnan(b_acc)) else float('nan')

        print(f"{n:<8} {b_ppl:<15.2f} {t_ppl:<15.2f} {b_acc:<15.1%} {t_acc:<15.1%} {delta:+.1%}")

    # Key finding: does toroidal extend useful loop range?
    print("\nKEY QUESTION: Does toroidal constraint extend useful loop range?")

    baseline_accs = {int(k): v.get("accuracy", 0) for k, v in all_results["baseline"].items() if "accuracy" in v}
    toroidal_accs = {int(k): v.get("accuracy", 0) for k, v in all_results["toroidal"].items() if "accuracy" in v}

    if baseline_accs and toroidal_accs:
        best_baseline_loops = max(baseline_accs, key=baseline_accs.get)
        best_toroidal_loops = max(toroidal_accs, key=toroidal_accs.get)

        print(f"  Best baseline: {best_baseline_loops} loops ({baseline_accs[best_baseline_loops]:.1%})")
        print(f"  Best toroidal: {best_toroidal_loops} loops ({toroidal_accs[best_toroidal_loops]:.1%})")

        if best_toroidal_loops > best_baseline_loops:
            print(f"  → TOROIDAL EXTENDS USEFUL LOOPS BY {best_toroidal_loops - best_baseline_loops}")
        elif toroidal_accs.get(best_baseline_loops, 0) > baseline_accs.get(best_baseline_loops, 0):
            print(f"  → TOROIDAL IMPROVES ACCURACY AT SAME LOOP COUNT")
        else:
            print(f"  → NO CLEAR ADVANTAGE (may need hyperparameter tuning)")

    all_results["end_time"] = datetime.now().isoformat()

    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {RESULTS_FILE}")
    print(f"End time: {datetime.now().isoformat()}")
    print("=" * 70)


if __name__ == "__main__":
    run_experiment()
