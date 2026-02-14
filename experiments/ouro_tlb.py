"""
Ouro-1.4B + Toroidal Logit Bias (TLB)

v3 showed that injecting toroidal bias into attention scores has zero effect
(too light) or is destructive (too strong). TLB works differently — it biases
the OUTPUT logits based on toroidal distance from recent context tokens.

This is the mechanism that showed +2.8pp on TruthfulQA in the published paper.
The hypothesis: Ouro's UT iterative loop + TLB at output = synergistic improvement.

Tests:
  1. Quick 20-prompt factual eval (matches v3 baselines)
  2. Perplexity on reference texts
  3. Per-prompt diff vs baseline to identify which answers TLB fixes/breaks

Author: Sylvain Cormier / Paraxiom Research
Date: 2026-02-14
"""

import sys
import os
import json
import time
import math
import traceback
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

# --- Device selection ---
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DTYPE = torch.float16
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    DEVICE = torch.device("cpu")
    DTYPE = torch.float32
    print("No GPU — running on CPU (will be slow)")

RESULTS_DIR = Path(__file__).parent.parent / "results"
if not RESULTS_DIR.exists():
    RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)
RESULTS_FILE = RESULTS_DIR / f"ouro_tlb_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"


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
    ["a\u00b2", "a^2", "square", "hypotenuse", "right triangle"],
]


def evaluate_model(model, tokenizer, device, logits_processor=None, max_new_tokens=30):
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
    print("OURO-1.4B + TOROIDAL LOGIT BIAS (TLB)")
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

    loop_counts = [4, 8]

    # TLB hyperparameter configs to sweep
    tlb_configs = [
        {"name": "tlb_a0.2_r2", "alpha": 0.2, "radius": 2.0, "max_tokens": 1440},
        {"name": "tlb_a0.3_r2", "alpha": 0.3, "radius": 2.0, "max_tokens": 1440},
        {"name": "tlb_a0.5_r2", "alpha": 0.5, "radius": 2.0, "max_tokens": 1440},
        {"name": "tlb_a0.3_r3", "alpha": 0.3, "radius": 3.0, "max_tokens": 1440},
        {"name": "tlb_a0.5_r3", "alpha": 0.5, "radius": 3.0, "max_tokens": 1440},
        {"name": "tlb_a1.0_r2", "alpha": 1.0, "radius": 2.0, "max_tokens": 1440},
        {"name": "tlb_a1.0_r3", "alpha": 1.0, "radius": 3.0, "max_tokens": 1440},
        # Higher coverage of vocab
        {"name": "tlb_a0.3_r2_v3k", "alpha": 0.3, "radius": 2.0, "max_tokens": 3000},
        {"name": "tlb_a0.5_r3_v3k", "alpha": 0.5, "radius": 3.0, "max_tokens": 3000},
    ]

    all_results = {
        "experiment": "ouro_tlb",
        "model": model_name,
        "device": str(device),
        "dtype": str(DTYPE),
        "date": datetime.now().isoformat(),
        "loop_counts": loop_counts,
        "tlb_configs": [c["name"] for c in tlb_configs],
        "method": "Toroidal Logit Bias on output logits (not attention)",
        "baseline": {},
        "tlb": {},
    }

    # --- BASELINES ---
    print("\n[1/2] Baselines...", flush=True)

    for n_loops in loop_counts:
        print(f"\n  --- Baseline: {n_loops} loops ---", flush=True)
        t1 = time.time()
        try:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            config.total_ut_steps = n_loops
            config.early_exit_threshold = 1.0
            if not hasattr(config, 'pad_token_id') or config.pad_token_id is None:
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
            print(f"  Acc: {acc:.0%} ({int(acc*20)}/20)", flush=True)

            all_results["baseline"][str(n_loops)] = {
                "perplexity": ppl, "accuracy": acc,
                "correct_count": int(acc * 20), "total_prompts": 20,
                "details": details,
            }
            del model
            import gc; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            traceback.print_exc()
            all_results["baseline"][str(n_loops)] = {"error": str(e)}

        with open(RESULTS_FILE, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

    # --- TLB SWEEP ---
    total_runs = len(tlb_configs) * len(loop_counts)
    print(f"\n[2/2] TLB sweep ({total_runs} configs)...", flush=True)

    run_idx = 0
    for tlb_cfg in tlb_configs:
        tlb_name = tlb_cfg["name"]

        for n_loops in loop_counts:
            run_idx += 1
            print(f"\n  [{run_idx}/{total_runs}] {tlb_name}, {n_loops} loops", flush=True)
            t1 = time.time()

            try:
                config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
                config.total_ut_steps = n_loops
                config.early_exit_threshold = 1.0
                if not hasattr(config, 'pad_token_id') or config.pad_token_id is None:
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

                elapsed = time.time() - t1
                print(f"  PPL: {ppl:.2f}  Acc: {acc:.0%} ({int(acc*20)}/20)  [{elapsed:.0f}s]", flush=True)

                # Compare vs matching baseline
                bl = all_results["baseline"].get(str(n_loops), {})
                bl_acc = bl.get("accuracy", 0)
                delta = acc - bl_acc
                if abs(delta) > 0.001:
                    print(f"  \u0394 vs baseline@{n_loops}: {delta:+.0%}", flush=True)

                # Show per-prompt changes
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

                if tlb_name not in all_results["tlb"]:
                    all_results["tlb"][tlb_name] = {}

                all_results["tlb"][tlb_name][str(n_loops)] = {
                    "perplexity": ppl, "accuracy": acc,
                    "correct_count": int(acc * 20), "total_prompts": 20,
                    "tlb_alpha": tlb_cfg["alpha"],
                    "tlb_radius": tlb_cfg["radius"],
                    "tlb_max_tokens": tlb_cfg["max_tokens"],
                    "details": details,
                }

                del model
                import gc; gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"  ERROR: {e}", flush=True)
                traceback.print_exc()
                if tlb_name not in all_results["tlb"]:
                    all_results["tlb"][tlb_name] = {}
                all_results["tlb"][tlb_name][str(n_loops)] = {"error": str(e)}

            with open(RESULTS_FILE, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)

    # --- SUMMARY ---
    print("\n" + "=" * 70)
    print("SUMMARY: OURO + TLB")
    print("=" * 70, flush=True)

    for n_loops in loop_counts:
        bl = all_results["baseline"].get(str(n_loops), {})
        bl_ppl = bl.get("perplexity", float('nan'))
        bl_acc = bl.get("accuracy", float('nan'))
        print(f"\nBaseline@{n_loops}: PPL={bl_ppl:.2f}  Acc={bl_acc:.0%}")

        print(f"  {'TLB Config':<25} {'PPL':<8} {'Acc':<8} {'\u0394 Acc':<8}")
        print(f"  {'-'*49}")

        for tlb_cfg in tlb_configs:
            tn = tlb_cfg["name"]
            r = all_results["tlb"].get(tn, {}).get(str(n_loops), {})
            if "error" in r:
                print(f"  {tn:<25} ERROR")
                continue
            rp = r.get("perplexity", float('nan'))
            ra = r.get("accuracy", float('nan'))
            da = ra - bl_acc if not math.isnan(ra) else float('nan')
            marker = " ***" if da > 0.001 else " ---" if da < -0.001 else ""
            print(f"  {tn:<25} {rp:<8.2f} {ra:<8.0%} {da:+.0%}{marker}")

    all_results["end_time"] = datetime.now().isoformat()
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults: {RESULTS_FILE}")
    print(f"End: {datetime.now().isoformat()}")
    print("=" * 70, flush=True)


if __name__ == "__main__":
    run_experiment()
