"""
Llama-3-8B + Toroidal Logit Bias (TLB)

Tests TLB on Meta's Llama-3-8B — a standard (non-UT) transformer at 8B scale.
Previous results: TLB showed +10% on Ouro-1.4B (Universal Transformer, 4 loops).
This experiment tests whether TLB generalizes to a larger, non-recurrent architecture.

Uses the same 20-prompt factual QA eval + the fine sweep alpha/radius grid
from the Ouro experiments for direct comparison.

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
RESULTS_FILE = RESULTS_DIR / f"llama3_tlb_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"


# --- Toroidal Logit Bias (same implementation as Ouro experiments) ---

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
                do_sample=False, temperature=1.0,
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
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
            n_tokens = inputs["input_ids"].shape[1]
            total_loss += loss * n_tokens
            total_tokens += n_tokens
    return math.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')


# --- Main ---

def run_experiment():
    print("=" * 70)
    print("LLAMA-3-8B + TOROIDAL LOGIT BIAS (TLB)")
    print("=" * 70)
    print(f"Device: {DEVICE} ({DTYPE})")
    print(f"Start: {datetime.now().isoformat()}")
    print(flush=True)

    device = DEVICE

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "meta-llama/Meta-Llama-3-8B"
    print(f"\nLoading {model_name}...", flush=True)
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=DTYPE,
        device_map="auto",
    )
    model.eval()
    print(f"Model loaded in {time.time()-t0:.1f}s", flush=True)

    # Same alpha/radius grid as Ouro fine sweep for direct comparison
    alphas = [0.30, 0.40, 0.50, 0.60, 0.70]
    radii = [2.5, 3.0, 3.5, 4.0]

    tlb_configs = []
    for a in alphas:
        for r in radii:
            tlb_configs.append({
                "alpha": a, "radius": r, "max_tokens": 1440,
                "name": f"a{a:.2f}_r{r:.1f}",
            })

    all_results = {
        "experiment": "llama3_tlb",
        "model": model_name,
        "device": str(device),
        "dtype": str(DTYPE),
        "date": datetime.now().isoformat(),
        "tlb_configs": [c["name"] for c in tlb_configs],
        "method": "Toroidal Logit Bias on output logits",
        "baseline": {},
        "tlb": {},
    }

    # --- BASELINE ---
    print("\n[1/2] Baseline (no TLB)...", flush=True)
    t1 = time.time()

    ppl = compute_perplexity(model, tokenizer, device)
    print(f"  PPL: {ppl:.2f}", flush=True)
    acc, details = evaluate_model(model, tokenizer, device)
    print(f"  Acc: {acc:.0%} ({int(acc*20)}/20)  [{time.time()-t1:.0f}s]", flush=True)

    # Show which prompts baseline gets right
    for d in details:
        marker = "+" if d["correct"] else "-"
        print(f"    [{marker}] {d['prompt']}: {d['answer'][:60]}", flush=True)

    all_results["baseline"] = {
        "perplexity": ppl, "accuracy": acc,
        "correct_count": int(acc * 20), "total_prompts": 20,
        "details": details,
    }

    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # --- TLB SWEEP ---
    total_runs = len(tlb_configs)
    print(f"\n[2/2] TLB sweep ({total_runs} configs)...", flush=True)

    bl_acc = acc
    bl_details = details

    for idx, tlb_cfg in enumerate(tlb_configs, 1):
        tlb_name = tlb_cfg["name"]
        print(f"\n  [{idx}/{total_runs}] alpha={tlb_cfg['alpha']:.2f}  radius={tlb_cfg['radius']:.1f}", flush=True)
        t1 = time.time()

        try:
            processor = ToroidalLogitProcessor(
                alpha=tlb_cfg["alpha"],
                radius=tlb_cfg["radius"],
                max_tokens=tlb_cfg["max_tokens"],
            )

            ppl_tlb = compute_perplexity(model, tokenizer, device)  # PPL unchanged (no TLB on PPL)
            acc_tlb, details_tlb = evaluate_model(model, tokenizer, device,
                                                  logits_processor=processor)

            elapsed = time.time() - t1
            delta = acc_tlb - bl_acc
            print(f"  PPL: {ppl_tlb:.2f}  Acc: {acc_tlb:.0%} ({int(acc_tlb*20)}/20)  [{elapsed:.0f}s]", flush=True)

            if abs(delta) > 0.001:
                delta_sym = "\u0394"
                print(f"  {delta_sym} vs baseline: {delta:+.0%}", flush=True)

            # Show per-prompt diffs
            fixes = []
            breaks = []
            for bd, td in zip(bl_details, details_tlb):
                if not bd["correct"] and td["correct"]:
                    fixes.append(td["prompt"])
                elif bd["correct"] and not td["correct"]:
                    breaks.append(td["prompt"])
            if fixes:
                print(f"  FIXED: {fixes}", flush=True)
            if breaks:
                print(f"  BROKE: {breaks}", flush=True)

            all_results["tlb"][tlb_name] = {
                "perplexity": ppl_tlb, "accuracy": acc_tlb,
                "correct_count": int(acc_tlb * 20), "total_prompts": 20,
                "tlb_alpha": tlb_cfg["alpha"],
                "tlb_radius": tlb_cfg["radius"],
                "tlb_max_tokens": tlb_cfg["max_tokens"],
                "delta_vs_baseline": delta,
                "details": details_tlb,
            }

        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            traceback.print_exc()
            all_results["tlb"][tlb_name] = {"error": str(e)}

        with open(RESULTS_FILE, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

    # --- SUMMARY ---
    print("\n" + "=" * 70)
    print("SUMMARY: LLAMA-3-8B + TLB")
    print("=" * 70, flush=True)

    bl_ppl = all_results["baseline"].get("perplexity", float('nan'))
    print(f"\nBaseline: PPL={bl_ppl:.2f}  Acc={bl_acc:.0%} ({int(bl_acc*20)}/20)")

    # Heatmap-style table
    delta_hdr = "\u0394 Acc"
    print(f"\n  {'alpha':<8} {'radius':<8} {'PPL':<8} {'Acc':<8} {delta_hdr:<10} {'Note'}")
    print(f"  {'-'*60}")

    for tlb_cfg in tlb_configs:
        tn = tlb_cfg["name"]
        r = all_results["tlb"].get(tn, {})
        if "error" in r:
            print(f"  {tlb_cfg['alpha']:<8.2f} {tlb_cfg['radius']:<8.1f} ERROR")
            continue
        rp = r.get("perplexity", float('nan'))
        ra = r.get("accuracy", float('nan'))
        da = ra - bl_acc if not math.isnan(ra) else float('nan')
        if da > 0.001:
            note = "*** STRONG" if da >= 0.10 else "* improved"
        elif da < -0.001:
            note = "!!! destructive"
        else:
            note = "= same"
        print(f"  {tlb_cfg['alpha']:<8.2f} {tlb_cfg['radius']:<8.1f} {rp:<8.2f} {ra:<8.0%} {da:+.0%}{'':>4} {note}")

    # Best config
    best_name = None
    best_acc = bl_acc
    for tn, r in all_results["tlb"].items():
        if "error" in r:
            continue
        ra = r.get("accuracy", 0)
        if ra > best_acc:
            best_acc = ra
            best_name = tn

    print(f"\n  BEST: {best_name or 'baseline (no TLB improvement)'}  Acc={best_acc:.0%}  " +
          f"{chr(0x0394)}={best_acc - bl_acc:+.0%}")

    all_results["end_time"] = datetime.now().isoformat()
    all_results["best_config"] = best_name
    all_results["best_accuracy"] = best_acc
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved: {RESULTS_FILE}")
    print(f"End: {datetime.now().isoformat()}")
    print("=" * 70, flush=True)


if __name__ == "__main__":
    run_experiment()
