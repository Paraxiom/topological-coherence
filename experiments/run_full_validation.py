#!/usr/bin/env python3
"""
FULL TOROIDAL COHERENCE VALIDATION
===================================
100 samples on Qwen 2.5-7B-Instruct and OLMo 1.7-7B-hf
For paper-quality confidence intervals.

Usage:
    python run_full_validation.py                    # Run both models
    python run_full_validation.py --model qwen       # Run only Qwen
    python run_full_validation.py --model olmo       # Run only OLMo
    python run_full_validation.py --samples 50       # Custom sample count
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from datetime import datetime
import os
import gc
import argparse

# ============================================================================
# TOROIDAL FUNCTIONS - VECTORIZED (FIXED VERSION)
# ============================================================================

def get_toroidal_logit_bias_fast(vocab_size, recent_tokens, grid_size=12, radius=2.0, alpha=1.0, device='cuda'):
    """
    FIXED: Vectorized toroidal bias applied to ALL vocab tokens.

    Each vocab token maps to a torus position via modulo.
    Tokens "near" recent tokens on torus get probability boost.
    """
    if len(recent_tokens) < 2:
        return torch.zeros(vocab_size, device=device, dtype=torch.float16)

    grid_cells = grid_size * grid_size  # 144 for 12x12

    # All vocab positions on torus
    vocab_positions = torch.arange(vocab_size, device=device) % grid_cells

    # Compute bias
    bias = torch.zeros(vocab_size, device=device, dtype=torch.float16)

    for i, token in enumerate(recent_tokens[-5:]):
        token_pos = token % grid_cells

        # Compute torus distance for all vocab tokens at once
        vx = vocab_positions % grid_size
        vy = vocab_positions // grid_size
        tx = token_pos % grid_size
        ty = token_pos // grid_size

        # Manhattan distance with wraparound
        dx = torch.minimum(torch.abs(vx - tx), grid_size - torch.abs(vx - tx))
        dy = torch.minimum(torch.abs(vy - ty), grid_size - torch.abs(vy - ty))
        dist = dx + dy

        weight = 1.0 / (i + 1)

        # Apply boost based on distance
        near_mask = dist <= radius
        mid_mask = (dist > radius) & (dist <= radius * 2)

        bias[near_mask] += alpha * (radius - dist[near_mask].float() + 1) * weight
        bias[mid_mask] += alpha * 0.3 * weight

    return bias

# ============================================================================
# GENERATION
# ============================================================================

def generate_baseline(model, tokenizer, prompt, max_tokens=30):
    """Standard greedy generation"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                            pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0], skip_special_tokens=True)

def generate_toroidal(model, tokenizer, prompt, max_tokens=30,
                      grid_size=12, radius=2.0, alpha=1.0):
    """Generation with toroidal logit bias"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generated = inputs['input_ids'][0].tolist()
    vocab_size = model.config.vocab_size

    for step in range(max_tokens):
        with torch.no_grad():
            outputs = model(torch.tensor([generated], device=model.device))
            logits = outputs.logits[0, -1, :]

            # Apply toroidal bias
            topo_bias = get_toroidal_logit_bias_fast(
                vocab_size, generated, grid_size, radius, alpha, model.device
            )
            logits = logits + topo_bias

            next_token = logits.argmax().item()

        generated.append(next_token)
        if next_token == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated, skip_special_tokens=True)

# ============================================================================
# TEST PROMPTS - Expanded for 100 samples
# ============================================================================

TEST_PROMPTS = [
    # Geography (20)
    ("The capital of France is", ["Paris"]),
    ("The capital of Japan is", ["Tokyo"]),
    ("The capital of Australia is", ["Canberra"]),
    ("The capital of Brazil is", ["Brasilia"]),
    ("The capital of Canada is", ["Ottawa"]),
    ("Mount Everest is in", ["Nepal", "Himalaya"]),
    ("The Amazon River is in", ["South America", "Brazil"]),
    ("The Great Wall is in", ["China"]),
    ("The Nile River flows through", ["Egypt", "Africa"]),
    ("The Sahara Desert is in", ["Africa"]),
    ("The largest ocean is the", ["Pacific"]),
    ("The longest river in the world is the", ["Nile", "Amazon"]),
    ("Australia is a", ["continent", "country"]),
    ("The Alps are in", ["Europe"]),
    ("The Dead Sea is between", ["Israel", "Jordan"]),
    ("Venice is famous for its", ["canals", "water"]),
    ("The Eiffel Tower is in", ["Paris", "France"]),
    ("The Great Barrier Reef is near", ["Australia"]),
    ("Tokyo is the capital of", ["Japan"]),
    ("The Statue of Liberty is in", ["New York", "USA"]),

    # Science (25)
    ("Water freezes at", ["0", "32", "zero"]),
    ("The largest planet is", ["Jupiter"]),
    ("Einstein developed the theory of", ["relativity"]),
    ("The chemical symbol for gold is", ["Au"]),
    ("DNA stands for", ["deoxyribonucleic"]),
    ("The atomic number of hydrogen is", ["1", "one"]),
    ("Photosynthesis converts", ["energy", "glucose", "sugar", "sunlight"]),
    ("Oxygen is about what percent of air", ["21", "20"]),
    ("Pi equals approximately", ["3.14"]),
    ("The speed of light is", ["300", "299", "186"]),
    ("The human heart has", ["four", "4"]),
    ("Newton discovered", ["gravity", "motion"]),
    ("The chemical symbol for water is", ["H2O"]),
    ("The boiling point of water is", ["100", "212"]),
    ("Electrons have a", ["negative"]),
    ("The sun is a", ["star"]),
    ("Diamonds are made of", ["carbon"]),
    ("The human body has how many bones", ["206"]),
    ("Sound travels faster in", ["water", "solid"]),
    ("The earth revolves around the", ["sun"]),
    ("Gravity was discovered by", ["Newton"]),
    ("The smallest unit of life is a", ["cell"]),
    ("Mitochondria are the", ["powerhouse"]),
    ("The pH of pure water is", ["7", "seven", "neutral"]),
    ("The chemical symbol for iron is", ["Fe"]),

    # History (20)
    ("World War II ended in", ["1945"]),
    ("The Declaration of Independence was signed in", ["1776"]),
    ("The first human on the moon was", ["Armstrong", "Neil"]),
    ("The Berlin Wall fell in", ["1989"]),
    ("World War I started in", ["1914"]),
    ("The French Revolution began in", ["1789"]),
    ("Columbus sailed to America in", ["1492"]),
    ("The Roman Empire fell in", ["476", "5th"]),
    ("The Renaissance began in", ["Italy", "14th", "15th"]),
    ("The printing press was invented by", ["Gutenberg"]),
    ("The American Civil War ended in", ["1865"]),
    ("The Soviet Union collapsed in", ["1991"]),
    ("The Titanic sank in", ["1912"]),
    ("Martin Luther King Jr gave his famous speech in", ["1963"]),
    ("The Great Depression started in", ["1929"]),
    ("The first airplane flight was by the", ["Wright"]),
    ("Mahatma Gandhi led", ["India", "independence"]),
    ("The Cold War was between", ["USA", "Soviet", "America", "Russia"]),
    ("Ancient Egypt was known for", ["pyramids", "pharaohs"]),
    ("The Industrial Revolution began in", ["Britain", "England", "18th"]),

    # Arts & Culture (15)
    ("The Mona Lisa was painted by", ["Leonardo", "Vinci"]),
    ("Shakespeare wrote", ["Hamlet", "Romeo", "Macbeth"]),
    ("Beethoven was a famous", ["composer", "musician"]),
    ("The currency of Japan is", ["yen"]),
    ("Vincent van Gogh painted", ["Starry Night", "sunflowers"]),
    ("Romeo and Juliet was written by", ["Shakespeare"]),
    ("The Sistine Chapel ceiling was painted by", ["Michelangelo"]),
    ("Mozart was born in", ["Austria", "Salzburg"]),
    ("Harry Potter was written by", ["Rowling"]),
    ("The Odyssey was written by", ["Homer"]),
    ("Don Quixote was written by", ["Cervantes"]),
    ("The currency of the UK is the", ["pound", "sterling"]),
    ("The currency of the EU is the", ["euro"]),
    ("Picasso was a famous", ["painter", "artist"]),
    ("The Louvre is in", ["Paris", "France"]),

    # Math & Computing (10)
    ("Binary uses only", ["0", "1", "two"]),
    ("A triangle has how many sides", ["three", "3"]),
    ("The square root of 144 is", ["12", "twelve"]),
    ("A byte contains how many bits", ["8", "eight"]),
    ("The programming language Python was created by", ["Guido", "Rossum"]),
    ("HTML stands for", ["HyperText", "Markup"]),
    ("The first computer programmer was", ["Ada", "Lovelace"]),
    ("A hexagon has how many sides", ["6", "six"]),
    ("The value of 2 to the power of 10 is", ["1024"]),
    ("CPU stands for", ["Central", "Processing"]),

    # General Knowledge (10)
    ("The official language of Brazil is", ["Portuguese"]),
    ("A decade is how many years", ["10", "ten"]),
    ("A century is how many years", ["100", "hundred"]),
    ("The Olympic Games originated in", ["Greece"]),
    ("The largest mammal is the", ["whale", "blue"]),
    ("Coffee beans come from", ["plant", "tree", "cherry"]),
    ("Bees produce", ["honey"]),
    ("The fastest land animal is the", ["cheetah"]),
    ("Silk comes from", ["silkworm", "worm"]),
    ("The Great Pyramid was built in", ["Egypt", "Giza"]),
]

# ============================================================================
# VALIDATION
# ============================================================================

def run_validation(model_name, num_samples=100, alpha=1.0):
    """Run full validation on a single model"""
    print("=" * 70)
    print(f"TOROIDAL COHERENCE VALIDATION - {model_name}")
    print("=" * 70)
    print(f"Samples: {num_samples}")
    print(f"Alpha: {alpha}")
    print(f"Grid: 12x12 torus")
    print(f"Radius: 2.0")

    # Load model
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    print(f"Loaded. GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB")
    print(f"Vocab size: {model.config.vocab_size}")

    # Expand prompts to reach num_samples
    prompts = (TEST_PROMPTS * ((num_samples // len(TEST_PROMPTS)) + 1))[:num_samples]

    results = {
        "model": model_name,
        "alpha": alpha,
        "num_samples": num_samples,
        "timestamp": datetime.now().isoformat(),
        "baseline": {"correct": 0, "total": 0},
        "toroidal": {"correct": 0, "total": 0},
        "comparisons": [],
        "changes": {"toro_fixed": [], "toro_broke": []}
    }

    print(f"\nRunning {num_samples} tests...")
    print("-" * 70)

    for i, (prompt, expected) in enumerate(prompts):
        # Baseline
        resp_b = generate_baseline(model, tokenizer, prompt)
        ok_b = any(e.lower() in resp_b.lower() for e in expected)
        results["baseline"]["total"] += 1
        if ok_b:
            results["baseline"]["correct"] += 1

        # Toroidal
        resp_t = generate_toroidal(model, tokenizer, prompt, alpha=alpha)
        ok_t = any(e.lower() in resp_t.lower() for e in expected)
        results["toroidal"]["total"] += 1
        if ok_t:
            results["toroidal"]["correct"] += 1

        # Track changes
        if ok_t and not ok_b:
            results["changes"]["toro_fixed"].append({"prompt": prompt, "baseline": resp_b[:80], "toroidal": resp_t[:80]})
        elif ok_b and not ok_t:
            results["changes"]["toro_broke"].append({"prompt": prompt, "baseline": resp_b[:80], "toroidal": resp_t[:80]})

        results["comparisons"].append({
            "prompt": prompt,
            "expected": expected,
            "baseline": resp_b[:120],
            "toroidal": resp_t[:120],
            "b_ok": ok_b, "t_ok": ok_t
        })

        m_b = "Y" if ok_b else "X"
        m_t = "Y" if ok_t else "X"
        diff = "SAME" if ok_b == ok_t else ("TORO+" if ok_t else "BASE+")
        print(f"[{i+1:3d}/{num_samples}] B:{m_b} T:{m_t} {diff:5s} | {prompt[:40]}...")

    # Calculate statistics
    b_acc = results["baseline"]["correct"] / results["baseline"]["total"]
    t_acc = results["toroidal"]["correct"] / results["toroidal"]["total"]
    b_err, t_err = 1 - b_acc, 1 - t_acc

    if b_err > 0:
        reduction = ((b_err - t_err) / b_err) * 100
    else:
        reduction = 0 if t_err == 0 else -100

    results["summary"] = {
        "baseline_accuracy": round(b_acc, 4),
        "toroidal_accuracy": round(t_acc, 4),
        "baseline_errors": results["baseline"]["total"] - results["baseline"]["correct"],
        "toroidal_errors": results["toroidal"]["total"] - results["toroidal"]["correct"],
        "error_reduction_pct": round(reduction, 2),
        "fixed_by_toroidal": len(results["changes"]["toro_fixed"]),
        "broken_by_toroidal": len(results["changes"]["toro_broke"]),
        "net_change": len(results["changes"]["toro_fixed"]) - len(results["changes"]["toro_broke"])
    }

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Baseline accuracy:  {b_acc:.1%} ({results['baseline']['correct']}/{results['baseline']['total']})")
    print(f"Toroidal accuracy:  {t_acc:.1%} ({results['toroidal']['correct']}/{results['toroidal']['total']})")
    print(f"Error reduction:    {reduction:+.1f}%")
    print(f"Fixed by toroidal:  {len(results['changes']['toro_fixed'])}")
    print(f"Broken by toroidal: {len(results['changes']['toro_broke'])}")
    print(f"Net improvement:    {results['summary']['net_change']:+d}")

    # Save results
    os.makedirs("./results", exist_ok=True)
    model_short = model_name.split("/")[-1].replace("-", "_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = f"./results/validation_{model_short}_{num_samples}samples_{ts}.json"
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {outfile}")

    # Cleanup
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return results

def main():
    parser = argparse.ArgumentParser(description="Full toroidal coherence validation")
    parser.add_argument("--model", choices=["qwen", "olmo", "both"], default="both",
                       help="Which model(s) to test")
    parser.add_argument("--samples", type=int, default=100,
                       help="Number of samples per model (default: 100)")
    parser.add_argument("--alpha", type=float, default=1.0,
                       help="Toroidal bias strength (default: 1.0)")
    args = parser.parse_args()

    models = {
        "qwen": "Qwen/Qwen2.5-7B-Instruct",
        "olmo": "allenai/OLMo-1.7-7B-hf"
    }

    all_results = {}

    if args.model == "both":
        test_models = ["qwen", "olmo"]
    else:
        test_models = [args.model]

    for model_key in test_models:
        model_name = models[model_key]
        print(f"\n{'#' * 70}")
        print(f"# Testing {model_key.upper()}: {model_name}")
        print(f"{'#' * 70}\n")

        results = run_validation(model_name, args.samples, args.alpha)
        all_results[model_key] = results["summary"]

        # Clear cache between models
        gc.collect()
        torch.cuda.empty_cache()
        print(f"\nCleared GPU cache. Available: {torch.cuda.mem_get_info()[0]/1e9:.1f}GB")

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY - ALL MODELS")
    print("=" * 70)

    for model_key, summary in all_results.items():
        print(f"\n{model_key.upper()}:")
        print(f"  Baseline: {summary['baseline_accuracy']:.1%}")
        print(f"  Toroidal: {summary['toroidal_accuracy']:.1%}")
        print(f"  Error Reduction: {summary['error_reduction_pct']:+.1f}%")
        print(f"  Net Change: {summary['net_change']:+d} answers")

    # Save combined summary
    os.makedirs("./results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"./results/validation_summary_{ts}.json"
    with open(summary_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "samples_per_model": args.samples,
            "alpha": args.alpha,
            "results": all_results
        }, f, indent=2)
    print(f"\nSummary saved: {summary_file}")

if __name__ == "__main__":
    main()
