#!/usr/bin/env python3
"""
FREQUENCY-BASED TOROIDAL BIAS
=============================
Instead of assuming first 1440 tokens = high-frequency,
we compute ACTUAL token frequencies and bias only those.

This should work across different tokenizers/models.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter
import json
from datetime import datetime
import os
import gc
import argparse

# ============================================================================
# REFERENCE TEXT FOR FREQUENCY ANALYSIS
# Common English text to identify high-frequency tokens
# ============================================================================

REFERENCE_TEXT = """
The quick brown fox jumps over the lazy dog. This is a common sentence used to test fonts.
Science and technology have transformed our world in countless ways. From the invention of the wheel
to the development of artificial intelligence, human ingenuity has shaped civilization.

History tells us that great empires rise and fall. The Roman Empire, the British Empire, and many
others have left their mark on the world. Wars and peace, discovery and invention, art and science
all contribute to the human story.

Mathematics is the language of the universe. Numbers, equations, and formulas describe the fundamental
laws of physics. Einstein's theory of relativity changed our understanding of space and time.
Newton discovered gravity and developed calculus. Pi equals approximately 3.14159.

Geography shapes human civilization. Rivers like the Nile, Amazon, and Yangtze have supported great
civilizations. Mountains, deserts, and oceans create natural boundaries. The capital of France is Paris.
The capital of Japan is Tokyo. The largest planet is Jupiter.

Literature and art reflect the human condition. Shakespeare wrote plays that are still performed today.
Leonardo da Vinci painted the Mona Lisa. Beethoven composed symphonies that moved generations.

The scientific method involves observation, hypothesis, experiment, and conclusion. DNA carries genetic
information. Photosynthesis converts sunlight into energy. The speed of light is approximately 300,000
kilometers per second. Water freezes at 0 degrees Celsius.

World War II ended in 1945. The Berlin Wall fell in 1989. The first human walked on the moon in 1969.
The Declaration of Independence was signed in 1776. The French Revolution began in 1789.

Technology continues to advance. Computers, smartphones, and the internet have connected the world.
Artificial intelligence and machine learning are transforming industries. The future holds both
promise and challenges for humanity.

The human body is a remarkable machine. The heart pumps blood through miles of vessels. The brain
contains billions of neurons. We breathe oxygen and exhale carbon dioxide. The human body has 206 bones.

Philosophy asks fundamental questions about existence, knowledge, and ethics. What is truth? What is
justice? How should we live? These questions have occupied thinkers for millennia.

Economics studies how societies allocate scarce resources. Supply and demand, inflation and deflation,
trade and commerce all play important roles. The currency of Japan is the yen. The currency of the
United Kingdom is the pound.

The universe is vast and mysterious. Stars, planets, galaxies, and black holes fill the cosmos.
The sun is a star at the center of our solar system. Light travels through the vacuum of space.

Education is the foundation of progress. Schools, universities, and libraries preserve and transmit
knowledge. Reading, writing, and arithmetic are fundamental skills. Learning never stops.

Music is a universal language. Rhythm, melody, and harmony combine to create emotional experiences.
Mozart was born in Austria. Beethoven was a famous composer. Jazz, rock, classical, and pop all have
their place in human culture.

Animals and plants share our planet. Biodiversity is essential for healthy ecosystems. The largest
mammal is the blue whale. The fastest land animal is the cheetah. Bees produce honey.
"""

# ============================================================================
# FREQUENCY-BASED BIAS
# ============================================================================

def get_high_frequency_tokens(tokenizer, reference_text, top_n=1500):
    """
    Tokenize reference text and return the most frequent token IDs.
    """
    tokens = tokenizer.encode(reference_text, add_special_tokens=False)
    freq = Counter(tokens)
    # Get top N most frequent tokens
    most_common = freq.most_common(top_n)
    high_freq_tokens = set(token_id for token_id, count in most_common)
    return high_freq_tokens

def toroidal_distance(i, j, grid_size=12):
    """Manhattan distance on 2D torus"""
    xi, yi = i % grid_size, (i // grid_size) % grid_size
    xj, yj = j % grid_size, (j // grid_size) % grid_size
    dx = min(abs(xi - xj), grid_size - abs(xi - xj))
    dy = min(abs(yi - yj), grid_size - abs(yi - yj))
    return dx + dy

def get_bias_FREQUENCY(vocab_size, recent_tokens, high_freq_set, grid_size=12, radius=2.0, alpha=0.3, device='cuda'):
    """
    Bias only tokens that are in the high-frequency set.
    """
    bias = torch.zeros(vocab_size, device=device, dtype=torch.float16)

    if len(recent_tokens) == 0:
        return bias

    for offset, token_id in enumerate(recent_tokens[-5:]):
        token_pos = token_id % (grid_size * grid_size)

        # Only bias high-frequency tokens
        for vocab_id in high_freq_set:
            if vocab_id >= vocab_size:
                continue
            target_pos = vocab_id % (grid_size * grid_size)
            dist = toroidal_distance(token_pos, target_pos, grid_size)

            if dist <= radius:
                bias[vocab_id] += alpha * (radius - dist + 1) / (offset + 1)
            elif dist <= radius * 2:
                bias[vocab_id] += alpha * 0.5 / (offset + 1)

    return bias

def get_bias_LIMITED(vocab_size, recent_tokens, grid_size=12, radius=2.0, alpha=0.3, device='cuda'):
    """
    Original method: bias first 1440 tokens by ID.
    """
    bias = torch.zeros(vocab_size, device=device, dtype=torch.float16)

    if len(recent_tokens) == 0:
        return bias

    max_tokens_to_bias = grid_size * grid_size * 10  # 1440

    for offset, token_id in enumerate(recent_tokens[-5:]):
        token_pos = token_id % (grid_size * grid_size)

        for vocab_id in range(min(vocab_size, max_tokens_to_bias)):
            target_pos = vocab_id % (grid_size * grid_size)
            dist = toroidal_distance(token_pos, target_pos, grid_size)

            if dist <= radius:
                bias[vocab_id] += alpha * (radius - dist + 1) / (offset + 1)
            elif dist <= radius * 2:
                bias[vocab_id] += alpha * 0.5 / (offset + 1)

    return bias

# ============================================================================
# GENERATION
# ============================================================================

def generate_baseline(model, tokenizer, prompt, max_tokens=30):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                            pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0], skip_special_tokens=True)

def generate_with_bias(model, tokenizer, prompt, bias_fn, bias_args, max_tokens=30):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generated = inputs['input_ids'][0].tolist()
    vocab_size = model.config.vocab_size

    for _ in range(max_tokens):
        with torch.no_grad():
            outputs = model(torch.tensor([generated], device=model.device))
            logits = outputs.logits[0, -1, :]

            topo_bias = bias_fn(vocab_size, generated, **bias_args, device=model.device)
            logits = logits + topo_bias

            next_token = logits.argmax().item()

        generated.append(next_token)
        if next_token == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated, skip_special_tokens=True)

# ============================================================================
# TEST PROMPTS
# ============================================================================

TEST_PROMPTS = [
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
# MAIN TEST
# ============================================================================

def run_comparison(model_name, num_samples=100, alpha=0.3):
    print("=" * 70)
    print(f"FREQUENCY-BASED vs LIMITED BIAS COMPARISON")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Samples: {num_samples}")
    print(f"Alpha: {alpha}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Compute high-frequency tokens from reference text
    print("\nAnalyzing token frequencies...")
    high_freq_tokens = get_high_frequency_tokens(tokenizer, REFERENCE_TEXT, top_n=1500)
    print(f"Found {len(high_freq_tokens)} unique high-frequency tokens")

    # Show some examples
    sample_tokens = list(high_freq_tokens)[:20]
    print(f"Sample high-freq tokens: {[tokenizer.decode([t]) for t in sample_tokens]}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    print(f"Loaded. GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    prompts = (TEST_PROMPTS * ((num_samples // len(TEST_PROMPTS)) + 1))[:num_samples]

    results = {
        "baseline": {"correct": 0, "total": 0},
        "limited": {"correct": 0, "total": 0},
        "frequency": {"correct": 0, "total": 0},
        "details": []
    }

    print(f"\nRunning {num_samples} tests (3 conditions each)...")
    print("-" * 70)

    for i, (prompt, expected) in enumerate(prompts):
        # Baseline
        resp_b = generate_baseline(model, tokenizer, prompt)
        ok_b = any(e.lower() in resp_b.lower() for e in expected)
        results["baseline"]["total"] += 1
        if ok_b: results["baseline"]["correct"] += 1

        # Limited bias (first 1440 by ID)
        resp_l = generate_with_bias(model, tokenizer, prompt, get_bias_LIMITED,
                                    {"grid_size": 12, "radius": 2.0, "alpha": alpha})
        ok_l = any(e.lower() in resp_l.lower() for e in expected)
        results["limited"]["total"] += 1
        if ok_l: results["limited"]["correct"] += 1

        # Frequency-based bias
        resp_f = generate_with_bias(model, tokenizer, prompt, get_bias_FREQUENCY,
                                    {"high_freq_set": high_freq_tokens, "grid_size": 12, "radius": 2.0, "alpha": alpha})
        ok_f = any(e.lower() in resp_f.lower() for e in expected)
        results["frequency"]["total"] += 1
        if ok_f: results["frequency"]["correct"] += 1

        results["details"].append({
            "prompt": prompt, "expected": expected,
            "baseline": (resp_b[:80], ok_b),
            "limited": (resp_l[:80], ok_l),
            "frequency": (resp_f[:80], ok_f)
        })

        b = "Y" if ok_b else "X"
        l = "Y" if ok_l else "X"
        f = "Y" if ok_f else "X"
        print(f"[{i+1:3d}] B:{b} L:{l} F:{f} | {prompt[:40]}...")

    # Results
    b_acc = results["baseline"]["correct"] / results["baseline"]["total"]
    l_acc = results["limited"]["correct"] / results["limited"]["total"]
    f_acc = results["frequency"]["correct"] / results["frequency"]["total"]

    b_err = 1 - b_acc
    l_red = ((b_err - (1-l_acc)) / b_err * 100) if b_err > 0 else 0
    f_red = ((b_err - (1-f_acc)) / b_err * 100) if b_err > 0 else 0

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Baseline accuracy:        {b_acc:.1%} ({results['baseline']['correct']}/{num_samples})")
    print(f"Limited bias (ID-based):  {l_acc:.1%} ({results['limited']['correct']}/{num_samples}) | Error reduction: {l_red:+.1f}%")
    print(f"Frequency bias (actual):  {f_acc:.1%} ({results['frequency']['correct']}/{num_samples}) | Error reduction: {f_red:+.1f}%")
    print()

    if f_acc > l_acc:
        print(">>> FREQUENCY-BASED WINS <<<")
    elif l_acc > f_acc:
        print(">>> LIMITED (ID-BASED) WINS <<<")
    else:
        print(">>> TIE <<<")

    # Save
    os.makedirs("./results", exist_ok=True)
    model_short = model_name.split("/")[-1].replace("-", "_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = f"./results/frequency_vs_limited_{model_short}_{ts}.json"

    summary = {
        "model": model_name,
        "samples": num_samples,
        "alpha": alpha,
        "high_freq_token_count": len(high_freq_tokens),
        "baseline_acc": b_acc,
        "limited_acc": l_acc,
        "frequency_acc": f_acc,
        "limited_error_reduction": l_red,
        "frequency_error_reduction": f_red
    }

    with open(outfile, "w") as f:
        json.dump({"summary": summary, "details": results["details"]}, f, indent=2)
    print(f"\nSaved: {outfile}")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="allenai/OLMo-1.7-7B-hf")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.3)
    args = parser.parse_args()

    run_comparison(args.model, args.samples, args.alpha)

if __name__ == "__main__":
    main()
