#!/usr/bin/env python3
"""
EMBEDDING-BASED TOROIDAL BIAS
=============================
Maps tokens to torus positions based on EMBEDDING SIMILARITY,
not arbitrary token IDs.

This gives semantic meaning to the toroidal structure:
- Nearby tokens on torus = semantically similar
- Toroidal bias encourages coherent token sequences
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import json
from datetime import datetime
import os
import gc
import argparse

# ============================================================================
# EMBEDDING-BASED TORUS MAPPING
# ============================================================================

def build_embedding_torus_map(model, tokenizer, grid_size=12, device='cuda'):
    """
    Build a mapping from token IDs to torus positions based on embedding similarity.

    1. Get embedding vectors for all tokens
    2. Cluster into grid_size² clusters using k-means
    3. Each cluster = one torus position
    """
    print("Building embedding-based torus mapping...")

    vocab_size = model.config.vocab_size
    n_clusters = grid_size * grid_size  # 144 for 12x12

    # Get the embedding matrix
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        embeddings = model.model.embed_tokens.weight.detach().cpu().numpy()
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
        embeddings = model.transformer.wte.weight.detach().cpu().numpy()
    elif hasattr(model, 'get_input_embeddings'):
        embeddings = model.get_input_embeddings().weight.detach().cpu().numpy()
    else:
        raise ValueError("Cannot find embedding layer")

    print(f"  Embedding shape: {embeddings.shape}")

    # Normalize embeddings for better clustering
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    embeddings_normalized = embeddings / norms

    # Cluster embeddings into grid_size² clusters
    print(f"  Clustering {vocab_size} tokens into {n_clusters} clusters...")
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1024, n_init=3)
    cluster_labels = kmeans.fit_predict(embeddings_normalized)

    # Map: token_id -> torus_position (cluster label)
    token_to_torus = torch.tensor(cluster_labels, device=device, dtype=torch.long)

    # Count tokens per cluster
    unique, counts = np.unique(cluster_labels, return_counts=True)
    print(f"  Tokens per cluster: min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}")

    # Show sample tokens from a few clusters
    for cluster_id in [0, 50, 100]:
        tokens_in_cluster = np.where(cluster_labels == cluster_id)[0][:5]
        token_strs = [tokenizer.decode([t]) for t in tokens_in_cluster]
        print(f"  Cluster {cluster_id}: {token_strs}")

    return token_to_torus

def toroidal_distance(i, j, grid_size=12):
    """Manhattan distance on 2D torus"""
    xi, yi = i % grid_size, (i // grid_size) % grid_size
    xj, yj = j % grid_size, (j // grid_size) % grid_size
    dx = min(abs(xi - xj), grid_size - abs(xi - xj))
    dy = min(abs(yi - yj), grid_size - abs(yi - yj))
    return dx + dy

# Pre-compute distance matrix
def get_torus_distance_matrix(grid_size=12, device='cuda'):
    """Pre-compute all pairwise distances on the torus"""
    n = grid_size * grid_size
    dist_matrix = torch.zeros(n, n, device=device, dtype=torch.float16)
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = toroidal_distance(i, j, grid_size)
    return dist_matrix

def get_bias_EMBEDDING(vocab_size, recent_tokens, token_to_torus, dist_matrix,
                       grid_size=12, radius=2.0, alpha=0.3, device='cuda'):
    """
    Apply toroidal bias using embedding-based torus mapping.
    """
    bias = torch.zeros(vocab_size, device=device, dtype=torch.float16)

    if len(recent_tokens) == 0:
        return bias

    n_cells = grid_size * grid_size

    # Get torus positions for all vocab tokens
    all_torus_pos = token_to_torus[:vocab_size]

    for offset, token_id in enumerate(recent_tokens[-5:]):
        if token_id >= vocab_size:
            continue

        # Get torus position of this token
        token_torus_pos = token_to_torus[token_id].item()

        # Get distances from this position to all positions
        distances = dist_matrix[token_torus_pos, all_torus_pos]

        # Apply bias based on distance
        weight = 1.0 / (offset + 1)

        near_mask = distances <= radius
        mid_mask = (distances > radius) & (distances <= radius * 2)

        bias[near_mask] += alpha * (radius - distances[near_mask] + 1) * weight
        bias[mid_mask] += alpha * 0.5 * weight

    return bias

def get_bias_LIMITED(vocab_size, recent_tokens, grid_size=12, radius=2.0, alpha=0.3, device='cuda'):
    """Original ID-based method for comparison"""
    bias = torch.zeros(vocab_size, device=device, dtype=torch.float16)

    if len(recent_tokens) == 0:
        return bias

    max_tokens_to_bias = grid_size * grid_size * 10

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

def generate_with_embedding_bias(model, tokenizer, prompt, token_to_torus, dist_matrix,
                                  max_tokens=30, grid_size=12, radius=2.0, alpha=0.3):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generated = inputs['input_ids'][0].tolist()
    vocab_size = model.config.vocab_size

    for _ in range(max_tokens):
        with torch.no_grad():
            outputs = model(torch.tensor([generated], device=model.device))
            logits = outputs.logits[0, -1, :]

            topo_bias = get_bias_EMBEDDING(
                vocab_size, generated, token_to_torus, dist_matrix,
                grid_size, radius, alpha, model.device
            )
            logits = logits + topo_bias

            next_token = logits.argmax().item()

        generated.append(next_token)
        if next_token == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated, skip_special_tokens=True)

def generate_with_limited_bias(model, tokenizer, prompt, max_tokens=30,
                               grid_size=12, radius=2.0, alpha=0.3):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generated = inputs['input_ids'][0].tolist()
    vocab_size = model.config.vocab_size

    for _ in range(max_tokens):
        with torch.no_grad():
            outputs = model(torch.tensor([generated], device=model.device))
            logits = outputs.logits[0, -1, :]

            topo_bias = get_bias_LIMITED(vocab_size, generated, grid_size, radius, alpha, model.device)
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
# MAIN
# ============================================================================

def run_comparison(model_name, num_samples=100, alpha=0.3, grid_size=12, radius=2.0):
    print("=" * 70)
    print("EMBEDDING-BASED vs LIMITED BIAS COMPARISON")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Samples: {num_samples}")
    print(f"Alpha: {alpha}")
    print(f"Grid: {grid_size}x{grid_size}")
    print(f"Radius: {radius}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    print(f"Loaded. GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    # Build embedding-based torus mapping
    token_to_torus = build_embedding_torus_map(model, tokenizer, grid_size, model.device)
    dist_matrix = get_torus_distance_matrix(grid_size, model.device)

    prompts = (TEST_PROMPTS * ((num_samples // len(TEST_PROMPTS)) + 1))[:num_samples]

    results = {
        "baseline": {"correct": 0, "total": 0},
        "limited": {"correct": 0, "total": 0},
        "embedding": {"correct": 0, "total": 0},
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

        # Limited (ID-based)
        resp_l = generate_with_limited_bias(model, tokenizer, prompt,
                                            grid_size=grid_size, radius=radius, alpha=alpha)
        ok_l = any(e.lower() in resp_l.lower() for e in expected)
        results["limited"]["total"] += 1
        if ok_l: results["limited"]["correct"] += 1

        # Embedding-based
        resp_e = generate_with_embedding_bias(model, tokenizer, prompt,
                                              token_to_torus, dist_matrix,
                                              grid_size=grid_size, radius=radius, alpha=alpha)
        ok_e = any(e.lower() in resp_e.lower() for e in expected)
        results["embedding"]["total"] += 1
        if ok_e: results["embedding"]["correct"] += 1

        results["details"].append({
            "prompt": prompt,
            "baseline": (resp_b[:80], ok_b),
            "limited": (resp_l[:80], ok_l),
            "embedding": (resp_e[:80], ok_e)
        })

        b = "Y" if ok_b else "X"
        l = "Y" if ok_l else "X"
        e = "Y" if ok_e else "X"
        print(f"[{i+1:3d}] B:{b} L:{l} E:{e} | {prompt[:40]}...")

    # Results
    b_acc = results["baseline"]["correct"] / results["baseline"]["total"]
    l_acc = results["limited"]["correct"] / results["limited"]["total"]
    e_acc = results["embedding"]["correct"] / results["embedding"]["total"]

    b_err = 1 - b_acc
    l_red = ((b_err - (1-l_acc)) / b_err * 100) if b_err > 0 else 0
    e_red = ((b_err - (1-e_acc)) / b_err * 100) if b_err > 0 else 0

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Baseline accuracy:         {b_acc:.1%} ({results['baseline']['correct']}/{num_samples})")
    print(f"Limited bias (ID-based):   {l_acc:.1%} ({results['limited']['correct']}/{num_samples}) | Error reduction: {l_red:+.1f}%")
    print(f"Embedding bias (semantic): {e_acc:.1%} ({results['embedding']['correct']}/{num_samples}) | Error reduction: {e_red:+.1f}%")
    print()

    if e_acc > l_acc and e_acc > b_acc:
        print(">>> EMBEDDING-BASED WINS <<<")
    elif l_acc > e_acc and l_acc > b_acc:
        print(">>> LIMITED WINS <<<")
    elif e_acc == l_acc and e_acc > b_acc:
        print(">>> TIE (both better than baseline) <<<")
    else:
        print(">>> BASELINE WINS (neither method helps) <<<")

    # Save
    os.makedirs("./results", exist_ok=True)
    model_short = model_name.split("/")[-1].replace("-", "_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = f"./results/embedding_vs_limited_{model_short}_{ts}.json"

    summary = {
        "model": model_name,
        "samples": num_samples,
        "alpha": alpha,
        "grid_size": grid_size,
        "radius": radius,
        "baseline_acc": b_acc,
        "limited_acc": l_acc,
        "embedding_acc": e_acc,
        "limited_error_reduction": l_red,
        "embedding_error_reduction": e_red
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
    parser.add_argument("--grid", type=int, default=12)
    parser.add_argument("--radius", type=float, default=2.0)
    args = parser.parse_args()

    run_comparison(args.model, args.samples, args.alpha, args.grid, args.radius)

if __name__ == "__main__":
    main()
