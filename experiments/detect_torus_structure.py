"""Detect toroidal topology in LLM hidden states.

Tests whether LLM representations naturally live on (or near) a torus,
using three independent methods:
  1. Persistent homology (Betti numbers: torus has β₁=2, β₂=1)
  2. Spectral analysis of k-NN graph Laplacian (torus eigenvalue signature)
  3. Circular statistics (angular structure in PCA projections)

Additionally tests whether truthful vs hallucinated answers separate
on any detected circular/toroidal structure.

Usage:
    # Full analysis (needs GPU for fast inference, or CPU with patience)
    python detect_torus_structure.py --output_dir results/torus_detection

    # Quick test with fewer samples
    python detect_torus_structure.py --max_samples 100 --output_dir results/torus_detection_quick

    # CPU-only (slow but works)
    python detect_torus_structure.py --device cpu --max_samples 200
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Hidden state extraction
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def extract_hidden_states(max_samples=None, device="auto", layers=None):
    """Extract hidden states from TruthfulQA through Qwen 2.5-0.5B.

    Returns:
        dict with keys:
            hidden_states: dict[layer_idx] -> (N, hidden_dim) numpy arrays
            labels: (N,) array, 1=truthful answer got highest prob, 0=not
            questions: list of question strings
            is_correct_answer: (N,) array, 1=this is the correct answer choice
            question_ids: (N,) array mapping each row to its question index
    """
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16,
        device_map=device if device != "cpu" else None,
        trust_remote_code=True, output_hidden_states=True,
    )
    if device == "cpu":
        model = model.float()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dev = next(model.parameters()).device
    print(f"Device: {dev}")

    dataset = load_dataset("truthfulqa/truthful_qa", "multiple_choice",
                           split="validation")
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    # Which layers to capture (default: last 4 + first)
    if layers is None:
        n_layers = model.config.num_hidden_layers
        layers = [0, n_layers // 2, n_layers - 3, n_layers - 2, n_layers - 1]
        layers = sorted(set(layers))

    all_hidden = {l: [] for l in layers}
    all_is_correct = []
    all_question_ids = []
    questions = []

    model.eval()
    with torch.no_grad():
        for q_idx, example in enumerate(tqdm(dataset, desc="Extracting hidden states")):
            question = example["question"]
            choices = example["mc1_targets"]["choices"]
            labels = example["mc1_targets"]["labels"]
            questions.append(question)

            for c_idx, choice in enumerate(choices):
                text = f"Q: {question}\nA: {choice}"
                inputs = tokenizer(text, return_tensors="pt", truncation=True,
                                   max_length=256).to(dev)

                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states  # tuple of (1, seq_len, dim)

                # Mean-pool across sequence for each captured layer
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                for l in layers:
                    h = hidden_states[l]  # (1, seq_len, dim)
                    pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                    all_hidden[l].append(pooled[0].cpu().float().numpy())

                all_is_correct.append(labels[c_idx])
                all_question_ids.append(q_idx)

    result = {
        "hidden_states": {l: np.stack(v) for l, v in all_hidden.items()},
        "is_correct_answer": np.array(all_is_correct),
        "question_ids": np.array(all_question_ids),
        "questions": questions,
        "layers": layers,
        "model": MODEL_NAME,
        "n_questions": len(questions),
    }
    print(f"Extracted {len(all_is_correct)} hidden states from {len(questions)} questions")
    return result


# ---------------------------------------------------------------------------
# Test 1: Persistent homology
# ---------------------------------------------------------------------------

def test_persistent_homology(hidden_states, max_points=500, max_dim=2):
    """Compute persistent homology and Betti numbers.

    A torus T^2 has: β₀=1, β₁=2, β₂=1
    A sphere S^2 has: β₀=1, β₁=0, β₂=1
    A circle S^1 has: β₀=1, β₁=1

    Uses Ripser for fast computation.
    """
    try:
        from ripser import ripser
    except ImportError:
        print("  ripser not installed. pip install ripser")
        return None

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    print(f"\n=== Persistent Homology (max_dim={max_dim}) ===")

    # Subsample if too many points
    n = hidden_states.shape[0]
    if n > max_points:
        idx = np.random.choice(n, max_points, replace=False)
        data = hidden_states[idx]
    else:
        data = hidden_states

    # Normalize and reduce dimensionality for computational feasibility
    data = StandardScaler().fit_transform(data)
    n_components = min(50, data.shape[1], data.shape[0])
    data_pca = PCA(n_components=n_components).fit_transform(data)

    print(f"  Points: {data_pca.shape[0]}, Dims: {data_pca.shape[1]}")

    t0 = time.time()
    result = ripser(data_pca, maxdim=max_dim, thresh=np.inf)
    elapsed = time.time() - t0
    print(f"  Ripser time: {elapsed:.1f}s")

    diagrams = result["dgms"]

    # Compute Betti numbers using persistence threshold
    betti = {}
    persistence_stats = {}
    for dim in range(max_dim + 1):
        dgm = diagrams[dim]
        # Filter out infinite death (the single connected component in dim 0)
        finite = dgm[dgm[:, 1] < np.inf] if len(dgm) > 0 else dgm

        persistences = finite[:, 1] - finite[:, 0] if len(finite) > 0 else np.array([])

        # Betti number = count of features with persistence > threshold
        # Use median persistence as threshold (robust)
        if len(persistences) > 0:
            threshold = np.median(persistences) + np.std(persistences)
            significant = persistences[persistences > threshold]
            betti[dim] = len(significant)
            persistence_stats[dim] = {
                "total_features": len(persistences),
                "significant_features": len(significant),
                "max_persistence": float(np.max(persistences)),
                "mean_persistence": float(np.mean(persistences)),
                "median_persistence": float(np.median(persistences)),
                "std_persistence": float(np.std(persistences)),
                "threshold": float(threshold),
            }
        else:
            betti[dim] = 0
            persistence_stats[dim] = {"total_features": 0}

        print(f"  H_{dim}: β_{dim}={betti[dim]} "
              f"(total={persistence_stats[dim].get('total_features', 0)}, "
              f"max_pers={persistence_stats[dim].get('max_persistence', 0):.4f})")

    # Interpretation
    torus_score = 0
    if betti.get(1, 0) >= 2:
        torus_score += 2
        print("  >> β₁ ≥ 2: STRONG toroidal signal (two independent cycles)")
    elif betti.get(1, 0) >= 1:
        torus_score += 1
        print("  >> β₁ = 1: circular structure detected (one cycle, not full torus)")
    else:
        print("  >> β₁ = 0: no circular structure detected")

    if betti.get(2, 0) >= 1:
        torus_score += 1
        print("  >> β₂ ≥ 1: void detected (consistent with torus or sphere)")

    return {
        "betti_numbers": betti,
        "persistence_stats": persistence_stats,
        "torus_score": torus_score,  # 0-3, higher = more toroidal
        "n_points": data_pca.shape[0],
        "n_dims_pca": data_pca.shape[1],
    }


# ---------------------------------------------------------------------------
# Test 2: Spectral analysis
# ---------------------------------------------------------------------------

def test_spectral_analysis(hidden_states, k_neighbors=10, n_eigenvalues=20,
                           max_points=1000):
    """Analyze eigenvalue structure of k-NN graph Laplacian.

    A torus C_N × C_M has eigenvalues:
        λ_{n,m} = (2-2cos(2πn/N)) + (2-2cos(2πm/M))
    The first nonzero eigenvalue has multiplicity 2 (from the two circles).
    """
    from sklearn.neighbors import kneighbors_graph
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from scipy.sparse.linalg import eigsh
    from scipy.sparse import csr_matrix

    print(f"\n=== Spectral Analysis (k={k_neighbors}) ===")

    n = hidden_states.shape[0]
    if n > max_points:
        idx = np.random.choice(n, max_points, replace=False)
        data = hidden_states[idx]
    else:
        data = hidden_states

    data = StandardScaler().fit_transform(data)
    n_components = min(50, data.shape[1], data.shape[0])
    data_pca = PCA(n_components=n_components).fit_transform(data)

    print(f"  Points: {data_pca.shape[0]}, Dims: {data_pca.shape[1]}")

    # Build k-NN graph
    A = kneighbors_graph(data_pca, k_neighbors, mode="connectivity",
                         include_self=False)
    A = ((A + A.T) > 0).astype(float)  # symmetrize

    # Graph Laplacian L = D - A
    degrees = np.array(A.sum(axis=1)).flatten()
    D = csr_matrix(np.diag(degrees))
    L = D - A

    # Compute smallest eigenvalues
    n_eig = min(n_eigenvalues, data_pca.shape[0] - 2)
    eigenvalues, eigenvectors = eigsh(L.toarray(), k=n_eig, which="SM")
    eigenvalues = np.sort(np.real(eigenvalues))

    print(f"  First {min(10, len(eigenvalues))} eigenvalues: "
          + " ".join(f"{e:.4f}" for e in eigenvalues[:10]))

    # Spectral gap = first nonzero eigenvalue
    # (eigenvalues[0] ≈ 0, so gap = eigenvalues[1])
    spectral_gap = float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
    print(f"  Spectral gap λ₁ = {spectral_gap:.6f}")

    # Check for multiplicity-2 pattern (torus signature)
    # On a torus, λ₁ = λ₂ (both circles contribute equally)
    if len(eigenvalues) > 2:
        ratio_12 = eigenvalues[2] / max(eigenvalues[1], 1e-10)
        print(f"  λ₂/λ₁ ratio = {ratio_12:.4f} (close to 1.0 suggests multiplicity-2)")

        # Check for torus-like eigenvalue spacing
        # On C_N × C_M: gap = 2-2cos(2π/N), next distinct = 2-2cos(2π/M)
        # If N=M: multiplicity-2 first nonzero eigenvalue
        multiplicity_score = 1.0 / (1.0 + abs(ratio_12 - 1.0))
    else:
        ratio_12 = None
        multiplicity_score = 0.0

    # Eigenvalue gap ratios (look for torus pattern)
    gap_ratios = []
    for i in range(2, min(8, len(eigenvalues))):
        if eigenvalues[1] > 1e-10:
            gap_ratios.append(float(eigenvalues[i] / eigenvalues[1]))

    print(f"  Gap ratios (λ_i/λ₁): {' '.join(f'{r:.3f}' for r in gap_ratios)}")

    # Compare to torus eigenvalue pattern
    # For T^2_{12}: λ_1 = 2-2cos(2π/12) ≈ 0.268
    # Normalized ratios for 12×12 torus: 1.00, 1.00, 2.73, 2.73, 3.73, 4.00, ...
    torus_12_ratios = [1.0, 1.0, 2.732, 2.732, 3.732, 4.0]
    if len(gap_ratios) >= 4:
        observed = np.array(gap_ratios[:4])
        expected = np.array(torus_12_ratios[:4])
        torus_fit = 1.0 / (1.0 + np.mean((observed - expected) ** 2))
        print(f"  Torus pattern fit (12×12): {torus_fit:.4f} (1.0 = perfect)")
    else:
        torus_fit = 0.0

    return {
        "spectral_gap": spectral_gap,
        "eigenvalues": eigenvalues[:n_eig].tolist(),
        "lambda2_lambda1_ratio": float(ratio_12) if ratio_12 is not None else None,
        "multiplicity_score": float(multiplicity_score),
        "gap_ratios": gap_ratios,
        "torus_12_fit": float(torus_fit),
        "n_points": data_pca.shape[0],
    }


# ---------------------------------------------------------------------------
# Test 3: Circular statistics
# ---------------------------------------------------------------------------

def test_circular_statistics(hidden_states, is_correct, max_points=2000):
    """Test for circular structure and truth/hallucination separation.

    Projects to 2D via PCA, converts to polar coordinates,
    tests for non-uniform angular distribution (Rayleigh test)
    and truth/hallucination angular separation.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    print(f"\n=== Circular Statistics ===")

    n = hidden_states.shape[0]
    if n > max_points:
        idx = np.random.choice(n, max_points, replace=False)
        data = hidden_states[idx]
        labels = is_correct[idx]
    else:
        data = hidden_states
        labels = is_correct

    data = StandardScaler().fit_transform(data)

    # Project to multiple 2D planes and test each
    pca = PCA(n_components=min(10, data.shape[1]))
    data_pca = pca.fit_transform(data)
    explained = pca.explained_variance_ratio_

    print(f"  PCA explained variance (first 5): "
          + " ".join(f"{v:.3f}" for v in explained[:5]))

    results = {}
    best_separation = 0.0
    best_plane = None

    # Test pairs of PCA components
    pairs = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3)]
    for i, j in pairs:
        if j >= data_pca.shape[1]:
            continue

        x = data_pca[:, i]
        y = data_pca[:, j]

        # Convert to angles
        angles = np.arctan2(y, x)

        # Rayleigh test for non-uniformity
        C = np.mean(np.cos(angles))
        S = np.mean(np.sin(angles))
        R = np.sqrt(C**2 + S**2)  # mean resultant length
        n_pts = len(angles)
        rayleigh_z = n_pts * R**2
        # p-value approximation
        rayleigh_p = np.exp(-rayleigh_z) if rayleigh_z < 50 else 0.0

        # Circular mean direction
        mean_dir = np.arctan2(S, C)

        # Separation: angular distance between truthful and hallucinated centroids
        truth_mask = labels == 1
        false_mask = labels == 0

        if truth_mask.sum() > 0 and false_mask.sum() > 0:
            truth_angles = angles[truth_mask]
            false_angles = angles[false_mask]

            truth_mean = np.arctan2(np.sin(truth_angles).mean(),
                                    np.cos(truth_angles).mean())
            false_mean = np.arctan2(np.sin(false_angles).mean(),
                                    np.cos(false_angles).mean())

            # Angular distance (wraps around)
            angular_sep = abs(np.arctan2(np.sin(truth_mean - false_mean),
                                         np.cos(truth_mean - false_mean)))
            angular_sep_deg = np.degrees(angular_sep)
        else:
            angular_sep_deg = 0.0

        plane_key = f"PC{i+1}_PC{j+1}"
        results[plane_key] = {
            "rayleigh_R": float(R),
            "rayleigh_z": float(rayleigh_z),
            "rayleigh_p": float(rayleigh_p),
            "mean_direction_deg": float(np.degrees(mean_dir)),
            "angular_separation_deg": float(angular_sep_deg),
            "explained_var": float(explained[i] + explained[j]),
        }

        is_nonuniform = "YES" if rayleigh_p < 0.01 else "no"
        print(f"  {plane_key}: R={R:.4f} (nonuniform: {is_nonuniform}), "
              f"truth/halluc sep={angular_sep_deg:.1f}°")

        if angular_sep_deg > best_separation:
            best_separation = angular_sep_deg
            best_plane = plane_key

    print(f"  Best truth/hallucination separation: {best_separation:.1f}° on {best_plane}")

    # Overall circular structure score
    max_R = max(r["rayleigh_R"] for r in results.values()) if results else 0
    circular_score = 0
    if max_R > 0.1:
        circular_score += 1  # significant non-uniformity
    if best_separation > 10:
        circular_score += 1  # meaningful angular separation
    if best_separation > 30:
        circular_score += 1  # strong angular separation

    return {
        "planes": results,
        "best_separation_deg": float(best_separation),
        "best_plane": best_plane,
        "max_rayleigh_R": float(max_R),
        "circular_score": circular_score,  # 0-3
        "pca_explained_variance": explained.tolist(),
    }


# ---------------------------------------------------------------------------
# Test 4: Truthful vs hallucinated on torus projection
# ---------------------------------------------------------------------------

def test_torus_projection(hidden_states, is_correct, question_ids):
    """Project hidden states through FourierTorusHead and test separation."""
    from sklearn.preprocessing import StandardScaler

    print(f"\n=== Torus Projection (FourierTorusHead) ===")

    data = StandardScaler().fit_transform(hidden_states)
    hidden_dim = data.shape[1]

    # Build a randomly initialized FourierTorusHead and project
    # (tests whether the linear structure maps cleanly to a torus)
    torus_dim = 2
    n_modes = 6

    # Simple linear projection to 2 angles (mimics FourierTorusHead)
    np.random.seed(42)
    W1 = np.random.randn(hidden_dim, 128) * 0.01
    b1 = np.zeros(128)
    W2 = np.random.randn(128, torus_dim) * 0.01
    b2 = np.zeros(torus_dim)

    # Forward: Linear -> ReLU -> Linear -> sigmoid -> 2π
    h = np.maximum(0, data @ W1 + b1)  # ReLU
    raw = h @ W2 + b2
    angles = 2 * np.pi / (1 + np.exp(-raw))  # sigmoid * 2π

    theta1 = angles[:, 0]
    theta2 = angles[:, 1]

    # Test: do truthful and hallucinated separate on the torus?
    truth_mask = is_correct == 1
    false_mask = is_correct == 0

    if truth_mask.sum() > 0 and false_mask.sum() > 0:
        # Circular mean for each group on each circle
        truth_mean_1 = np.arctan2(np.sin(theta1[truth_mask]).mean(),
                                   np.cos(theta1[truth_mask]).mean())
        false_mean_1 = np.arctan2(np.sin(theta1[false_mask]).mean(),
                                   np.cos(theta1[false_mask]).mean())
        truth_mean_2 = np.arctan2(np.sin(theta2[truth_mask]).mean(),
                                   np.cos(theta2[truth_mask]).mean())
        false_mean_2 = np.arctan2(np.sin(theta2[false_mask]).mean(),
                                   np.cos(theta2[false_mask]).mean())

        sep_1 = abs(np.arctan2(np.sin(truth_mean_1 - false_mean_1),
                                np.cos(truth_mean_1 - false_mean_1)))
        sep_2 = abs(np.arctan2(np.sin(truth_mean_2 - false_mean_2),
                                np.cos(truth_mean_2 - false_mean_2)))

        # Toroidal distance (L1 on angles)
        torus_sep = sep_1 + sep_2
        torus_sep_deg = np.degrees(torus_sep)

        print(f"  Circle 1 separation: {np.degrees(sep_1):.1f}°")
        print(f"  Circle 2 separation: {np.degrees(sep_2):.1f}°")
        print(f"  Total toroidal separation: {torus_sep_deg:.1f}°")

        # Circular variance within each group (lower = more concentrated)
        truth_var_1 = 1 - np.sqrt(np.sin(theta1[truth_mask]).mean()**2 +
                                   np.cos(theta1[truth_mask]).mean()**2)
        false_var_1 = 1 - np.sqrt(np.sin(theta1[false_mask]).mean()**2 +
                                   np.cos(theta1[false_mask]).mean()**2)

        print(f"  Truthful circular variance (S¹): {truth_var_1:.4f}")
        print(f"  Hallucinated circular variance (S¹): {false_var_1:.4f}")
    else:
        torus_sep_deg = 0.0
        truth_var_1 = 0.0
        false_var_1 = 0.0

    return {
        "torus_separation_deg": float(torus_sep_deg),
        "circle1_sep_deg": float(np.degrees(sep_1)) if truth_mask.sum() > 0 else 0,
        "circle2_sep_deg": float(np.degrees(sep_2)) if truth_mask.sum() > 0 else 0,
        "truthful_circular_variance": float(truth_var_1),
        "hallucinated_circular_variance": float(false_var_1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Detect toroidal structure in LLM hidden states")
    parser.add_argument("--output_dir", type=str, default="results/torus_detection")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max TruthfulQA questions (None=all 817)")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--layer", type=int, default=None,
                        help="Specific layer to analyze (default: multiple)")
    parser.add_argument("--skip_extraction", action="store_true",
                        help="Load cached hidden states from output_dir")

    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract or load hidden states
    cache_path = out / "hidden_states.npz"
    if args.skip_extraction and cache_path.exists():
        print(f"Loading cached hidden states from {cache_path}")
        cached = np.load(cache_path, allow_pickle=True)
        data = {
            "hidden_states": {int(k.split("_")[1]): cached[k]
                              for k in cached.files if k.startswith("layer_")},
            "is_correct_answer": cached["is_correct_answer"],
            "question_ids": cached["question_ids"],
            "layers": sorted(int(k.split("_")[1]) for k in cached.files
                             if k.startswith("layer_")),
        }
    else:
        data = extract_hidden_states(max_samples=args.max_samples, device=args.device)
        # Cache
        save_dict = {
            f"layer_{l}": v for l, v in data["hidden_states"].items()
        }
        save_dict["is_correct_answer"] = data["is_correct_answer"]
        save_dict["question_ids"] = data["question_ids"]
        np.savez_compressed(cache_path, **save_dict)
        print(f"Cached hidden states to {cache_path}")

    # Analyze each layer
    layers_to_analyze = [args.layer] if args.layer is not None else data["layers"]
    all_results = {}

    for layer_idx in layers_to_analyze:
        if layer_idx not in data["hidden_states"]:
            print(f"Layer {layer_idx} not in extracted data, skipping")
            continue

        hidden = data["hidden_states"][layer_idx]
        is_correct = data["is_correct_answer"]
        question_ids = data["question_ids"]

        print(f"\n{'='*60}")
        print(f"LAYER {layer_idx} — shape {hidden.shape}")
        print(f"{'='*60}")
        print(f"  Correct answers: {is_correct.sum()} / {len(is_correct)}")

        layer_results = {"layer": layer_idx, "shape": list(hidden.shape)}

        # Test 1: Persistent homology
        ph = test_persistent_homology(hidden)
        if ph:
            layer_results["persistent_homology"] = ph

        # Test 2: Spectral analysis
        spectral = test_spectral_analysis(hidden)
        layer_results["spectral_analysis"] = spectral

        # Test 3: Circular statistics
        circular = test_circular_statistics(hidden, is_correct)
        layer_results["circular_statistics"] = circular

        # Test 4: Torus projection
        torus = test_torus_projection(hidden, is_correct, question_ids)
        layer_results["torus_projection"] = torus

        # Overall score
        total_score = (
            (ph["torus_score"] if ph else 0) +
            (1 if spectral["multiplicity_score"] > 0.7 else 0) +
            circular["circular_score"]
        )
        max_score = 7  # ph:3 + spectral:1 + circular:3
        layer_results["torus_evidence_score"] = total_score
        layer_results["torus_evidence_max"] = max_score

        print(f"\n  TORUS EVIDENCE SCORE: {total_score}/{max_score}")
        if total_score >= 5:
            print("  >> STRONG toroidal structure detected")
        elif total_score >= 3:
            print("  >> MODERATE toroidal structure detected")
        elif total_score >= 1:
            print("  >> WEAK toroidal structure detected")
        else:
            print("  >> NO toroidal structure detected")

        all_results[f"layer_{layer_idx}"] = layer_results

    # Save results
    results_path = out / "torus_detection_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for key, res in all_results.items():
        score = res["torus_evidence_score"]
        mx = res["torus_evidence_max"]
        ph_b1 = res.get("persistent_homology", {}).get("betti_numbers", {}).get(1, "?")
        sg = res.get("spectral_analysis", {}).get("spectral_gap", "?")
        sep = res.get("circular_statistics", {}).get("best_separation_deg", "?")
        print(f"  {key}: score={score}/{mx}, β₁={ph_b1}, "
              f"spectral_gap={sg}, best_sep={sep}°")


if __name__ == "__main__":
    main()
