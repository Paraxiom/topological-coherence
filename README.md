# Topological Constraints for Coherent Language Models

**Why Geometry Prevents Hallucination**

Sylvain Cormier | Paraxiom Research | January 2026

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](cormier_topological_coherence_2026.pdf)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

---

## Abstract

Residual geometry determines whether reasoning is stable. We show that transformer latent dynamics, operating on unconstrained vector spaces, lack the conserved quantities necessary for bounded inference. This establishes a hierarchy of sufficient conditions:

```
mHC (Birkhoff) ⊂ ERLHS (Hamiltonian) ⊂ Karmonic (Toroidal + Spectral)
```

The practical consequence—reduced drift, and thereby reduced hallucination—follows from the geometry when these conditions are satisfied.

---

## Key Theoretical Contributions

### 1. Hallucination as Geometry Problem

We argue that hallucination is not a training data problem, an alignment failure, or an inherent limitation of autoregressive generation. **Hallucination is a geometry problem**: unconstrained latent dynamics permit arbitrary drift through latent space.

### 2. Hierarchy of Constraints

| Level | Adds | Solves |
|-------|------|--------|
| **mHC** (Birkhoff polytope) | Bounded mixing | Training stability |
| **ERLHS** (Hamiltonian) | Conserved flow | Inference coherence |
| **Karmonic** (Toroidal + Spectral) | Spectral gap | Noise suppression |

### 3. Spectral Alignment (Resonance)

Modes that align with the manifold's eigenstructure persist under repeated composition. Non-resonant modes decay as e^(-λt).

**Epistemic boundary**: Spectral alignment *filters*, *stabilizes*, and *selects*. It does not alone guarantee semantic correctness. A resonant mode may be stably wrong.

---

## Experimental Validation

### Setup

- **Model**: 2-layer transformer, d_model=64, 4 attention heads
- **Task**: Next-token prediction on sequences with controlled semantic drift (Tonnetz-adjacent valid continuations)
- **Conditions**: Baseline, mHC, Toroidal, Random (negative control)
- **Hardware**: CPU only (~4 minutes total)

### Results

```
============================================================
RESULTS SUMMARY
============================================================

Condition    | Final Drift  | Final Coh.Var  | Grad Norm
------------------------------------------------------------
baseline     | 0.0100       | 35.76          | 0.27
mhc          | 0.0133       | 1010.54        | 1.60
toroidal     | 0.0060       | 41.93          | 0.22
random       | 0.1673       | 113.88         | 0.78
```

### Key Findings

| Metric | Winner | Interpretation |
|--------|--------|----------------|
| **Drift Rate** | Toroidal (0.006) | 40% lower than baseline, **96% lower than random** |
| **Grad Norm** | Toroidal (0.22) | Most stable training |
| **Coherence Var** | Baseline (35.8) | But mHC exploded (1010!) |

### Critical Insight: Negative Control

**Random graph masking (same sparsity, no topological structure) has drift rate 0.167 vs toroidal's 0.006.**

That's a **28x difference**.

This proves:
- It's not "any constraint" that works
- It's specifically **topological structure**
- Sparsity alone is insufficient; geometry is necessary

### Interpretation

1. **Toroidal constraint reduces long-range semantic jumps** under a topology-aligned task
2. **mHC increases drift slightly** despite being more "regularized" — confirms that **constraint ≠ structure**
3. **Gradient stability improves under local topological constraints** but degrades under global doubly-stochastic coupling
4. **Baseline minimizes raw hidden-state variance but does not prevent semantic drift**; toroidal attention trades a small increase in variance for a substantial reduction in drift

**The catastrophic coherence variance under mHC (1010 vs ~40) suggests that doubly-stochastic constraints without spectral or geometric locality introduce global coupling instabilities.**

> **Note**: Absolute values are task- and scale-dependent; we report relative trends across conditions.

---

## Repository Structure

```
topological-coherence/
├── cormier_topological_coherence_2026.pdf   # Paper (15 pages)
├── cormier_topological_coherence_2026.tex   # LaTeX source
├── experiments/
│   ├── tonnetz_validation.py                # Minimal validation experiment
│   └── venv/                                # Python environment (not tracked)
├── README.md                                # This file
└── LICENSE                                  # Apache 2.0
```

---

## Running the Experiment

### Prerequisites

- Python 3.8+
- ~500MB disk space for PyTorch

### Installation

```bash
cd experiments
python3 -m venv venv
source venv/bin/activate
pip install torch numpy
```

### Run

```bash
python tonnetz_validation.py
```

**Expected runtime**: ~4 minutes on CPU (no GPU required)

### Expected Output

The experiment trains 4 models (baseline, mHC, toroidal, random) and reports:
- Drift rate (lower = better semantic coherence)
- Coherence variance (hidden state stability)
- Gradient norm (training stability)

---

## Theoretical Background

### Tonnetz Topology

The Tonnetz is a 2D torus where:
- Horizontal edges connect by perfect fifths
- Vertical edges connect by major thirds
- Diagonal edges connect by minor thirds

We use it as a **constructive existence proof** of a low-genus manifold with constant spectral gap—not as a claim about semantic universals.

### Spectral Gap

For a d-dimensional torus T^d_N:

```
λ₁ = 2 - 2cos(2π/N) = Θ(1)
```

for fixed side length N, independent of total nodes N^d.

**Important caveat**: This holds for fixed torus side length N. Scaling N reintroduces gap decay as O(1/N²).

### Why Not Implicit Smoothing?

Standard transformer components (LayerNorm, softmax temperature, multi-head averaging) provide some implicit spectral filtering. However, none impose *topological* constraints—they operate pointwise or via soft weighting, not via manifold structure. They smooth without providing a conserved quantity or spectral gap guarantee.

The distinction is between **ad-hoc regularization** (which helps) and **geometric constraint** (which bounds).

---

## Citation

```bibtex
@misc{cormier2026topological,
  author = {Cormier, Sylvain},
  title = {Topological Constraints for Coherent Language Models: Why Geometry Prevents Hallucination},
  year = {2026},
  publisher = {Zenodo},
  url = {https://github.com/Paraxiom/topological-coherence}
}
```

---

## Related Work

| Paper | Topic | Link |
|-------|-------|------|
| ERLHS | Hamiltonian framework for coherence-preserving ML | [DOI: 10.5281/zenodo.17928909](https://doi.org/10.5281/zenodo.17928909) |
| Karmonic Mesh | Spectral consensus on toroidal manifolds | [DOI: 10.5281/zenodo.17928991](https://doi.org/10.5281/zenodo.17928991) |
| mHC | Manifold-Constrained Hyper-Connections | [arXiv:2512.24880](https://arxiv.org/abs/2512.24880) |
| Graph Signal Processing | Spectral methods on graphs | [Shuman et al., 2013](https://ieeexplore.ieee.org/document/6494675) |

---

## Key Equations

### Toroidal Attention Mask (Eq. 17)

```
M_Tonnetz(i, j) = 1                           if d_Tonnetz(i, j) ≤ r
                  exp(-α · d_Tonnetz(i,j))    otherwise
```

### Learned Toroidal Projection (Eq. 20)

```
φ_θ(e) = ( σ(W₁e) mod 1, σ(W₂e) mod 1 )
```

### Adjacency Loss (Eq. 21)

```
L_topo = E[(a,b)~co-occur][d_T(φ(a), φ(b))] - λ · E[(a,c)~random][d_T(φ(a), φ(c))]
```

---

## Limitations

1. **Embedding complexity**: Mapping tokens to Tonnetz positions requires learning or heuristics
2. **Recall-coherence tradeoff**: Suppressing long-range attention may hurt tasks requiring non-local retrieval
3. **Task dependence**: Optimal radius r and decay rate α are task-dependent
4. **Scale**: Results shown on toy model; validation at scale is future work

---

## Future Work

1. Scale to larger models (7B+ parameters)
2. Evaluate on standard benchmarks (TruthfulQA, HaluEval)
3. Compare with other geometric constraints (hyperbolic, spherical)
4. Develop efficient Tonnetz embedding algorithms
5. Investigate task-dependent optimal topology

---

## License

Apache 2.0

---

## Contact

- **Author**: Sylvain Cormier
- **Email**: sylvain@paraxiom.org
- **Organization**: [Paraxiom Research](https://paraxiom.org)

---

*"Geometric constraints provide one principled path to coherent artificial intelligence—not the only path, but a formally grounded one."*
