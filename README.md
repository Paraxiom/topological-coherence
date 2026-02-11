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

## Empirical Results

### TruthfulQA v2 — Multi-Model Benchmark (817 samples, LLM-judged)

Toroidal logit bias produces consistent improvements across all 4 models tested:

| Model | Baseline T&I | Toroidal T&I | Delta |
|-------|-------------|-------------|-------|
| Qwen 0.5B | 16.9% | 17.1% | **+0.2pp** |
| Qwen 1.5B | 32.2% | 32.8% | **+0.6pp** |
| Qwen 7B | 75.6% | 77.7% | **+2.1pp** |
| Mistral 7B | 74.4% | 77.2% | **+2.8pp** |

**Key finding**: Improvement scales with model capacity — larger models benefit more from toroidal constraints.

### Toy Model Validation

Training-time toroidal attention masks on a 2-layer transformer:

| Condition | Drift Rate | Interpretation |
|-----------|-----------|----------------|
| Baseline | 0.0100 | Control |
| Toroidal | **0.0060** | **40% lower drift** |
| Random sparse | 0.1673 | 28x worse — proves topology matters, not sparsity |

### Critical Insight: Negative Control

**Random graph masking (same sparsity, no topological structure) has drift rate 0.167 vs toroidal's 0.006.** This proves it's specifically **topological structure** that matters — sparsity alone is insufficient.

---

## Repository Structure

```
topological-coherence/
├── src/
│   ├── topological_coherence/          # Python package (PyPI)
│   │   ├── logit_bias.py              # ToroidalLogitProcessor
│   │   ├── tonnetz.py                 # Tonnetz topology
│   │   ├── masks.py                   # Toroidal mask generation
│   │   ├── attention.py               # Attention layer variants
│   │   ├── drift.py                   # Drift measurement
│   │   └── tests/                     # Unit tests
│   └── lib.rs                         # Rust crate (crates.io)
├── paper/
│   ├── toroidal_hallucination_reduction_2026.tex  # v2 paper (multi-model)
│   └── toroidal_hallucination_reduction_2026.pdf
├── cormier_topological_coherence_2026.tex   # Theory paper (LaTeX)
├── cormier_topological_coherence_2026.pdf   # Theory paper (PDF)
├── results/                            # v2 benchmark data & charts
├── experiments/                        # Validation scripts
├── diagrams/                           # Result visualizations
├── docs/                               # Unified theory & diagrams
├── huggingface-space/                  # HuggingFace Space demo
├── presentation/                       # HTML presentation
├── Cargo.toml                          # Rust crate config
├── pyproject.toml                      # Python package config
└── LICENSE                             # Apache 2.0
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
| **Unified Theory** | Conservative composition across ML, blockchain, consensus | [docs/UNIFIED_THEORY.md](docs/UNIFIED_THEORY.md) |
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

1. **Benchmark scope**: Tested on factual truthfulness (TruthfulQA). Open-ended generation untested.
2. **Recall-coherence tradeoff**: Suppressing long-range attention may hurt tasks requiring non-local retrieval
3. **Hyperparameter sensitivity**: Each model family requires tuning
4. **Judge bias**: LLM-judged evaluation uses Qwen-7B as both subject and judge

---

## Future Work

1. Scale to 70B+ models (scaling trend is encouraging)
2. Compare with other geometric constraints (hyperbolic, spherical)
3. Cross-model judging to eliminate judge bias
4. Investigate task-dependent optimal topology

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
