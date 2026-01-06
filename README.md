# Topological Constraints for Coherent Language Models

**Why Geometry Prevents Hallucination**

Sylvain Cormier | Paraxiom Research | January 2026

---

## Abstract

Residual geometry determines whether reasoning is stable. We show that transformer latent dynamics, operating on unconstrained vector spaces, lack the conserved quantities necessary for bounded inference. This establishes a hierarchy of sufficient conditions:

```
mHC (Birkhoff) ⊂ ERLHS (Hamiltonian) ⊂ Karmonic (Toroidal + Spectral)
```

## Key Results

| Condition | Drift Rate | Coherence Var | Grad Norm |
|-----------|------------|---------------|-----------|
| Baseline | 0.0100 | 35.76 | 0.27 |
| mHC | 0.0133 | 1010.54 | 1.60 |
| **Toroidal** | **0.0060** | 41.93 | **0.22** |

**Toroidal attention reduces drift by 40%** compared to unconstrained baseline.

## Repository Structure

```
topological-coherence/
├── cormier_topological_coherence_2026.pdf   # Paper
├── cormier_topological_coherence_2026.tex   # LaTeX source
├── experiments/
│   ├── tonnetz_validation.py                # Minimal validation experiment
│   └── venv/                                # Python environment
└── README.md
```

## Running the Experiment

```bash
cd experiments
python3 -m venv venv
source venv/bin/activate
pip install torch numpy
python tonnetz_validation.py
```

Runs in ~3 minutes on CPU. No GPU required.

## Citation

```bibtex
@misc{cormier2026topological,
  author = {Cormier, Sylvain},
  title = {Topological Constraints for Coherent Language Models: Why Geometry Prevents Hallucination},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.XXXXXXX}
}
```

## Related Work

- [ERLHS: Hamiltonian Framework for Coherence-Preserving ML](https://doi.org/10.5281/zenodo.17928909)
- [Karmonic Mesh: Spectral Consensus on Toroidal Manifolds](https://doi.org/10.5281/zenodo.17928991)
- [mHC: Manifold-Constrained Hyper-Connections (DeepSeek, 2026)](https://arxiv.org/abs/2512.24880)

## License

Apache 2.0
