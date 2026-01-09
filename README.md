---
title: Topological Coherence
emoji: 🔮
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Geometric constraints reduce LLM hallucination
---

# Topological Constraints for Coherent Language Models

**Why Geometry Prevents Hallucination**

*Sylvain Cormier | Paraxiom Research | January 2026*

## Key Result

- **40% lower semantic drift** than baseline attention
- **28x lower drift** than random sparsity (negative control)
- Proves: **topology matters, not just compute reduction**

## Abstract

Residual geometry determines whether reasoning is stable. We show that transformer latent dynamics, operating on unconstrained vector spaces, lack the conserved quantities necessary for bounded inference.

Toroidal (periodic) constraints on attention provide a spectral gap guarantee that suppresses non-resonant modes, reducing semantic drift.

## Links

- [Paper (Zenodo)](https://doi.org/10.5281/zenodo.18187835)
- [Code (GitHub)](https://github.com/Paraxiom/topological-coherence)
- [Rust Crate](https://crates.io/crates/topological-coherence)

## Citation

```bibtex
@misc{cormier2026topological,
  author = {Cormier, Sylvain},
  title = {Topological Constraints for Coherent Language Models},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.18187835}
}
```
