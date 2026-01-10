# Topological Constraints for Coherent Language Models

**Why Geometry Prevents Hallucination**

*Sylvain Cormier | Paraxiom Research | January 2026*

## Key Result

- **40% lower semantic drift** than baseline attention
- **28x lower drift** than random sparsity (negative control)
- Proves: **topology matters, not just compute reduction**

## Installation

```bash
pip install topological-coherence
```

## Usage

```python
from topological_coherence import ToroidalAttention, TinyTransformer

# Use toroidal attention in your model
attn = ToroidalAttention(d_model=64, n_heads=4, max_seq_len=64)

# Or use the full demo transformer
model = TinyTransformer(
    vocab_size=144,
    d_model=64,
    n_heads=4,
    attention_type="toroidal"  # or "baseline", "random"
)
```

## Abstract

Residual geometry determines whether reasoning is stable. We show that transformer latent dynamics, operating on unconstrained vector spaces, lack the conserved quantities necessary for bounded inference.

Toroidal (periodic) constraints on attention provide a spectral gap guarantee that suppresses non-resonant modes, reducing semantic drift.

## Links

- [Paper (Zenodo)](https://doi.org/10.5281/zenodo.18187835)
- [Live Demo (HuggingFace)](https://huggingface.co/spaces/paraxiom/topological-coherence)
- [Code (GitHub)](https://github.com/Paraxiom/topological-coherence)

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
