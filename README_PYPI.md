# Topological Coherence

**Toroidal attention constraints for reducing LLM hallucination**

*Sylvain Cormier | Paraxiom Research | 2026*

## Key Results (v2 — February 2026)

- **+2.8pp on Mistral 7B**, +2.1pp on Qwen 7B — TruthfulQA (817 samples, LLM-judged)
- **4/4 models improved** across 2 architectures and 3 parameter scales
- **Improvement scales with model capacity** — larger models benefit more
- **28x lower drift** than random sparsity (proves topology matters, not just compute reduction)
- Paper: [DOI: 10.5281/zenodo.18516477](https://doi.org/10.5281/zenodo.18516477)

## Installation

```bash
pip install topological-coherence

# With HuggingFace transformers support
pip install topological-coherence[hf]
```

## Quick Start: Toroidal Logit Bias

Drop-in logit processor for any HuggingFace model — no fine-tuning required:

```python
from topological_coherence import ToroidalLogitProcessor
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

processor = ToroidalLogitProcessor(grid_size=12, radius=2.0, alpha=0.3)

inputs = tokenizer("The quantum nature of", return_tensors="pt")
outputs = model.generate(
    **inputs,
    logits_processor=[processor],
    max_new_tokens=100
)
print(tokenizer.decode(outputs[0]))
```

## Core API

### Tonnetz Geometry

```python
from topological_coherence import Tonnetz, distance_matrix

# Create a 12x12 torus topology
t = Tonnetz(grid_size=12)
t.distance(0, 5)        # L1 toroidal distance with wraparound
t.spectral_gap()         # First eigenvalue of the torus Laplacian

# Vectorized distance matrix (numpy, fast)
dm = distance_matrix(n_tokens=64, grid_size=12)  # (64, 64)
```

### Attention Masks (3 variants)

```python
from topological_coherence import ToroidalMask, sinkhorn_knopp

mask = ToroidalMask.hybrid(seq_len=64, radius=2.0, alpha=1.0)   # default
mask = ToroidalMask.hard_cutoff(seq_len=64, radius=2.0)          # binary
mask = ToroidalMask.soft_exponential(seq_len=64, alpha=1.0)      # smooth decay

tensor = mask.to_tensor()                     # torch.Tensor for attention
ds = sinkhorn_knopp(tensor, n_iters=50)       # project to doubly-stochastic
```

### Drift Measurement

```python
from topological_coherence import DriftMeter

meter = DriftMeter(threshold=2, grid_size=12)
meter.record(pred=5, target=8)
meter.record(pred=5, target=100)
print(f"Drift rate: {meter.rate():.3f}")
```

### Toroidal Attention (PyTorch)

```python
from topological_coherence import ToroidalAttention, TinyTransformer

# Drop-in attention replacement
attn = ToroidalAttention(d_model=64, n_heads=4, max_seq_len=64)

# Full demo transformer with swappable attention
model = TinyTransformer(
    vocab_size=144, d_model=64, n_heads=4,
    attention_type="toroidal"  # or "baseline", "random"
)
```

## Theory

Hallucination is a geometry problem. Unconstrained latent dynamics permit arbitrary drift through embedding space. Toroidal constraints provide a **constant spectral gap** that suppresses non-resonant modes:

```
λ₁ = 2 - 2cos(2π/N) = Θ(1)    for fixed grid size N
```

This bounds semantic drift without reducing model capacity.

**Hierarchy:** mHC (Birkhoff) ⊂ ERLHS (Hamiltonian) ⊂ Karmonic (Toroidal + Spectral)

## Links

- [Paper (Zenodo)](https://doi.org/10.5281/zenodo.18187835)
- [Toroidal Logit Bias Paper](https://doi.org/10.5281/zenodo.18516477)
- [Live Demo (HuggingFace)](https://huggingface.co/spaces/paraxiom-research/topological-coherence)
- [Source (GitHub)](https://github.com/Paraxiom/topological-coherence)
- [Rust Crate (crates.io)](https://crates.io/crates/topological-coherence)

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

## License

Apache-2.0
