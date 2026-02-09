# Social Media Posts — v2 Multi-Model Results (Feb 2026)

## Links
- **Paper (Zenodo)**: https://doi.org/10.5281/zenodo.18516477
- **Theory Paper**: https://doi.org/10.5281/zenodo.18187835
- **Code**: https://github.com/Paraxiom/topological-coherence
- **PyPI**: https://pypi.org/project/topological-coherence/
- **Rust Crate**: https://crates.io/crates/topological-coherence
- **HuggingFace Demo**: https://huggingface.co/spaces/paraxiom-research/topological-coherence

---

## Twitter/X Thread

### Tweet 1:
```
Toroidal Logit Bias now validated on 4 models.

TruthfulQA (817 samples, LLM-judged):
  Mistral 7B: +2.8pp
  Qwen 7B:    +2.1pp
  Qwen 1.5B:  +0.6pp
  Qwen 0.5B:  +0.2pp

4/4 models improved. Scales with model capacity.

No fine-tuning. ~5% overhead. Inference-time only.

Paper: doi.org/10.5281/zenodo.18516477
Code: github.com/Paraxiom/topological-coherence
```

### Tweet 2:
```
How it works:

Map tokens to a 12x12 torus (Tonnetz topology).
Bias logits toward "nearby" tokens at each generation step.
Only bias the first ~1400 high-frequency tokens.

Same geometry that makes music consonant — applied to token selection.

Full-vocabulary bias destroys performance. Limited bias improves it.
```

### Tweet 3:
```
The scaling result is the most interesting part:

0.5B → +0.2pp
1.5B → +0.6pp
7B   → +2.1pp

Toroidal constraints amplify existing model capabilities.
Bigger models have better representations → benefit more from geometric coherence.
```

### Tweet 4:
```
Try it yourself:

pip install topological-coherence

from topological_coherence import ToroidalLogitProcessor
processor = ToroidalLogitProcessor(grid_size=12, radius=2.0, alpha=0.3)
# plug into any HuggingFace model.generate()

Live demo: huggingface.co/spaces/paraxiom-research/topological-coherence
```

### Tweet 5:
```
What's next:

- Scale to 70B+ (scaling trend is encouraging)
- Cross-model judging (eliminate judge bias)
- Integration as inference middleware (vLLM, TGI)

This is an operational contribution: a cheap, reliable knob for production.

DM if working on AI reliability or hallucination reduction.
```

---

## LinkedIn Post

```
Toroidal Logit Bias — now validated on 4 models, 2 architectures, 3 parameter scales.

We impose toroidal topological constraints on token selection at inference time. No fine-tuning. ~5% latency overhead.

Results on TruthfulQA (817 samples, LLM-judged):

  Mistral 7B:  74.4% -> 77.2%  (+2.8pp)
  Qwen 7B:     75.6% -> 77.7%  (+2.1pp)
  Qwen 1.5B:   32.2% -> 32.8%  (+0.6pp)
  Qwen 0.5B:   16.9% -> 17.1%  (+0.2pp)

Key insight: improvement scales with model capacity. Larger models benefit more from geometric coherence constraints.

The same Tonnetz topology from music theory — periodic boundaries, no edge effects, constant spectral gap — applied to LLM token generation.

Paper: https://doi.org/10.5281/zenodo.18516477
Code: https://github.com/Paraxiom/topological-coherence
pip install topological-coherence
Live demo: https://huggingface.co/spaces/paraxiom-research/topological-coherence

Looking to connect with teams working on AI reliability, enterprise LLM deployment, or inference optimization.

For consulting inquiries: sylvain@paraxiom.org

#AI #MachineLearning #LLM #AISafety #HallucinationReduction #TopologicalCoherence
```

---

## Reddit (r/MachineLearning)

```
Title: [R] Toroidal Logit Bias for Hallucination Reduction — validated on 4 models, improvement scales with model size

We present an inference-time intervention that reduces LLM hallucination by imposing toroidal topological constraints on token selection.

**Results on TruthfulQA (817 samples, LLM-judged, Truthful & Informative metric):**

| Model | Baseline | Toroidal | Delta |
|-------|----------|----------|-------|
| Qwen 0.5B | 16.9% | 17.1% | +0.2pp |
| Qwen 1.5B | 32.2% | 32.8% | +0.6pp |
| Qwen 7B | 75.6% | 77.7% | +2.1pp |
| Mistral 7B | 74.4% | 77.2% | +2.8pp |

**Method**: Map token IDs to positions on a 12x12 torus. At each generation step, boost logits for tokens "near" recent tokens in toroidal distance. Only bias the first ~1400 high-frequency tokens (full-vocabulary bias is harmful).

**Key findings**:
- 4/4 models improved (2 architectures, 3 parameter scales)
- Improvement scales with model capacity
- No fine-tuning, ~5% latency overhead
- Random sparsity has no effect (proves topology matters)

**Limitations**: LLM-judged (Qwen-7B as judge), hyperparameter sensitivity per model family.

Paper: https://doi.org/10.5281/zenodo.18516477
Code + pip package: https://github.com/Paraxiom/topological-coherence
Live demo: https://huggingface.co/spaces/paraxiom-research/topological-coherence
```

---

## Reddit (r/LocalLLaMA)

```
Title: Toroidal Logit Bias — inference-time trick that improves TruthfulQA on Qwen and Mistral, pip installable

Tested a geometric logit bias on 4 models. Maps tokens to a torus, boosts nearby tokens at each step. No fine-tuning.

Results (TruthfulQA, 817 samples):
- Mistral 7B: +2.8pp (74.4% -> 77.2%)
- Qwen 7B: +2.1pp (75.6% -> 77.7%)
- Scales with model size

pip install topological-coherence

```python
from topological_coherence import ToroidalLogitProcessor
processor = ToroidalLogitProcessor(grid_size=12, radius=2.0, alpha=0.3)
outputs = model.generate(**inputs, logits_processor=[processor])
```

Live demo: https://huggingface.co/spaces/paraxiom-research/topological-coherence
Paper: https://doi.org/10.5281/zenodo.18516477

Would love feedback from anyone running it on other models.
```

---

## Posting Checklist

1. [ ] Upload v2 PDF to Zenodo (new version on existing DOI)
2. [ ] `pip install twine && python -m build && twine upload dist/*` (PyPI v0.2.1)
3. [ ] `cargo publish` (crates.io v0.1.2)
4. [ ] Push HuggingFace Space updates (`cd huggingface-space && git push`)
5. [ ] Post Twitter/X thread
6. [ ] Post LinkedIn
7. [ ] Post Reddit r/MachineLearning
8. [ ] Post Reddit r/LocalLLaMA
9. [ ] Submit to arXiv (cs.LG primary, cs.CL secondary) once Zenodo v2 is live
