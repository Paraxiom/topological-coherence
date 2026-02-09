# Zenodo Upload: Toroidal Logit Bias for Hallucination Reduction (v2)

**DOI: https://doi.org/10.5281/zenodo.18512373**

## Title
Toroidal Logit Bias for Hallucination Reduction in Large Language Models

## Authors
Sylvain Cormier (Paraxiom Research)

## Description

An inference-time intervention that reduces factual hallucination in large language models by imposing toroidal topological constraints on token selection.

**v2 Results — Multi-Model TruthfulQA Benchmark (817 samples, LLM-judged):**

| Model | Baseline T&I | Toroidal T&I | Delta |
|-------|-------------|-------------|-------|
| Qwen 0.5B | 16.9% | 17.1% | +0.2pp |
| Qwen 1.5B | 32.2% | 32.8% | +0.6pp |
| Qwen 7B | 75.6% | 77.7% | +2.1pp |
| Mistral 7B | 74.4% | 77.2% | +2.8pp |

**Key findings:**
- Consistent positive improvement across all 4 models and 3 parameter scales
- Improvement scales with model capacity
- Cross-architecture validation (Qwen + Mistral)

**Method:**
- Map vocabulary tokens to positions on a 12x12 torus
- Bias logits toward tokens "near" recent tokens in toroidal space
- Only bias high-frequency tokens (first 1K-3K of vocabulary)
- Model-specific hyperparameter tuning required

**Properties:**
- No fine-tuning required
- ~5% latency overhead
- Works across architectures with tuning

**Scope:** This work focuses narrowly on an inference-time intervention for hallucination reduction. It makes no claims about ontology, training dynamics, or universal representations. The contribution is operational and empirical.

## Keywords
- large language models
- hallucination reduction
- inference-time intervention
- logit bias
- toroidal topology
- factual accuracy
- TruthfulQA

## Related Identifiers
- GitHub: https://github.com/Paraxiom/topological-coherence
- PyPI: https://pypi.org/project/topological-coherence/
- Crates.io: https://crates.io/crates/topological-coherence
- HuggingFace Demo: https://huggingface.co/spaces/paraxiom-research/topological-coherence
- Related work: DOI 10.5281/zenodo.18187835 (Topological Constraints theory paper)

## License
CC-BY-4.0

## Upload Type
Publication / Preprint

## Publication Date
February 2026

---

## Files to Upload
1. `toroidal_hallucination_reduction_2026.pdf` - Main paper (v2, 10 pages)
2. `toroidal_hallucination_reduction_2026.tex` - LaTeX source (optional)

## Upload Instructions
1. Go to https://zenodo.org/deposit/18512373 (existing record)
2. Click "New version"
3. Upload new PDF
4. Copy description above into the Description field
5. Publish
