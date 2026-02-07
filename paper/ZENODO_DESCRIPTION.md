# Zenodo Upload: Toroidal Logit Bias for Hallucination Reduction

## Title
Toroidal Logit Bias for Hallucination Reduction in Large Language Models

## Authors
Sylvain Cormier (Paraxiom Research)

## Description

An inference-time intervention that reduces factual hallucination in large language models by imposing toroidal topological constraints on token selection.

**Key Results:**
- Qwen 2.5-7B-Instruct: +40% error reduction (95% → 97% accuracy)
- OLMo 1.7-7B: +15.4% error reduction (87% → 89% accuracy)

**Method:**
- Map vocabulary tokens to positions on a 12×12 torus
- Bias logits toward tokens "near" recent tokens in toroidal space
- Only bias high-frequency tokens (first 1K–3K of vocabulary)
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

## Related Identifiers
- GitHub: https://github.com/Paraxiom/topological-coherence
- Related work: [Link to your other Zenodo papers in the coherence/topology line]

## License
CC-BY-4.0

## Upload Type
Publication / Preprint

## Publication Date
February 2026

---

## Files to Upload
1. `toroidal_hallucination_reduction_2026.pdf` - Main paper
2. `toroidal_hallucination_reduction_2026.tex` - LaTeX source (optional)

## Suggested Cross-Links

In your existing Zenodo papers, consider adding a note in future versions:

> "For empirical validation of toroidal constraints at inference time, see: [this paper's DOI]"

This creates the narrative thread without forcing readers to accept the full arc.
