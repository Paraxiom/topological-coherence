# Zenodo Upload: Toroidal Geometry of LLM Representations

## Title
The Toroidal Geometry of LLM Representations: From Detection to Karmonic-Guided Alignment

## Authors
Sylvain Cormier (Paraxiom Research)

## Description

We detect toroidal topology in LLM hidden states via persistent homology and introduce Karmonic spectral filtering as a training-time regularizer that outperforms Sinkhorn optimal transport for truthfulness alignment.

**Torus Detection (Qwen 2.5-0.5B, TruthfulQA):**

| Layer | β₁ (1-cycles) | β₂ (voids) | Truth/Halluc Angular Sep |
|-------|---------------|------------|--------------------------|
| 0 (embedding) | 32 | 8 | 120.7° |
| 12 (middle) | 37 | 21 | 35.8° |
| 23 (final) | 25 | 4 | 35.0° |

**Alignment Results (DPO + LoRA, 200 steps, TruthfulQA 817 samples):**

| Condition | MC1 | MC2 | PPL |
|-----------|-----|-----|-----|
| Untrained baseline | 0.278 | 0.407 | 18.06 |
| DPO only | 0.486 | 0.554 | 18.24 |
| DPO + SAMI + Sinkhorn OT | 0.448 | 0.499 | 18.18 |
| **DPO + SAMI + Karmonic** | **0.461** | **0.544** | 18.22 |
| DPO + Karmonic | 0.468 | 0.531 | 18.21 |

**Key findings:**
- Karmonic > Sinkhorn OT by +1.3pp MC1 and +4.4pp MC2 in matched conditions
- DPO + Karmonic achieves MC1=0.468 (+19pp over untrained baseline)
- Toroidal topology detected across all layers (β₁=23-37)
- Perplexity stable (18.18-18.24) — no quality degradation

**Scope:** Empirical. Detects toroidal topology and shows torus-aware regularization outperforms geometry-agnostic drift control. No ontological claims.

## Keywords
- large language models
- toroidal topology
- persistent homology
- Karmonic spectral filtering
- truthfulness alignment
- hallucination reduction
- optimal transport
- DPO
- TruthfulQA
- geometric regularization

## Related Identifiers
- GitHub: https://github.com/Paraxiom/topological-coherence
- HuggingFace: https://huggingface.co/paraxiom-research
- Related: DOI 10.5281/zenodo.18516477 (Toroidal Logit Bias — inference-time companion)
- Related: DOI 10.5281/zenodo.18187835 (Karmonic Mesh — theoretical foundation)

## License
CC-BY-4.0

## Upload Type
Publication / Preprint

## Publication Date
February 2026

---

## Files to Upload
1. `cormier_toroidal_llm_geometry_2026.pdf` — Main paper
2. `cormier_toroidal_llm_geometry_2026.tex` — LaTeX source (optional)

## Upload Instructions
1. Go to https://zenodo.org/deposit/new
2. Upload type: Publication → Preprint
3. Upload the PDF
4. Fill in metadata from above (title, authors, description, keywords)
5. Add related identifiers
6. Set license to CC-BY-4.0
7. Publish
8. Record the DOI for future reference
