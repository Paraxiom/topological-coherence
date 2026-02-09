---
title: Toroidal Logit Bias
emoji: 🍩
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: true
license: apache-2.0
suggested_hardware: t4-small
---

# Toroidal Logit Bias for Hallucination Reduction

Interactive demo comparing baseline vs toroidal-biased LLM generation on factual prompts.

**Method**: Map tokens to a 12x12 torus. Bias logits toward nearby tokens at inference time. No fine-tuning required.

**Results** (v2 — 4 models, 817 TruthfulQA samples, LLM-judged):
- +2.8pp Mistral 7B, +2.1pp Qwen 7B (Truthful & Informative)
- Consistent improvement across all 4 models (0.5B → 7B)
- Improvement scales with model capacity
- ~5% latency overhead

**Paper**: [DOI: 10.5281/zenodo.18516477](https://doi.org/10.5281/zenodo.18516477)
**Code**: [github.com/Paraxiom/topological-coherence](https://github.com/Paraxiom/topological-coherence)
**Author**: Sylvain Cormier, Paraxiom Research
