# Toroidal Coherence Validation Results

## OLMo 1.7-7B - February 6, 2026

### Configuration
- **Model**: allenai/OLMo-1.7-7B-hf (7B parameters)
- **Method**: Toroidal logit bias (Tonnetz topology)
- **Grid**: 12×12 torus
- **Radius**: 2.0
- **Alpha**: 0.3
- **Hardware**: RunPod A40 (48GB VRAM)
- **Precision**: float16
- **Samples**: 20 factual prompts

### Results

| Condition | Correct | Total | Accuracy |
|-----------|---------|-------|----------|
| Baseline  | 15      | 20    | 75.0%    |
| Toroidal  | 15      | 20    | 75.0%    |

**Error Reduction: 0%**

### Detailed Comparison

| # | Prompt | Baseline | Toroidal | Result |
|---|--------|----------|----------|--------|
| 1 | The capital of France is | ✓ | ✓ | SAME |
| 2 | Water freezes at | ✓ | ✓ | SAME |
| 3 | The largest planet is | ✓ | ✓ | SAME |
| 4 | Einstein developed the theory of | ✓ | ✓ | SAME |
| 5 | The chemical symbol for gold is | ✓ | ✓ | SAME |
| 6 | World War II ended in | ✓ | ✓ | SAME |
| 7 | DNA stands for | ✗ | ✗ | SAME |
| 8 | The Mona Lisa was painted by | ✓ | ✓ | SAME |
| 9 | Shakespeare wrote | ✗ | ✗ | SAME |
| 10 | Mount Everest is in | ✓ | ✓ | SAME |
| 11 | The atomic number of hydrogen is | ✓ | ✓ | SAME |
| 12 | Photosynthesis converts | ✓ | ✓ | SAME |
| 13 | The currency of Japan is | ✓ | ✓ | SAME |
| 14 | Newton discovered | ✗ | ✗ | SAME |
| 15 | The Amazon River is in | ✓ | ✓ | SAME |
| 16 | The Great Wall is in | ✓ | ✓ | SAME |
| 17 | Oxygen is about what percent of air | ✓ | ✓ | SAME |
| 18 | Pi equals approximately | ✗ | ✗ | SAME |
| 19 | The speed of light is | ✗ | ✗ | SAME |
| 20 | The human heart has | ✓ | ✓ | SAME |

### Key Findings

1. **No improvement** - Toroidal constraint had no effect on OLMo
2. **No regression** - Toroidal constraint also caused no harm
3. **Architecture-dependent** - Same constraint that gave 100% error reduction on Qwen had 0% effect on OLMo

### Cross-Model Comparison (Same Day)

| Model | Baseline | Toroidal | Error Reduction |
|-------|----------|----------|-----------------|
| Qwen 2.5-7B-Instruct | 90% | 100% | **+100%** |
| OLMo 1.7-7B-hf | 75% | 75% | 0% |

### Interpretation

This result is consistent with the Divergence Note findings:
- Phi-2 (2.7B): 50% reduction
- TinyLlama (1.1B): 180% increase (negative)
- Qwen 2.5-7B: 100% reduction
- OLMo 1.7-7B: 0% change

**Hypothesis**: The toroidal constraint's effectiveness depends on:
1. Model architecture (attention patterns, layer structure)
2. Training data characteristics
3. Potential alignment between Tonnetz topology and model's learned representations

OLMo's fully transparent training data (web + academic) may have different distributional properties than Qwen's training mix.

### Research Implications

1. **Topology-architecture matching** is critical
2. Different models may need different topological constraints
3. The Tonnetz (12×12 musical) topology works well for some models but not universally
4. Future work: explore alternative topologies (E8/Atlas, different grid sizes)

### Raw Output

```
============================================================
TOROIDAL COHERENCE - LOGIT BIAS METHOD
============================================================
Model: allenai/OLMo-1.7-7B-hf
Samples: 20
Loaded. GPU: 13.8GB

Running tests...
[ 1] B:✓ T:✓ SAME  | The capital of France is...
[ 2] B:✓ T:✓ SAME  | Water freezes at...
[ 3] B:✓ T:✓ SAME  | The largest planet is...
[ 4] B:✓ T:✓ SAME  | Einstein developed the theory of...
[ 5] B:✓ T:✓ SAME  | The chemical symbol for gold is...
[ 6] B:✓ T:✓ SAME  | World War II ended in...
[ 7] B:✗ T:✗ SAME  | DNA stands for...
[ 8] B:✓ T:✓ SAME  | The Mona Lisa was painted by...
[ 9] B:✗ T:✗ SAME  | Shakespeare wrote...
[10] B:✓ T:✓ SAME  | Mount Everest is in...
[11] B:✓ T:✓ SAME  | The atomic number of hydrogen is...
[12] B:✓ T:✓ SAME  | Photosynthesis converts...
[13] B:✓ T:✓ SAME  | The currency of Japan is...
[14] B:✗ T:✗ SAME  | Newton discovered...
[15] B:✓ T:✓ SAME  | The Amazon River is in...
[16] B:✓ T:✓ SAME  | The Great Wall is in...
[17] B:✓ T:✓ SAME  | Oxygen is about what percent of air...
[18] B:✗ T:✗ SAME  | Pi equals approximately...
[19] B:✗ T:✗ SAME  | The speed of light is...
[20] B:✓ T:✓ SAME  | The human heart has...

============================================================
RESULTS
============================================================
Baseline accuracy: 75.0%
Toroidal accuracy: 75.0%
Error reduction:   +0.0%
```

---

*Sylvain Cormier / Paraxiom Research / February 6, 2026*
