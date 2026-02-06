# Toroidal Coherence Validation Results

## Qwen 2.5-7B-Instruct - February 6, 2026

### Configuration
- **Model**: Qwen/Qwen2.5-7B-Instruct (7B parameters)
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
| Baseline  | 18      | 20    | 90.0%    |
| Toroidal  | 20      | 20    | 100.0%   |

**Error Reduction: 100%**

### Detailed Comparison

| # | Prompt | Baseline | Toroidal | Result |
|---|--------|----------|----------|--------|
| 1 | The capital of France is | ✓ | ✓ | SAME |
| 2 | Water freezes at | ✓ | ✓ | SAME |
| 3 | The largest planet is | ✓ | ✓ | SAME |
| 4 | Einstein developed the theory of | ✓ | ✓ | SAME |
| 5 | The chemical symbol for gold is | ✓ | ✓ | SAME |
| 6 | World War II ended in | ✓ | ✓ | SAME |
| 7 | DNA stands for | ✓ | ✓ | SAME |
| 8 | The Mona Lisa was painted by | ✓ | ✓ | SAME |
| 9 | Shakespeare wrote | ✗ | ✓ | **TORO+** |
| 10 | Mount Everest is in | ✓ | ✓ | SAME |
| 11 | The atomic number of hydrogen is | ✓ | ✓ | SAME |
| 12 | Photosynthesis converts | ✓ | ✓ | SAME |
| 13 | The currency of Japan is | ✓ | ✓ | SAME |
| 14 | Newton discovered | ✗ | ✓ | **TORO+** |
| 15 | The Amazon River is in | ✓ | ✓ | SAME |
| 16 | The Great Wall is in | ✓ | ✓ | SAME |
| 17 | Oxygen is about what percent of air | ✓ | ✓ | SAME |
| 18 | Pi equals approximately | ✓ | ✓ | SAME |
| 19 | The speed of light is | ✓ | ✓ | SAME |
| 20 | The human heart has | ✓ | ✓ | SAME |

### Key Findings

1. **Toroidal constraint fixed both baseline errors** (Shakespeare, Newton questions)
2. **No regressions** - all correct baseline answers remained correct
3. **Consistent improvement** - toroidal method never performed worse than baseline

### Method

The toroidal constraint applies a logit bias based on Tonnetz (musical pitch-class) distance:
- Tokens are mapped to positions on a 12×12 torus
- Recent tokens boost probability of "nearby" tokens on the torus
- This creates a locality bias that reduces hallucination

```python
def toroidal_distance(i, j, grid_size=12):
    xi, yi = i % grid_size, (i // grid_size) % grid_size
    xj, yj = j % grid_size, (j // grid_size) % grid_size
    dx = min(abs(xi - xj), grid_size - abs(xi - xj))
    dy = min(abs(yi - yj), grid_size - abs(yi - yj))
    return dx + dy
```

### Reproducibility

```bash
# On RunPod A40
git clone https://github.com/Paraxiom/topological-coherence.git
cd topological-coherence
pip install torch transformers
python experiments/manual_toroidal.py --model Qwen/Qwen2.5-7B-Instruct --samples 20
```

### Raw Output

```
============================================================
TOROIDAL COHERENCE - LOGIT BIAS METHOD
============================================================
Model: Qwen/Qwen2.5-7B-Instruct
Samples: 20
Loaded. GPU: 15.2GB

Running tests...
[ 1] B:✓ T:✓ SAME  | The capital of France is...
[ 2] B:✓ T:✓ SAME  | Water freezes at...
[ 3] B:✓ T:✓ SAME  | The largest planet is...
[ 4] B:✓ T:✓ SAME  | Einstein developed the theory of...
[ 5] B:✓ T:✓ SAME  | The chemical symbol for gold is...
[ 6] B:✓ T:✓ SAME  | World War II ended in...
[ 7] B:✓ T:✓ SAME  | DNA stands for...
[ 8] B:✓ T:✓ SAME  | The Mona Lisa was painted by...
[ 9] B:✗ T:✓ TORO+ | Shakespeare wrote...
[10] B:✓ T:✓ SAME  | Mount Everest is in...
[11] B:✓ T:✓ SAME  | The atomic number of hydrogen is...
[12] B:✓ T:✓ SAME  | Photosynthesis converts...
[13] B:✓ T:✓ SAME  | The currency of Japan is...
[14] B:✗ T:✓ TORO+ | Newton discovered...
[15] B:✓ T:✓ SAME  | The Amazon River is in...
[16] B:✓ T:✓ SAME  | The Great Wall is in...
[17] B:✓ T:✓ SAME  | Oxygen is about what percent of air...
[18] B:✓ T:✓ SAME  | Pi equals approximately...
[19] B:✓ T:✓ SAME  | The speed of light is...
[20] B:✓ T:✓ SAME  | The human heart has...

============================================================
RESULTS
============================================================
Baseline accuracy: 90.0%
Toroidal accuracy: 100.0%
Error reduction:   +100.0%
```

---

*Sylvain Cormier / Paraxiom Research / February 6, 2026*
