# Topological Coherence Experiments

Empirical validation of: **"Topological Constraints Reduce Hallucinations in Language Models"**

## Hypothesis

Geometric/topological constraints on attention reduce hallucinations by preventing attention from "jumping" to unrelated concepts.

## Experimental Design

### Conditions

| Condition | Description | Purpose |
|-----------|-------------|---------|
| `baseline` | Standard causal attention | Control |
| `local_window` | Local window mask (same radius, no wraparound) | Tests if it's locality or topology |
| `random` | Random sparse mask (same sparsity) | Negative control |
| `toroidal` | Toroidal/Tonnetz attention mask | **Treatment** |

### Key Insight

The `local_window` control is critical. If toroidal outperforms local_window, we can claim:

> "It's not just locality — it's topology."

### Model

- **Phi-2 (2.7B)** with LoRA fine-tuning
- Strong reasoning, fits on single A100, credible for mechanism paper

### Benchmarks

- **TruthfulQA**: Measures tendency to produce false but plausible answers
- **HaluEval**: Measures ability to distinguish factual from hallucinated content

## Quick Start

### On RunPod A100

```bash
# Clone repo
git clone https://github.com/Paraxiom/topological-coherence.git
cd topological-coherence/experiments

# Run full experiment
bash run_on_runpod.sh
```

### Single Condition

```bash
python train_phi2.py --mask_type toroidal --output_dir ./results/toroidal
```

### All Conditions

```bash
python train_phi2.py --run_all --output_dir ./results
```

## Expected Output

```
EXPERIMENT SUMMARY
============================================
Condition       TruthfulQA      HaluEval
---------------------------------------------
baseline        XX.XX%          XX.XX%
local_window    XX.XX%          XX.XX%
random          XX.XX%          XX.XX%
toroidal        XX.XX%          XX.XX%
```

## Files

- `topological_attention.py` - GPU-optimized attention masks
- `train_phi2.py` - Training and evaluation script
- `requirements.txt` - Python dependencies
- `run_on_runpod.sh` - One-command experiment runner

## Citation

```bibtex
@article{cormier2026topological,
  title={Topological Constraints for Coherent Language Models: Why Geometry Prevents Hallucination},
  author={Cormier, Sylvain},
  year={2026},
  doi={10.5281/zenodo.18168012}
}
```
