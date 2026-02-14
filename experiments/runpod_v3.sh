#!/bin/bash
# =============================================================================
# Ouro + Toroidal v3 — RunPod GPU Runner
# =============================================================================
#
# One-liner from a RunPod pod (PyTorch template):
#
#   curl -sSL https://raw.githubusercontent.com/Paraxiom/topological-coherence/main/experiments/runpod_v3.sh | bash
#
# Or manually:
#   git clone https://github.com/Paraxiom/topological-coherence.git
#   cd topological-coherence/experiments
#   bash runpod_v3.sh
#
# =============================================================================

set -e

echo "=============================================="
echo "Ouro-1.4B + Toroidal Attention v3"
echo "Light Strength Sweep on GPU"
echo "=============================================="

# Check GPU
echo ""
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || echo "No GPU detected"
echo ""

# If not already in the repo, clone it
if [ ! -f "ouro_toroidal_v3.py" ]; then
    if [ ! -d "topological-coherence" ]; then
        echo "Cloning repo..."
        git clone https://github.com/Paraxiom/topological-coherence.git
    fi
    cd topological-coherence/experiments
fi

# Install minimal deps (RunPod PyTorch templates already have torch)
echo "Installing dependencies..."
# Pin transformers<5 — Ouro's custom rope config breaks on transformers 5.x
pip install -q "transformers>=4.36.0,<5.0.0" numpy
echo ""

# Verify
echo "Verifying setup..."
python3 -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
import transformers
print(f'Transformers {transformers.__version__}')
"
echo ""

# Make results dir
mkdir -p ../results

# Run the experiment
echo "Starting v3 experiment..."
echo "Expected time: ~30 min on T4, ~15 min on A40/A100"
echo ""
python3 ouro_toroidal_v3.py 2>&1 | tee ../results/v3_run.log

echo ""
echo "=============================================="
echo "Done! Results in ../results/"
echo "=============================================="
ls -la ../results/ouro_toroidal_v3_*.json 2>/dev/null
