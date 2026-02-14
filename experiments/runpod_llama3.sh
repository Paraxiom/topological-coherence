#!/bin/bash
# =============================================================================
# Llama-3-8B + TLB — RunPod GPU Runner
# =============================================================================
#
# IMPORTANT: Llama-3 is a gated model. You need a HuggingFace token:
#   export HF_TOKEN=hf_xxxxx
#
# One-liner from a RunPod pod (PyTorch template, 24GB+ VRAM):
#
#   export HF_TOKEN=hf_xxxxx && curl -sSL https://raw.githubusercontent.com/Paraxiom/topological-coherence/main/experiments/runpod_llama3.sh | bash
#
# Or manually:
#   git clone https://github.com/Paraxiom/topological-coherence.git
#   cd topological-coherence/experiments
#   export HF_TOKEN=hf_xxxxx
#   bash runpod_llama3.sh
#
# =============================================================================

set -e

echo "=============================================="
echo "Llama-3-8B + Toroidal Logit Bias (TLB)"
echo "20 configs: alpha x radius sweep"
echo "=============================================="

# Check GPU
echo ""
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || echo "No GPU detected"
echo ""

# Check HF token
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN not set. Llama-3 is a gated model."
    echo "Run: export HF_TOKEN=hf_xxxxx"
    echo "Get your token at: https://huggingface.co/settings/tokens"
    echo ""
fi

# If not already in the repo, clone it
if [ ! -f "llama3_tlb.py" ]; then
    if [ ! -d "topological-coherence" ]; then
        echo "Cloning repo..."
        git clone https://github.com/Paraxiom/topological-coherence.git
    fi
    cd topological-coherence/experiments
fi

# Install deps — Llama-3 works fine with latest transformers
echo "Installing dependencies..."
pip install -q "transformers>=4.40.0" "huggingface_hub>=0.20.0" numpy accelerate
echo ""

# Login to HuggingFace if token is set
if [ -n "$HF_TOKEN" ]; then
    echo "Logging in to HuggingFace..."
    huggingface-cli login --token "$HF_TOKEN" 2>/dev/null || python3 -c "from huggingface_hub import login; login(token='$HF_TOKEN')"
    echo ""
fi

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

# Run
echo "Starting Llama-3-8B + TLB experiment..."
echo "Expected: ~16GB VRAM, ~30-45 min for 20 configs on 4090"
echo ""
python3 llama3_tlb.py 2>&1 | tee ../results/llama3_run.log

echo ""
echo "=============================================="
echo "Done! Results in ../results/"
echo "=============================================="
ls -la ../results/llama3_tlb_*.json 2>/dev/null
