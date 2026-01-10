#!/bin/bash
# =============================================================================
# Topological Coherence Experiment - RunPod Runner
# =============================================================================
#
# This script runs the full experiment on RunPod A100.
#
# Setup on RunPod:
#   1. Create a new pod with:
#      - Template: RunPod Pytorch 2.0
#      - GPU: A100 (40GB or 80GB)
#      - Disk: 50GB minimum
#
#   2. SSH into the pod and run:
#      git clone https://github.com/Paraxiom/topological-coherence.git
#      cd topological-coherence/experiments
#      bash run_on_runpod.sh
#
# =============================================================================

set -e

echo "=============================================="
echo "Topological Coherence Experiment"
echo "=============================================="
echo ""

# Check GPU
echo "Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt
pip install -q einops  # Phi-2 needs this
echo ""

# Verify imports
echo "Verifying imports..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "from topological_attention import TopologicalAttentionMask; print('Topological attention: OK')"
echo ""

# Test mask generation
echo "Testing mask generation..."
python topological_attention.py
echo ""

# Create results directory
RESULTS_DIR="./results_$(date +%Y%m%d_%H%M%S)"
mkdir -p $RESULTS_DIR
echo "Results will be saved to: $RESULTS_DIR"
echo ""

# Run all four conditions
echo "=============================================="
echo "Starting full experiment (4 conditions)"
echo "=============================================="
echo ""
echo "Conditions:"
echo "  1. baseline     - Standard attention (control)"
echo "  2. local_window - Local window (tests locality alone)"
echo "  3. random       - Random mask (negative control)"
echo "  4. toroidal     - Toroidal mask (treatment)"
echo ""

# Run experiment
python train_phi2.py \
    --run_all \
    --output_dir $RESULTS_DIR \
    --epochs 3 \
    --batch_size 4 \
    --lr 2e-5 \
    --decay 0.3 \
    --grid_size 12

echo ""
echo "=============================================="
echo "Experiment Complete!"
echo "=============================================="
echo ""
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "Key files:"
echo "  - $RESULTS_DIR/combined_results.json"
echo "  - $RESULTS_DIR/baseline/results.json"
echo "  - $RESULTS_DIR/local_window/results.json"
echo "  - $RESULTS_DIR/random/results.json"
echo "  - $RESULTS_DIR/toroidal/results.json"
echo ""

# Print summary
echo "Final Results:"
cat $RESULTS_DIR/combined_results.json
