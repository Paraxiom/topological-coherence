#!/bin/bash
# TOROIDAL COHERENCE - FULL VALIDATION SETUP
# Run this on a fresh RunPod (RTX 4090, 100GB SSD)

set -e

echo "============================================"
echo "TOROIDAL COHERENCE VALIDATION SETUP"
echo "============================================"

# Clone repo
cd /workspace
if [ -d "topological-coherence" ]; then
    echo "Repo exists, pulling latest..."
    cd topological-coherence
    git pull origin main
else
    echo "Cloning repo..."
    git clone https://github.com/Paraxiom/topological-coherence.git
    cd topological-coherence
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install torch transformers accelerate

# Check GPU
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Run validation
echo ""
echo "============================================"
echo "STARTING 100-SAMPLE VALIDATION"
echo "============================================"
python experiments/run_full_validation.py --model both --samples 100

echo ""
echo "============================================"
echo "DONE! Results saved in ./results/"
echo "============================================"
ls -la results/
