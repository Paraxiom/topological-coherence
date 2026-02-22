#!/bin/bash
# Karmonic-Constrained LLM Fine-Tuning — RunPod launcher
#
# Usage on RunPod:
#   nohup bash run_karmonic_llm.sh > karmonic_llm.log 2>&1 &
#
# Prerequisites: RunPod RTX 4090, ~4GB VRAM for Qwen 2.5-0.5B + LoRA
set -euo pipefail

echo "=============================================="
echo "Karmonic LLM Fine-Tuning Experiment"
echo "$(date)"
echo "=============================================="

# GPU info
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU detected"

# Install dependencies
echo ""
echo "[1/4] Installing dependencies..."
pip install -q torch transformers peft datasets accelerate tqdm numpy

# Clone or update repo
REPO_DIR="topological-coherence"
echo ""
echo "[2/4] Setting up repo..."
if [ -d "$REPO_DIR" ]; then
    cd "$REPO_DIR"
    git pull --ff-only || true
    cd ..
else
    git clone https://github.com/Paraxiom/topological-coherence.git
fi

# Run all 4 conditions
SCRIPT="$REPO_DIR/experiments/train_karmonic_llm.py"
OUTPUT_DIR="results/karmonic_llm"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "[3/4] Running all 4 conditions..."
echo ""

echo "--- Condition 1/4: baseline ---"
python "$SCRIPT" --condition baseline \
    --output_dir "$OUTPUT_DIR" \
    --train_samples 5000 \
    2>&1 | tee "$OUTPUT_DIR/baseline.log"
echo ""

echo "--- Condition 2/4: lora_only ---"
python "$SCRIPT" --condition lora_only \
    --output_dir "$OUTPUT_DIR" \
    --train_samples 5000 \
    2>&1 | tee "$OUTPUT_DIR/lora_only.log"
echo ""

echo "--- Condition 3/4: lora_karmonic ---"
python "$SCRIPT" --condition lora_karmonic \
    --output_dir "$OUTPUT_DIR" \
    --lambda_karmonic 0.05 \
    --grad_scale 0.1 \
    --train_samples 5000 \
    2>&1 | tee "$OUTPUT_DIR/lora_karmonic.log"
echo ""

echo "--- Condition 4/4: lora_karmonic_tlb ---"
python "$SCRIPT" --condition lora_karmonic_tlb \
    --output_dir "$OUTPUT_DIR" \
    --lambda_karmonic 0.05 \
    --grad_scale 0.1 \
    --train_samples 5000 \
    2>&1 | tee "$OUTPUT_DIR/lora_karmonic_tlb.log"
echo ""

# Summary
echo ""
echo "[4/4] Printing summary..."
echo "=============================================="
echo "RESULTS SUMMARY"
echo "=============================================="
echo ""

python3 -c "
import json, glob, os
results = []
for f in sorted(glob.glob('$OUTPUT_DIR/*/eval_results.json')):
    with open(f) as fh:
        results.append(json.load(fh))

print(f\"{'Condition':<22} {'MC1':>8} {'MC2':>8} {'PPL':>10} {'Train(s)':>10}\")
print('-' * 62)
for r in results:
    print(f\"{r['condition']:<22} {r['mc1_accuracy']:>8.4f} {r['mc2_score']:>8.4f} \"
          f\"{r['perplexity']:>10.2f} {r['train_time_s']:>10.1f}\")

# Save combined
with open('$OUTPUT_DIR/combined_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f\"\nCombined: $OUTPUT_DIR/combined_results.json\")
"

echo ""
echo "Done! $(date)"
echo "Results in: $OUTPUT_DIR/"
