#!/bin/bash
# ENIGMA-Karmonic Hybrid — RunPod launcher
# Usage: nohup bash run_enigma_karmonic.sh > enigma_karmonic.log 2>&1 &
#
# Runs all 5 conditions sequentially:
#   1. GRPO only (ENIGMA ablation baseline)
#   2. GRPO + SAMI (ENIGMA without OT)
#   3. GRPO + SAMI + OT (full ENIGMA reproduction)
#   4. GRPO + SAMI + Karmonic (our hypothesis)
#   5. GRPO + Karmonic (no SAMI)

set -e

echo "=========================================="
echo "ENIGMA-Karmonic Hybrid Experiment"
echo "$(date)"
echo "=========================================="

# Install deps
pip install -q transformers datasets peft accelerate torch tqdm

# Setup
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK_DIR="/root/karmonic_exp"
mkdir -p "$WORK_DIR"

# Check if script exists locally or in repo
if [ -f "$SCRIPT_DIR/train_enigma_karmonic.py" ]; then
    SCRIPT="$SCRIPT_DIR/train_enigma_karmonic.py"
elif [ -f "$WORK_DIR/train_enigma_karmonic.py" ]; then
    SCRIPT="$WORK_DIR/train_enigma_karmonic.py"
else
    echo "ERROR: train_enigma_karmonic.py not found"
    exit 1
fi

OUTDIR="$WORK_DIR/results/enigma_karmonic"
mkdir -p "$OUTDIR"

echo "Script: $SCRIPT"
echo "Output: $OUTDIR"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Common args
COMMON="--output_dir $OUTDIR --n_steps 200 --train_samples 500 --lambda_karmonic 0.01 --lambda_sami 0.05 --lambda_ot 0.01 --log_every 20"

# Phase 2 first: Torus detection (if script available)
if [ -f "$SCRIPT_DIR/detect_torus_structure.py" ] || [ -f "$WORK_DIR/detect_torus_structure.py" ]; then
    DETECT="${SCRIPT_DIR}/detect_torus_structure.py"
    [ -f "$DETECT" ] || DETECT="$WORK_DIR/detect_torus_structure.py"
    echo ""
    echo "=========================================="
    echo "Phase 2: Torus Detection"
    echo "$(date)"
    echo "=========================================="
    python "$DETECT" --output_dir "$WORK_DIR/results/torus_detection" --max_samples 200 2>&1 | tee "$WORK_DIR/results/torus_detection.log"
fi

# Run all 5 conditions
for COND in grpo_only grpo_sami grpo_sami_ot grpo_sami_karmonic grpo_karmonic; do
    echo ""
    echo "=========================================="
    echo "Condition: $COND"
    echo "$(date)"
    echo "=========================================="
    python "$SCRIPT" --condition "$COND" $COMMON 2>&1 | tee "$OUTDIR/${COND}.log"
done

# Print summary
echo ""
echo "=========================================="
echo "ALL CONDITIONS COMPLETE"
echo "$(date)"
echo "=========================================="

# Combine results
python -c "
import json, glob, os
results = []
for f in sorted(glob.glob('$OUTDIR/*/eval_results.json')):
    with open(f) as fh:
        results.append(json.load(fh))
with open('$OUTDIR/combined_results.json', 'w') as fh:
    json.dump(results, fh, indent=2)
print('Combined results saved to $OUTDIR/combined_results.json')
print()
print(f\"{'Condition':<24} {'MC1':>8} {'MC2':>8} {'PPL':>10}\")
print('-' * 56)
for r in results:
    print(f\"{r['condition']:<24} {r['mc1_accuracy']:>8.4f} {r['mc2_score']:>8.4f} {r['perplexity']:>10.2f}\")
"

echo ""
echo "Done! Results at: $OUTDIR/combined_results.json"
