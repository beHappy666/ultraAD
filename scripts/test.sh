#!/bin/bash
# Test VAD model
# Usage: bash scripts/test.sh [config] [checkpoint] [gpu_id]

CONFIG="${1:-configs/VAD_base_debug.py}"
CHECKPOINT="${2:-}"
GPU_ID="${3:-0}"

if [ -z "$CHECKPOINT" ]; then
    WORK_DIR="work_dirs/$(basename $CONFIG .py)"
    CHECKPOINT=$(ls -t $WORK_DIR/*.pth 2>/dev/null | head -1)
    if [ -z "$CHECKPOINT" ]; then
        echo "[ERROR] No checkpoint found. Provide path explicitly."
        exit 1
    fi
fi

echo "============================================="
echo " VAD Testing"
echo "============================================="
echo " Config: $CONFIG"
echo " Checkpoint: $CHECKPOINT"
echo " GPU: $GPU_ID"
echo ""

export CUDA_VISIBLE_DEVICES=$GPU_ID

cd third_party/VAD

python tools/test.py \
    ../../$CONFIG \
    $CHECKPOINT \
    --eval bbox \
    --show-dir ../../logs/vis_results

echo ""
echo "Testing complete. Results in: logs/vis_results"
