#!/bin/bash
# Train VAD model
# Usage: bash scripts/train.sh [config] [gpu_id]

CONFIG="${1:-configs/VAD_base_debug.py}"
GPU_ID="${2:-0}"
WORK_DIR="work_dirs/$(basename $CONFIG .py)"

echo "============================================="
echo " VAD Training"
echo "============================================="
echo " Config: $CONFIG"
echo " GPU: $GPU_ID"
echo " Work dir: $WORK_DIR"
echo ""

export CUDA_VISIBLE_DEVICES=$GPU_ID

cd third_party/VAD

python tools/train.py \
    ../../$CONFIG \
    --work-dir ../../$WORK_DIR \
    --gpus 1 \
    --seed 0

echo ""
echo "Training complete. Logs and checkpoints in: $WORK_DIR"
