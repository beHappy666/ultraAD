#!/bin/bash
# Prepare nuScenes data for VAD
# Usage: bash scripts/prepare_data.sh [data_root]

DATA_ROOT="${1:-data/nuscenes}"
CANBUS_ROOT="$(dirname $DATA_ROOT)"
EXTRA_TAG="nuscenes"

echo "============================================="
echo " nuScenes Data Preparation for VAD"
echo "============================================="
echo " Data root: $DATA_ROOT"
echo " Canbus root: $CANBUS_ROOT"
echo ""

# Check if data exists
if [ ! -d "$DATA_ROOT" ]; then
    echo "[ERROR] nuScenes data not found at $DATA_ROOT"
    echo "Please download nuScenes and place it there."
    echo ""
    echo "Expected structure:"
    echo "  $DATA_ROOT/"
    echo "    v1.0-trainval/"
    echo "    v1.0-test/"
    echo "    samples/"
    echo "    sweeps/"
    exit 1
fi

cd third_party/VAD

echo "[1/2] Creating nuScenes infos (trainval + test)..."
python tools/create_data.py nuscenes \
    --root-path $DATA_ROOT \
    --canbus $CANBUS_ROOT \
    --version v1.0 \
    --out-dir $DATA_ROOT \
    --extra-tag $EXTRA_TAG \
    --max-sweeps 10

echo ""
echo "[2/2] Checking generated files..."
for f in vad_nuscenes_infos_temporal_train.pkl vad_nuscenes_infos_temporal_val.pkl; do
    if [ -f "$DATA_ROOT/$f" ]; then
        echo "  OK: $DATA_ROOT/$f"
    else
        echo "  MISSING: $DATA_ROOT/$f"
    fi
done

echo ""
echo "Data preparation complete!"
