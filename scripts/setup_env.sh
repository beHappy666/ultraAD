#!/bin/bash
# Setup conda environment for VAD development
# Usage: bash scripts/setup_env.sh

set -e

ENV_NAME="vad_debug"

echo "============================================="
echo " VAD Environment Setup"
echo "============================================="

# Check conda
if ! command -v conda &> /dev/null; then
    echo "[ERROR] conda not found. Please install Anaconda/Miniconda first."
    exit 1
fi

echo "[1/6] Creating conda environment: $ENV_NAME"
conda create -y -n $ENV_NAME python=3.8

echo "[2/6] Activating environment..."
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

echo "[3/6] Installing PyTorch (CUDA 11.3)..."
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

echo "[4/6] Installing mmcv/mmdet/mmdet3d..."
pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html
pip install mmdet==2.25.0
pip install mmdet3d==1.0.0rc6
pip install mmsegmentation==0.29.0

echo "[5/6] Installing additional dependencies..."
pip install nuscenes-devkit==1.1.9
pip install numpy==1.23.5
pip install scipy==1.10.1
pip install matplotlib==3.7.1
pip install opencv-python==4.7.0.72
pip install shapely==2.0.1
pip install numba==0.57.0
pip install pillow==9.5.0

echo "[6/6] Installing debug dependencies..."
pip install tensorboard==2.13.0
pip install tqdm==4.65.0
pip install pandas==2.0.2

echo ""
echo "============================================="
echo " Environment setup complete!"
echo " Activate with: conda activate $ENV_NAME"
echo "============================================="
echo ""
echo "Next steps:"
echo "  1. Download nuScenes dataset to data/nuscenes/"
echo "  2. Run: bash scripts/prepare_data.sh"
echo "  3. Run: python scripts/debug_one_sample.py configs/VAD_tiny_debug.py"
echo "  4. Run: bash scripts/train.sh configs/VAD_tiny_debug.py"
