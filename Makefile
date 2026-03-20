.PHONY: setup data debug train test viz help

ENV_NAME := vad_debug
CONFIG := configs/VAD_tiny_debug.py

help:
	@echo "ultraAD - End-to-End Autonomous Driving Debug System"
	@echo ""
	@echo "Usage:"
	@echo "  make setup          - Create conda environment"
	@echo "  make data           - Prepare nuScenes data"
	@echo "  make debug          - Debug single sample"
	@echo "  make train          - Train with debug hooks"
	@echo "  make test           - Test model"
	@echo "  make viz            - Visualize results"
	@echo ""
	@echo "Variables:"
	@echo "  CONFIG=$(CONFIG)"
	@echo "  ENV_NAME=$(ENV_NAME)"

setup:
	bash scripts/setup_env.sh

data:
	bash scripts/prepare_data.sh data/nuscenes

debug:
	conda run -n $(ENV_NAME) python scripts/debug_one_sample.py $(CONFIG)

train:
	conda run -n $(ENV_NAME) python scripts/train_debug.py $(CONFIG)

test:
	bash scripts/test.sh $(CONFIG)

train-vad:
	bash scripts/train.sh $(CONFIG)

viz:
	@echo "Visualization outputs in logs/debug_one_sample/"
	@ls -la logs/debug_one_sample/ 2>/dev/null || echo "Run 'make debug' first."
