"""Unified configuration management using Hydra."""

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import yaml
from omegaconf import OmegaConf, DictConfig


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "vad_tiny"
    backbone: str = "resnet50"
    checkpoint: Optional[str] = None
    freeze_backbone: bool = False

    # BEVFormer specific
    bev_h: int = 200
    bev_w: int = 200
    bev_embed_dim: int = 256

    # VAD specific
    use_temporal: bool = True
    fut_ts: int = 6
    fut_mode: int = 6


@dataclass
class DataConfig:
    """Data configuration."""
    dataset: str = "nuscenes"
    data_root: str = "data/nuscenes"

    # Dataloader
    batch_size: int = 1
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = False

    # Preprocessing
    img_scale: List[int] = field(default_factory=lambda: [1600, 900])
    multiscale_mode: str = "range"

    # Augmentation
    use_augmentation: bool = True
    random_flip: bool = True
    random_crop: bool = True


@dataclass
class TrainerConfig:
    """Trainer configuration."""
    max_epochs: int = 20

    # Optimizer
    optimizer: str = "adamw"
    lr: float = 2e-4
    weight_decay: float = 0.01

    # Scheduler
    scheduler: str = "cosine"
    warmup_epochs: int = 2
    warmup_lr: float = 2e-6

    # Mixed precision
    use_amp: bool = True
    amp_opt_level: str = "O1"

    # Gradient clipping
    grad_clip: float = 35.0

    # Checkpointing
    save_interval: int = 1
    max_keep_ckpts: int = 3

    # Evaluation
    eval_interval: int = 1


@dataclass
class DebugConfig:
    """Debug configuration."""
    enabled: bool = True
    log_level: str = "INFO"

    # Debug hooks
    check_nan: bool = True
    check_inf: bool = True
    check_interval: int = 50

    # Visualization
    visualize_bev: bool = True
    visualize_pipeline: bool = False
    save_interval: int = 100

    # Profiling
    profile_memory: bool = False
    profile_time: bool = False

    # AI Doctor
    enable_ai_doctor: bool = True
    auto_fix: bool = False


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)

    # Experiment
    name: str = "experiment"
    work_dir: str = "work_dirs"
    seed: int = 0

    def __post_init__(self):
        """Validate and setup configuration."""
        # Setup work directory
        self.work_dir = os.path.join(self.work_dir, self.name)
        os.makedirs(self.work_dir, exist_ok=True)

        # Set random seed
        if self.seed is not None:
            import random
            import numpy as np
            import torch
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "model": self.model.__dict__,
            "data": self.data.__dict__,
            "trainer": self.trainer.__dict__,
            "debug": self.debug.__dict__,
            "name": self.name,
            "work_dir": self.work_dir,
            "seed": self.seed,
        }

    def save(self, path: str):
        """Save config to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def from_file(cls, path: str) -> "Config":
        """Load config from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        """Create config from dictionary."""
        model_config = ModelConfig(**config_dict.get("model", {}))
        data_config = DataConfig(**config_dict.get("data", {}))
        trainer_config = TrainerConfig(**config_dict.get("trainer", {}))
        debug_config = DebugConfig(**config_dict.get("debug", {}))

        return cls(
            model=model_config,
            data=data_config,
            trainer=trainer_config,
            debug=debug_config,
            name=config_dict.get("name", "experiment"),
            work_dir=config_dict.get("work_dir", "work_dirs"),
            seed=config_dict.get("seed", 0),
        )


def load_config(config_path: str = None, overrides: dict = None) -> Config:
    """Load configuration with optional overrides.

    Args:
        config_path: Path to config file
        overrides: Dictionary of config overrides

    Returns:
        Config object
    """
    if config_path and os.path.exists(config_path):
        config = Config.from_file(config_path)
    else:
        config = Config()

    # Apply overrides
    if overrides:
        for key, value in overrides.items():
            parts = key.split('.')
            obj = config
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)

    return config