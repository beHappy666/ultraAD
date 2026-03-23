"""ultraAD - 端到端自动驾驶算法开发调试系统."""

__version__ = "0.2.0"
__author__ = "ultraAD Team"

from ultraad.core.config import Config
from ultraad.core.trainer import SmartTrainer

__all__ = ["Config", "SmartTrainer"]