"""
Core type definitions for paper-to-plugin skill.
"""

from enum import Enum, auto
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
from datetime import datetime


class PaperSource(Enum):
    """论文来源类型"""
    ARXIV = "arxiv"
    PDF = "pdf"
    URL = "url"
    TEXT = "text"
    AUTO = "auto"


class InnovationType(Enum):
    """创新点类型"""
    ARCHITECTURE = "architecture"
    TRAINING = "training"
    PERCEPTION = "perception"
    PREDICTION = "prediction"
    PLANNING = "planning"
    EFFICIENCY = "efficiency"
    DATA = "data"
    UNKNOWN = "unknown"


class PluginType(Enum):
    """插件类型"""
    PERCEPTION = "perception"
    TRAINING = "training"
    PLANNING = "planning"
    ARCHITECTURE = "architecture"
    EFFICIENCY = "efficiency"
    DATA = "data"
    GENERIC = "generic"


class TestPhase(Enum):
    """测试阶段"""
    SANDBOX = "sandbox"
    MODULE = "module"
    INTEGRATION = "integration"
    SYSTEM = "system"


class TestStatus(Enum):
    """测试状态"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


class FeasibilityLevel(Enum):
    """可行性等级"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


# Data Classes

@dataclass
class PaperMetadata:
    """论文元数据"""
    title: str
    authors: List[str]
    abstract: str
    publication_date: Optional[datetime] = None
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    keywords: List[str] = field(default_factory=list)


@dataclass
class TestMetric:
    """测试指标"""
    name: str
    value: Union[int, float, str]
    unit: Optional[str] = None
    threshold: Optional[float] = None
    status: TestStatus = TestStatus.PENDING


@dataclass
class ValidationResult:
    """验证结果"""
    check_name: str
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None
    severity: str = "info"  # info, warning, error, critical


@dataclass
class ResourceEstimate:
    """资源估算"""
    compute_hours: float
    gpu_memory_gb: float
    storage_gb: float
    data_size_gb: float
    estimated_cost_usd: Optional[float] = None


@dataclass
class DependencySpec:
    """依赖规格"""
    name: str
    version_spec: str
    optional: bool = False
    install_command: Optional[str] = None
    purpose: Optional[str] = None
