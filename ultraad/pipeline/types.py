"""流水线核心类型定义"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class PaperSourceType(Enum):
    """论文来源类型"""
    ARXIV = "arxiv"
    PDF = "pdf"
    URL = "url"


class InnovationCategory(Enum):
    """创新点类别"""
    ARCHITECTURE = "architecture"
    ATTENTION = "attention"
    TEMPORAL = "temporal"
    PLANNING = "planning"
    EFFICIENCY = "efficiency"
    DATA = "data"


@dataclass
class PaperContent:
    """论文内容"""
    paper_id: str
    title: str
    authors: List[str]
    abstract: str
    full_text: str
    sections: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Innovation:
    """创新点"""
    id: str
    name: str
    description: str
    category: InnovationCategory
    implementation_details: Optional[str] = None
    feasibility_score: float = 0.0
    complexity_score: float = 0.0
    impact_score: float = 0.0


@dataclass
class GeneratedCode:
    """生成的代码"""
    innovation_id: str
    module_name: str
    files: Dict[str, str]  # 文件路径 -> 内容
    dependencies: List[str] = field(default_factory=list)


@dataclass
class Metrics:
    """性能指标"""
    mAP: Optional[float] = None
    mATE: Optional[float] = None
    mASE: Optional[float] = None
    mAOE: Optional[float] = None
    mAVE: Optional[float] = None
    mAAE: Optional[float] = None
    NDS: Optional[float] = None
    fps: Optional[float] = None
    gpu_memory_gb: Optional[float] = None
    training_time_hours: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'mAP': self.mAP,
            'mATE': self.mATE,
            'mASE': self.mASE,
            'mAOE': self.mAOE,
            'mAVE': self.mAVE,
            'mAAE': self.mAAE,
            'NDS': self.NDS,
            'fps': self.fps,
            'gpu_memory_gb': self.gpu_memory_gb,
            'training_time_hours': self.training_time_hours,
        }


@dataclass
class MetricsDiff:
    """指标差异"""
    mAP_delta: Optional[float] = None
    mAP_pct: Optional[float] = None
    NDS_delta: Optional[float] = None
    NDS_pct: Optional[float] = None
    fps_delta: Optional[float] = None
    fps_pct: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'mAP_delta': self.mAP_delta,
            'mAP_pct': self.mAP_pct,
            'NDS_delta': self.NDS_delta,
            'NDS_pct': self.NDS_pct,
            'fps_delta': self.fps_delta,
            'fps_pct': self.fps_pct,
        }


@dataclass
class ExperimentResults:
    """实验结果"""
    experiment_id: str
    config_name: str
    metrics: Metrics
    checkpoints: List[str] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


@dataclass
class ComparisonResult:
    """对比结果"""
    baseline: ExperimentResults
    experiment: ExperimentResults
    diff: MetricsDiff
    overall_improvement: str  # "positive", "negative", "neutral"
    significance: str  # "high", "medium", "low"


@dataclass
class Report:
    """最终报告"""
    report_id: str
    paper: PaperContent
    innovations: List[Innovation]
    comparison: ComparisonResult
    generated_at: datetime
    output_path: str
    summary: str
