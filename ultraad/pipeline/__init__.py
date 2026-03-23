"""自动化流水线模块 - 从论文到报告的一键式解决方案"""

from .auto_pipeline import AutoPipeline
from .paper_analyzer import PaperAnalyzer, PaperContent
from .innovation_extractor import InnovationExtractor, Innovation
from .code_generator import CodeGenerator, GeneratedCode
from .vad_integrator import VADIntegrator
from .experiment_runner import ExperimentRunner, ExperimentResults
from .performance_comparator import PerformanceComparator, ComparisonResult
from .report_generator import ReportGenerator, Report

__all__ = [
    'AutoPipeline',
    'PaperAnalyzer',
    'PaperContent',
    'InnovationExtractor',
    'Innovation',
    'CodeGenerator',
    'GeneratedCode',
    'VADIntegrator',
    'ExperimentRunner',
    'ExperimentResults',
    'PerformanceComparator',
    'ComparisonResult',
    'ReportGenerator',
    'Report',
]
