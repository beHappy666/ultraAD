"""
Paper-to-Plugin Skill Core Module

核心模块包含论文解析、创新点提取、插件设计和代码生成的主要功能。
"""

from .paper_analyzer import PaperAnalyzer, PaperContent, Section
from .innovation_extractor import (
    InnovationExtractor,
    Innovation,
    FeasibilityReport,
    Contribution
)
from .plugin_designer import (
    PluginDesigner,
    PluginDesign,
    Interface,
    Dependency
)
from .plugin_generator import (
    PluginGenerator,
    GeneratedPlugin
)
from .integration_tester import (
    IntegrationTester,
    TestReport,
    PhaseResult,
    TestConfig
)
from .report_generator import (
    ReportGenerator,
    FinalReport
)
from .types import (
    PaperSource,
    InnovationType,
    PluginType,
    TestPhase,
    TestStatus
)

__all__ = [
    # Paper Analysis
    'PaperAnalyzer',
    'PaperContent',
    'Section',

    # Innovation Extraction
    'InnovationExtractor',
    'Innovation',
    'FeasibilityReport',
    'Contribution',

    # Plugin Design
    'PluginDesigner',
    'PluginDesign',
    'Interface',
    'Dependency',

    # Code Generation
    'PluginGenerator',
    'GeneratedPlugin',

    # Testing
    'IntegrationTester',
    'TestReport',
    'PhaseResult',
    'TestConfig',

    # Reporting
    'ReportGenerator',
    'FinalReport',

    # Types
    'PaperSource',
    'InnovationType',
    'PluginType',
    'TestPhase',
    'TestStatus',
]

__version__ = '0.1.0'
