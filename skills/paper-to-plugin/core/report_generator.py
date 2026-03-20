"""
Report Generator Module

Responsible for generating comprehensive reports from the paper-to-plugin pipeline.
"""

import os
import json
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .paper_analyzer import PaperContent
from .innovation_extractor import Innovation, FeasibilityReport
from .plugin_designer import PluginDesign
from .plugin_generator import GeneratedPlugin
from .integration_tester import TestReport


@dataclass
class PipelineSummary:
    """Summary of the paper-to-plugin pipeline execution"""
    paper_title: str = ""
    paper_url: str = ""
    innovations_found: int = 0
    innovations_selected: int = 0
    plugins_generated: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    execution_time: float = 0.0
    status: str = "pending"  # pending, running, completed, failed


@dataclass
class FinalReport:
    """Complete final report from paper-to-plugin pipeline"""
    # Metadata
    report_id: str
    generated_at: datetime
    pipeline_version: str = "0.1.0"

    # Input
    paper_content: Optional[PaperContent] = None

    # Analysis
    innovations: List[Innovation] = field(default_factory=list)
    feasibility_reports: Dict[str, FeasibilityReport] = field(default_factory=dict)

    # Design & Generation
    plugin_designs: List[PluginDesign] = field(default_factory=list)
    generated_plugins: List[GeneratedPlugin] = field(default_factory=list)

    # Testing
    test_reports: List[TestReport] = field(default_factory=list)

    # Summary
    summary: PipelineSummary = field(default_factory=PipelineSummary)

    # Artifacts
    output_directory: str = ""
    artifact_paths: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary"""
        return {
            'report_id': self.report_id,
            'generated_at': self.generated_at.isoformat(),
            'pipeline_version': self.pipeline_version,
            'paper': {
                'title': self.paper_content.metadata.title if self.paper_content else None,
                'abstract': self.paper_content.abstract if self.paper_content else None,
            } if self.paper_content else None,
            'innovations': [inv.to_dict() for inv in self.innovations],
            'plugins_generated': len(self.generated_plugins),
            'tests_summary': {
                'total': sum(len(tr.phases) for tr in self.test_reports),
                'passed': sum(
                    1 for tr in self.test_reports
                    for ph in tr.phases.values()
                    if ph.status.value == 'passed'
                )
            },
            'summary': {
                'innovations_found': self.summary.innovations_found,
                'plugins_generated': self.summary.plugins_generated,
                'tests_passed': self.summary.tests_passed,
                'execution_time': self.summary.execution_time,
                'status': self.summary.status
            },
            'output_directory': self.output_directory,
            'artifacts': self.artifact_paths
        }


class ReportGenerator:
    """
    Main class for generating comprehensive reports.
    """

    def __init__(self, output_dir: str):
        """
        Initialize the generator.

        Args:
            output_dir: Output directory for reports
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate(self, report: FinalReport) -> List[str]:
        """
        Generate all report formats.

        Args:
            report: Final report to generate from

        Returns:
            List of generated file paths
        """
        generated_files = []

        # JSON report
        json_path = os.path.join(self.output_dir, 'report.json')
        self._generate_json_report(report, json_path)
        generated_files.append(json_path)

        # Markdown report
        md_path = os.path.join(self.output_dir, 'report.md')
        self._generate_markdown_report(report, md_path)
        generated_files.append(md_path)

        # HTML report
        html_path = os.path.join(self.output_dir, 'report.html')
        self._generate_html_report(report, html_path)
        generated_files.append(html_path)

        # Design documents
        for i, design in enumerate(report.plugin_designs):
            design_doc_path = os.path.join(
                self.output_dir,
                f'design_{design.name}_{i+1}.md'
            )
            self._generate_design_document(design, design_doc_path)
            generated_files.append(design_doc_path)

        # Integration guides
        for i, plugin in enumerate(report.generated_plugins):
            guide_path = os.path.join(
                self.output_dir,
                f'integration_guide_{plugin.name}_{i+1}.md'
            )
            self._generate_integration_guide(plugin, guide_path)
            generated_files.append(guide_path)

        return generated_files

    def _generate_json_report(self, report: FinalReport, path: str) -> None:
        """Generate JSON format report"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)

    def _generate_markdown_report(self, report: FinalReport, path: str) -> None:
        """Generate Markdown format report"""
        lines = [
            f"# Paper-to-Plugin Pipeline Report",
            "",
            f"**Report ID:** {report.report_id}  ",
            f"**Generated:** {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}  ",
            f"**Pipeline Version:** {report.pipeline_version}",
            "",
            "---",
            "",
            "## Executive Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Paper Title | {report.paper_content.metadata.title if report.paper_content else 'N/A'} |",
            f"| Innovations Found | {report.summary.innovations_found} |",
            f"| Plugins Generated | {report.summary.plugins_generated} |",
            f"| Tests Passed | {report.summary.tests_passed} |",
            f"| Tests Failed | {report.summary.tests_failed} |",
            f"| Execution Time | {report.summary.execution_time:.2f}s |",
            f"| Overall Status | {report.summary.status} |",
            "",
            "---",
            "",
            "## Innovations",
            ""
        ]

        for i, innovation in enumerate(report.innovations, 1):
            lines.extend([
                f"### {i}. {innovation.name}",
                "",
                f"**Type:** {innovation.innovation_type.value}  ",
                f"**Source:** {innovation.source_paper} (Section: {innovation.source_section})  ",
                f"**Confidence:** {innovation.confidence_score:.2f}",
                "",
                f"**Description:**  ",
                f"{innovation.description}",
                "",
                "**Expected Benefits:**  "
            ])

            for benefit in innovation.expected_benefits:
                lines.append(f"- {benefit}")

            lines.append("")

            # Add feasibility report if available
            if innovation.id in report.feasibility_reports:
                feasibility = report.feasibility_reports[innovation.id]
                lines.extend([
                    "**Feasibility Assessment:**  ",
                    f"- Overall Feasibility: {feasibility.overall_feasibility.value}",
                    f"- Technical Complexity: {feasibility.technical_complexity}",
                    f"- Estimated Effort: {feasibility.implementation_effort}",
                    ""
                ])

            lines.append("---")
            lines.append("")

        lines.extend([
            "## Generated Plugins",
            ""
        ])

        for i, plugin in enumerate(report.generated_plugins, 1):
            lines.extend([
                f"### {i}. {plugin.name}",
                "",
                f"**Version:** {plugin.version}  ",
                f"**Output Directory:** {plugin.output_dir}",
                "",
                "**Files Generated:**  "
            ])

            for file in plugin.files[:20]:  # Show first 20 files
                lines.append(f"- `{file.path}`")

            if len(plugin.files) > 20:
                lines.append(f"- ... and {len(plugin.files) - 20} more files")

            lines.append("")

            # Add test results if available
            test_report = next(
                (tr for tr in report.test_reports if tr.plugin_name == plugin.name),
                None
            )
            if test_report:
                lines.extend([
                    "**Test Results:**  ",
                    f"- Overall Status: {test_report.overall_status.value}",
                    f"- Total Tests: {sum(len(phase.test_cases) for phase in test_report.phases.values())}",
                    ""
                ])

            lines.append("---")
            lines.append("")

        # Add artifacts section
        lines.extend([
            "## Artifacts",
            "",
            f"All generated artifacts are located in: `{self.output_dir}`",
            "",
            "### Generated Files",
            ""
        ])

        for artifact in report.artifact_paths:
            lines.append(f"- `{artifact}`")

        lines.append("")

        # Add recommendations
        if report.test_reports:
            lines.extend([
                "## Recommendations",
                ""
            ])

            all_recommendations = []
            for test_report in report.test_reports:
                all_recommendations.extend(test_report.recommendations)

            # Remove duplicates while preserving order
            seen = set()
            unique_recommendations = []
            for rec in all_recommendations:
                if rec not in seen:
                    seen.add(rec)
                    unique_recommendations.append(rec)

            for i, rec in enumerate(unique_recommendations, 1):
                lines.append(f"{i}. {rec}")

            lines.append("")

        # Footer
        lines.extend([
            "---",
            "",
            f"*Report generated by paper-to-plugin v{report.pipeline_version}*  ",
            f"*Generated on: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}*"
        ])

        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

    def _generate_html_report(self, report: FinalReport, path: str) -> None:
        """Generate HTML format report"""
        # Simple HTML report that can be enhanced later
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Paper-to-Plugin Report - {report.plugin_name if hasattr(report, 'plugin_name') else 'Unknown'}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .card {{
            background: white;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .card h2 {{
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .status-badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9em;
        }}
        .status-passed {{
            background: #d4edda;
            color: #155724;
        }}
        .status-failed {{
            background: #f8d7da;
            color: #721c24;
        }}
        .status-pending {{
            background: #fff3cd;
            color: #856404;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        .metric-label {{
            color: #666;
            margin-top: 5px;
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            margin-top: 30px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Paper-to-Plugin Pipeline Report</h1>
        <p>Automated paper analysis and plugin generation</p>
    </div>

    <div class="card">
        <h2>Executive Summary</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">{len(report.innovations)}</div>
                <div class="metric-label">Innovations Found</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{len(report.generated_plugins)}</div>
                <div class="metric-label">Plugins Generated</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{sum(tr.summary.get('passed_tests', 0) for tr in report.test_reports)}</div>
                <div class="metric-label">Tests Passed</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{report.summary.execution_time:.1f}s</div>
                <div class="metric-label">Execution Time</div>
            </div>
        </div>
    </div>

    <div class="card">
        <h2>Status Overview</h2>
        <p>
            <span class="status-badge status-{report.summary.status}">
                {report.summary.status.upper()}
            </span>
        </p>
    </div>

    <div class="footer">
        <p>Generated by paper-to-plugin v{report.pipeline_version}</p>
        <p>{report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
</body>
</html>"""

        with open(path, 'w', encoding='utf-8') as f:
            f.write(html)

    def _generate_design_document(self, design: PluginDesign, path: str) -> None:
        """Generate design document for a plugin"""
        lines = [
            f"# {design.name} Design Document",
            "",
            f"**Version:** {design.version}  ",
            f"**Type:** {design.plugin_type.value}  ",
            f"**Base Class:** {design.base_class}",
            "",
            "## Description",
            "",
            design.description,
            "",
            "## Architecture",
            "",
            "### Interfaces",
            ""
        ]

        for interface in design.interfaces:
            lines.extend([
                f"#### {interface.name}",
                "",
                f"- **Type:** {interface.type}",
                f"- **Description:** {interface.description}",
                f"- **Required:** {interface.required}",
                "",
                "**Schema:**",
                "```json",
                json.dumps(interface.schema, indent=2),
                "```",
                ""
            ])

        lines.extend([
            "### Dependencies",
            "",
            "| Package | Version | Required | Purpose |",
            "|---------|---------|----------|---------|"
        ])

        for dep in design.dependencies:
            req = "Yes" if not dep.optional else "No"
            lines.append(f"| {dep.name} | {dep.version_spec} | {req} | {dep.purpose or 'N/A'} |")

        lines.extend([
            "",
            "### Configuration",
            "",
            "| Option | Type | Default | Required | Description |",
            "|--------|------|---------|----------|-------------|"
        ])

        for schema in design.config_schema:
            req = "Yes" if schema.required else "No"
            default = str(schema.default) if schema.default is not None else "None"
            lines.append(f"| {schema.name} | {schema.type} | {default} | {req} | {schema.description} |")

        lines.extend([
            "",
            "## Testing",
            "",
            "### Test Plan",
            "",
            f"**Unit Tests:** {len(design.test_plan.unit_tests)}  ",
            f"**Integration Tests:** {len(design.test_plan.integration_tests)}  ",
            f"**System Tests:** {len(design.test_plan.system_tests)}  ",
            f"**Performance Tests:** {len(design.test_plan.performance_tests)}",
            "",
            "## Implementation Notes",
            "",
            "### Base Classes",
            "",
            f"This plugin extends `{design.base_class}`."
        ]

        if design.mixins:
            lines.extend([
                "",
                "### Mixins",
                "",
                "The following mixins are included:"
            ])
            for mixin in design.mixins:
                lines.append(f"- {mixin}")

        lines.extend([
            "",
            "## License",
            "",
            f"This plugin is licensed under the {design.license} license.",
            "",
            f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        ])

        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

    def _generate_integration_guide(self, plugin: GeneratedPlugin, path: str) -> None:
        """Generate integration guide for a plugin"""
        lines = [
            f"# {plugin.name} Integration Guide",
            "",
            f"This guide will help you integrate the `{plugin.name}` plugin into your ultraAD system.",
            "",
            "## Table of Contents",
            "",
            "1. [Prerequisites](#prerequisites)",
            "2. [Installation](#installation)",
            "3. [Configuration](#configuration)",
            "4. [Verification](#verification)",
            "5. [Troubleshooting](#troubleshooting)",
            "",
            "## Prerequisites",
            "",
            "### System Requirements",
            "",
            "- Python 3.8 or higher",
            "- PyTorch 1.9 or higher",
            "- ultraAD system installed and configured",
            "",
            "### Dependencies",
            "",
            "The following dependencies will be installed automatically:"
        ]

        # Add dependencies from design
        if plugin.design:
            for dep in plugin.design.dependencies[:10]:  # Show first 10
                opt = " (optional)" if dep.optional else ""
                lines.append(f"- {dep.name}{dep.version_spec}{opt}")

        lines.extend([
            "",
            "## Installation",
            "",
            "### Method 1: Direct Installation (Recommended)",
            "",
            "```bash",
            f"cd {plugin.output_dir}",
            "pip install -e .",
            "```",
            "",
            "### Method 2: Installation from Source",
            "",
            "```bash",
            f"cd {plugin.output_dir}",
            "python setup.py install",
            "```",
            "",
            "### Method 3: Development Installation",
            "",
            "```bash",
            f"cd {plugin.output_dir}",
            "pip install -e .[dev]",
            "```",
            "",
            "## Configuration",
            "",
            "### ultraAD Configuration",
            "",
            "Add the following to your ultraAD configuration file:"
        ])

        if plugin.design:
            lines.extend([
                "",
                "```python",
                "# configs/feature_flags.py",
                "",
                "ENABLED_PLUGINS = [",
                f"    '{plugin.design.name}',",
                "    # ... other plugins",
                "]",
                "",
                "PLUGIN_CONFIGS = {",
                f"    '{plugin.design.name}': {{",
                "        'enabled': True,",
                "        'debug_mode': False,",
            ])

            for schema in plugin.design.config_schema[:5]:
                default_val = repr(schema.default) if schema.default is not None else "None"
                lines.append(f"        '{schema.name}': {default_val},")

            lines.extend([
                "    }",
                "}",
                "```",
                ""
            ])

        lines.extend([
            "### Plugin-specific Configuration",
            "",
            f"The `{plugin.name}` plugin can be configured through:"
        ])

        if plugin.design and plugin.design.config_schema:
            lines.extend([
                "",
                "| Option | Type | Default | Description |",
                "|--------|------|---------|-------------|"
            ])

            for schema in plugin.design.config_schema:
                default = str(schema.default) if schema.default is not None else "None"
                lines.append(
                    f"| {schema.name} | {schema.type} | {default} | {schema.description} |"
                )

        lines.extend([
            "",
            "## Verification",
            "",
            "### 1. Check Installation",
            "",
            "```bash",
            f"python -c \"import {plugin.name}; print({plugin.name}.__version__)\"",
            "```",
            "",
            "Expected output: `", plugin.version, "`",
            "",
            "### 2. Run Tests",
            "",
            "```bash",
            f"cd {plugin.output_dir}",
            "python -m pytest tests/ -v",
            "```",
            "",
            "### 3. Integration Test",
            "",
            "```bash",
            "# In your ultraAD environment",
            "python scripts/debug_one_sample.py configs/VAD_tiny_debug.py",
            "```",
            "",
            "## Troubleshooting",
            "",
            "### Common Issues",
            "",
            "#### Issue 1: Import Error",
            "",
            "**Symptom:** `ModuleNotFoundError: No module named '", plugin.name, "'`",
            "",
            "**Solution:**",
            "```bash",
            f"cd {plugin.output_dir}",
            "pip install -e .",
            "```",
            "",
            "#### Issue 2: Dependency Conflict",
            "",
            "**Symptom:** Version conflicts with existing packages",
            "",
            "**Solution:**",
            "```bash",
            "# Create a fresh virtual environment",
            "python -m venv venv_plugin",
            "source venv_plugin/bin/activate  # On Windows: venv_plugin\\Scripts\\activate",
            f"cd {plugin.output_dir}",
            "pip install -e .",
            "```",
            "",
            "#### Issue 3: ultraAD Integration Failure",
            "",
            "**Symptom:** Plugin not recognized by ultraAD",
            "",
            "**Solution:**",
            "1. Check that the plugin is in `ENABLED_PLUGINS` list",
            "2. Verify the plugin configuration in `PLUGIN_CONFIGS`",
            "3. Restart the ultraAD application",
            "4. Check the logs for error messages",
            "",
            "### Getting Help",
            "",
            "If you encounter issues not covered here:"
        ])

        if plugin.design:
            lines.extend([
                "",
                "1. Check the design document: `design_",
                f"{plugin.design.name}_1.md`",
                "2. Review the API documentation in the `docs/` directory",
                "3. Run the test suite: `python -m pytest tests/ -v`",
                "4. Check the logs in the artifacts directory"
            ])

        lines.extend([
            "",
            "## Next Steps",
            "",
            "After successful integration:"
        ])

        if plugin.design:
            lines.extend([
                "",
                f"1. **Customize Configuration**: Adjust the plugin settings in `PLUGIN_CONFIGS['{plugin.design.name}']`",
                "2. **Monitor Performance**: Check the plugin's impact on system performance",
                "3. **Fine-tune Parameters**: Optimize hyperparameters for your specific use case",
                "4. **Contribute**: Consider contributing improvements back to the community"
            ])

        lines.extend([
            "",
            "---",
            "",
            "## Appendix",
            "",
            "### A. Plugin Structure",
            "",
            "```",
            plugin.name,
            "/",
        ])

        if plugin.files:
            structure = {}
            for file in plugin.files:
                parts = file.path.split('/')
                current = structure
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = None

            def print_tree(d, prefix=''):
                lines = []
                items = list(d.items())
                for i, (k, v) in enumerate(items):
                    is_last = i == len(items) - 1
                    connector = '└── ' if is_last else '├── '
                    lines.append(f"{prefix}{connector}{k}")
                    if isinstance(v, dict):
                        extension = '    ' if is_last else '│   '
                        lines.extend(print_tree(v, prefix + extension))
                return lines

            lines.extend(print_tree(structure))

        lines.extend([
            "```",
            "",
            "### B. Configuration Reference",
            "",
            "All configuration options can be set via:"
        ])

        if plugin.design:
            lines.extend([
                "",
                "1. **Python Configuration**: In your `feature_flags.py`:",
                "```python",
                "PLUGIN_CONFIGS = {",
                f"    '{plugin.design.name}': {{",
            ])

            for schema in plugin.design.config_schema[:5]:
                default = repr(schema.default) if schema.default is not None else "None"
                lines.append(f"        '{schema.name}': {default},")

            lines.extend([
                "    }",
                "}",
                "```",
                "",
                "2. **JSON Configuration**: Create a `config.json` file:",
                "```json",
                "{",
                f"  \"{plugin.design.name}\": {{",
            ])

            for schema in plugin.design.config_schema[:5]:
                if schema.default is not None:
                    if isinstance(schema.default, str):
                        lines.append(f"    \"{schema.name}\": \"{schema.default}\",")
                    else:
                        lines.append(f"    \"{schema.name}\": {schema.default},")

            lines.extend([
                "  }",
                "}",
                "```",
                "",
                "3. **Environment Variables**: Use the format `PLUGIN_<NAME>_<OPTION>`:",
                "```bash",
            ])

            for schema in plugin.design.config_schema[:3]:
                env_name = f"PLUGIN_{plugin.design.name.upper()}_{schema.name.upper()}"
                default = str(schema.default) if schema.default is not None else "<value>"
                lines.append(f"export {env_name}={default}")

            lines.extend([
                "```",
                ""
            ])

        lines.extend([
            "### C. API Reference",
            "",
            "#### Main Classes",
            "",
        ])

        if plugin.design:
            class_name = self._to_pascal_case(plugin.design.name)
            lines.extend([
                f"##### `{class_name}`",
                "",
                f"Main plugin class that extends `{plugin.design.base_class}`.",
                "",
                "**Constructor:**",
                "```python",
                f"plugin = {class_name}(",
                "    config=None,",
                "    logger=None,",
                ")",
                "```",
                "",
                "**Methods:**",
                "",
                "| Method | Description |",
                "|--------|-------------|"
            ])

            # Add method descriptions based on plugin type
            if plugin.design.plugin_type.value == "training":
                lines.extend([
                    "| `train()` | Start training process |",
                    "| `validate()` | Run validation |",
                    "| `save_checkpoint()` | Save model checkpoint |",
                    "| `load_checkpoint()` | Load model checkpoint |"
                ])
            elif plugin.design.plugin_type.value == "perception":
                lines.extend([
                    "| `process()` | Process input data |",
                    "| `detect()` | Run detection |",
                    "| `extract_features()` | Extract features |",
                    "| `post_process()` | Post-process results |"
                ])
            else:
                lines.extend([
                    "| `initialize()` | Initialize plugin |",
                    "| `process()` | Process input |",
                    "| `forward()` | Forward pass |",
                    "| `cleanup()` | Cleanup resources |"
                ])

            lines.extend([
                "",
                "**Events:**",
                "",
                "The plugin emits the following events:"
            ])

            if plugin.design.plugin_type.value == "training":
                lines.extend([
                    "- `training_start`: Training has started",
                    "- `epoch_end`: Epoch completed",
                    "- `validation_end`: Validation completed",
                    "- `training_end`: Training has ended",
                    "- `checkpoint_saved`: Checkpoint saved"
                ])
            else:
                lines.extend([
                    "- `initialization_complete`: Plugin initialized",
                    "- `processing_start`: Processing started",
                    "- `processing_complete`: Processing completed",
                    "- `error`: Error occurred"
                ])

        lines.extend([
            "",
            "### D. Troubleshooting",
            "",
            "#### Common Issues and Solutions",
            "",
            "**Issue:** Plugin not loading  ",
            "**Solution:** Check that the plugin is in the `ENABLED_PLUGINS` list and the configuration is correct.",
            "",
            "**Issue:** Import errors  ",
            "**Solution:** Ensure all dependencies are installed. Run `pip install -e .` in the plugin directory.",
            "",
            "**Issue:** Configuration not loading  ",
            "**Solution:** Verify the configuration format in `PLUGIN_CONFIGS`. Check for syntax errors.",
            "",
            "**Issue:** Tests failing  ",
            "**Solution:** Run tests individually to identify the failing test. Check the test logs for details.",
            "",
            "#### Getting Help",
            "",
            "If you encounter issues not covered here:"
        ])

        if plugin.design:
            lines.extend([
                "",
                "1. Check the design document: `docs/design_" + plugin.design.name + "_1.md`",
                "2. Review the API documentation in the `docs/` directory",
                "3. Run the test suite: `python -m pytest tests/ -v`",
                "4. Check the logs in the artifacts directory",
                "5. Enable debug mode in the configuration for verbose logging"
            ])

        lines.extend([
            "",
            "#### Debug Mode",
            "",
            "To enable debug mode:"
        ])

        if plugin.design:
            lines.extend([
                "",
                "```python",
                "# In your configuration",
                "PLUGIN_CONFIGS = {",
                f"    '{plugin.design.name}': {{",
                "        'debug_mode': True,",
                "        'log_level': 'DEBUG',",
                "    }",
                "}",
                "```",
                "",
                "Or via environment variable:",
                f"```bash",
                f"export PLUGIN_{plugin.design.name.upper()}_DEBUG_MODE=true",
                f"```"
            ])

        lines.extend([
            "",
            "### E. Performance Considerations",
            "",
            "#### Resource Requirements",
            "",
        ])

        if plugin.design and plugin.design.config_schema:
            lines.extend([
                "The following resources are recommended:"
            ])

            # Add resource requirements based on plugin type
            if plugin.design.plugin_type.value == "training":
                lines.extend([
                    "",
                    "- **GPU**: NVIDIA GPU with at least 8GB VRAM",
                    "- **RAM**: 16GB minimum, 32GB recommended",
                    "- **Storage**: 50GB free space for checkpoints and data",
                    "- **CPU**: Multi-core processor (8+ cores recommended)"
                ])
            elif plugin.design.plugin_type.value == "perception":
                lines.extend([
                    "",
                    "- **GPU**: NVIDIA GPU with at least 4GB VRAM (for real-time inference)",
                    "- **RAM**: 8GB minimum, 16GB recommended",
                    "- **Storage**: 10GB free space for models",
                    "- **CPU**: Multi-core processor (4+ cores)"
                ])
            else:
                lines.extend([
                    "",
                    "- **RAM**: 8GB minimum",
                    "- **Storage**: 5GB free space",
                    "- **CPU**: Multi-core processor"
                ])

        lines.extend([
            "",
            "#### Optimization Tips",
            "",
            "1. **Batch Processing**: When processing multiple items, use batch operations for better throughput",
            "2. **Memory Management**: Release unused resources promptly to free up memory",
            "3. **Caching**: Use appropriate caching strategies for frequently accessed data",
            "4. **Parallel Processing**: Utilize multi-processing for CPU-bound tasks",
            "",
            "#### Monitoring",
            "",
            "Monitor the following metrics during operation:"
        ])

        if plugin.design:
            if plugin.design.plugin_type.value == "training":
                lines.extend([
                    "- Training loss and validation metrics",
                    "- GPU utilization and memory usage",
                    "- Checkpoint save times",
                    "- Data loading throughput"
                ])
            elif plugin.design.plugin_type.value == "perception":
                lines.extend([
                    "- Inference latency (per image)",
                    "- Detection accuracy metrics",
                    "- GPU utilization",
                    "- Memory usage"
                ])
            else:
                lines.extend([
                    "- Execution time",
                    "- Memory usage",
                    "- Error rates"
                ])

        lines.extend([
            "",
            "### F. Changelog",
            "",
            f"## Version {plugin.version}",
            "",
            "### Initial Release",
            "",
            "- First version of the plugin",
            "- Core functionality implemented",
            "- Documentation and tests added",
            "",
            "### Known Issues",
            "",
            "None at this time."
        ])

        # Save license section
        if plugin.design:
            lines.extend([
                "",
                "## License",
                "",
                f"This plugin is released under the {plugin.design.license} license.",
                "",
                "### Third-Party Licenses",
                "",
                "This plugin uses the following third-party libraries:"
            ])

            # List dependencies with their licenses
            for dep in plugin.design.dependencies[:5]:
                lines.append(f"- {dep.name}: See package for license details")

        lines.extend([
            "",
            "### Contributing",
            "",
            "Contributions are welcome! Please follow these guidelines:"
        ])

        if plugin.design:
            lines.extend([
                "",
                "1. **Code Style**: Follow PEP 8 for Python code",
                "2. **Documentation**: Update documentation for any changes",
                "3. **Testing**: Add tests for new features",
                f"4. **Commits**: Use clear, descriptive commit messages",
                "5. **Pull Requests**: Include a clear description of changes"
            ])

        lines.extend([
            "",
            "### Contact",
            "",
            "For questions, issues, or contributions:"
        ])

        if plugin.design:
            lines.extend([
                "",
                "- **Issues**: Please open an issue in the project repository",
                f"- **Email**: Contact the author at {plugin.design.author}",
                "- **Documentation**: See the `docs/` directory for detailed documentation"
            ])

        lines.extend([
            "",
            "---",
            "",
            "*This integration guide was automatically generated by the paper-to-plugin pipeline.*",
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            ""
        ])

        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

    @staticmethod
    def _to_pascal_case(text: str) -> str:
        """Convert text to PascalCase"""
        words = re.split(r'[\s\-_]+', text)
        return ''.join(word.capitalize() for word in words)


# Convenience function
def generate_report(
    paper_content: Optional[PaperContent] = None,
    innovations: Optional[List[Innovation]] = None,
    generated_plugins: Optional[List[GeneratedPlugin]] = None,
    test_reports: Optional[List[TestReport]] = None,
    output_dir: str = "./reports"
) -> FinalReport:
    """
    Convenience function to generate a complete report.

    Args:
        paper_content: Parsed paper content
        innovations: Extracted innovations
        generated_plugins: Generated plugins
        test_reports: Test reports
        output_dir: Output directory

    Returns:
        Complete final report
    """
    import uuid

    report = FinalReport(
        report_id=str(uuid.uuid4()),
        generated_at=datetime.now(),
        paper_content=paper_content,
        innovations=innovations or [],
        generated_plugins=generated_plugins or [],
        test_reports=test_reports or [],
        output_directory=output_dir
    )

    # Populate summary
    report.summary.innovations_found = len(innovations) if innovations else 0
    report.summary.plugins_generated = len(generated_plugins) if generated_plugins else 0
    report.summary.tests_passed = sum(
        sum(1 for phase in tr.phases.values() if phase.status.value == 'passed')
        for tr in (test_reports or [])
    )

    # Generate files
    generator = ReportGenerator(output_dir)
    report.artifact_paths = generator.generate(report)

    return report
