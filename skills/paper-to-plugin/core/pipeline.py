"""
Main Pipeline Module

Orchestrates the entire paper-to-plugin workflow.
"""

import os
import time
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

from .paper_analyzer import PaperAnalyzer, PaperContent
from .innovation_extractor import InnovationExtractor, Innovation, FeasibilityReport
from .plugin_designer import PluginDesigner, PluginDesign
from .plugin_generator import PluginGenerator, GeneratedPlugin
from .integration_tester import IntegrationTester, TestReport, TestConfig
from .report_generator import ReportGenerator, FinalReport
from .types import PaperSource


class PaperToPluginPipeline:
    """
    Orchestrates the entire workflow from paper analysis to plugin generation and testing.
    """

    def __init__(self, output_dir: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the pipeline.

        Args:
            output_dir: Root directory for all generated artifacts
            config: Pipeline configuration
        """
        self.output_dir = output_dir
        self.config = config or {}
        self.start_time = time.time()

        # Initialize components
        self.paper_analyzer = PaperAnalyzer(cache_dir=os.path.join(output_dir, 'cache'))
        self.innovation_extractor = InnovationExtractor()
        self.plugin_designer = PluginDesigner()
        self.plugin_generator = PluginGenerator(
            template_dir=self.config.get('template_dir')
        )
        self.integration_tester = IntegrationTester(
            config=TestConfig(**self.config.get('test_config', {}))
        )
        self.report_generator = ReportGenerator(
            output_dir=os.path.join(output_dir, 'reports')
        )

        # Initialize final report
        self.final_report = FinalReport(
            report_id=str(uuid.uuid4()),
            generated_at=datetime.now(),
            output_directory=output_dir
        )

    def run(self, source: str,
            source_type: Optional[PaperSource] = None) -> FinalReport:
        """
        Run the full pipeline.

        Args:
            source: Paper source (URL, path, or text)
            source_type: Type of source

        Returns:
            Final report of the pipeline execution
        """
        self.final_report.summary.status = "running"

        try:
            # 1. Analyze Paper
            print("--- 1. Analyzing Paper ---")
            paper_content = self.paper_analyzer.analyze(source, source_type)
            self.final_report.paper_content = paper_content
            print(f"  -> Title: {paper_content.metadata.title}")

            # 2. Extract Innovations
            print("\n--- 2. Extracting Innovations ---")
            innovations = self.innovation_extractor.extract(paper_content)
            self.final_report.innovations = innovations
            self.final_report.summary.innovations_found = len(innovations)
            print(f"  -> Found {len(innovations)} potential innovations.")

            # 3. Evaluate Feasibility
            print("\n--- 3. Evaluating Feasibility ---")
            for innovation in innovations:
                feasibility = self.innovation_extractor.evaluate_feasibility(innovation)
                self.final_report.feasibility_reports[innovation.id] = feasibility
                print(f"  -> Innovation '{innovation.name[:30]}...' feasibility: {feasibility.overall_feasibility.value}")

            # Select innovations to implement (for now, take the first one)
            selected_innovations = innovations[:1] if innovations else []
            self.final_report.summary.innovations_selected = len(selected_innovations)

            # 4. Design & Generate Plugins
            print("\n--- 4. Designing & Generating Plugins ---")
            for innovation in selected_innovations:
                # Design
                plugin_design = self.plugin_designer.design(innovation)
                self.final_report.plugin_designs.append(plugin_design)
                print(f"  -> Designed plugin: {plugin_design.name}")

                # Generate
                plugin_output_dir = os.path.join(self.output_dir, 'plugins', plugin_design.name)
                generated_plugin = self.plugin_generator.generate(plugin_design, plugin_output_dir)
                generated_plugin.write_all()
                self.final_report.generated_plugins.append(generated_plugin)
                self.final_report.summary.plugins_generated += 1
                print(f"  -> Generated plugin package at: {plugin_output_dir}")

            # 5. Test Plugins
            print("\n--- 5. Testing Plugins ---")
            for plugin in self.final_report.generated_plugins:
                test_report = self.integration_tester.run(plugin)
                self.final_report.test_reports.append(test_report)
                print(f"  -> Test for {plugin.name} completed with status: {test_report.overall_status.value}")

            # Update test summary
            self.final_report.summary.tests_passed = sum(
                tr.summary.get('passed_tests', 0) for tr in self.final_report.test_reports
            )
            self.final_report.summary.tests_failed = sum(
                tr.summary.get('failed_tests', 0) for tr in self.final_report.test_reports
            )

            # 6. Generate Final Report
            print("\n--- 6. Generating Final Report ---")
            self.final_report.summary.execution_time = time.time() - self.start_time
            self.final_report.summary.status = "completed"
            report_paths = self.report_generator.generate(self.final_report)
            self.final_report.artifact_paths.extend(report_paths)
            print(f"  -> Final report generated at: {self.report_generator.output_dir}")

            print("\n--- Pipeline Finished ---")

        except Exception as e:
            print(f"\n--- Pipeline Failed ---")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            self.final_report.summary.status = "failed"

        return self.final_report
