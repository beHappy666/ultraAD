"""
Integration Tester Module

Responsible for testing plugins through progressive phases: sandbox, module, integration, and system tests.
"""

import os
import sys
import json
import time
import shutil
import subprocess
import tempfile
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum

from .plugin_generator import GeneratedPlugin
from .plugin_designer import TestPlan


class TestStatus(Enum):
    """Test execution status"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class TestPhase(Enum):
    """Testing phase"""
    SANDBOX = "sandbox"
    MODULE = "module"
    INTEGRATION = "integration"
    SYSTEM = "system"


@dataclass
class TestMetric:
    """Individual test metric"""
    name: str
    value: Union[int, float, str, bool]
    unit: Optional[str] = None
    threshold: Optional[float] = None
    passed: bool = True
    details: Optional[Dict[str, Any]] = None


@dataclass
class TestCaseResult:
    """Result of a single test case"""
    name: str
    status: TestStatus
    duration: float  # seconds
    message: str = ""
    traceback: Optional[str] = None
    metrics: List[TestMetric] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)  # Paths to artifacts
    logs: List[str] = field(default_factory=list)


@dataclass
class PhaseResult:
    """Result of a testing phase"""
    phase: TestPhase
    status: TestStatus = TestStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: float = 0.0  # seconds
    test_cases: List[TestCaseResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    artifacts_dir: Optional[str] = None
    logs: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def passed_count(self) -> int:
        return sum(1 for tc in self.test_cases if tc.status == TestStatus.PASSED)

    @property
    def failed_count(self) -> int:
        return sum(1 for tc in self.test_cases if tc.status == TestStatus.FAILED)

    @property
    def skipped_count(self) -> int:
        return sum(1 for tc in self.test_cases if tc.status == TestStatus.SKIPPED)


@dataclass
class TestReport:
    """Complete test report"""
    plugin_name: str
    plugin_version: str
    test_date: datetime = field(default_factory=datetime.now)
    phases: Dict[TestPhase, PhaseResult] = field(default_factory=dict)
    overall_status: TestStatus = TestStatus.PENDING
    summary: Dict[str, Any] = field(default_factory=dict)
    artifacts_base_dir: Optional[str] = None
    recommendations: List[str] = field(default_factory=list)

    def get_phase(self, phase: TestPhase) -> Optional[PhaseResult]:
        return self.phases.get(phase)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'plugin_name': self.plugin_name,
            'plugin_version': self.plugin_version,
            'test_date': self.test_date.isoformat(),
            'phases': {
                phase.value: {
                    'status': result.status.value,
                    'duration': result.duration,
                    'test_cases': len(result.test_cases),
                    'passed': result.passed_count,
                    'failed': result.failed_count
                }
                for phase, result in self.phases.items()
            },
            'overall_status': self.overall_status.value,
            'summary': self.summary
        }


@dataclass
class TestConfig:
    """Test configuration"""
    # Directories
    sandbox_root: str = field(default_factory=lambda: tempfile.gettempdir())
    integration_artifacts_dir: str = "experiments/integration"
    keep_artifacts: bool = True
    keep_sandbox: bool = False

    # Timeouts (seconds)
    sandbox_timeout: int = 600
    module_test_timeout: int = 300
    integration_timeout: int = 600
    system_test_timeout: int = 1200

    # Test options
    run_sandbox: bool = True
    run_module_tests: bool = True
    run_integration_tests: bool = True
    run_system_tests: bool = True
    fail_fast: bool = False

    # Environment
    python_version: str = "3.8"
    cuda_required: bool = False
    ultraad_path: Optional[str] = None

    # Reporting
    verbose: bool = True
    generate_html_report: bool = True
    generate_junit_xml: bool = True


class IntegrationTester:
    """
    Main class for testing plugins through progressive phases.
    """

    TEST_PHASES = [TestPhase.SANDBOX, TestPhase.MODULE,
                   TestPhase.INTEGRATION, TestPhase.SYSTEM]

    def __init__(self, config: Optional[TestConfig] = None):
        """
        Initialize the tester.

        Args:
            config: Test configuration
        """
        self.config = config or TestConfig()
        self.current_report: Optional[TestReport] = None

    def run(self, plugin: GeneratedPlugin,
            phases: Optional[List[TestPhase]] = None) -> TestReport:
        """
        Run tests for a plugin.

        Args:
            plugin: Plugin to test
            phases: Specific phases to run (default: all)

        Returns:
            Test report
        """
        # Create report
        report = TestReport(
            plugin_name=plugin.name,
            plugin_version=plugin.version,
            artifacts_base_dir=os.path.join(
                self.config.integration_artifacts_dir,
                f"{plugin.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        )
        self.current_report = report

        # Create artifacts directory
        os.makedirs(report.artifacts_base_dir, exist_ok=True)

        # Determine phases to run
        phases_to_run = phases or self.TEST_PHASES

        # Run each phase
        for phase in phases_to_run:
            print(f"\n{'='*60}")
            print(f"Running phase: {phase.value}")
            print('='*60)

            phase_result = self._run_phase(phase, plugin)
            report.phases[phase] = phase_result

            # Check if phase failed and fail_fast is enabled
            if phase_result.status == TestStatus.FAILED and self.config.fail_fast:
                print(f"\n❌ Phase '{phase.value}' failed. Stopping tests.")
                break

        # Calculate overall status
        report.overall_status = self._calculate_overall_status(report)

        # Generate summary
        report.summary = self._generate_summary(report)

        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)

        # Save report
        self._save_report(report)

        return report

    def _run_phase(self, phase: TestPhase,
                  plugin: GeneratedPlugin) -> PhaseResult:
        """Run a specific test phase"""
        result = PhaseResult(phase=phase)
        result.start_time = datetime.now()
        result.artifacts_dir = os.path.join(
            self.current_report.artifacts_base_dir,
            phase.value
        )
        os.makedirs(result.artifacts_dir, exist_ok=True)

        try:
            if phase == TestPhase.SANDBOX:
                result = self._run_sandbox_test(result, plugin)
            elif phase == TestPhase.MODULE:
                result = self._run_module_test(result, plugin)
            elif phase == TestPhase.INTEGRATION:
                result = self._run_integration_test(result, plugin)
            elif phase == TestPhase.SYSTEM:
                result = self._run_system_test(result, plugin)

            result.status = TestStatus.PASSED if result.failed_count == 0 else TestStatus.FAILED

        except Exception as e:
            result.status = TestStatus.ERROR
            result.errors.append(str(e))
            import traceback
            result.errors.append(traceback.format_exc())

        finally:
            result.end_time = datetime.now()
            result.duration = (result.end_time - result.start_time).total_seconds()

        return result

    def _run_sandbox_test(self, result: PhaseResult,
                         plugin: GeneratedPlugin) -> PhaseResult:
        """Run sandbox tests in isolated environment"""
        import tempfile
        import venv

        # Create sandbox directory
        sandbox_dir = os.path.join(
            self.config.sandbox_root,
            f"sandbox_{plugin.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(sandbox_dir, exist_ok=True)

        try:
            # Create virtual environment
            venv_dir = os.path.join(sandbox_dir, 'venv')
            venv.create(venv_dir, with_pip=True)

            pip = os.path.join(venv_dir, 'bin', 'pip')
            if not os.path.exists(pip):
                pip = os.path.join(venv_dir, 'Scripts', 'pip.exe')

            # Install dependencies
            test_case = TestCaseResult(
                name='install_dependencies',
                status=TestStatus.RUNNING,
                duration=0.0,
                message='Installing dependencies...'
            )

            start_time = time.time()

            # Create requirements file
            reqs_file = os.path.join(sandbox_dir, 'requirements.txt')
            with open(reqs_file, 'w') as f:
                for dep in plugin.design.dependencies:
                    if not dep.optional:
                        f.write(f"{dep.name}{dep.version_spec}\n")

            # Install requirements
            result_code = subprocess.call(
                [pip, 'install', '-r', reqs_file],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=self.config.sandbox_timeout
            )

            test_case.duration = time.time() - start_time

            if result_code == 0:
                test_case.status = TestStatus.PASSED
                test_case.message = 'Dependencies installed successfully'
            else:
                test_case.status = TestStatus.FAILED
                test_case.message = 'Failed to install dependencies'

            result.test_cases.append(test_case)

            # Import test
            if test_case.status == TestStatus.PASSED:
                import_test = self._run_import_test(plugin, pip, venv_dir, sandbox_dir)
                result.test_cases.append(import_test)

            # Unit tests
            if any(tc.status == TestStatus.PASSED for tc in result.test_cases):
                unit_tests = self._run_unit_tests(plugin, pip, venv_dir, sandbox_dir)
                result.test_cases.extend(unit_tests)

        except subprocess.TimeoutExpired:
            error_test = TestCaseResult(
                name='sandbox_execution',
                status=TestStatus.ERROR,
                duration=0.0,
                message=f'Sandbox test timed out after {self.config.sandbox_timeout} seconds'
            )
            result.test_cases.append(error_test)

        except Exception as e:
            error_test = TestCaseResult(
                name='sandbox_execution',
                status=TestStatus.ERROR,
                duration=0.0,
                message=f'Sandbox test error: {str(e)}'
            )
            result.test_cases.append(error_test)

        finally:
            # Cleanup sandbox
            if not self.config.keep_sandbox:
                shutil.rmtree(sandbox_dir, ignore_errors=True)

        return result

    def _run_import_test(self, plugin: GeneratedPlugin, pip: str,
                        venv_dir: str, sandbox_dir: str) -> TestCaseResult:
        """Test that plugin can be imported"""
        test_case = TestCaseResult(
            name='import_test',
            status=TestStatus.RUNNING,
            duration=0.0,
            message='Testing plugin import...'
        )

        start_time = time.time()

        # Create test script
        test_script = f'''
import sys
sys.path.insert(0, '{plugin.output_dir}')

try:
    import {plugin.name}
    print(f"Successfully imported {plugin.name}")
    print(f"Version: {{getattr({plugin.name}, '__version__', 'unknown')}}")
    sys.exit(0)
except Exception as e:
    print(f"Import error: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''
        script_path = os.path.join(sandbox_dir, 'import_test.py')
        with open(script_path, 'w') as f:
            f.write(test_script)

        # Run test
        python = os.path.join(venv_dir, 'bin', 'python')
        if not os.path.exists(python):
            python = os.path.join(venv_dir, 'Scripts', 'python.exe')

        try:
            result = subprocess.run(
                [python, script_path],
                capture_output=True,
                text=True,
                timeout=60
            )

            test_case.duration = time.time() - start_time

            if result.returncode == 0:
                test_case.status = TestStatus.PASSED
                test_case.message = 'Plugin imported successfully'
                test_case.logs = result.stdout.split('\n')
            else:
                test_case.status = TestStatus.FAILED
                test_case.message = f'Import failed: {result.stderr}'
                test_case.logs = result.stderr.split('\n')

        except subprocess.TimeoutExpired:
            test_case.duration = time.time() - start_time
            test_case.status = TestStatus.ERROR
            test_case.message = 'Import test timed out'

        return test_case

    def _run_unit_tests(self, plugin: GeneratedPlugin, pip: str,
                       venv_dir: str, sandbox_dir: str) -> List[TestCaseResult]:
        """Run unit tests"""
        results = []

        # Install pytest
        subprocess.run([pip, 'install', 'pytest', '-q'],
                     capture_output=True)

        # Find test files
        test_dir = os.path.join(plugin.output_dir, 'tests')
        if not os.path.exists(test_dir):
            return results

        # Run tests
        python = os.path.join(venv_dir, 'bin', 'python')
        if not os.path.exists(python):
            python = os.path.join(venv_dir, 'Scripts', 'python.exe')

        result = subprocess.run(
            [python, '-m', 'pytest', test_dir, '-v', '--tb=short',
             f'--junitxml={sandbox_dir}/junit.xml'],
            capture_output=True,
            text=True,
            timeout=300
        )

        # Parse results
        for line in result.stdout.split('\n'):
            if 'PASSED' in line:
                test_name = line.split(' ')[0]
                results.append(TestCaseResult(
                    name=test_name,
                    status=TestStatus.PASSED,
                    duration=0.0,
                    message='Test passed'
                ))
            elif 'FAILED' in line:
                test_name = line.split(' ')[0]
                results.append(TestCaseResult(
                    name=test_name,
                    status=TestStatus.FAILED,
                    duration=0.0,
                    message='Test failed'
                ))

        return results

    def _run_module_test(self, result: PhaseResult,
                        plugin: GeneratedPlugin) -> PhaseResult:
        """Run module-level tests"""
        # Module tests are already run in sandbox
        # This phase can include additional module-specific tests

        test_case = TestCaseResult(
            name='module_validation',
            status=TestStatus.RUNNING,
            duration=0.0,
            message='Validating module structure...'
        )

        start_time = time.time()

        try:
            # Check package structure
            package_dir = os.path.join(plugin.output_dir, self._to_snake_case(plugin.name))

            required_files = ['__init__.py', 'plugin.py', 'core.py']
            missing_files = []

            for file in required_files:
                if not os.path.exists(os.path.join(package_dir, file)):
                    missing_files.append(file)

            if missing_files:
                test_case.status = TestStatus.FAILED
                test_case.message = f'Missing required files: {missing_files}'
            else:
                test_case.status = TestStatus.PASSED
                test_case.message = 'Module structure validated'

        except Exception as e:
            test_case.status = TestStatus.ERROR
            test_case.message = f'Validation error: {str(e)}'

        test_case.duration = time.time() - start_time
        result.test_cases.append(test_case)

        return result

    def _run_integration_test(self, result: PhaseResult,
                             plugin: GeneratedPlugin) -> PhaseResult:
        """Run integration tests with ultraAD"""
        # Check if ultraAD is available
        ultraad_path = self.config.ultraad_path

        if not ultraad_path or not os.path.exists(ultraad_path):
            test_case = TestCaseResult(
                name='ultraad_integration',
                status=TestStatus.SKIPPED,
                duration=0.0,
                message='ultraAD not available, skipping integration tests'
            )
            result.test_cases.append(test_case)
            return result

        # Run integration tests
        test_case = TestCaseResult(
            name='ultraad_integration',
            status=TestStatus.RUNNING,
            duration=0.0,
            message='Running integration tests with ultraAD...'
        )

        start_time = time.time()

        try:
            # Create test configuration
            test_config_path = os.path.join(
                result.artifacts_dir,
                'integration_config.py'
            )

            # Run ultraAD debug script
            cmd = [
                sys.executable,
                os.path.join(ultraad_path, 'scripts/debug_one_sample.py'),
                test_config_path,
                '--output-dir', os.path.join(result.artifacts_dir, 'debug_output')
            ]

            result_code = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.integration_timeout,
                cwd=ultraad_path
            )

            test_case.duration = time.time() - start_time

            if result_code.returncode == 0:
                test_case.status = TestStatus.PASSED
                test_case.message = 'Integration test passed'
            else:
                test_case.status = TestStatus.FAILED
                test_case.message = f'Integration test failed: {result_code.stderr}'

        except subprocess.TimeoutExpired:
            test_case.duration = time.time() - start_time
            test_case.status = TestStatus.ERROR
            test_case.message = 'Integration test timed out'

        except Exception as e:
            test_case.duration = time.time() - start_time
            test_case.status = TestStatus.ERROR
            test_case.message = f'Integration test error: {str(e)}'

        result.test_cases.append(test_case)

        return result

    def _run_system_test(self, result: PhaseResult,
                        plugin: GeneratedPlugin) -> PhaseResult:
        """Run system-level tests"""
        test_case = TestCaseResult(
            name='system_validation',
            status=TestStatus.RUNNING,
            duration=0.0,
            message='Running system validation...'
        )

        start_time = time.time()

        try:
            # Validate plugin can be installed
            cmd = [sys.executable, '-m', 'pip', 'install', '-e', plugin.output_dir]
            install_result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )

            if install_result.returncode != 0:
                test_case.status = TestStatus.FAILED
                test_case.message = f'Installation failed: {install_result.stderr}'
            else:
                test_case.status = TestStatus.PASSED
                test_case.message = 'System validation passed'

        except Exception as e:
            test_case.status = TestStatus.ERROR
            test_case.message = f'System test error: {str(e)}'

        test_case.duration = time.time() - start_time
        result.test_cases.append(test_case)

        return result

    @staticmethod
    def _to_snake_case(text: str) -> str:
        """Convert text to snake_case"""
        text = re.sub(r'[\s\-]+', '_', text)
        text = re.sub(r'([a-z])([A-Z])', r'\1_\2', text)
        text = text.lower()
        text = re.sub(r'[^a-z0-9_]', '', text)
        return text

    def _calculate_overall_status(self, report: TestReport) -> TestStatus:
        """Calculate overall test status"""
        if not report.phases:
            return TestStatus.PENDING

        all_passed = True
        any_failed = False

        for phase_result in report.phases.values():
            if phase_result.status == TestStatus.FAILED:
                any_failed = True
                all_passed = False
            elif phase_result.status != TestStatus.PASSED:
                all_passed = False

        if any_failed:
            return TestStatus.FAILED
        elif all_passed:
            return TestStatus.PASSED
        else:
            return TestStatus.ERROR

    def _generate_summary(self, report: TestReport) -> Dict[str, Any]:
        """Generate test summary"""
        summary = {
            'total_phases': len(report.phases),
            'passed_phases': 0,
            'failed_phases': 0,
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'skipped_tests': 0,
            'total_duration': 0.0
        }

        for phase_result in report.phases.values():
            if phase_result.status == TestStatus.PASSED:
                summary['passed_phases'] += 1
            elif phase_result.status == TestStatus.FAILED:
                summary['failed_phases'] += 1

            summary['total_tests'] += len(phase_result.test_cases)
            summary['passed_tests'] += phase_result.passed_count
            summary['failed_tests'] += phase_result.failed_count
            summary['skipped_tests'] += phase_result.skipped_count
            summary['total_duration'] += phase_result.duration

        return summary

    def _generate_recommendations(self, report: TestReport) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []

        if report.overall_status == TestStatus.PASSED:
            recommendations.append("All tests passed. The plugin is ready for integration.")
        elif report.overall_status == TestStatus.FAILED:
            # Analyze failures
            for phase, result in report.phases.items():
                if result.status == TestStatus.FAILED:
                    if phase == TestPhase.SANDBOX:
                        recommendations.append(
                            "Sandbox tests failed. Check dependencies and basic functionality."
                        )
                    elif phase == TestPhase.INTEGRATION:
                        recommendations.append(
                            "Integration tests failed. Check compatibility with ultraAD."
                        )

        # Add general recommendations
        summary = report.summary
        if summary.get('failed_tests', 0) > 0:
            recommendations.append(
                f"Address {summary['failed_tests']} failed test(s) before integration."
            )

        if summary.get('skipped_tests', 0) > 0:
            recommendations.append(
                f"Review {summary['skipped_tests']} skipped test(s) for relevance."
            )

        return recommendations

    def _save_report(self, report: TestReport) -> None:
        """Save test report to disk"""
        if not report.artifacts_base_dir:
            return

        # Save as JSON
        json_path = os.path.join(report.artifacts_base_dir, 'test_report.json')
        with open(json_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)

        # Save as Markdown
        md_path = os.path.join(report.artifacts_base_dir, 'test_report.md')
        with open(md_path, 'w') as f:
            f.write(self._generate_markdown_report(report))

    def _generate_markdown_report(self, report: TestReport) -> str:
        """Generate Markdown format test report"""
        lines = [
            f"# Test Report: {report.plugin_name}",
            "",
            f"**Version:** {report.plugin_version}  ",
            f"**Date:** {report.test_date.strftime('%Y-%m-%d %H:%M:%S')}  ",
            f"**Overall Status:** {report.overall_status.value.upper()}",
            "",
            "## Summary",
            "",
            f"- **Total Phases:** {report.summary.get('total_phases', 0)}",
            f"- **Passed Phases:** {report.summary.get('passed_phases', 0)}",
            f"- **Failed Phases:** {report.summary.get('failed_phases', 0)}",
            f"- **Total Duration:** {report.summary.get('total_duration', 0):.2f} seconds",
            "",
            "## Phase Results",
            ""
        ]

        for phase, phase_result in report.phases.items():
            lines.extend([
                f"### {phase.value.upper()}",
                "",
                f"**Status:** {phase_result.status.value}  ",
                f"**Duration:** {phase_result.duration:.2f} seconds  ",
                f"**Tests:** {len(phase_result.test_cases)} total, "
                f"{phase_result.passed_count} passed, "
                f"{phase_result.failed_count} failed, "
                f"{phase_result.skipped_count} skipped",
                ""
            ])

            if phase_result.test_cases:
                lines.extend([
                    "#### Test Details",
                    "",
                    "| Test | Status | Duration | Message |",
                    "|------|--------|----------|----------|"
                ])

                for tc in phase_result.test_cases:
                    lines.append(
                        f"| {tc.name} | {tc.status.value} | "
                        f"{tc.duration:.2f}s | {tc.message[:50]}... |"
                    )

                lines.append("")

            if phase_result.errors:
                lines.extend([
                    "#### Errors",
                    ""
                ])
                for error in phase_result.errors:
                    lines.extend([
                        f"```",
                        error,
                        f"```",
                        ""
                    ])

        if report.recommendations:
            lines.extend([
                "## Recommendations",
                ""
            ])
            for i, rec in enumerate(report.recommendations, 1):
                lines.append(f"{i}. {rec}")
            lines.append("")

        lines.extend([
            "---",
            "",
            f"*Report generated by paper-to-plugin integration tester*",
            f"*Artifacts: {report.artifacts_base_dir}*"
        ])

        return '\n'.join(lines)


# Convenience function
def test_plugin(plugin: GeneratedPlugin,
               config: Optional[TestConfig] = None,
               phases: Optional[List[TestPhase]] = None) -> TestReport:
    """
    Convenience function to test a plugin.

    Args:
        plugin: Plugin to test
        config: Test configuration
        phases: Specific phases to run

    Returns:
        Test report
    """
    tester = IntegrationTester(config=config)
    return tester.run(plugin, phases=phases)
