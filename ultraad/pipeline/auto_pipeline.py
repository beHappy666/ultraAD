"""一键式自动流水线 - 从论文到报告"""

import uuid
import time
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from .paper_analyzer import PaperAnalyzer
from .innovation_extractor import InnovationExtractor
from .code_generator import CodeGenerator
from .vad_integrator import VADIntegrator
from .experiment_runner import ExperimentRunner
from .performance_comparator import PerformanceComparator
from .report_generator import ReportGenerator

from .types import Report

console = Console()


class AutoPipeline:
    """一键式自动流水线"""

    def __init__(self, output_dir: str = None, gpu_id: int = 0):
        """
        初始化流水线

        Args:
            output_dir: 输出目录
            gpu_id: GPU ID
        """
        self.output_dir = Path(output_dir) if output_dir else Path("auto_output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gpu_id = gpu_id
        self.start_time = time.time()

        # 初始化各组件
        self.analyzer = PaperAnalyzer(cache_dir=str(self.output_dir / "cache"))
        self.extractor = InnovationExtractor()
        self.generator = CodeGenerator()
        self.integrator = VADIntegrator()
        self.runner = ExperimentRunner(work_dir=str(self.output_dir / "work_dirs"), gpu_id=gpu_id)
        self.comparator = PerformanceComparator()
        self.reporter = ReportGenerator(output_dir=str(self.output_dir / "reports"))

    def run(self, paper_source: str, baseline_config: str = None,
              epochs: int = 20) -> Report:
        """
        运行完整流水线

        Args:
            paper_source: 论文路径或URL
            baseline_config: 基线配置路径
            epochs: 训练轮数

        Returns:
            最终报告
        """
        report_id = str(uuid.uuid4())[:8]

        console.print(Panel(
            f"[bold cyan]ultraAD 一键式自动流水线[/]\n"
            f"报告ID: {report_id}\n"
            f"输入: {paper_source}",
            title="AutoPipeline"
        ))

        try:
            # 步骤1: 解析论文
            with self._step_progress("解析论文", 1, 6):
                paper = self.analyzer.parse(paper_source)

            # 步骤2: 提取创新点
            with self._step_progress("提取创新点", 2, 6):
                innovations = self.extractor.extract(paper)
                best_innovation = self.extractor.select_best(innovations)

            # 步骤3: 生成代码
            with self._step_progress("生成代码", 3, 6):
                code = self.generator.generate(best_innovation)

            # 步骤4: 集成到 VAD
            with self._step_progress("集成到 VAD", 4, 6):
                module_path = self.integrator.integrate(code)

            # 步骤5: 运行实验
            with self._step_progress("运行实验", 5, 6):
                console.print()
                console.print(f"[dim]→ 运行基线实验...[/]")

                if baseline_config is None:
                    baseline_config = self._get_default_baseline_config()

                baseline_results = self.runner.run_baseline(baseline_config, epochs)

                console.print()
                console.print(f"[dim]→ 运行创新点实验...[/]")

                experiment_results = self.runner.run_experiment(
                    baseline_config, module_path, epochs
                )

            # 步骤6: 对比性能
            with self._step_progress("对比性能", 6, 6):
                comparison = self.comparator.compare(baseline_results, experiment_results)

            # 步骤7: 生成报告
            with self._step_progress("生成报告", 7, 7):
                report = self.reporter.generate(paper, innovations, comparison)

            console.print()
            console.print(Panel(
                f"[bold green]✓ 流水线完成![/]\n\n"
                f"总耗时: {time.time() - self.start_time:.1f}s\n"
                f"报告路径: {report.output_path}\n\n"
                f"[dim]提示: 在浏览器中打开 HTML 报告查看详细结果[/]",
                title="完成"
            ))

            return report

        except Exception as e:
            console.print(f"\n[bold red]✗ 流水线失败:[/]")
            console.print(f"[red]{e}[/]")
            import traceback
            console.print(traceback.format_exc())
            raise

    def _step_progress(self, description: str, current: int, total: int):
        """创建步骤进度上下文"""
        class _StepProgress:
            def __init__(self, desc, cur, tot):
                self.desc = desc
                self.cur = cur
                self.tot = tot

            def __enter__(self):
                console.print(f"\n[bold cyan][{self.cur}/{self.tot}] {self.desc}...[/]")

            def __exit__(self, exc_type, exc_val, exc_tb):
                if exc_type is None:
                    console.print(f"[green]✓ {self.desc}完成[/]")
                else:
                    console.print(f"[red]✗ {self.desc}失败[/]")

        return _StepProgress(description, current, total)

    def _get_default_baseline_config(self) -> str:
        """获取默认基线配置"""
        # 查找默认配置
        candidates = [
            "third_party:VAD/projects/configs/VAD/VAD_tiny_stage_1.py",
            "configs/baseline/vad_tiny.yaml",
        ]

        for candidate in candidates:
            if Path(candidate).exists():
                return candidate

        # 返回相对路径
        return "third_party/VAD/projects/configs/VAD/VAD_tiny_stage_1.py"
