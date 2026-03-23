"""性能对比器 - 对比基线和实验的性能"""

from typing import Optional
from rich.console import Console
from rich.table import Table

from .types import ExperimentResults, MetricsDiff, ComparisonResult

console = Console()


class PerformanceComparator:
    """性能对比器"""

    def __init__(self):
        """初始化对比器"""
        pass

    def compare(self, baseline: ExperimentResults,
               experiment: ExperimentResults) -> ComparisonResult:
        """
        对比两个实验结果

        Args:
            baseline: 基线结果
            experiment: 实验结果

        Returns:
            对比结果
        """
        console.print(f"[bold]对比性能[/]")

        # 计算指标差异
        diff = self._compute_diff(baseline.metrics, experiment.metrics)

        # 判断总体改进
        improvement = self._judge_improvement(diff)
        significance = self._judge_significance(diff)

        # 打印对比结果
        self._print_comparison_table(baseline, experiment, diff)

        result = ComparisonResult(
            baseline=baseline,
            experiment=experiment,
            diff=diff,
            overall_improvement=improvement,
            significance=significance
        )

        return result

    def _compute_diff(self, baseline_metrics, experiment_metrics) -> MetricsDiff:
        """计算指标差异"""
        diff = MetricsDiff()

        # mAP
        if baseline_metrics.mAP and experiment_metrics.mAP:
            diff.mAP_delta = experiment_metrics.mAP - baseline_metrics.mAP
            diff.mAP_pct = (diff.mAP_delta / baseline_metrics.mAP) * 100

        # NDS
        if baseline_metrics.NDS and experiment_metrics.NDS:
            diff.NDS_delta = experiment_metrics.NDS - baseline_metrics.NDS
            diff.NDS_pct = (diff.NDS_delta / baseline_metrics.NDS) * 100

        # FPS
        if baseline_metrics.fps and experiment_metrics.fps:
            diff.fps_delta = experiment_metrics.fps - baseline_metrics.fps
            diff.fps_pct = (diff.fps_delta / baseline_metrics.fps) * 100

        return diff

    def _judge_improvement(self, diff: MetricsDiff) -> str:
        """判断总体改进"""
        # 主要看 mAP 和 NDS 的提升
        score = 0.0

        if diff.mAP_delta:
            score += diff.mAP_delta * 10  # mAP 权重更高
        if diff.NDS_delta:
            score += diff.NDS_delta * 10

        if diff.fps_delta and diff.fps_delta < 0:
            score -= abs(diff.fps_delta) * 0.5  # FPS 降低会有一定惩罚

        if score > 0.002:  # 提升 > 0.2%
            return "positive"
        elif score < -0.002:  # 下降 > 0.2%
            return "negative"
        else:
            return "neutral"

    def _judge_significance(self, diff: MetricsDiff) -> str:
        """判断显著性"""
        max_abs_pct = 0.0

        if diff.mAP_pct:
            max_abs_pct = max(max_abs_pct, abs(diff.mAP_pct))
        if diff.NDS_pct:
            max_abs_pct = max(max_abs_pct, abs(diff.NDS_pct))

        if max_abs_pct > 2.0:
            return "high"
        elif max_abs_pct > 1.0:
            return "medium"
        else:
            return "low"

    def _print_comparison_table(self, baseline: ExperimentResults,
                            experiment: ExperimentResults, diff: MetricsDiff):
        """打印表格"""
        table = Table(title="性能对比")
        table.add_column("指标", style="cyan")
        table.add_column("基线", justify="right", style="dim")
        table.add_column("实验", justify="right", style="green")
        table.add_column("变化", justify="right")

        # mAP
        if baseline.metrics.mAP and experiment.metrics.mAP:
            change_str = self._format_change(diff.mAP_delta, diff.mAP_pct)
            table.add_row(
                "mAP",
                f"{baseline.metrics.mAP:.4f}",
                f"{experiment.metrics.mAP:.4f}",
                change_str
            )

        # NDS
        if baseline.metrics.NDS and experiment.metrics.NDS:
            change_str = self._format_change(diff.NDS_delta, diff.NDS_pct)
            table.add_row(
                "NDS",
                f"{baseline.metrics.NDS:.4f}",
                f"{experiment.metrics.NDS:.4f}",
                change_str
            )

        # FPS
        if baseline.metrics.fps and experiment.metrics.fps:
            change_str = self._format_change(diff.fps_delta, diff.fps_pct)
            table.add_row(
                "FPS",
                f"{baseline.metrics.fps:.2f}",
                f"{experiment.metrics.fps:.2f}",
                change_str
            )

        console.print(table)

    def _format_change(self, delta: Optional[float],
                    pct: Optional[float]) -> str:
        """格式化变化"""
        if delta is None:
            return "N/A"

        pct_str = f"({pct:+.1f}%)" if pct is not None else ""

        if delta > 0:
            return f"[green]+{delta:.4f}[/] {pct_str}"
        elif delta < 0:
            return f"[red]{delta:.4f}[/] {pct_str}"
        else:
            return f"[dim]{delta:.4f}[/] {pct_str}"
