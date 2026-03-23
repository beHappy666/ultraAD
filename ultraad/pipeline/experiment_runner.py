"""实验器 - 运行真实训练"""

import os
import sys
import subprocess
import json
import re
from datetime import datetime
from typing import Optional
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from .types import ExperimentResults, Metrics

console = Console()


class ExperimentRunner:
    """实验运行器"""

    def __init__(self, work_dir: str = None, gpu_id: int = 0):
        """
        初始化运行器

        Args:
            work_dir: 工作目录
            gpu_id: GPU ID
        """
        self.work_dir = Path(work_dir) if work_dir else Path("work_dirs") / "experiments"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.gpu_id = gpu_id

    def run_baseline(self, config_path: str, epochs: int = 20) -> ExperimentResults:
        """
        运行基线实验

        Args:
            config_path: 配置文件路径
            epochs: 训练轮数

        Returns:
            实验结果
        """
        experiment_id = "baseline"
        output_dir = self.work_dir / experiment_id

        console.print(f"[bold]运行基线实验[/]")
        console.print(f"  配置: {config_path}")
        console.print(f"  输出: {output_dir}")

        start_time = datetime.now()

        # 运行真实训练
        results = self._run_training(
            experiment_id,
            config_path,
            output_dir,
            epochs,
            is_baseline=True
        )

        end_time = datetime.now()
        results.start_time = start_time
        results.end_time = end_time

        console.print(f"[green]✓ 基线实验完成[/]")
        console.print(f"  mAP: {results.metrics.mAP:.4f}")
        console.print(f"  NDS: {results.metrics.NDS:.4f}")

        return results

    def run_experiment(self, config_path: str, module_path: str, epochs: int = 20) -> ExperimentResults:
        """
        运行创新点实验

        Args:
            config_path: 配置文件路径
            module_path: 生成的模块路径
            epochs: 训练轮数

        Returns:
            实验结果
        """
        experiment_id = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir = self.work_dir / experiment_id

        console.print(f"[bold]运行实验[/]")
        console.print(f"  配置: {config_path}")
        console.print(f"  模块: {module_path}")
        console.print(f"  输出: {output_dir}")

        start_time = datetime.now()

        # 运行真实训练
        results = self._run_training(
            experiment_id,
            config_path,
            output_dir,
            epochs,
            is_baseline=False,
            module_path=module_path
        )

        end_time = datetime.now()
        results.start_time = start_time
        results.end_time = end_time

        console.print(f"[green]✓ 实验完成[/]")
        console.print(f"  mAP: {results.metrics.mAP:.4f}")
        console.print(f"  NDS: {results.metrics.NDS:.4f}")

        return results

    def _run_training(self, experiment_id: str, config_path: str,
                     output_dir: Path, epochs: int, is_baseline: bool,
                     module_path: str = None) -> ExperimentResults:
        """
        运行真实训练

        Args:
            experiment_id: 实验ID
            config_path: 配置文件路径
            output_dir: 输出目录
            epochs: 训练轮数
            is_baseline: 是否为基线
            module_path: 生成的模块路径（实验用）
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # 构建训练命令
        cmd = [
            sys.executable, "-m", "ultraad.cli.main", "train",
            config_path,
            "--work-dir", str(output_dir)
        ]

        # 添加 GPU 参数
        cmd.extend(["--gpu-ids", str(self.gpu_id)])

        console.print(f"[dim]执行命令: {' '.join(cmd)}[/]")

        # 运行训练
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # 实时输出日志
        log_file = output_dir / "training.log"
        start_time = datetime.now()

        with open(log_file, 'w') as log_f:
            for line in iter(process.stdout.readline, ''):
                if not line:
                    break
                print(line, end='')  # 输出到控制台
                log_f.write(line)
                log_f.flush()

        process.wait()

        # 计算训练时间
        training_hours = (datetime.now() - start_time).total_seconds() / 3600

        # 检查训练是否成功
        if process.returncode != 0:
            raise RuntimeError(f"训练失败，返回码: {process.returncode}")

        # 解析结果
        metrics = self._parse_results(output_dir, log_file)
        metrics.training_time_hours = training_hours

        # 获取检查点
        checkpoints = list(output_dir.glob("*.pth"))

        return ExperimentResults(
            experiment_id=experiment_id,
            config_name=config_path,
            metrics=metrics,
            checkpoints=[str(c) for c in checkpoints],
            logs=[str(log_file)]
        )

    def _parse_results(self, output_dir: Path, log_file: Path) -> Metrics:
        """从训练结果解析指标"""
        metrics = Metrics()

        # 尝试从结果JSON读取
        result_file = output_dir / "results.json"
        if result_file.exists():
            with open(result_file) as f:
                data = json.load(f)
                metrics.mAP = data.get("mAP")
                metrics.mATE = data.get("mATE")
                metrics.mASE = data.get("mASE")
                metrics.mAOE = data.get("mAOE")
                metrics.mAVE = data.get("mAVE")
                metrics.mAAE = data.get("mAAE")
                metrics.NDS = data.get("NDS")
                metrics.fps = data.get("fps")
                metrics.gpu_memory_gb = data.get("gpu_memory_gb")
                return metrics

        # 从日志解析
        if log_file.exists():
            with open(log_file) as f:
                log_text = f.read()

            # 解析 mAP
            map_match = re.search(r'mAP[=:]\s*([\d.]+)', log_text)
            if map_match:
                metrics.mAP = float(map_match.group(1))

            # 解析 NDS
            nds_match = re.search(r'NDS[=:]\s*([\d.]+)', log_text)
            if nds_match:
                metrics.NDS = float(nds_match.group(1))

            # 解析 FPS
            fps_match = re.search(r'FPS[=:]\s*([\d.]+)', log_text)
            if fps_match:
                metrics.fps = float(fps_match.group(1))

        # 检查是否获取到关键指标
        if metrics.mAP is None or metrics.NDS is None:
            console.print("[yellow]警告: 未能从输出中提取到 mAP 和 NDS 指标[/]")
            console.print("[dim]请确保训练脚本输出包含这些指标[/]")

        return metrics
