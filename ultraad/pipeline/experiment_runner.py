"""实验运行器 - 运行训练（支持真实和模拟模式）"""

import os
import sys
import subprocess
import json
import re
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from .types import ExperimentResults, Metrics

console = Console()


class ExperimentRunner:
    """实验运行器"""

    def __init__(self, work_dir: str = None, gpu_id: int = 0, use_mock: bool = False):
        """
        初始化运行器

        Args:
            work_dir: 工作目录
            gpu_id: GPU ID
            use_mock: 是否使用模拟训练（不调用真实训练）
        """
        self.work_dir = Path(work_dir) if work_dir else Path("work_dirs") / "experiments"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.gpu_id = gpu_id
        self.use_mock = use_mock or self._check_force_mock()

        if self.use_mock:
            console.print("[yellow]使用模拟训练模式[/]")
        else:
            console.print("[dim]使用真实训练模式[/]")

    def _check_force_mock(self) -> bool:
        """检查是否强制使用模拟模式"""
        # 如果 torch 未安装，使用模拟
        try:
            import torch
            return False
        except ImportError:
            console.print("[yellow]torch 未安装，使用模拟训练模式[/]")
            return True

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

        # 运行训练
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

        # 运行训练
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
        运行训练（真实或模拟）

        Args:
            experiment_id: 实验ID
            config_path: 配置文件路径
            output_dir: 输出目录
            epochs: 训练轮数
            is_baseline: 是否为基线
            module_path: 生成的模块路径（实验用）
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.use_mock:
            # 使用模拟训练
            return self._run_mock_training(
                experiment_id, config_path, output_dir,
                epochs, is_baseline, module_path
            )
        else:
            # 使用真实训练
            return self._run_real_training(
                experiment_id, config_path, output_dir,
                epochs, is_baseline, module_path
            )

    def _run_real_training(self, experiment_id: str, config_path: str,
                         output_dir: Path, epochs: int, is_baseline: bool,
                         module_path: str = None) -> ExperimentResults:
        """运行真实训练"""
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

    def _run_mock_training(self, experiment_id: str, config_path: str,
                        output_dir: Path, epochs: int, is_baseline: bool,
                        module_path: str = None) -> ExperimentResults:
        """运行模拟训练"""
        console.print(f"[dim]模拟训练 {epochs} 轮...[/]")

        # 模拟训练进度
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("训练中...", total=epochs)

            import time
            for epoch in range(epochs):
                time.sleep(0.05)  # 模拟训练时间
                progress.update(task, advance=1)

        # 生成模拟指标
        metrics = self._get_mock_metrics(is_baseline)

        # 创建模拟检查点和日志
        ckpt_path = output_dir / "latest.pth"
        with open(ckpt_path, 'w') as f:
            f.write(f"# 模拟检查点 - {experiment_id}\n")

        log_file = output_dir / "training.log"
        with open(log_file, 'w') as f:
            f.write(f"# 模拟训练日志 - {experiment_id}\n")
            for epoch in range(epochs):
                loss = 0.5 - epoch * 0.02
                f.write(f"Epoch {epoch+1}/{epochs}\n")
                f.write(f"  loss: {loss:.4f}\n")
            f.write(f"\n最终指标:\n")
            f.write(f"  mAP: {metrics.mAP:.4f}\n")
            f.write(f"  NDS: {metrics.NDS:.4f}\n")
            f.write(f"  FPS: {metrics.fps:.2f}\n")

        return ExperimentResults(
            experiment_id=experiment_id,
            config_name=config_path,
            metrics=metrics,
            checkpoints=[str(ckpt_path)],
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

    def _get_mock_metrics(self, is_baseline: bool) -> Metrics:
        """获取模拟指标"""
        if is_baseline:
            return Metrics(
                mAP=0.352,
                mATE=0.730,
                mASE=0.279,
                mAOE=0.459,
                mAVE=0.308,
                mAAE=0.321,
                NDS=0.456,
                fps=12.5,
                gpu_memory_gb=8.2,
                training_time_hours=3.2
            )
        else:
            import random
            improvement = random.uniform(0.015, 0.025)
            return Metrics(
                mAP=0.352 * (1 + improvement),
                mATE=0.730 * (1 - improvement * 0.5),
                mASE=0.279 * (1 - improvement * 0.3),
                mAOE=0.459 * (1 - improvement * 0.4),
                mAVE=0.308 * (1 - improvement * 0.5),
                mAAE=0.321 * (1 - improvement * 0.3),
                NDS=0.456 * (1 + improvement * 0.8),
                fps=12.5 * (1 - improvement * 2),
                gpu_memory_gb=8.5,
                training_time_hours=3.5
            )
