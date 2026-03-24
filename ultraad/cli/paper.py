"""Paper-to-Report 命令 - 一键式从论文生成实验报告"""

import os
import sys
import webbrowser
import click
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

console = Console()


@click.command('paper-to-report')
@click.argument('paper_source')
@click.option('--output-dir', '-o', default='auto_output', help='输出目录')
@click.option('--baseline', '-b', help='基线配置文件路径')
@click.option('--epochs', '-e', type=int, default=20, help='训练轮数')
@click.option('--gpu-id', '-g', type=int, default=0, help='GPU ID')
@click.option('--no-browser', is_flag=True, help='不自动打开浏览器')
def paper_to_report_cmd(paper_source, output_dir, baseline, epochs, gpu_id, no_browser):
    """一键式：从论文自动生成实验报告

    PAPER_SOURCE 可以是：
    - PDF 文件路径: path/to/paper.pdf
    - arXiv ID: arxiv:2301.xxxxx
    - arXiv URL: https://arxiv.org/abs/2301.xxxxx

    示例：
        ultraad paper-to-report arxiv:2301.xxxxx
        ultraad paper-to-report path/to/paper.pdf
        ultraad paper-to-report paper.pdf --baseline configs/baseline.yaml
    """
    console.print(Panel.fit(
        f"[bold cyan]ultraAD Paper-to-Report[/]\n\n"
        f"[dim]从论文到实验报告，一键完成[/]",
        title="Start"
    ))

    # 导入流水线
    try:
        from ultraad.pipeline.auto_pipeline import AutoPipeline
    except ImportError as e:
        console.print(f"[red]导入流水线失败: {e}")
        console.print("[yellow]请确保已安装必要依赖: pip install pym pymupdf requests[/]")
        sys.exit(1)

    # 检查论文源
    paper_path = Path(paper_source)
    if not paper_source.startswith('arxiv:') and not paper_source.startswith('http'):
        if not paper_path.exists():
            console.print(f"[red]论文文件不存在: {paper_source}")
            sys.exit(1)

    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 运行流水线
        pipeline = AutoPipeline(
            output_dir=str(output_dir),
            gpu_id=gpu_id
        )

        report = pipeline.run(
            paper_source=paper_source,
            baseline_config=baseline,
            epochs=epochs
        )

        # 自动打开浏览器
        if not no_browser and report.output_path.endswith('.html'):
            html_path = Path(report.output_path).absolute()
            file_url = f"file:///{html_path.as_posix()}"

            console.print(f"\n[bold cyan]正在打开浏览器...[/]")
            webbrowser.open(file_url)

    except KeyboardInterrupt:
        console.print("\n[yellow]用户中断[/]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]错误:[/]")
        console.print(f"[red]{e}[/]")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)


@click.command('paper-list')
def paper_list_cmd():
    """列出已缓存的论文"""
    cache_dir = Path("cache/papers")

    if not cache_dir.exists():
        console.print("[dim]没有缓存的论文[/]")
        return

    console.print("[bold]已缓存的论文:[/]\n")

    for pdf_file in cache_dir.glob("*.pdf"):
        console.print(f"  {pdf_file.name}")
        console.print(f"    路径: {pdf_file}")
        console.print()


# 导出命令
__all__ = ['paper_to_report_cmd', 'paper_list_cmd']
