"""Main CLI entry point for ultraAD."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import click
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from ultraad.cli.train import train_cmd
from ultraad.cli.debug import debug_cmd
from ultraad.cli.viz import viz_cmd
from ultraad.cli.studio import studio_cmd
from ultraad.cli.doctor import doctor_cmd

console = Console()


def print_banner():
    """Print ultraAD banner."""
    banner = """
    ██╗   ██╗██╗  ████████╗██████╗  █████╗ ██████╗
    ██║   ██║██║  ╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗
    ██║   ██║██║     ██║   ██████╔╝███████║██║  ██║
    ██║   ██║██║     ██║   ██╔══██╗██╔══██║██║  ██║
    ╚██████╔╝███████╗██║   ██║  ██║██║  ██║██████╔╝
     ╚═════╝ ╚══════╝╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝
    """
    console.print(banner, style="bold cyan")
    console.print("    End-to-End Autonomous Driving Development Platform\n", style="dim")


@click.group(invoke_without_command=True)
@click.option('--version', is_flag=True, help='Show version and exit.')
@click.pass_context
def app(ctx, version):
    """ultraAD - End-to-End Autonomous Driving Development Platform.

    Quick Start:
        ultraad debug configs/vad_tiny.yaml    # Debug single sample
        ultraad train configs/vad_tiny.yaml   # Start training
        ultraad studio                        # Launch web studio

    For more help: ultraad <command> --help
    """
    if version:
        from ultraad import __version__
        console.print(f"ultraAD version {__version__}")
        return

    if ctx.invoked_subcommand is None:
        print_banner()
        console.print(ctx.get_help())


# Register subcommands
app.add_command(train_cmd, name='train')
app.add_command(debug_cmd, name='debug')
app.add_command(viz_cmd, name='viz')
app.add_command(studio_cmd, name='studio')
app.add_command(doctor_cmd, name='doctor')


@app.command('init')
@click.argument('project_name', default='my_project')
def init_project(project_name):
    """Initialize a new ultraAD project."""
    console.print(f"[bold green]Initializing project: {project_name}[/]")

    # Create project structure
    dirs = [
        f"{project_name}/configs",
        f"{project_name}/scripts",
        f"{project_name}/notebooks",
        f"{project_name}/outputs",
    ]

    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        console.print(f"  Created: {d}")

    # Create sample config
    config_content = f'''# {project_name} configuration

name: {project_name}
work_dir: ./outputs/{project_name}

model:
  name: vad_tiny
  checkpoint: null

data:
  dataset: nuscenes
  data_root: data/nuscenes
  batch_size: 1
  num_workers: 0

trainer:
  max_epochs: 20
  lr: 2e-4
  use_amp: true

debug:
  enabled: true
  check_interval: 50
'''

    config_path = f"{project_name}/configs/default.yaml"
    with open(config_path, 'w') as f:
        f.write(config_content)
    console.print(f"  Created: {config_path}")

    console.print(f"\n[bold green]Project {project_name} initialized successfully![/]")
    console.print(f"\nNext steps:")
    console.print(f"  cd {project_name}")
    console.print(f"  ultraad debug configs/default.yaml")


@app.command('info')
def show_info():
    """Show system information."""
    import torch
    import platform

    table = Table(title="System Information", box=box.ROUNDED)
    table.add_column("Component", style="cyan")
    table.add_column("Version/Info", style="green")

    table.add_row("Python", platform.python_version())
    table.add_row("PyTorch", torch.__version__)
    table.add_row("CUDA", torch.version.cuda if torch.cuda.is_available() else "N/A")
    table.add_row("ultraAD", "0.2.0")

    if torch.cuda.is_available():
        table.add_row("GPU", torch.cuda.get_device_name(0))
        table.add_row("GPU Memory", f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    console.print(table)


if __name__ == '__main__':
    app()