"""Training command for ultraAD."""

import os
import sys
import click
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

console = Console()


@click.command('train')
@click.argument('config', required=False)
@click.option('--work-dir', '-w', help='Working directory for outputs')
@click.option('--resume', '-r', help='Resume from checkpoint')
@click.option('--gpu-ids', '-g', multiple=True, type=int, help='GPU IDs to use')
@click.option('--debug', '-d', is_flag=True, help='Enable debug mode')
@click.option('--dry-run', is_flag=True, help='Show config without running')
def train_cmd(config, work_dir, resume, gpu_ids, debug, dry_run):
    """Train a model with specified configuration.

    CONFIG is the path to config file. If not provided, uses default.

    Examples:
        ultraad train configs/vad_tiny.yaml
        ultraad train configs/vad_base.yaml -w ./outputs/exp1 -g 0 1
        ultraad train --resume work_dirs/exp1/latest.pth
    """
    # Find config file
    if config is None:
        config = find_default_config()

    config_path = Path(config)
    if not config_path.exists():
        console.print(f"[red]Error: Config file not found: {config}")
        sys.exit(1)

    # Load and process config
    from ultraad.core.config import load_config

    overrides = {}
    if work_dir:
        overrides['work_dir'] = work_dir
    if gpu_ids:
        overrides['gpu_ids'] = list(gpu_ids)

    cfg = load_config(str(config_path), overrides)

    # Set debug mode
    if debug:
        cfg.debug.enabled = True
        cfg.data.num_workers = 0  # For easier debugging
        console.print("[yellow]Debug mode enabled")

    # Dry run - just show config
    if dry_run:
        console.print(Panel.fit(str(cfg.to_dict()), title="Configuration"))
        return

    # Start training
    console.print(Panel.fit(
        f"[bold green]Starting Training[/]\n"
        f"Config: {config_path}\n"
        f"Work Dir: {cfg.work_dir}\n"
        f"Max Epochs: {cfg.trainer.max_epochs}\n"
        f"Batch Size: {cfg.data.batch_size}",
        title="ultraAD Training"
    ))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Setup training
        task = progress.add_task("Initializing...", total=None)

        try:
            # Initialize trainer
            from ultraad.core.trainer import SmartTrainer
            trainer = SmartTrainer(cfg)

            # Resume if specified
            if resume:
                progress.update(task, description=f"Resuming from {resume}")
                trainer.resume_from(resume)

            # Start training
            progress.update(task, description="Training started")
            trainer.train()

            console.print(f"[bold green]Training completed! Results saved to: {cfg.work_dir}")

        except Exception as e:
            console.print(f"[bold red]Training failed: {e}")
            if cfg.debug.enabled:
                import traceback
                console.print(traceback.format_exc())
            sys.exit(1)


def find_default_config() -> str:
    """Find default configuration file."""
    candidates = [
        "configs/vad_tiny.yaml",
        "configs/default.yaml",
        "config.yaml",
    ]

    for candidate in candidates:
        if Path(candidate).exists():
            return candidate

    console.print("[red]Error: No config file found. Please specify one.")
    sys.exit(1)