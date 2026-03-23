"""Debug command implementation."""

import click
from rich.console import Console

console = Console()


@click.command('debug')
@click.argument('config', required=False)
@click.option('--sample-idx', '-i', type=int, default=0, help='Sample index to debug')
@click.option('--output-dir', '-o', default='logs/debug', help='Output directory')
@click.option('--stage', '-s', type=click.Choice(['data', 'model', 'train', 'all']),
              default='all', help='Debug stage')
@click.option('--gpu-id', '-g', type=int, default=0, help='GPU ID')
@click.option('--checkpoint', '-c', help='Checkpoint to load')
def debug_cmd(config, sample_idx, output_dir, stage, gpu_id, checkpoint):
    """Debug a single sample through the full pipeline.

    CONFIG is the path to config file. If not provided, uses default.

    Examples:
        ultraad debug configs/vad_tiny.yaml
        ultraad debug configs/vad_base.yaml -s data -i 42
        ultraad debug --checkpoint work_dirs/exp1/latest.pth -s model
    """
    console.print("[bold cyan]ultraAD Debug Mode[/]")
    console.print(f"Config: {config}")
    console.print(f"Stage: {stage}")

    # TODO: Implement full debug logic from main.py
    console.print("[yellow]Full debug implementation loading...")

    # For now, delegate to the implementation in main.py
    from ultraad.cli.main import debug as main_debug
    ctx = click.Context(main_debug)
    ctx.invoke(main_debug,
               config=config,
               sample_idx=sample_idx,
               output_dir=output_dir,
               stage=stage,
               gpu_id=gpu_id,
               checkpoint=checkpoint)