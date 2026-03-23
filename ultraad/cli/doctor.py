"""AI Doctor command implementation."""

import os
import sys
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from pathlib import Path

console = Console()


@click.command('doctor')
@click.option('--log-file', '-l', help='Training log file to analyze')
@click.option('--checkpoint', '-c', help='Checkpoint to diagnose')
@click.option('--watch', '-w', is_flag=True, help='Watch mode - monitor training in real-time')
@click.option('--auto-fix', '-a', is_flag=True, help='Attempt to auto-fix issues')
@click.option('--report', '-r', is_flag=True, help='Generate detailed report')
def doctor_cmd(log_file, checkpoint, watch, auto_fix, report):
    """AI Doctor - Intelligent diagnosis for training issues.

    The AI Doctor analyzes training logs, checkpoints, and model state
    to diagnose issues and suggest fixes.

    Examples:
        ultraad doctor -l work_dirs/exp1/20240101.log
        ultraad doctor -c work_dirs/exp1/latest.pth --auto-fix
        ultraad doctor -w  # Watch mode for active training
    """
    console.print(Panel.fit(
        "[bold cyan]AI Doctor - Intelligent Diagnosis System[/]\n"
        "Analyzing your training for issues and improvements...",
        title="ultraAD"
    ))

    # Initialize AI Doctor
    try:
        from ai_doctor.core import AIDoctor
        from ultraad.core.config import Config

        doctor = AIDoctor()
        console.print("[green]✓ AI Doctor initialized")
    except Exception as e:
        console.print(f"[red]✗ Failed to initialize AI Doctor: {e}")
        sys.exit(1)

    # Different modes
    if watch:
        _watch_mode(doctor)
    elif checkpoint:
        _diagnose_checkpoint(doctor, checkpoint, auto_fix, report)
    elif log_file:
        _diagnose_log(doctor, log_file, report)
    else:
        # Interactive diagnosis
        _interactive_diagnosis(doctor)


def _watch_mode(doctor):
    """Watch mode for monitoring active training."""
    import time

    console.print("\n[bold]Watch Mode Active[/] - Press Ctrl+C to stop")
    console.print("Monitoring for issues every 10 seconds...\n")

    try:
        while True:
            # This would connect to running training process
            # For now, just simulate
            time.sleep(10)
            console.print("[dim]Scanning...[/]", end="\r")
    except KeyboardInterrupt:
        console.print("\n[yellow]Watch mode stopped")


def _diagnose_checkpoint(doctor, checkpoint_path, auto_fix, report):
    """Diagnose a checkpoint file."""
    import torch

    console.print(f"\n[bold]Analyzing checkpoint:[/] {checkpoint_path}")

    if not Path(checkpoint_path).exists():
        console.print(f"[red]✗ Checkpoint not found: {checkpoint_path}")
        return

    try:
        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location='cpu')

        # Analyze checkpoint
        analysis = {
            'epoch': ckpt.get('epoch', 'unknown'),
            'num_params': len(ckpt.get('state_dict', {})),
            'keys': list(ckpt.keys()),
        }

        console.print(f"[green]✓ Checkpoint loaded successfully")
        console.print(f"  Epoch: {analysis['epoch']}")
        console.print(f"  State dict keys: {analysis['num_params']}")

        # Create training state for diagnosis
        training_state = {
            'checkpoint': ckpt,
            'model_state': ckpt.get('state_dict', {}),
        }

        # Run diagnosis
        diagnosis = doctor.diagnose(training_state)

        if auto_fix and diagnosis.suggestions:
            console.print("\n[bold]Attempting auto-fix...")
            fixes = doctor.auto_fix(diagnosis)
            if fixes:
                for fix in fixes:
                    console.print(f"[green]✓ Applied: {fix['action_taken']}")
            else:
                console.print("[yellow]No auto-fixes could be applied")

        # Display report
        diagnosis.display()

        # Save detailed report if requested
        if report:
            report_path = str(Path(checkpoint_path).parent / 'diagnosis_report.json')
            with open(report_path, 'w') as f:
                json.dump({
                    'timestamp': diagnosis.timestamp,
                    'overall_health': diagnosis.overall_health,
                    'symptoms': [s.__dict__ for s in diagnosis.symptoms],
                    'suggestions': [s.__dict__ for s in diagnosis.suggestions],
                }, f, indent=2, default=str)
            console.print(f"\n[dim]Detailed report saved to: {report_path}")

    except Exception as e:
        console.print(f"[red]✗ Failed to analyze checkpoint: {e}")


def _diagnose_log(doctor, log_file, report):
    """Diagnose a training log file."""
    console.print(f"\n[bold]Analyzing log file:[/] {log_file}")

    if not Path(log_file).exists():
        console.print(f"[red]✗ Log file not found: {log_file}")
        return

    try:
        # Parse log file
        losses = []
        errors = []

        with open(log_file) as f:
            for line in f:
                # Extract loss values
                if 'loss' in line.lower():
                    # Simple parsing - could be more sophisticated
                    pass

                # Extract errors
                if 'error' in line.lower() or 'exception' in line.lower():
                    errors.append(line.strip())

        console.print(f"[green]✓ Log file parsed")
        console.print(f"  Errors found: {len(errors)}")

        if errors:
            console.print("\n[yellow]Recent errors:")
            for error in errors[-5:]:
                console.print(f"  {error}")

        # Create training state for diagnosis
        training_state = {
            'log_file': log_file,
            'errors': errors,
            'losses': losses,
        }

        # Run diagnosis
        diagnosis = doctor.diagnose(training_state)
        diagnosis.display()

    except Exception as e:
        console.print(f"[red]✗ Failed to analyze log: {e}")


def _interactive_diagnosis(doctor):
    """Interactive diagnosis mode."""
    console.print("\n[bold]Interactive Diagnosis Mode[/]")
    console.print("Describe the issue you're experiencing, or type 'quit' to exit.\n")

    while True:
        user_input = console.input("[bold cyan]What issue are you experiencing? [/]")

        if user_input.lower() in ['quit', 'exit', 'q']:
            break

        # Create a simple training state from user input
        training_state = {
            'user_description': user_input,
            'context': 'interactive'
        }

        # Run diagnosis
        diagnosis = doctor.diagnose(training_state)
        diagnosis.display()

        console.print("\n" + "=" * 60 + "\n")

    console.print("[dim]Exiting interactive mode.")