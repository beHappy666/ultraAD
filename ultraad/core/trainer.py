"""SmartTrainer - Intelligent training engine for ultraAD."""

import os
import sys
import time
import json
from typing import Dict, List, Optional, Callable, Any
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel

from ultraad.core.config import Config

console = Console()


@dataclass
class TrainingState:
    """Current training state."""
    epoch: int
    step: int
    global_step: int
    loss: float
    learning_rate: float
    best_loss: float = float('inf')
    patience_counter: int = 0


class SmartTrainer:
    """Intelligent training engine with auto-diagnosis and optimization."""

    def __init__(self, config: Config):
        """Initialize SmartTrainer.

        Args:
            config: Training configuration
        """
        self.config = config
        self.state = TrainingState(
            epoch=0,
            step=0,
            global_step=0,
            loss=0.0,
            learning_rate=config.trainer.lr
        )

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.train_loader = None
        self.val_loader = None

        self.early_stopping_enabled = True
        self.early_stopping_patience = 5

        # Setup
        self._setup_device()
        self._setup_training()

        # AI Doctor integration
        self.ai_doctor = None
        if config.debug.get('enable_ai_doctor', True):
            try:
                from ai_doctor.core import AIDoctor
                self.ai_doctor = AIDoctor()
                console.print("[green]AI Doctor enabled for training monitoring")
            except ImportError:
                console.print("[yellow]AI Doctor not available")

    def _setup_device(self):
        """Setup training device."""
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{self.config.gpu_ids[0]}' if self.config.gpu_ids else 'cuda:0')
            torch.cuda.set_device(self.device)
            console.print(f"[green]Using GPU: {torch.cuda.get_device_name(self.device)}")
        else:
            self.device = torch.device('cpu')
            console.print("[yellow]Using CPU")

    def _setup_training(self):
        """Setup training components."""
        # Mixed precision
        if self.config.trainer.use_amp:
            self.scaler = GradScaler()
            console.print("[green]Mixed precision training enabled")

    def train(self, model=None, train_loader=None, val_loader=None,
              optimizer=None, scheduler=None, epochs=None):
        """Run training loop.

        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            epochs: Number of epochs (overrides config)

        Returns:
            Training history
        """
        # Use provided components or stored ones
        if model is not None:
            self.model = model.to(self.device)
        if train_loader is not None:
            self.train_loader = train_loader
        if val_loader is not None:
            self.val_loader = val_loader
        if optimizer is not None:
            self.optimizer = optimizer
        if scheduler is not None:
            self.scheduler = scheduler

        epochs = epochs or self.config.trainer.max_epochs

        # Validate setup
        if self.model is None:
            raise ValueError("Model not provided")
        if self.train_loader is None:
            raise ValueError("Training data loader not provided")
        if self.optimizer is None:
            raise ValueError("Optimizer not provided")

        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch_times': []
        }

        # Training loop
        console.print(f"\n[bold]Starting training for {epochs} epochs...[/]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:

            epoch_task = progress.add_task("Training...", total=epochs)

            for epoch in range(epochs):
                self.state.epoch = epoch
                epoch_start_time = time.time()

                progress.update(epoch_task, description=f"Epoch {epoch+1}/{epochs}")

                # Training epoch
                train_loss = self._train_epoch(progress)
                history['train_loss'].append(train_loss)

                # Validation
                if self.val_loader is not None:
                    val_loss = self._validate_epoch(progress)
                    history['val_loss'].append(val_loss)

                # Update learning rate
                if self.scheduler is not None:
                    if hasattr(self.scheduler, 'step'):
                        self.scheduler.step()
                    history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])

                # Epoch time
                epoch_time = time.time() - epoch_start_time
                history['epoch_times'].append(epoch_time)

                # Update progress
                progress.update(epoch_task, advance=1)

                # Console output
                console.print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Time: {epoch_time:.2f}s")

                # Run AI Doctor check
                if self.ai_doctor and (epoch + 1) % 5 == 0:
                    self._run_ai_doctor_check(history)

                # Early stopping check
                if self.early_stopping_enabled and self.val_loader is not None:
                    if self._check_early_stopping(history):
                        console.print("[yellow]Early stopping triggered")
                        break

        console.print(f"\n[bold green]Training complete![/]")
        console.print(f"Best training loss: {min(history['train_loss']):.4f}")
        if history['val_loss']:
            console.print(f"Best validation loss: {min(history['val_loss']):.4f}")

        return history

    def _train_epoch(self, progress=None) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Forward pass
            self.optimizer.zero_grad()

            if self.scaler is not None:
                # Mixed precision training
                with autocast():
                    outputs = self.model(return_loss=True, **batch)
                    loss = outputs['loss'] if isinstance(outputs, dict) else outputs

                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.config.trainer.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.trainer.grad_clip
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Full precision training
                outputs = self.model(return_loss=True, **batch)
                loss = outputs['loss'] if isinstance(outputs, dict) else outputs

                loss.backward()

                # Gradient clipping
                if self.config.trainer.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.trainer.grad_clip
                    )

                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            self.state.step += 1
            self.state.global_step += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def _validate_epoch(self, progress=None) -> float:
        """Run one validation epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                # Forward pass
                if self.scaler is not None:
                    with autocast():
                        outputs = self.model(return_loss=True, **batch)
                        loss = outputs['loss'] if isinstance(outputs, dict) else outputs
                else:
                    outputs = self.model(return_loss=True, **batch)
                    loss = outputs['loss'] if isinstance(outputs, dict) else outputs

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def _run_ai_doctor_check(self, history: Dict):
        """Run AI Doctor diagnosis on current training state."""
        if self.ai_doctor is None:
            return

        # Prepare training state for diagnosis
        training_state = {
            'epoch': self.state.epoch,
            'step': self.state.step,
            'loss': history['train_loss'][-1] if history['train_loss'] else 0,
            'loss_history': history['train_loss'],
            'learning_rate': self.optimizer.param_groups[0]['lr'],
        }

        # Run diagnosis
        try:
            report = self.ai_doctor.diagnose(training_state)

            # Display if there are issues
            if report.symptoms:
                console.print("\n[bold yellow]AI Doctor Alert:[/]")
                report.display()
        except Exception as e:
            console.print(f"[dim]AI Doctor check failed: {e}")

    def _check_early_stopping(self, history: Dict) -> bool:
        """Check if early stopping should be triggered."""
        if not history['val_loss']:
            return False

        # Get minimum validation loss
        min_val_loss = min(history['val_loss'])
        current_val_loss = history['val_loss'][-1]

        # Check if no improvement
        if current_val_loss > min_val_loss:
            self.state.patience_counter += 1
            if self.state.patience_counter >= self.early_stopping_patience:
                return True
        else:
            self.state.patience_counter = 0

        return False

    def save_checkpoint(self, path: str, **kwargs):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.state.epoch,
            'step': self.state.step,
            'global_step': self.state.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.state.best_loss,
            'config': self.config.to_dict(),
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        checkpoint.update(kwargs)

        torch.save(checkpoint, path)
        console.print(f"[green]Checkpoint saved: {path}")

    def resume_from(self, checkpoint_path: str):
        """Resume training from checkpoint."""
        console.print(f"[bold]Resuming from: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Restore model
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Restore optimizer
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Restore scheduler
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Restore scaler
        if 'scaler_state_dict' in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        # Restore state
        self.state.epoch = checkpoint.get('epoch', 0)
        self.state.step = checkpoint.get('step', 0)
        self.state.global_step = checkpoint.get('global_step', 0)
        self.state.best_loss = checkpoint.get('best_loss', float('inf'))

        console.print(f"[green]✓ Resumed from epoch {self.state.epoch}")