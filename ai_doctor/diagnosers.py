"""Diagnoser implementations for AI Doctor."""

import re
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from ai_doctor.core import Symptom, Severity


class BaseDiagnoser:
    """Base class for all diagnosers."""

    def __init__(self):
        self.name = self.__class__.__name__

    def diagnose(self, training_state: Dict, context: Optional[Dict] = None) -> List[Symptom]:
        """Diagnose issues based on training state.

        Args:
            training_state: Current state of training
            context: Additional context

        Returns:
            List of detected symptoms
        """
        raise NotImplementedError

    def _safe_get(self, data: Dict, key: str, default: Any = None) -> Any:
        """Safely get value from dictionary."""
        return data.get(key, default)


class GradientDiagnoser(BaseDiagnoser):
    """Diagnose gradient-related issues."""

    def diagnose(self, training_state: Dict, context: Optional[Dict] = None) -> List[Symptom]:
        symptoms = []

        # Check for gradient statistics
        grad_stats = self._safe_get(training_state, 'gradients', {})

        if not grad_stats:
            # No gradient information available
            return symptoms

        # Check for vanishing gradients
        max_grad = grad_stats.get('max', 0)
        mean_grad = grad_stats.get('mean', 0)

        if max_grad < 1e-7 and mean_grad < 1e-8:
            symptoms.append(Symptom(
                name="vanishing_gradient",
                description="Gradients are near zero. Model may not be learning.",
                severity=Severity.ERROR,
                category="gradient",
                metrics={'max_gradient': max_grad, 'mean_gradient': mean_grad}
            ))
        elif max_grad < 1e-5:
            symptoms.append(Symptom(
                name="small_gradient",
                description="Gradients are very small. Learning may be slow.",
                severity=Severity.WARNING,
                category="gradient",
                metrics={'max_gradient': max_grad}
            ))

        # Check for exploding gradients
        if max_grad > 1e3 or np.isinf(max_grad):
            symptoms.append(Symptom(
                name="exploding_gradient",
                description="Gradients are too large. Training may be unstable.",
                severity=Severity.CRITICAL,
                category="gradient",
                metrics={'max_gradient': max_grad}
            ))

        # Check for NaN/Inf in gradients
        if grad_stats.get('has_nan', False):
            symptoms.append(Symptom(
                name="nan_gradient",
                description="NaN detected in gradients. Training will fail.",
                severity=Severity.CRITICAL,
                category="gradient",
                metrics={}
            ))

        if grad_stats.get('has_inf', False):
            symptoms.append(Symptom(
                name="inf_gradient",
                description="Inf detected in gradients. Training may be unstable.",
                severity=Severity.ERROR,
                category="gradient",
                metrics={}
            ))

        return symptoms


class LossDiagnoser(BaseDiagnoser):
    """Diagnose loss-related issues."""

    def diagnose(self, training_state: Dict, context: Optional[Dict] = None) -> List[Symptom]:
        symptoms = []

        losses = self._safe_get(training_state, 'losses', {})
        loss_history = self._safe_get(training_state, 'loss_history', [])

        if not losses and not loss_history:
            return symptoms

        # Get total loss
        total_loss = losses.get('loss', 0) if isinstance(losses, dict) else 0

        # Check for NaN/Inf loss
        if np.isnan(total_loss):
            symptoms.append(Symptom(
                name="nan_loss",
                description="Loss is NaN. Training has diverged.",
                severity=Severity.CRITICAL,
                category="loss",
                metrics={'loss': 'NaN'}
            ))

        if np.isinf(total_loss):
            symptoms.append(Symptom(
                name="inf_loss",
                description="Loss is Inf. Check learning rate and gradients.",
                severity=Severity.CRITICAL,
                category="loss",
                metrics={'loss': 'Inf'}
            ))

        # Check loss plateau
        if len(loss_history) >= 10:
            recent_losses = loss_history[-10:]
            loss_variance = np.var(recent_losses)

            if loss_variance < 1e-6:
                symptoms.append(Symptom(
                    name="loss_plateau",
                    description="Loss has plateaued. Model may be stuck.",
                    severity=Severity.WARNING,
                    category="loss",
                    metrics={'loss_variance': loss_variance}
                ))

        # Check for increasing loss
        if len(loss_history) >= 5:
            if loss_history[-1] > loss_history[-5] * 1.5:
                symptoms.append(Symptom(
                    name="increasing_loss",
                    description="Loss is increasing. Check learning rate.",
                    severity=Severity.WARNING,
                    category="loss",
                    metrics={'loss_ratio': loss_history[-1] / loss_history[-5]}
                ))

        # Check individual loss components
        if isinstance(losses, dict):
            for loss_name, loss_value in losses.items():
                if loss_name == 'loss':
                    continue

                if np.isnan(loss_value) or np.isinf(loss_value):
                    symptoms.append(Symptom(
                        name=f"nan_inf_component_{loss_name}",
                        description=f"Loss component '{loss_name}' is {loss_value}.",
                        severity=Severity.ERROR,
                        category="loss",
                        metrics={loss_name: loss_value}
                    ))

        return symptoms


class DataDiagnoser(BaseDiagnoser):
    """Diagnose data-related issues."""

    def diagnose(self, training_state: Dict, context: Optional[Dict] = None) -> List[Symptom]:
        symptoms = []

        # Check for data loading issues
        data_info = self._safe_get(training_state, 'data_info', {})
        dataloader_stats = self._safe_get(training_state, 'dataloader_stats', {})

        # Check data loading time
        loading_time = dataloader_stats.get('loading_time', 0)
        if loading_time > 10:  # More than 10 seconds
            symptoms.append(Symptom(
                name="slow_data_loading",
                description=f"Data loading is slow ({loading_time:.2f}s per batch). Consider increasing num_workers.",
                severity=Severity.WARNING,
                category="data",
                metrics={'loading_time': loading_time}
            ))

        # Check for empty batches
        if dataloader_stats.get('empty_batches', 0) > 0:
            symptoms.append(Symptom(
                name="empty_batches",
                description=f"{dataloader_stats['empty_batches']} empty batches detected.",
                severity=Severity.ERROR,
                category="data",
                metrics={'empty_batches': dataloader_stats['empty_batches']}
            ))

        # Check for NaN/Inf in data
        if data_info.get('has_nan', False):
            symptoms.append(Symptom(
                name="nan_in_data",
                description="NaN values detected in input data.",
                severity=Severity.CRITICAL,
                category="data",
                metrics={}
            ))

        if data_info.get('has_inf', False):
            symptoms.append(Symptom(
                name="inf_in_data",
                description="Inf values detected in input data.",
                severity=Severity.CRITICAL,
                category="data",
                metrics={}
            ))

        return symptoms


class ModelDiagnoser(BaseDiagnoser):
    """Diagnose model-related issues."""

    def diagnose(self, training_state: Dict, context: Optional[Dict] = None) -> List[Symptom]:
        symptoms = []

        model_info = self._safe_get(training_state, 'model_info', {})

        # Check model size
        num_params = model_info.get('num_parameters', 0)
        if num_params > 1e9:  # More than 1B parameters
            symptoms.append(Symptom(
                name="large_model",
                description=f"Model has {num_params/1e9:.2f}B parameters. Ensure sufficient GPU memory.",
                severity=Severity.INFO,
                category="model",
                metrics={'num_parameters': num_params}
            ))

        # Check for unused parameters
        unused_params = model_info.get('unused_parameters', [])
        if unused_params:
            symptoms.append(Symptom(
                name="unused_parameters",
                description=f"{len(unused_params)} parameters are not receiving gradients.",
                severity=Severity.WARNING,
                category="model",
                metrics={'unused_parameters': len(unused_params)}
            ))

        # Check for frozen parameters
        frozen_params = model_info.get('frozen_parameters', [])
        if frozen_params:
            symptoms.append(Symptom(
                name="frozen_parameters",
                description=f"{len(frozen_params)} parameters are frozen.",
                severity=Severity.INFO,
                category="model",
                metrics={'frozen_parameters': len(frozen_params)}
            ))

        # Check weight initialization
        weight_stats = model_info.get('weight_stats', {})
        if weight_stats.get('has_nan', False):
            symptoms.append(Symptom(
                name="nan_weights",
                description="NaN values detected in model weights.",
                severity=Severity.CRITICAL,
                category="model",
                metrics={}
            ))

        return symptoms