"""Core AI Doctor implementation."""

import os
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()


class Severity(Enum):
    """Severity levels for diagnosis."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Symptom:
    """A detected symptom."""
    name: str
    description: str
    severity: Severity
    category: str  # 'gradient', 'loss', 'data', 'model'
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Suggestion:
    """A suggested fix."""
    description: str
    action: str  # What to do
    code_snippet: Optional[str] = None
    confidence: float = 0.5  # 0-1
    auto_fixable: bool = False


@dataclass
class DiagnosisReport:
    """Complete diagnosis report."""
    timestamp: str
    symptoms: List[Symptom] = field(default_factory=list)
    suggestions: List[Suggestion] = field(default_factory=list)
    severity_counts: Dict[str, int] = field(default_factory=dict)
    overall_health: str = "healthy"  # healthy, warning, critical

    def __post_init__(self):
        if not self.severity_counts:
            self.severity_counts = {s.value: 0 for s in Severity}
            for symptom in self.symptoms:
                self.severity_counts[symptom.severity.value] += 1

        # Determine overall health
        if self.severity_counts[Severity.CRITICAL.value] > 0:
            self.overall_health = "critical"
        elif self.severity_counts[Severity.ERROR.value] > 0:
            self.overall_health = "warning"
        elif self.severity_counts[Severity.WARNING.value] > 0:
            self.overall_health = "caution"

    def display(self):
        """Display the report using Rich."""
        # Overall status
        if self.overall_health == "healthy":
            status_color = "green"
        elif self.overall_health == "caution":
            status_color = "yellow"
        elif self.overall_health == "warning":
            status_color = "orange"
        else:
            status_color = "red"

        console.print(Panel.fit(
            f"[bold {status_color}]System Status: {self.overall_health.upper()}[/]\n"
            f"Timestamp: {self.timestamp}\n"
            f"Symptoms: {len(self.symptoms)} | Suggestions: {len(self.suggestions)}",
            title="Diagnosis Report"
        ))

        # Severity counts
        counts_table = Table(title="Severity Distribution", box=box.ROUNDED)
        counts_table.add_column("Severity", style="bold")
        counts_table.add_column("Count", justify="right")

        for severity, count in self.severity_counts.items():
            color = {
                'info': 'blue',
                'warning': 'yellow',
                'error': 'red',
                'critical': 'red'
            }.get(severity, 'white')
            counts_table.add_row(f"[{color}]{severity.upper()}[/{color}]", str(count))

        console.print(counts_table)

        # Symptoms
        if self.symptoms:
            symptoms_table = Table(title="Detected Symptoms", box=box.ROUNDED)
            symptoms_table.add_column("Category", style="cyan")
            symptoms_table.add_column("Symptom", style="yellow")
            symptoms_table.add_column("Severity", style="bold")

            for symptom in self.symptoms[:10]:  # Show top 10
                color = {
                    Severity.INFO: 'blue',
                    Severity.WARNING: 'yellow',
                    Severity.ERROR: 'red',
                    Severity.CRITICAL: 'red'
                }.get(symptom.severity, 'white')

                symptoms_table.add_row(
                    symptom.category,
                    symptom.name,
                    f"[{color}]{symptom.severity.value.upper()}[/{color}]"
                )

            console.print(symptoms_table)

        # Suggestions
        if self.suggestions:
            console.print(f"\n[bold]Top Suggestions:[/]")
            for i, suggestion in enumerate(self.suggestions[:5], 1):
                auto_fix = "[auto-fixable]" if suggestion.auto_fixable else ""
                console.print(f"{i}. {suggestion.description} {auto_fix}")
                if suggestion.code_snippet:
                    console.print(f"   Code: {suggestion.code_snippet[:100]}...")


class AIDoctor:
    """AI Doctor for diagnosing and fixing issues in ultraAD."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize AI Doctor.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.diagnosers = []
        self.knowledge_base = self._load_knowledge_base()
        self.diagnosis_history = []

        # Register diagnosers
        from ai_doctor.diagnosers import (
            GradientDiagnoser,
            LossDiagnoser,
            DataDiagnoser,
            ModelDiagnoser
        )

        self.diagnosers = [
            GradientDiagnoser(),
            LossDiagnoser(),
            DataDiagnoser(),
            ModelDiagnoser(),
        ]

        console.print("[green]AI Doctor initialized with {} diagnosers".format(
            len(self.diagnosers)))

    def _load_knowledge_base(self) -> Dict:
        """Load knowledge base of common issues and solutions."""
        kb_path = Path(__file__).parent / "knowledge" / "knowledge_base.json"

        if kb_path.exists():
            with open(kb_path) as f:
                return json.load(f)

        # Return default knowledge base
        return {
            "gradient_vanishing": {
                "symptoms": ["gradients near zero", "weights not updating"],
                "solutions": [
                    "Use BatchNorm/LayerNorm",
                    "Change activation to LeakyReLU/GELU",
                    "Check for dead ReLUs",
                    "Use residual connections"
                ]
            },
            "gradient_exploding": {
                "symptoms": ["loss is nan", "very large gradients"],
                "solutions": [
                    "Use gradient clipping",
                    "Reduce learning rate",
                    "Check for numerical instability",
                    "Use mixed precision carefully"
                ]
            }
        }

    def diagnose(self, training_state: Dict, context: Optional[Dict] = None) -> DiagnosisReport:
        """Run diagnosis on training state.

        Args:
            training_state: Current state of training (losses, gradients, etc.)
            context: Additional context information

        Returns:
            DiagnosisReport with symptoms and suggestions
        """
        from datetime import datetime

        symptoms = []
        suggestions = []

        # Run all diagnosers
        for diagnoser in self.diagnosers:
            diagnoser_symptoms = diagnoser.diagnose(training_state, context)
            symptoms.extend(diagnoser_symptoms)

            # Get suggestions for each symptom
            for symptom in diagnoser_symptoms:
                symptom_suggestions = self._get_suggestions(symptom)
                suggestions.extend(symptom_suggestions)

        # Remove duplicate suggestions
        seen = set()
        unique_suggestions = []
        for s in suggestions:
            key = s.description
            if key not in seen:
                seen.add(key)
                unique_suggestions.append(s)

        # Sort by confidence
        unique_suggestions.sort(key=lambda x: x.confidence, reverse=True)

        # Create report
        report = DiagnosisReport(
            timestamp=datetime.now().isoformat(),
            symptoms=symptoms,
            suggestions=unique_suggestions[:10]  # Top 10 suggestions
        )

        # Store in history
        self.diagnosis_history.append(report)

        return report

    def _get_suggestions(self, symptom: Symptom) -> list:
        """Get suggestions for a symptom from knowledge base."""
        suggestions = []

        # Match against knowledge base
        for issue_name, issue_data in self.knowledge_base.items():
            # Check if symptom matches
            if any(sym in symptom.name.lower() or sym in symptom.description.lower()
                   for sym in issue_data.get('symptoms', [])):
                # Create suggestions
                for solution in issue_data.get('solutions', []):
                    suggestions.append(Suggestion(
                        description=f"{issue_name}: {solution}",
                        action=solution,
                        confidence=0.7 if symptom.severity == Severity.ERROR else 0.5,
                        auto_fixable=False
                    ))

        # Add generic suggestions based on severity
        if symptom.severity == Severity.CRITICAL:
            suggestions.append(Suggestion(
                description="CRITICAL: Consider stopping training and investigating",
                action="Stop training and check logs",
                confidence=0.9,
                auto_fixable=False
            ))

        return suggestions

    def auto_fix(self, report: DiagnosisReport) -> list:
        """Attempt to automatically fix issues.

        Args:
            report: Diagnosis report

        Returns:
            List of applied fixes
        """
        applied_fixes = []

        for suggestion in report.suggestions:
            if suggestion.auto_fixable:
                # Attempt to apply fix
                fix_result = self._apply_fix(suggestion)
                if fix_result['success']:
                    applied_fixes.append({
                        'suggestion': suggestion.description,
                        'action_taken': fix_result['action']
                    })

        return applied_fixes

    def _apply_fix(self, suggestion: Suggestion) -> dict:
        """Apply a fix suggestion."""
        # This would implement actual fix logic
        # For now, just return success
        return {
            'success': True,
            'action': f"Applied: {suggestion.action}"
        }

    def get_statistics(self) -> dict:
        """Get statistics about diagnosis history."""
        if not self.diagnosis_history:
            return {}

        total_reports = len(self.diagnosis_history)
        total_symptoms = sum(len(r.symptoms) for r in self.diagnosis_history)

        severity_counts = {s.value: 0 for s in Severity}
        for report in self.diagnosis_history:
            for s in Severity:
                severity_counts[s.value] += report.severity_counts.get(s.value, 0)

        return {
            'total_reports': total_reports,
            'total_symptoms': total_symptoms,
            'avg_symptoms_per_report': total_symptoms / total_reports,
            'severity_distribution': severity_counts
        }