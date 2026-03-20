"""
Innovation Extractor Module

Responsible for extracting innovation points from paper content
and evaluating their feasibility for autonomous driving applications.
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from .types import (
    InnovationType,
    FeasibilityLevel,
    ResourceEstimate,
    DependencySpec,
    ValidationResult
)
from .paper_analyzer import PaperContent, Section


@dataclass
class Contribution:
    """Represents a specific contribution or innovation from the paper"""
    title: str
    description: str
    section_ref: str  # Reference to the section where this is described
    innovation_type: InnovationType = InnovationType.UNKNOWN
    technical_details: Dict[str, Any] = field(default_factory=dict)
    performance_claims: List[Dict[str, Any]] = field(default_factory=list)
    related_work_comparison: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'title': self.title,
            'description': self.description,
            'section_ref': self.section_ref,
            'innovation_type': self.innovation_type.value,
            'technical_details': self.technical_details,
            'performance_claims': self.performance_claims
        }


@dataclass
class Innovation:
    """Represents an extracted innovation point ready for implementation"""
    id: str
    name: str
    description: str
    innovation_type: InnovationType
    source_paper: str
    source_section: str

    # Technical details
    algorithm_description: str = ""
    mathematical_formulation: Optional[str] = None
    pseudo_code: Optional[str] = None

    # Implementation details
    input_spec: Dict[str, Any] = field(default_factory=dict)
    output_spec: Dict[str, Any] = field(default_factory=dict)
    hyperparameters: List[Dict[str, Any]] = field(default_factory=list)

    # Evaluation
    expected_benefits: List[str] = field(default_factory=list)
    performance_targets: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    extraction_timestamp: datetime = field(default_factory=datetime.now)
    confidence_score: float = 0.0  # Confidence in extraction quality

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'innovation_type': self.innovation_type.value,
            'source_paper': self.source_paper,
            'source_section': self.source_section,
            'algorithm_description': self.algorithm_description,
            'input_spec': self.input_spec,
            'output_spec': self.output_spec,
            'hyperparameters': self.hyperparameters,
            'expected_benefits': self.expected_benefits,
            'confidence_score': self.confidence_score
        }


@dataclass
class FeasibilityReport:
    """Report on the feasibility of implementing an innovation"""
    innovation_id: str

    # Overall assessment
    overall_feasibility: FeasibilityLevel = FeasibilityLevel.UNKNOWN
    confidence_score: float = 0.0

    # Technical feasibility
    technical_complexity: str = "unknown"  # simple, moderate, complex
    implementation_effort: str = "unknown"  # days, weeks, months
    required_expertise: List[str] = field(default_factory=list)

    # Compatibility
    compatibility: Dict[str, Any] = field(default_factory=lambda: {
        'vad_compatible': False,
        'ultraad_compatible': False,
        'breaking_changes': [],
        'deprecation_warnings': []
    })

    # Resource requirements
    resources_needed: ResourceEstimate = field(default_factory=lambda: ResourceEstimate(
        compute_hours=0.0,
        gpu_memory_gb=0.0,
        storage_gb=0.0,
        data_size_gb=0.0
    ))

    # Dependencies
    dependencies: List[DependencySpec] = field(default_factory=list)

    # Risks
    risks: List[Dict[str, Any]] = field(default_factory=list)

    # Recommendations
    recommendations: List[str] = field(default_factory=list)

    # Validation
    validation_checks: List[ValidationResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'innovation_id': self.innovation_id,
            'overall_feasibility': self.overall_feasibility.value,
            'confidence_score': self.confidence_score,
            'technical_complexity': self.technical_complexity,
            'implementation_effort': self.implementation_effort,
            'required_expertise': self.required_expertise,
            'compatibility': self.compatibility,
            'resources_needed': {
                'compute_hours': self.resources_needed.compute_hours,
                'gpu_memory_gb': self.resources_needed.gpu_memory_gb,
                'storage_gb': self.resources_needed.storage_gb,
                'data_size_gb': self.resources_needed.data_size_gb,
            },
            'dependencies': [
                {'name': d.name, 'version_spec': d.version_spec}
                for d in self.dependencies
            ],
            'risks': self.risks,
            'recommendations': self.recommendations,
            'validation_checks': [
                {'check_name': v.check_name, 'passed': v.passed, 'message': v.message}
                for v in self.validation_checks
            ]
        }


class InnovationExtractor:
    """
    Main class for extracting innovation points from papers
    and evaluating their feasibility.
    """

    def __init__(self, llm_client=None):
        """
        Initialize the extractor.

        Args:
            llm_client: Optional LLM client for enhanced analysis
        """
        self.llm = llm_client
        self.innovation_counter = 0

    def extract(self, paper_content: PaperContent) -> List[Innovation]:
        """
        Extract innovation points from paper content.

        Args:
            paper_content: Structured paper content

        Returns:
            List of extracted innovations
        """
        innovations = []

        # Step 1: Identify contributions from paper structure
        contributions = self._identify_contributions(paper_content)

        # Step 2: Extract innovations from contributions
        for contribution in contributions:
            innovation = self._contribution_to_innovation(
                contribution,
                paper_content
            )
            if innovation:
                innovations.append(innovation)

        # Step 3: Extract additional innovations from sections
        section_innovations = self._extract_from_sections(paper_content)
        innovations.extend(section_innovations)

        # Step 4: Deduplicate and rank
        innovations = self._deduplicate_and_rank(innovations)

        return innovations

    def evaluate_feasibility(self, innovation: Innovation,
                            target_system: str = 'ultraAD') -> FeasibilityReport:
        """
        Evaluate the feasibility of implementing an innovation.

        Args:
            innovation: The innovation to evaluate
            target_system: Target system for implementation

        Returns:
            Feasibility report
        """
        report = FeasibilityReport(
            innovation_id=innovation.id
        )

        # 1. Technical complexity analysis
        report.technical_complexity = self._analyze_complexity(innovation)
        report.implementation_effort = self._estimate_effort(innovation)

        # 2. Compatibility analysis
        report.compatibility = self._check_compatibility(
            innovation,
            target_system
        )

        # 3. Resource requirements
        report.resources_needed = self._estimate_resources(innovation)

        # 4. Dependencies analysis
        report.dependencies = self._analyze_dependencies(innovation)

        # 5. Risk assessment
        report.risks = self._assess_risks(innovation, report)

        # 6. Overall feasibility
        report.overall_feasibility = self._calculate_overall_feasibility(report)

        # 7. Recommendations
        report.recommendations = self._generate_recommendations(report)

        return report

    def _identify_contributions(self, paper_content: PaperContent) -> List[Contribution]:
        """Identify contributions from paper content"""
        contributions = []

        # Look for contribution sections
        for section in paper_content.sections:
            # Check if section is about contributions
            if self._is_contribution_section(section):
                section_contributions = self._parse_contributions_from_section(
                    section,
                    paper_content
                )
                contributions.extend(section_contributions)

            # Also look for contributions in introduction
            if 'intro' in section.title.lower():
                intro_contributions = self._parse_contributions_from_intro(
                    section,
                    paper_content
                )
                contributions.extend(intro_contributions)

        return contributions

    def _is_contribution_section(self, section: Section) -> bool:
        """Check if section is about contributions"""
        contribution_keywords = [
            'contribution',
            'contributions',
            'our work',
            'main contribution',
            'key contribution'
        ]

        section_title = section.title.lower()
        return any(keyword in section_title for keyword in contribution_keywords)

    def _parse_contributions_from_section(self, section: Section,
                                         paper_content: PaperContent) -> List[Contribution]:
        """Parse contributions from a contribution section"""
        contributions = []
        content = section.get_full_text()

        # Look for numbered or bulleted lists
        # Pattern 1: Numbered list (1., 2., 3. or (1), (2), (3))
        numbered_pattern = r'(?:^|\n)\s*(?:\(?\d+[\.\)]\s+|\-\s+|\*\s+)([^\n]+(?:\n(?!(?:\(?\d+[\.\)]\s+|\-\s+|\*\s+))[^\n]+)*)'

        matches = re.finditer(numbered_pattern, content, re.MULTILINE)

        for i, match in enumerate(matches, 1):
            contribution_text = match.group(1).strip()

            # Try to extract title and description
            lines = contribution_text.split('\n')
            title = lines[0][:100]  # First line as title
            description = '\n'.join(lines[1:]) if len(lines) > 1 else contribution_text

            contribution = Contribution(
                title=title,
                description=description,
                section_ref=section.title
            )

            contributions.append(contribution)

        return contributions

    def _parse_contributions_from_intro(self, section: Section,
                                        paper_content: PaperContent) -> List[Contribution]:
        """Parse contributions from introduction section"""
        contributions = []
        content = section.get_full_text()

        # Look for phrases like "our contributions are" or "we make the following contributions"
        contribution_phrases = [
            r'our contributions?(?:\s+are)?:?\s*',
            r'we make the following contributions:',
            r'the main contributions? (?:of this work|are):',
            r'our main contributions? (?:are|include):'
        ]

        for phrase in contribution_phrases:
            match = re.search(phrase, content, re.IGNORECASE)
            if match:
                # Extract text after the phrase
                start = match.end()
                # Find the end (next section or paragraph break)
                end_match = re.search(r'\n\n[A-Z]', content[start:start+2000])
                if end_match:
                    end = start + end_match.start()
                else:
                    end = start + 2000

                contribution_text = content[start:end].strip()

                # Split into individual contributions
                # Look for numbered items or bullet points
                items = re.split(r'\n\s*(?:\d+[\.\)]|\-|\*)\s+', contribution_text)

                for i, item in enumerate(items[1:], 1):  # Skip first empty item
                    if len(item.strip()) > 20:  # Only consider substantial items
                        lines = item.strip().split('\n')
                        title = lines[0][:100]
                        description = '\n'.join(lines[1:]) if len(lines) > 1 else item.strip()

                        contribution = Contribution(
                            title=title,
                            description=description,
                            section_ref=section.title
                        )
                        contributions.append(contribution)

                break  # Only process first match

        return contributions

    def _contribution_to_innovation(self, contribution: Contribution,
                                    paper_content: PaperContent) -> Optional[Innovation]:
        """Convert a contribution to an innovation"""
        self.innovation_counter += 1

        # Determine innovation type
        innovation_type = self._classify_innovation_type(contribution)

        # Generate innovation
        innovation = Innovation(
            id=f"innovation_{self.innovation_counter:03d}",
            name=contribution.title,
            description=contribution.description,
            innovation_type=innovation_type,
            source_paper=paper_content.metadata.title or "Unknown",
            source_section=contribution.section_ref,
            algorithm_description=self._extract_algorithm_description(contribution),
            input_spec=self._extract_input_spec(contribution),
            output_spec=self._extract_output_spec(contribution),
            hyperparameters=self._extract_hyperparameters(contribution),
            expected_benefits=self._extract_expected_benefits(contribution),
            confidence_score=0.7  # Default confidence
        )

        return innovation

    def _classify_innovation_type(self, contribution: Contribution) -> InnovationType:
        """Classify the type of innovation"""
        title = contribution.title.lower()
        description = contribution.description.lower()
        text = title + " " + description

        # Check for architecture-related keywords
        arch_keywords = ['architecture', 'backbone', 'encoder', 'decoder',
                          'network design', 'model structure', 'transformer',
                          'attention mechanism', 'feature extraction']
        if any(kw in text for kw in arch_keywords):
            return InnovationType.ARCHITECTURE

        # Check for training-related keywords
        training_keywords = ['training', 'optimization', 'loss function',
                            'learning rate', 'data augmentation', 'regularization',
                            'pre-training', 'fine-tuning', 'curriculum learning']
        if any(kw in text for kw in training_keywords):
            return InnovationType.TRAINING

        # Check for perception-related keywords
        perception_keywords = ['detection', 'segmentation', 'tracking',
                              'object recognition', '3d detection', 'bev',
                              'bird\'s eye view', 'depth estimation', 'lidar']
        if any(kw in text for kw in perception_keywords):
            return InnovationType.PERCEPTION

        # Check for prediction-related keywords
        prediction_keywords = ['prediction', 'forecasting', 'trajectory',
                              'motion prediction', 'behavior prediction',
                              'future prediction', 'multi-modal prediction']
        if any(kw in text for kw in prediction_keywords):
            return InnovationType.PREDICTION

        # Check for planning-related keywords
        planning_keywords = ['planning', 'decision making', 'path planning',
                            'trajectory planning', 'motion planning',
                            'behavior planning', 'route planning']
        if any(kw in text for kw in planning_keywords):
            return InnovationType.PLANNING

        # Check for efficiency-related keywords
        efficiency_keywords = ['efficiency', 'speed', 'latency', 'inference time',
                              'memory', 'compression', 'quantization',
                              'pruning', 'distillation', 'lightweight']
        if any(kw in text for kw in efficiency_keywords):
            return InnovationType.EFFICIENCY

        # Check for data-related keywords
        data_keywords = ['data', 'dataset', 'annotation', 'synthetic',
                          'simulation', 'augmentation', 'preprocessing',
                          'sampling', 'imbalance']
        if any(kw in text for kw in data_keywords):
            return InnovationType.DATA

        return InnovationType.UNKNOWN

    def _extract_algorithm_description(self, contribution: Contribution) -> str:
        """Extract algorithm description from contribution"""
        # This would use LLM or pattern matching to extract algorithm details
        # For now, return the description
        return contribution.description

    def _extract_input_spec(self, contribution: Contribution) -> Dict[str, Any]:
        """Extract input specification from contribution"""
        # Pattern matching to find input specifications
        spec = {
            'description': 'Input not specified',
            'shape': None,
            'type': None,
            'constraints': []
        }

        text = contribution.description.lower()

        # Look for input shape information
        shape_patterns = [
            r'input[^.]*?(?:\(|\[)([^\)\]]+)(?:\)|\])',
            r'(?:shape|size)[^.]*?(\d+[×x,\s\d]+)',
            r'(\d+)\s*×\s*(\d+)\s*×\s*(\d+)'
        ]

        for pattern in shape_patterns:
            match = re.search(pattern, text)
            if match:
                spec['shape'] = match.group(1)
                break

        return spec

    def _extract_output_spec(self, contribution: Contribution) -> Dict[str, Any]:
        """Extract output specification from contribution"""
        spec = {
            'description': 'Output not specified',
            'shape': None,
            'type': None
        }
        return spec

    def _extract_hyperparameters(self, contribution: Contribution) -> List[Dict[str, Any]]:
        """Extract hyperparameters from contribution"""
        hyperparams = []

        # Look for hyperparameter patterns
        text = contribution.description

        # Pattern: parameter name followed by value
        param_patterns = [
            r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([\d.]+)',
            r'([a-zA-Z_][a-zA-Z0-9_]*)\s+of\s+([\d.]+)',
            r'set\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+to\s+([\d.]+)'
        ]

        for pattern in param_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                name = match.group(1)
                value = match.group(2)

                # Try to infer type
                try:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                    value_type = 'numeric'
                except ValueError:
                    value_type = 'string'

                hyperparams.append({
                    'name': name,
                    'default_value': value,
                    'type': value_type,
                    'description': f'Extracted from paper'
                })

        return hyperparams

    def _extract_expected_benefits(self, contribution: Contribution) -> List[str]:
        """Extract expected benefits from contribution"""
        benefits = []

        # Look for benefit indicators
        text = contribution.description.lower()

        benefit_patterns = [
            r'improve[s\s]+([^.,]+)',
            r'achieve[s\s]+([^.,]+)',
            r'enhance[s\s]+([^.,]+)',
            r'reduce[s\s]+([^.,]+)',
            r'increase[s\s]+([^.,]+)',
            r'better\s+([^.,]+)',
            r'outperform[s\s]+([^.,]+)'
        ]

        for pattern in benefit_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                benefit = match.group(1).strip()
                if benefit and len(benefit) > 3:
                    benefits.append(f"{match.group(0).split()[0].capitalize()} {benefit}")

        # Remove duplicates while preserving order
        seen = set()
        unique_benefits = []
        for b in benefits:
            if b.lower() not in seen:
                seen.add(b.lower())
                unique_benefits.append(b)

        return unique_benefits[:5]  # Return top 5 benefits

    def _extract_from_sections(self, paper_content: PaperContent) -> List[Innovation]:
        """Extract additional innovations from individual sections"""
        innovations = []

        # Focus on method and experiments sections
        method_section = paper_content.get_method_section()
        exp_section = paper_content.get_experiments_section()

        # Extract from method section
        if method_section:
            method_innovations = self._extract_from_method_section(
                method_section,
                paper_content
            )
            innovations.extend(method_innovations)

        # Extract from experiments section
        if exp_section:
            exp_innovations = self._extract_from_experiments_section(
                exp_section,
                paper_content
            )
            innovations.extend(exp_innovations)

        return innovations

    def _extract_from_method_section(self, section: Section,
                                       paper_content: PaperContent) -> List[Innovation]:
        """Extract innovations from method section"""
        innovations = []

        # Look for novel techniques in method section
        text = section.get_full_text()

        # Look for "we propose", "we introduce", etc.
        proposal_patterns = [
            r'we\s+propose[d\s]+([^.,]+(?:\.[^.,]+)?)',
            r'we\s+introduce[d\s]+([^.,]+(?:\.[^.,]+)?)',
            r'we\s+present[d\s]+([^.,]+(?:\.[^.,]+)?)',
            r'we\s+develop[d\s]+([^.,]+(?:\.[^.,]+)?)',
        ]

        for pattern in proposal_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                proposal_text = match.group(1).strip()

                # Create innovation
                self.innovation_counter += 1

                innovation = Innovation(
                    id=f"innovation_{self.innovation_counter:03d}",
                    name=proposal_text[:100],
                    description=proposal_text,
                    innovation_type=InnovationType.ARCHITECTURE,  # Default to architecture
                    source_paper=paper_content.metadata.title or "Unknown",
                    source_section=section.title,
                    confidence_score=0.6
                )

                innovations.append(innovation)

        return innovations

    def _extract_from_experiments_section(self, section: Section,
                                           paper_content: PaperContent) -> List[Innovation]:
        """Extract innovations from experiments section"""
        # Usually fewer innovations in experiments section
        # Mainly looking for novel evaluation methods or metrics
        return []

    def _deduplicate_and_rank(self, innovations: List[Innovation]) -> List[Innovation]:
        """Deduplicate and rank innovations by importance"""
        # Remove duplicates based on similarity
        unique_innovations = []
        seen_descriptions = set()

        for innovation in innovations:
            # Create key from description (lowercase, first 50 chars)
            key = innovation.description.lower()[:50]

            if key not in seen_descriptions:
                seen_descriptions.add(key)
                unique_innovations.append(innovation)

        # Rank by confidence score
        ranked = sorted(
            unique_innovations,
            key=lambda x: x.confidence_score,
            reverse=True
        )

        return ranked

    def _analyze_complexity(self, innovation: Innovation) -> str:
        """Analyze technical complexity"""
        description = innovation.description.lower()
        algorithm = innovation.algorithm_description.lower()
        text = description + " " + algorithm

        # Count complexity indicators
        complexity_score = 0

        # Complex indicators
        complex_indicators = [
            'multi-stage', 'hierarchical', 'recursive', 'attention',
            'transformer', 'graph neural', 'reinforcement learning',
            'generative', 'adversarial', 'variational', 'bayesian',
            'optimization', 'differential', 'end-to-end'
        ]
        for indicator in complex_indicators:
            if indicator in text:
                complexity_score += 2

        # Simple indicators
        simple_indicators = [
            'simple', 'lightweight', 'efficient', 'fast',
            'straightforward', 'baseline', 'standard'
        ]
        for indicator in simple_indicators:
            if indicator in text:
                complexity_score -= 1

        # Determine complexity level
        if complexity_score >= 4:
            return "complex"
        elif complexity_score >= 1:
            return "moderate"
        else:
            return "simple"

    def _estimate_effort(self, innovation: Innovation) -> str:
        """Estimate implementation effort"""
        complexity = self._analyze_complexity(innovation)

        effort_map = {
            "simple": "1-3 days",
            "moderate": "1-2 weeks",
            "complex": "1-2 months"
        }

        return effort_map.get(complexity, "unknown")

    def _check_compatibility(self, innovation: Innovation,
                            target_system: str) -> Dict[str, Any]:
        """Check compatibility with target system"""
        return {
            'vad_compatible': True,  # Would need actual implementation
            'ultraad_compatible': True,
            'breaking_changes': [],
            'deprecation_warnings': [],
            'notes': [
                f"Innovation type: {innovation.innovation_type.value}",
                f"Implementation effort: {self._estimate_effort(innovation)}"
            ]
        }

    def _estimate_resources(self, innovation: Innovation) -> ResourceEstimate:
        """Estimate resource requirements"""
        complexity = self._analyze_complexity(innovation)

        resource_map = {
            "simple": ResourceEstimate(
                compute_hours=8.0,
                gpu_memory_gb=8.0,
                storage_gb=10.0,
                data_size_gb=1.0
            ),
            "moderate": ResourceEstimate(
                compute_hours=80.0,
                gpu_memory_gb=16.0,
                storage_gb=50.0,
                data_size_gb=10.0
            ),
            "complex": ResourceEstimate(
                compute_hours=320.0,
                gpu_memory_gb=32.0,
                storage_gb=200.0,
                data_size_gb=50.0
            )
        }

        return resource_map.get(complexity, ResourceEstimate(
            compute_hours=0.0,
            gpu_memory_gb=0.0,
            storage_gb=0.0,
            data_size_gb=0.0
        ))

    def _analyze_dependencies(self, innovation: Innovation) -> List[DependencySpec]:
        """Analyze dependencies"""
        dependencies = []

        # Common dependencies for deep learning
        dl_deps = [
            ('torch', '>=1.9.0', 'PyTorch framework'),
            ('numpy', '>=1.19.0', 'Numerical computations'),
            ('mmcv', '>=1.3.0', 'Computer vision utilities')
        ]

        for name, version, purpose in dl_deps:
            dependencies.append(DependencySpec(
                name=name,
                version_spec=version,
                optional=False,
                purpose=purpose
            ))

        # Add type-specific dependencies
        type_deps = {
            InnovationType.ARCHITECTURE: [
                ('einops', '>=0.3.0', 'Tensor operations')
            ],
            InnovationType.TRAINING: [
                ('wandb', '>=0.12.0', 'Experiment tracking')
            ],
            InnovationType.PERCEPTION: [
                ('opencv-python', '>=4.5.0', 'Image processing')
            ]
        }

        for dep in type_deps.get(innovation.innovation_type, []):
            dependencies.append(DependencySpec(
                name=dep[0],
                version_spec=dep[1],
                optional=False,
                purpose=dep[2]
            ))

        return dependencies

    def _assess_risks(self, innovation: Innovation,
                     report: FeasibilityReport) -> List[Dict[str, Any]]:
        """Assess implementation risks"""
        risks = []

        # Technical complexity risk
        if report.technical_complexity == "complex":
            risks.append({
                'category': 'technical',
                'description': 'High technical complexity may lead to implementation challenges',
                'probability': 'high',
                'impact': 'high',
                'mitigation': 'Consider breaking down into smaller components'
            })

        # Resource risk
        if report.resources_needed.compute_hours > 160:  # > 1 month
            risks.append({
                'category': 'resource',
                'description': 'High resource requirements may exceed available capacity',
                'probability': 'medium',
                'impact': 'medium',
                'mitigation': 'Plan for phased implementation'
            })

        # Compatibility risk
        if not report.compatibility.get('ultraad_compatible', True):
            risks.append({
                'category': 'compatibility',
                'description': 'Potential compatibility issues with ultraAD system',
                'probability': 'medium',
                'impact': 'high',
                'mitigation': 'Thorough testing in isolated environment'
            })

        # Dependency risk
        if len(report.dependencies) > 5:
            risks.append({
                'category': 'dependency',
                'description': f'High number of dependencies ({len(report.dependencies)}) increases maintenance burden',
                'probability': 'low',
                'impact': 'medium',
                'mitigation': 'Regular dependency updates and compatibility checks'
            })

        return risks

    def _calculate_overall_feasibility(self, report: FeasibilityReport) -> FeasibilityLevel:
        """Calculate overall feasibility level"""
        # Scoring factors
        score = 0

        # Technical complexity (0-30 points)
        complexity_scores = {
            "simple": 30,
            "moderate": 20,
            "complex": 10
        }
        score += complexity_scores.get(report.technical_complexity, 15)

        # Compatibility (0-30 points)
        if report.compatibility.get('ultraad_compatible', False):
            score += 30
        elif report.compatibility.get('vad_compatible', False):
            score += 20
        else:
            score += 10

        # Resource requirements (0-20 points)
        if report.resources_needed.compute_hours < 40:
            score += 20
        elif report.resources_needed.compute_hours < 160:
            score += 15
        elif report.resources_needed.compute_hours < 320:
            score += 10
        else:
            score += 5

        # Risk level (0-20 points)
        high_risks = sum(1 for r in report.risks if r.get('impact') == 'high')
        if high_risks == 0:
            score += 20
        elif high_risks <= 2:
            score += 15
        elif high_risks <= 4:
            score += 10
        else:
            score += 5

        # Map score to feasibility level
        if score >= 75:
            return FeasibilityLevel.HIGH
        elif score >= 50:
            return FeasibilityLevel.MEDIUM
        else:
            return FeasibilityLevel.LOW

    def _generate_recommendations(self, report: FeasibilityReport) -> List[str]:
        """Generate implementation recommendations"""
        recommendations = []

        # Based on complexity
        if report.technical_complexity == "complex":
            recommendations.append(
                "Consider breaking down the implementation into smaller, manageable components"
            )

        # Based on resources
        if report.resources_needed.gpu_memory_gb > 16:
            recommendations.append(
                "Ensure adequate GPU memory is available; consider gradient checkpointing for memory efficiency"
            )

        # Based on dependencies
        if len(report.dependencies) > 5:
            recommendations.append(
                f"Manage the {len(report.dependencies)} dependencies carefully; consider pinning versions for reproducibility"
            )

        # Based on risks
        high_impact_risks = [r for r in report.risks if r.get('impact') == 'high']
        if high_impact_risks:
            recommendations.append(
                f"Address the {len(high_impact_risks)} high-impact risks early in the implementation"
            )

        # Based on feasibility
        if report.overall_feasibility == FeasibilityLevel.LOW:
            recommendations.append(
                "Consider an alternative approach or simplify the scope due to low feasibility"
            )
        elif report.overall_feasibility == FeasibilityLevel.HIGH:
            recommendations.append(
                "High feasibility indicates this can be implemented with confidence"
            )

        # General recommendations
        recommendations.extend([
            "Start with a prototype to validate the approach before full implementation",
            "Document the implementation thoroughly for future maintainers",
            "Include comprehensive tests covering edge cases and failure modes"
        ])

        return recommendations
