"""
Plugin Designer Module

Responsible for designing plugin architecture and interfaces based on extracted innovations.
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

from .types import PluginType, InnovationType
from .innovation_extractor import Innovation


@dataclass
class Interface:
    """Plugin interface definition"""
    name: str
    type: str  # 'data', 'control', 'event', 'config'
    schema: Dict[str, Any]
    description: str = ""
    required: bool = True
    default_value: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'type': self.type,
            'schema': self.schema,
            'description': self.description,
            'required': self.required,
            'default_value': self.default_value
        }


@dataclass
class Dependency:
    """Plugin dependency definition"""
    name: str
    version_spec: str
    optional: bool = False
    install_command: Optional[str] = None
    purpose: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'version_spec': self.version_spec,
            'optional': self.optional,
            'purpose': self.purpose
        }


@dataclass
class ConfigSchema:
    """Plugin configuration schema"""
    name: str
    type: str
    description: str = ""
    default: Any = None
    required: bool = False
    validation_rules: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'type': self.type,
            'description': self.description,
            'default': self.default,
            'required': self.required,
            'validation_rules': self.validation_rules
        }


@dataclass
class TestPlan:
    """Plugin test plan"""
    unit_tests: List[Dict[str, Any]] = field(default_factory=list)
    integration_tests: List[Dict[str, Any]] = field(default_factory=list)
    system_tests: List[Dict[str, Any]] = field(default_factory=list)
    performance_tests: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'unit_tests': self.unit_tests,
            'integration_tests': self.integration_tests,
            'system_tests': self.system_tests,
            'performance_tests': self.performance_tests
        }


@dataclass
class PluginDesign:
    """Complete plugin design specification"""
    # Identity
    name: str
    version: str = "0.1.0"
    description: str = ""

    # Classification
    plugin_type: PluginType = PluginType.GENERIC
    innovation_type: InnovationType = InnovationType.UNKNOWN

    # Architecture
    interfaces: List[Interface] = field(default_factory=list)
    dependencies: List[Dependency] = field(default_factory=list)
    config_schema: List[ConfigSchema] = field(default_factory=list)

    # Implementation
    base_class: str = "VADPluginBase"
    mixins: List[str] = field(default_factory=list)
    overrides: Dict[str, str] = field(default_factory=dict)

    # Testing
    test_plan: TestPlan = field(default_factory=TestPlan)

    # Metadata
    author: str = "paper-to-plugin"
    license: str = "MIT"
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'plugin_type': self.plugin_type.value,
            'innovation_type': self.innovation_type.value,
            'interfaces': [i.to_dict() for i in self.interfaces],
            'dependencies': [d.to_dict() for d in self.dependencies],
            'config_schema': [c.to_dict() for c in self.config_schema],
            'base_class': self.base_class,
            'test_plan': self.test_plan.to_dict(),
            'author': self.author,
            'license': self.license,
            'tags': self.tags
        }


class PluginDesigner:
    """
    Main class for designing plugin architecture based on innovations.
    """

    # Mapping from innovation types to plugin types
    TYPE_MAPPING = {
        InnovationType.ARCHITECTURE: PluginType.ARCHITECTURE,
        InnovationType.TRAINING: PluginType.TRAINING,
        InnovationType.PERCEPTION: PluginType.PERCEPTION,
        InnovationType.PREDICTION: PluginType.PERCEPTION,
        InnovationType.PLANNING: PluginType.PLANNING,
        InnovationType.EFFICIENCY: PluginType.ARCHITECTURE,
        InnovationType.DATA: PluginType.ARCHITECTURE,
        InnovationType.UNKNOWN: PluginType.GENERIC
    }

    # Base classes for different plugin types
    BASE_CLASSES = {
        PluginType.PERCEPTION: 'VADPerceptionHead',
        PluginType.TRAINING: 'VADTrainer',
        PluginType.PLANNING: 'VADPlanner',
        PluginType.ARCHITECTURE: 'VADBackbone',
        PluginType.GENERIC: 'VADPluginBase'
    }

    def __init__(self, template_dir: Optional[str] = None):
        """
        Initialize the designer.

        Args:
            template_dir: Directory containing plugin templates
        """
        self.template_dir = template_dir

    def design(self, innovation: Innovation) -> PluginDesign:
        """
        Design a plugin based on an innovation.

        Args:
            innovation: The innovation to design a plugin for

        Returns:
            Plugin design specification
        """
        design = PluginDesign(
            name=self._generate_plugin_name(innovation),
            description=innovation.description,
            innovation_type=innovation.innovation_type
        )

        # Set plugin type
        design.plugin_type = self.TYPE_MAPPING.get(
            innovation.innovation_type,
            PluginType.GENERIC
        )

        # Set base class
        design.base_class = self.BASE_CLASSES.get(
            design.plugin_type,
            'VADPluginBase'
        )

        # Design interfaces
        design.interfaces = self._design_interfaces(innovation, design)

        # Design dependencies
        design.dependencies = self._design_dependencies(innovation, design)

        # Design configuration schema
        design.config_schema = self._design_config_schema(innovation, design)

        # Design test plan
        design.test_plan = self._design_test_plan(innovation, design)

        # Set tags
        design.tags = self._generate_tags(innovation, design)

        return design

    def _generate_plugin_name(self, innovation: Innovation) -> str:
        """Generate a plugin name from innovation"""
        # Clean and format the innovation name
        name = innovation.name

        # Remove special characters
        name = re.sub(r'[^\w\s]', '', name)

        # Convert to snake_case
        name = name.lower().replace(' ', '_')

        # Limit length
        name = name[:50]

        # Add prefix if needed
        if not name.startswith('vad_'):
            name = f'vad_{name}'

        return name

    def _design_interfaces(self, innovation: Innovation,
                          design: PluginDesign) -> List[Interface]:
        """Design plugin interfaces"""
        interfaces = []

        # Input interface (common for all plugins)
        input_interface = Interface(
            name='input',
            type='data',
            schema=self._infer_input_schema(innovation),
            description='Input data interface',
            required=True
        )
        interfaces.append(input_interface)

        # Output interface (common for all plugins)
        output_interface = Interface(
            name='output',
            type='data',
            schema=self._infer_output_schema(innovation),
            description='Output data interface',
            required=True
        )
        interfaces.append(output_interface)

        # Control interface (optional)
        if self._requires_control(innovation):
            control_interface = Interface(
                name='control',
                type='control',
                schema={
                    'type': 'object',
                    'properties': {
                        'enabled': {'type': 'boolean'},
                        'mode': {'type': 'string'}
                    }
                },
                description='Control interface',
                required=False,
                default_value={'enabled': True, 'mode': 'default'}
            )
            interfaces.append(control_interface)

        # Config interface (for configurable plugins)
        if design.config_schema:
            config_interface = Interface(
                name='config',
                type='config',
                schema={
                    'type': 'object',
                    'properties': {
                        schema.name: {
                            'type': schema.type,
                            'description': schema.description,
                            'default': schema.default
                        }
                        for schema in design.config_schema
                    }
                },
                description='Configuration interface',
                required=False
            )
            interfaces.append(config_interface)

        return interfaces

    def _infer_input_schema(self, innovation: Innovation) -> Dict[str, Any]:
        """Infer input schema from innovation"""
        # Default schema
        schema = {
            'type': 'object',
            'properties': {
                'data': {
                    'type': 'array',
                    'description': 'Input data'
                }
            },
            'required': ['data']
        }

        # Try to infer from innovation type
        type_schemas = {
            InnovationType.PERCEPTION: {
                'type': 'object',
                'properties': {
                    'images': {
                        'type': 'array',
                        'items': {'type': 'string'},
                        'description': 'Input image paths or tensors'
                    },
                    'camera_params': {
                        'type': 'object',
                        'description': 'Camera calibration parameters'
                    }
                },
                'required': ['images']
            },
            InnovationType.TRAINING: {
                'type': 'object',
                'properties': {
                    'model': {
                        'type': 'object',
                        'description': 'Model to train'
                    },
                    'train_data': {
                        'type': 'object',
                        'description': 'Training data loader'
                    },
                    'val_data': {
                        'type': 'object',
                        'description': 'Validation data loader'
                    }
                },
                'required': ['model', 'train_data']
            },
            InnovationType.PLANNING: {
                'type': 'object',
                'properties': {
                    'ego_state': {
                        'type': 'object',
                        'description': 'Ego vehicle state'
                    },
                    'perception_output': {
                        'type': 'object',
                        'description': 'Perception module output'
                    },
                    'map_data': {
                        'type': 'object',
                        'description': 'HD map data'
                    }
                },
                'required': ['ego_state']
            }
        }

        return type_schemas.get(innovation.innovation_type, schema)

    def _infer_output_schema(self, innovation: Innovation) -> Dict[str, Any]:
        """Infer output schema from innovation"""
        # Default schema
        schema = {
            'type': 'object',
            'properties': {
                'result': {
                    'type': 'object',
                    'description': 'Output result'
                }
            },
            'required': ['result']
        }

        # Type-specific schemas
        type_schemas = {
            InnovationType.PERCEPTION: {
                'type': 'object',
                'properties': {
                    'detections': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'properties': {
                                'class': {'type': 'string'},
                                'confidence': {'type': 'number'},
                                'bbox': {'type': 'array', 'items': {'type': 'number'}}
                            }
                        },
                        'description': 'Detected objects'
                    },
                    'features': {
                        'type': 'object',
                        'description': 'Extracted features'
                    }
                },
                'required': ['detections']
            },
            InnovationType.TRAINING: {
                'type': 'object',
                'properties': {
                    'trained_model': {
                        'type': 'object',
                        'description': 'Trained model checkpoint'
                    },
                    'training_history': {
                        'type': 'object',
                        'description': 'Training metrics history'
                    },
                    'final_metrics': {
                        'type': 'object',
                        'description': 'Final evaluation metrics'
                    }
                },
                'required': ['trained_model']
            },
            InnovationType.PLANNING: {
                'type': 'object',
                'properties': {
                    'trajectory': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'properties': {
                                'x': {'type': 'number'},
                                'y': {'type': 'number'},
                                'heading': {'type': 'number'},
                                'velocity': {'type': 'number'},
                                'timestamp': {'type': 'number'}
                            }
                        },
                        'description': 'Planned trajectory'
                    },
                    'confidence': {
                        'type': 'number',
                        'description': 'Planning confidence score'
                    },
                    'alternative_paths': {
                        'type': 'array',
                        'description': 'Alternative trajectory options'
                    }
                },
                'required': ['trajectory']
            }
        }

        return type_schemas.get(innovation.innovation_type, schema)

    def _requires_control(self, innovation: Innovation) -> bool:
        """Check if innovation requires control interface"""
        # Control-related keywords
        control_keywords = [
            'enable', 'disable', 'mode', 'control', 'configure',
            'adjust', 'tune', 'parameter', 'setting'
        ]

        text = (innovation.description + " " + innovation.name).lower()
        return any(kw in text for kw in control_keywords)

    def _design_dependencies(self, innovation: Innovation,
                            design: PluginDesign) -> List[Dependency]:
        """Design plugin dependencies"""
        dependencies = []

        # Core dependencies (always required)
        core_deps = [
            Dependency(
                name='torch',
                version_spec='>=1.9.0',
                optional=False,
                purpose='Deep learning framework'
            ),
            Dependency(
                name='numpy',
                version_spec='>=1.19.0',
                optional=False,
                purpose='Numerical computations'
            ),
            Dependency(
                name='mmcv',
                version_spec='>=1.3.0',
                optional=False,
                purpose='Computer vision utilities'
            )
        ]
        dependencies.extend(core_deps)

        # Type-specific dependencies
        type_deps = {
            InnovationType.ARCHITECTURE: [
                Dependency(
                    name='einops',
                    version_spec='>=0.3.0',
                    optional=True,
                    purpose='Tensor operations'
                )
            ],
            InnovationType.TRAINING: [
                Dependency(
                    name='wandb',
                    version_spec='>=0.12.0',
                    optional=True,
                    purpose='Experiment tracking'
                ),
                Dependency(
                    name='tensorboard',
                    version_spec='>=2.8.0',
                    optional=True,
                    purpose='Training visualization'
                )
            ],
            InnovationType.PERCEPTION: [
                Dependency(
                    name='opencv-python',
                    version_spec='>=4.5.0',
                    optional=False,
                    purpose='Image processing'
                )
            ],
            InnovationType.PLANNING: [
                Dependency(
                    name='scipy',
                    version_spec='>=1.7.0',
                    optional=False,
                    purpose='Optimization algorithms'
                )
            ]
        }

        if innovation.innovation_type in type_deps:
            dependencies.extend(type_deps[innovation.innovation_type])

        # Check for specific algorithm requirements
        text = innovation.description.lower()

        if 'attention' in text or 'transformer' in text:
            dependencies.append(Dependency(
                name='transformers',
                version_spec='>=4.15.0',
                optional=True,
                purpose='Transformer models'
            ))

        if 'graph' in text or 'gnn' in text:
            dependencies.append(Dependency(
                name='torch-geometric',
                version_spec='>=2.0.0',
                optional=True,
                purpose='Graph neural networks'
            ))

        return dependencies

    def _design_config_schema(self, innovation: Innovation,
                             design: PluginDesign) -> List[ConfigSchema]:
        """Design plugin configuration schema"""
        config_schemas = []

        # Extract hyperparameters as config
        for hp in innovation.hyperparameters:
            schema = ConfigSchema(
                name=hp.get('name', 'param'),
                type=hp.get('type', 'string'),
                description=hp.get('description', ''),
                default=hp.get('default_value'),
                required=False
            )
            config_schemas.append(schema)

        # Add standard config options based on plugin type
        type_configs = {
            PluginType.TRAINING: [
                ConfigSchema(
                    name='learning_rate',
                    type='number',
                    description='Learning rate for training',
                    default=0.001,
                    required=False
                ),
                ConfigSchema(
                    name='batch_size',
                    type='integer',
                    description='Batch size for training',
                    default=32,
                    required=False
                ),
                ConfigSchema(
                    name='num_epochs',
                    type='integer',
                    description='Number of training epochs',
                    default=100,
                    required=False
                )
            ],
            PluginType.PERCEPTION: [
                ConfigSchema(
                    name='confidence_threshold',
                    type='number',
                    description='Confidence threshold for detections',
                    default=0.5,
                    required=False
                ),
                ConfigSchema(
                    name='nms_threshold',
                    type='number',
                    description='NMS threshold',
                    default=0.5,
                    required=False
                )
            ],
            PluginType.PLANNING: [
                ConfigSchema(
                    name='planning_horizon',
                    type='number',
                    description='Planning horizon in seconds',
                    default=8.0,
                    required=False
                ),
                ConfigSchema(
                    name='time_step',
                    type='number',
                    description='Time step for planning',
                    default=0.5,
                    required=False
                )
            ]
        }

        if design.plugin_type in type_configs:
            config_schemas.extend(type_configs[design.plugin_type])

        # Add common config
        common_configs = [
            ConfigSchema(
                name='enabled',
                type='boolean',
                description='Whether the plugin is enabled',
                default=True,
                required=False
            ),
            ConfigSchema(
                name='debug_mode',
                type='boolean',
                description='Enable debug mode for verbose logging',
                default=False,
                required=False
            ),
            ConfigSchema(
                name='log_level',
                type='string',
                description='Logging level',
                default='INFO',
                required=False,
                validation_rules=[
                    {'type': 'enum', 'values': ['DEBUG', 'INFO', 'WARNING', 'ERROR']}
                ]
            )
        ]

        config_schemas.extend(common_configs)

        return config_schemas

    def _design_test_plan(self, innovation: Innovation,
                         design: PluginDesign) -> TestPlan:
        """Design plugin test plan"""
        test_plan = TestPlan()

        # Unit tests
        test_plan.unit_tests = [
            {
                'name': 'test_initialization',
                'description': 'Test plugin initialization',
                'type': 'unit'
            },
            {
                'name': 'test_config_loading',
                'description': 'Test configuration loading',
                'type': 'unit'
            },
            {
                'name': 'test_input_validation',
                'description': 'Test input validation',
                'type': 'unit'
            },
            {
                'name': 'test_output_format',
                'description': 'Test output format correctness',
                'type': 'unit'
            }
        ]

        # Add type-specific unit tests
        if design.plugin_type == PluginType.TRAINING:
            test_plan.unit_tests.extend([
                {
                    'name': 'test_training_step',
                    'description': 'Test single training step',
                    'type': 'unit'
                },
                {
                    'name': 'test_loss_computation',
                    'description': 'Test loss computation',
                    'type': 'unit'
                }
            ])
        elif design.plugin_type == PluginType.PERCEPTION:
            test_plan.unit_tests.extend([
                {
                    'name': 'test_detection_output',
                    'description': 'Test detection output format',
                    'type': 'unit'
                },
                {
                    'name': 'test_feature_extraction',
                    'description': 'Test feature extraction',
                    'type': 'unit'
                }
            ])

        # Integration tests
        test_plan.integration_tests = [
            {
                'name': 'test_vad_integration',
                'description': 'Test integration with VAD framework',
                'type': 'integration',
                'dependencies': ['VAD']
            },
            {
                'name': 'test_ultraad_integration',
                'description': 'Test integration with ultraAD debugging tools',
                'type': 'integration',
                'dependencies': ['ultraAD']
            },
            {
                'name': 'test_config_integration',
                'description': 'Test configuration system integration',
                'type': 'integration'
            }
        ]

        # System tests
        test_plan.system_tests = [
            {
                'name': 'test_end_to_end',
                'description': 'Test complete end-to-end workflow',
                'type': 'system',
                'dataset': 'nuscenes_mini'
            },
            {
                'name': 'test_error_handling',
                'description': 'Test error handling and recovery',
                'type': 'system'
            }
        ]

        # Performance tests
        test_plan.performance_tests = [
            {
                'name': 'test_inference_speed',
                'description': 'Test inference speed',
                'type': 'performance',
                'metrics': ['fps', 'latency']
            },
            {
                'name': 'test_memory_usage',
                'description': 'Test memory usage',
                'type': 'performance',
                'metrics': ['gpu_memory', 'cpu_memory']
            },
            {
                'name': 'test_accuracy',
                'description': 'Test accuracy metrics',
                'type': 'performance',
                'metrics': ['accuracy', 'precision', 'recall', 'f1']
            }
        ]

        return test_plan

    def _generate_tags(self, innovation: Innovation, design: PluginDesign) -> List[str]:
        """Generate tags for the plugin"""
        tags = []

        # Add innovation type
        tags.append(innovation.innovation_type.value)

        # Add plugin type
        tags.append(design.plugin_type.value)

        # Add keywords from innovation name
        keywords = innovation.name.lower().split()
        tags.extend([k for k in keywords if len(k) > 3])

        # Add common tags
        if 'attention' in innovation.description.lower():
            tags.append('attention')
        if 'transformer' in innovation.description.lower():
            tags.append('transformer')
        if 'cnn' in innovation.description.lower() or 'convolution' in innovation.description.lower():
            tags.append('cnn')
        if 'graph' in innovation.description.lower() or 'gnn' in innovation.description.lower():
            tags.append('gnn')

        # Remove duplicates and limit
        unique_tags = list(dict.fromkeys(tags))
        return unique_tags[:10]


# Convenience function
def design_plugin(innovation: Innovation,
                   template_dir: Optional[str] = None) -> PluginDesign:
    """
    Convenience function to design a plugin from an innovation.

    Args:
        innovation: The innovation to design a plugin for
        template_dir: Directory containing templates

    Returns:
        Plugin design specification
    """
    designer = PluginDesigner(template_dir=template_dir)
    return designer.design(innovation)
