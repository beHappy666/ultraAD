"""
Plugin Generator Module

Responsible for generating complete, installable plugin packages from plugin designs.
"""

import os
import re
import json
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, PackageLoader, select_autoescape

from .plugin_designer import PluginDesign, Interface, Dependency, ConfigSchema, TestPlan
from .innovation_extractor import Innovation


@dataclass
class GeneratedFile:
    """Represents a generated file"""
    path: str
    content: str
    file_type: str = "python"  # python, markdown, yaml, json, etc.
    executable: bool = False

    def write(self, base_dir: str) -> str:
        """Write file to disk"""
        full_path = os.path.join(base_dir, self.path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(self.content)

        if self.executable:
            os.chmod(full_path, 0o755)

        return full_path


@dataclass
class GeneratedPlugin:
    """Complete generated plugin package"""
    name: str
    version: str
    output_dir: str
    design: PluginDesign
    files: List[GeneratedFile] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    author: str = "paper-to-plugin"

    def write_all(self) -> List[str]:
        """Write all files to disk"""
        written_paths = []
        for file in self.files:
            path = file.write(self.output_dir)
            written_paths.append(path)
        return written_paths

    def get_structure(self) -> Dict[str, Any]:
        """Get plugin directory structure"""
        structure = {}
        for file in self.files:
            parts = file.path.split('/')
            current = structure
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = file.file_type
        return structure


class PluginGenerator:
    """
    Main class for generating plugin packages from designs.
    """

    def __init__(self, template_dir: Optional[str] = None):
        """
        Initialize the generator.

        Args:
            template_dir: Directory containing templates
        """
        self.template_dir = template_dir or self._get_default_template_dir()
        self.jinja_env = self._setup_jinja_env()

    def _get_default_template_dir(self) -> str:
        """Get default template directory"""
        # Look for templates relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        templates_dir = os.path.join(current_dir, '..', 'templates')
        return os.path.abspath(templates_dir)

    def _setup_jinja_env(self) -> Environment:
        """Setup Jinja2 environment"""
        if os.path.exists(self.template_dir):
            loader = FileSystemLoader(self.template_dir)
        else:
            # Fall back to package loader
            loader = PackageLoader('paper_to_plugin', 'templates')

        env = Environment(
            loader=loader,
            autoescape=select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True
        )

        # Add custom filters
        env.filters['snake_case'] = self._to_snake_case
        env.filters['camel_case'] = self._to_camel_case
        env.filters['pascal_case'] = self._to_pascal_case
        env.filters['kebab_case'] = self._to_kebab_case

        return env

    @staticmethod
    def _to_snake_case(text: str) -> str:
        """Convert text to snake_case"""
        # Replace spaces and hyphens with underscores
        text = re.sub(r'[\s\-]+', '_', text)
        # Insert underscores between camelCase
        text = re.sub(r'([a-z])([A-Z])', r'\1_\2', text)
        # Convert to lowercase
        text = text.lower()
        # Remove non-alphanumeric characters
        text = re.sub(r'[^a-z0-9_]', '', text)
        return text

    @staticmethod
    def _to_camel_case(text: str) -> str:
        """Convert text to camelCase"""
        words = re.split(r'[\s\-_]+', text)
        return words[0].lower() + ''.join(word.capitalize() for word in words[1:])

    @staticmethod
    def _to_pascal_case(text: str) -> str:
        """Convert text to PascalCase"""
        words = re.split(r'[\s\-_]+', text)
        return ''.join(word.capitalize() for word in words)

    @staticmethod
    def _to_kebab_case(text: str) -> str:
        """Convert text to kebab-case"""
        text = re.sub(r'[\s_]+', '-', text)
        text = re.sub(r'([a-z])([A-Z])', r'\1-\2', text)
        text = text.lower()
        text = re.sub(r'[^a-z0-9\-]', '', text)
        return text

    def generate(self, design: PluginDesign,
                output_dir: str) -> GeneratedPlugin:
        """
        Generate a complete plugin package.

        Args:
            design: Plugin design specification
            output_dir: Output directory for generated files

        Returns:
            Generated plugin package
        """
        # Create plugin object
        plugin = GeneratedPlugin(
            name=design.name,
            version=design.version,
            output_dir=output_dir,
            design=design
        )

        # Generate all files
        plugin.files.extend(self._generate_package_files(design))
        plugin.files.extend(self._generate_source_files(design))
        plugin.files.extend(self._generate_test_files(design))
        plugin.files.extend(self._generate_doc_files(design))
        plugin.files.extend(self._generate_config_files(design))

        return plugin

    def _generate_package_files(self, design: PluginDesign) -> List[GeneratedFile]:
        """Generate package metadata files"""
        files = []

        # Setup.py
        setup_content = self._render_template('setup.py.j2', design=design)
        files.append(GeneratedFile(
            path='setup.py',
            content=setup_content,
            file_type='python'
        ))

        # setup.cfg
        setup_cfg_content = self._render_template('setup.cfg.j2', design=design)
        files.append(GeneratedFile(
            path='setup.cfg',
            content=setup_cfg_content,
            file_type='config'
        ))

        # pyproject.toml
        pyproject_content = self._render_template('pyproject.toml.j2', design=design)
        files.append(GeneratedFile(
            path='pyproject.toml',
            content=pyproject_content,
            file_type='toml'
        ))

        # MANIFEST.in
        manifest_content = self._render_template('MANIFEST.in.j2', design=design)
        files.append(GeneratedFile(
            path='MANIFEST.in',
            content=manifest_content,
            file_type='text'
        ))

        # LICENSE
        license_content = self._render_template('LICENSE.j2', design=design)
        files.append(GeneratedFile(
            path='LICENSE',
            content=license_content,
            file_type='text'
        ))

        # README.md
        readme_content = self._render_template('README.md.j2', design=design)
        files.append(GeneratedFile(
            path='README.md',
            content=readme_content,
            file_type='markdown'
        ))

        # .gitignore
        gitignore_content = self._render_template('gitignore.j2', design=design)
        files.append(GeneratedFile(
            path='.gitignore',
            content=gitignore_content,
            file_type='text'
        ))

        return files

    def _generate_source_files(self, design: PluginDesign) -> List[GeneratedFile]:
        """Generate Python source files"""
        files = []

        package_name = self._to_snake_case(design.name)

        # __init__.py
        init_content = self._render_template('package/__init__.py.j2', design=design)
        files.append(GeneratedFile(
            path=f'{package_name}/__init__.py',
            content=init_content,
            file_type='python'
        ))

        # Main module
        main_content = self._render_template('package/plugin.py.j2', design=design)
        files.append(GeneratedFile(
            path=f'{package_name}/plugin.py',
            content=main_content,
            file_type='python'
        ))

        # Core implementation
        core_content = self._render_template('package/core.py.j2', design=design)
        files.append(GeneratedFile(
            path=f'{package_name}/core.py',
            content=core_content,
            file_type='python'
        ))

        # Utilities
        utils_content = self._render_template('package/utils.py.j2', design=design)
        files.append(GeneratedFile(
            path=f'{package_name}/utils.py',
            content=utils_content,
            file_type='python'
        ))

        # Types
        types_content = self._render_template('package/types.py.j2', design=design)
        files.append(GeneratedFile(
            path=f'{package_name}/types.py',
            content=types_content,
            file_type='python'
        ))

        # Exceptions
        exceptions_content = self._render_template('package/exceptions.py.j2', design=design)
        files.append(GeneratedFile(
            path=f'{package_name}/exceptions.py',
            content=exceptions_content,
            file_type='python'
        ))

        return files

    def _generate_test_files(self, design: PluginDesign) -> List[GeneratedFile]:
        """Generate test files"""
        files = []

        package_name = self._to_snake_case(design.name)

        # tests/__init__.py
        test_init_content = self._render_template('tests/__init__.py.j2', design=design)
        files.append(GeneratedFile(
            path='tests/__init__.py',
            content=test_init_content,
            file_type='python'
        ))

        # conftest.py
        conftest_content = self._render_template('tests/conftest.py.j2', design=design)
        files.append(GeneratedFile(
            path='tests/conftest.py',
            content=conftest_content,
            file_type='python'
        ))

        # Unit tests
        unit_test_content = self._render_template('tests/test_unit.py.j2', design=design)
        files.append(GeneratedFile(
            path='tests/test_unit.py',
            content=unit_test_content,
            file_type='python'
        ))

        # Integration tests
        integration_test_content = self._render_template('tests/test_integration.py.j2', design=design)
        files.append(GeneratedFile(
            path='tests/test_integration.py',
            content=integration_test_content,
            file_type='python'
        ))

        # Performance tests
        perf_test_content = self._render_template('tests/test_performance.py.j2', design=design)
        files.append(GeneratedFile(
            path='tests/test_performance.py',
            content=perf_test_content,
            file_type='python'
        ))

        # Test data
        test_data_content = self._render_template('tests/data/.gitkeep.j2', design=design)
        files.append(GeneratedFile(
            path='tests/data/.gitkeep',
            content=test_data_content,
            file_type='text'
        ))

        return files

    def _generate_doc_files(self, design: PluginDesign) -> List[GeneratedFile]:
        """Generate documentation files"""
        files = []

        # API documentation
        api_doc_content = self._render_template('docs/API.md.j2', design=design)
        files.append(GeneratedFile(
            path='docs/API.md',
            content=api_doc_content,
            file_type='markdown'
        ))

        # Usage guide
        usage_content = self._render_template('docs/USAGE.md.j2', design=design)
        files.append(GeneratedFile(
            path='docs/USAGE.md',
            content=usage_content,
            file_type='markdown'
        ))

        # Configuration guide
        config_doc_content = self._render_template('docs/CONFIGURATION.md.j2', design=design)
        files.append(GeneratedFile(
            path='docs/CONFIGURATION.md',
            content=config_doc_content,
            file_type='markdown'
        ))

        # Development guide
        dev_doc_content = self._render_template('docs/DEVELOPMENT.md.j2', design=design)
        files.append(GeneratedFile(
            path='docs/DEVELOPMENT.md',
            content=dev_doc_content,
            file_type='markdown'
        ))

        # Changelog
        changelog_content = self._render_template('docs/CHANGELOG.md.j2', design=design)
        files.append(GeneratedFile(
            path='docs/CHANGELOG.md',
            content=changelog_content,
            file_type='markdown'
        ))

        return files

    def _generate_config_files(self, design: PluginDesign) -> List[GeneratedFile]:
        """Generate configuration files"""
        files = []

        # Default configuration
        default_config = {
            'plugin': {
                'name': design.name,
                'version': design.version,
                'enabled': True
            },
            'config': {}
        }

        # Add configuration schema defaults
        for schema in design.config_schema:
            default_config['config'][schema.name] = schema.default

        config_content = json.dumps(default_config, indent=2)
        files.append(GeneratedFile(
            path='config/default.json',
            content=config_content,
            file_type='json'
        ))

        # Production configuration template
        prod_config = default_config.copy()
        prod_config['plugin']['debug_mode'] = False
        prod_config['plugin']['log_level'] = 'INFO'

        prod_content = json.dumps(prod_config, indent=2)
        files.append(GeneratedFile(
            path='config/production.json',
            content=prod_content,
            file_type='json'
        ))

        # Development configuration
        dev_config = default_config.copy()
        dev_config['plugin']['debug_mode'] = True
        dev_config['plugin']['log_level'] = 'DEBUG'

        dev_content = json.dumps(dev_config, indent=2)
        files.append(GeneratedFile(
            path='config/development.json',
            content=dev_content,
            file_type='json'
        ))

        # YAML configuration alternative
        try:
            import yaml
            yaml_content = yaml.dump(default_config, default_flow_style=False)
            files.append(GeneratedFile(
                path='config/default.yaml',
                content=yaml_content,
                file_type='yaml'
            ))
        except ImportError:
            pass

        return files

    def _render_template(self, template_name: str, **kwargs) -> str:
        """Render a Jinja2 template"""
        try:
            template = self.jinja_env.get_template(template_name)
            return template.render(**kwargs)
        except Exception as e:
            # Fallback: generate basic content
            print(f"Warning: Could not render template {template_name}: {e}")
            return self._generate_fallback_content(template_name, **kwargs)

    def _generate_fallback_content(self, template_name: str, **kwargs) -> str:
        """Generate fallback content when template is missing"""
        design = kwargs.get('design')
        if not design:
            return f"# {template_name}\n# Template not found"

        # Basic fallback for common files
        if 'setup.py' in template_name:
            return f'''"""Setup script for {design.name}"""
from setuptools import setup, find_packages

setup(
    name="{design.name}",
    version="{design.version}",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
    ],
)
'''
        elif '__init__.py' in template_name:
            return f'''"""{design.name} - {design.description}"""

__version__ = "{design.version}"
__author__ = "{design.author}"

from .plugin import {self._to_pascal_case(design.name)}Plugin

__all__ = ['{self._to_pascal_case(design.name)}Plugin']
'''
        else:
            return f'''"""{template_name}
Generated for {design.name}
"""
# TODO: Implement this file
'''


# Convenience function
def generate_plugin(design: PluginDesign,
                   output_dir: str,
                   template_dir: Optional[str] = None) -> GeneratedPlugin:
    """
    Convenience function to generate a plugin from a design.

    Args:
        design: Plugin design specification
        output_dir: Output directory
        template_dir: Template directory

    Returns:
        Generated plugin package
    """
    generator = PluginGenerator(template_dir=template_dir)
    return generator.generate(design, output_dir)
