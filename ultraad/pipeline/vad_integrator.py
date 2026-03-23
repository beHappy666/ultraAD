"""VAD 集成器 - 将生成的代码集成到 VAD 框架"""

import os
import shutil
from pathlib import Path
from typing import Optional
from rich.console import Console

from .types import GeneratedCode

console = Console()


class VADIntegrator:
    """VAD 集成器"""

    def __init__(self, vad_root: str = None):
        """
        初始化集成器

        Args:
            vad_root: VAD 根目录
        """
        self.vad_root = Path(vad_root) if vad_root else Path(__file__).parent.parent.parent / "third_party" / "VAD" / "projects" / "mmdet3d_plugin"
        self.output_dir = self.vad_root / "generated"

    def integrate(self, code: GeneratedCode) -> str:
        """
        集成代码到 VAD

        Args:
            code: 生成的代码

        Returns:
            集成后的模块路径
        """
        console.print(f"[dim]正在集成到 VAD...[/]")

        # 创建输出目录
        module_dir = self.output_dir / code.module_name
        module_dir.mkdir(parents=True, exist_ok=True)

        # 写入文件
        for file_path, content in code.files.items():
            full_path = module_dir / file_path
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)

        # 创建 __init__.py
        init_path = module_dir / "__init__.py"
        if not init_path.exists():
            with open(init_path, 'w') as f:
                f.write(f'"""Generated module: {code.module_name}""')

        console.print(f"[green]✓ 代码集成完成[/]")
        console.print(f"  路径: {module_dir}")

        return str(module_dir)

    def update_vad_config(self, module_name: str, config_path: str = None):
        """
        更新 VAD 配置文件

        Args:
            module_name: 模块名称
            config_path: 配置文件路径
        """
        console.print(f"[dim]正在更新 VAD 配置...[/]")

        if config_path is None:
            # 使用默认配置路径
            vad_configs = Path(__file__).parent.parent.parent / "third_party" / "VAD" / "projects" / "configs" / "VAD"
            config_path = str(vad_configs / "VAD_tiny_stage_1.py")

        console.print(f"[dim]配置文件: {config_path}[/]")
        console.print(f"[yellow]⚠️  请手动检查并更新配置文件[/]")

        return config_path
