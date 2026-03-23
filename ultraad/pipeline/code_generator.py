"""代码生成器 - 根据创新点生成VAD集成代码"""

from typing import Dict
from rich.console import Console

from .types import Innovation, GeneratedCode, InnovationCategory

console = Console()


class CodeGenerator:
    """代码生成器"""

    def __init__(self):
        """初始化代码生成器"""
        pass

    def generate(self, innovation: Innovation) -> GeneratedCode:
        """
        生成代码

        Args:
            innovation: 创新点

        Returns:
            生成的代码
        """
        console.print(f"[dim]正在生成代码: {innovation.name}...[/]")

        # 根据创新点类别选择模板
        if innovation.category == InnovationCategory.TEMPORAL:
            files = self._generate_temporal_fusion_code(innovation)
        elif innovation.category == InnovationCategory.ATTENTION:
            files = self._generate_attention_code(innovation)
        elif innovation.category == InnovationCategory.PLANNING:
            files = self._generate_planning_code(innovation)
        else:
            files = self._generate_generic_code(innovation)

        module_name = f"vad_{innovation.id}"

        code = GeneratedCode(
            innovation_id=innovation.id,
            module_name=module_name,
            files=files,
            dependencies=["torch", "torch.nn"]
        )

        console.print(f"[green]✓ 代码生成完成: {module_name}[/]")
        console.print(f"  文件: {len(files)} 个")

        return code

    def _generate_temporal_fusion_code(self, innovation: Innovation) -> Dict[str, str]:
        """生成时间融合代码"""
        code = f'''"""
{innovation.name}

{innovation.description}
"""

import torch
import torch.nn as nn
from einops import rearrange


class EnhancedTemporalFusion(nn.Module):
    """增强的时间融合模块"""

    def __init__(self, embed_dim=256, num_frames=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_frames = num_frames

        # 自适应权重
        self.temporal_weight = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, num_frames),
            nn.Softmax(dim=-1)
        )

        # 时间注意力
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim, num_heads=8, dropout=0.1
        )

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * num_frames, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, bev_features):
        """
        Args:
            bev_features: [B, T, H*W, C] 时间序列特征

        Returns:
            fused_features: [B, H*W, C] 融合后特征
        """
        B, T, HW, C = bev_features.shape

        # 计算自适应权重
        mean_feat = bev_features.mean(dim=1)  # [B, HW, C]
        weights = self.temporal_weight(mean_feat)  # [B, HW, T]

        # 加权聚合
        weights = weights.unsqueeze(-1)  # [B, HW, T, 1]
        weighted = bev_features * weights  # [B, T, HW, C]
        aggregated = weighted.sum(dim=1)  # [B, HW, C]

        # 应用时间注意力
        attn_out, _ = self.temporal_attn(
            aggregated, aggregated, aggregated
        )

        # 残差连接
        output = aggregated + attn_out

        return output


def get_temporal_fusion_module(embed_dim=256, num_frames=3):
    """获取时间融合模块"""
    return EnhancedTemporalFusion(embed_dim, num_frames)
'''

        return {
            f"{innovation.id}_temporal.py": code,
            f"{innovation.id}_README.md": f"""
# {innovation.name}

## 描述
{innovation.description}

## 使用方法
```python
from {innovation.id}_temporal import get_temporal_fusion_module

fusion_module = get_temporal_fusion_module(embed_dim=256, num_frames=3)
output = fusion_module(input_features)
```

## 集成到 VAD
在 `projects/mmdet3d_plugin/VAD/VAD.py` 中:
1. 替换原有的时间融合模块
2. 更新配置文件
"""
        }

    def _generate_attention_code(self, innovation: Innovation) -> Dict[str, str]:
        """生成注意力代码"""
        code = f'''"""
{innovation.name}

{innovation.description}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseBEVAttention(nn.Module):
    """稀疏BEV注意力模块"""

    def __init__(self, embed_dim=256, num_heads=8, sparsity_ratio=0.5):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.sparsity_ratio = sparsity_ratio

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

        self.scale = (embed_dim // num_heads) ** -0.5

    def forward(self, x):
        """
        Args:
            x: [B, N, C] BEV特征

        Returns:
            out: [B, N, C] 注意力输出
        """
        B, N, C = x.shape

        # 生成Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 稀疏化：只计算部分位置的注意力
        k_sparse = self._sparsify(k, self.sparsity_ratio)

        # 注意力计算
        attn = (q @ k_sparse.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # 应用注意力
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)

        return out

    def _sparsify(self, x, ratio):
        """稀疏化张量"""
        # 简单的随机稀疏化
        mask = torch.rand_like(x[..., :1]) > ratio
        return x * mask.float()


def get_sparse_attention(embed_dim=256, num_heads=8):
    """获取稀疏注意力模块"""
    return SparseBEVAttention(embed_dim, num_heads)
'''

        return {
            f"{innovation.id}_attention.py": code
        }

    def _generate_planning_code(self, innovation: Innovation) -> Dict[str, str]:
        """生成规划代码"""
        code = f'''"""
{innovation.name}

{innovation.description}
"""

import torch
import torch.nn as nn


class EfficientTrajectoryDecoder(nn.Module):
    """高效的轨迹解码器"""

    def __init__(self, input_dim=256, num_modes=6, num_future_steps=6):
        super().__init__()
        self.num_modes = num_modes
        self.num_future_steps = num_future_steps

        # 轻量级解码头
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_modes * num_future_steps * 2)  # x, y坐标
        )

        # 模式权重
        self.mode_weights = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_modes)
        )

    def forward(self, bev_feat):
        """
        Args:
            bev_feat: [B, C] BEV特征

        Returns:
            trajectories: [B, num_modes, num_future_steps, 2]
            weights: [B, num_modes]
        """
        # 解码轨迹
        traj_flat = self.decoder(bev_feat)  # [B, M*T*2]
        trajectories = traj_flat.reshape(
            -1, self.num_modes, self.num_future_steps, 2
        )

        # 计算模式权重
        weights = self.mode_weights(bev_feat)
        weights = weights.softmax(dim=-1)  # [B, M]

        return trajectories, weights


def get_trajectory_decoder(input_dim=256):
    """获取轨迹解码器"""
    return EfficientTrajectoryDecoder(input_dim)
'''

        return {
            f"{innovation.id}_planning.py": code
        }

    def _generate_generic_code(self, innovation: Innovation) -> Dict[str, str]:
        """生成通用代码模板"""
        code = f'''"""
{innovation.name}

{innovation.description}

这是一个通用模板，需要根据具体创新点进行定制。
"""

import torch
import torch.nn as nn


class {innovation.name.replace(" ", "")}Module(nn.Module):
    """创新点模块"""

    def __init__(self, **kwargs):
        super().__init__()
        # TODO: 根据创新点实现模块
        pass

    def forward(self, x):
        # TODO: 实现前向传播
        return x
'''

        return {
            f"{innovation.id}_module.py": code
        }
