# CLAUDE.md - ultraAD 项目指南

## 项目概述

ultraAD 是基于 VAD (Vectorized Scene Representation) 的端到端自动驾驶算法开发调试系统。支持 nuScenes 数据集，后续将适配 nuPlan。

## 技术栈

- Python 3.8, PyTorch 1.12+ (CUDA 11.3)
- mmcv-full 1.6.0, mmdet 2.25.0, mmdet3d 1.0.0rc6, mmsegmentation 0.29.0
- nuscenes-devkit 1.1.9

## 常用命令

```bash
# 环境配置
bash scripts/setup_env.sh && conda activate vad_debug

# 数据准备
bash scripts/prepare_data.sh data/nuscenes

# 单样本调试
python scripts/debug_one_sample.py configs/VAD_tiny_debug.py

# 训练 (带调试钩子)
python scripts/train_debug.py configs/VAD_tiny_debug.py --debug-interval 50

# 训练/测试 (标准 VAD)
bash scripts/train.sh configs/VAD_base_debug.py
bash scripts/test.sh configs/VAD_base_debug.py <checkpoint>
```

## 项目结构

- `third_party/VAD/` - VAD 源码，不直接修改
- `configs/` - 调试用配置，继承 VAD 原始配置
- `debug_tools/` - BEV 可视化、管道检查、损失分析、模型检查
- `scripts/` - 环境配置、数据准备、训练、测试脚本

## 代码规范

- 修改 VAD 源码前先复制到项目目录，不要直接改 third_party
- 新增模块放在 `debug_tools/` 或新建目录，保持与 VAD 源码分离
- 配置文件通过 `_base_` 继承 VAD 原始配置，只覆盖需要修改的部分
