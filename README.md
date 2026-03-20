# ultraAD

基于 [VAD (Vectorized Scene Representation for Efficient Autonomous Driving)](https://github.com/hustvl/VAD) 的端到端自动驾驶算法开发调试系统。

## 项目结构

```
ultraAD/
├── third_party/VAD/           # VAD 源码 (已克隆)
├── configs/                   # 调试用配置文件
│   ├── VAD_base_debug.py      # Base 模型调试配置
│   └── VAD_tiny_debug.py      # Tiny 模型调试配置
├── debug_tools/               # 调试工具集
│   ├── visualize_bev.py       # BEV 可视化 (检测框/地图/规划轨迹)
│   ├── visualize_pipeline.py  # 数据管道逐阶段检查
│   ├── loss_analyzer.py       # 损失曲线/梯度流/权重分布分析
│   └── model_inspector.py     # 模型架构/参数/推理时间分析
├── scripts/                   # 运行脚本
│   ├── setup_env.sh           # 环境配置
│   ├── prepare_data.sh        # 数据准备
│   ├── train.sh               # VAD 标准训练
│   ├── train_debug.py         # 带调试钩子的训练
│   ├── test.sh                # 模型测试
│   └── debug_one_sample.py    # 单样本全管道调试
├── data/nuscenes/             # nuScenes 数据 (需下载)
├── logs/                      # 日志和可视化输出
├── checkpoints/               # 预训练权重
└── work_dirs/                 # 训练输出
```

## 快速开始

### 1. 环境配置

```bash
bash scripts/setup_env.sh
conda activate vad_debug
```

### 2. 数据准备

下载 nuScenes 数据集到 `data/nuscenes/`，然后：

```bash
bash scripts/prepare_data.sh data/nuscenes
```

### 3. 单样本调试

检查整个管道是否正常工作：

```bash
python scripts/debug_one_sample.py configs/VAD_tiny_debug.py
```

输出包括：
- 数据管道各阶段的 tensor shape/stats
- 模型架构总结和参数量
- BEV 特征热力图
- GT 标注可视化

### 4. 带调试钩子的训练

```bash
python scripts/train_debug.py configs/VAD_tiny_debug.py --debug-interval 50
```

调试功能：
- 梯度 NaN/Inf 检测
- 梯度统计监控
- 损失逐组件分解
- 训练结束后自动绘制梯度流图和权重分布图

### 5. 标准 VAD 训练/测试

```bash
# 训练
bash scripts/train.sh configs/VAD_base_debug.py

# 测试
bash scripts/test.sh configs/VAD_base_debug.py <checkpoint_path>
```

## 调试工具 API

### BEV 可视化

```python
from debug_tools import BEVVisualizer

vis = BEVVisualizer(pc_range=[-15, -30, -2, 15, 30, 2])

# 可视化检测结果
vis.visualize_sample(img_metas, bbox_results, gt_bboxes_3d, gt_labels_3d,
                     save_path='output.png')

# 可视化 BEV 特征
vis.visualize_bev_feature(bev_embedding, save_path='bev.png')
```

### 数据管道检查

```python
from debug_tools import PipelineInspector

inspector = PipelineInspector(output_dir='logs/pipeline_debug')

# 检查 dataloader 输出
inspector.inspect_dataloader_sample(data_batch, step=0)

# 检查 image features
inspector.inspect_image_features(img_feats, step=1)

# 检查 head 输出
inspector.inspect_head_outputs(head_outs, step=2)

# 检查损失分解
inspector.inspect_losses(losses, step=3)
```

### 损失分析

```python
from debug_tools import LossAnalyzer

analyzer = LossAnalyzer()

# 绘制损失曲线
analyzer.plot_loss_curves('work_dirs/xxx/20240101.log', save_path='loss.png')

# 检查梯度流
analyzer.check_gradient_flow(model, save_path='grad.png')

# 检查权重分布
analyzer.check_weight_distribution(model, save_path='weights.png')
```

### 模型检查

```python
from debug_tools import ModelInspector

inspector = ModelInspector(model)

# 架构总结
inspector.summary()

# 设备/数据类型检查
inspector.check_device_dtype()

# 推理时间分析
inspector.measure_inference_time(dummy_input)
```

## VAD 架构概要

```
Input: 6 camera images [B, 6, 3, H, W]
  │
  ├─ ResNet50 Backbone → FPN Neck
  │   └─ Multi-scale features [B, 6, C, Hi, Wi]
  │
  ├─ BEVFormer Encoder (Temporal + Spatial attention)
  │   └─ BEV features [B, H_bev, W_bev, C]
  │
  └─ VAD Head
      ├─ Perception: 3D detection + trajectory prediction
      ├─ Map: vectorized map element prediction
      └─ Planning: ego trajectory prediction
```

## 注意事项

- VAD 原始依赖 Python 3.7 / PyTorch 1.9，调试配置已适配到 Python 3.8 / PyTorch 1.12
- 调试配置默认 `workers_per_gpu=0`（禁用多进程），方便断点调试
- 如需断点调试，在 `scripts/train_debug.py` 或 `scripts/debug_one_sample.py` 中加 `import pdb; pdb.set_trace()`
- nuPlan 适配将在后续阶段添加
