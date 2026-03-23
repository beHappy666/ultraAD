# ultraAD v0.2.0

基于 [VAD (Vectorized Scene Representation for Efficient Autonomous Driving)](https://github.com/hustvl/VAD) 的企业级端到端自动驾驶算法开发平台。

![Version](https://img.shields.io/badge/version-0.2.0-blue)
![Python](https://img.shields.io/badge/python-3.8+-green)
![PyTorch](https://img.shields.io/badge/pytorch-1.12+-red)

## 核心特性

- **统一配置管理** - 基于 Python dataclass 的类型安全配置系统
- **现代化 CLI** - 统一的命令行工具链 (train/debug/studio/doctor/init)
- **AI 智能诊断** - 自动检测训练问题并提供修复建议
- **Web 可视化** - 基于 Web 的交互式调试和监控平台
- **智能训练引擎** - 支持混合精度、早停、自动恢复等高级功能

## 项目结构

```
ultraAD/
├── ultraad/                   # 核心框架
│   ├── cli/                   # CLI 命令实现
│   │   ├── main.py            # 主入口
│   │   ├── train.py           # 训练命令
│   │   ├── debug.py           # 调试命令
│   │   ├── studio.py          # Web Studio
│   │   └── doctor.py          # AI 诊断
│   ├── core/                  # 核心模块
│   │   ├── config.py          # 配置管理
│   │   └── trainer.py         # SmartTrainer
│   └── plugins/               # 插件系统
│
├── ai_doctor/                 # AI 诊断系统
│   ├── core.py                # 诊断引擎
│   └── diagnosers.py          # 专业诊断器
│
├── web/                       # Web 平台
│   ├── backend/               # FastAPI 后端
│   └── frontend/              # React 前端
│
├── third_party/VAD/           # VAD 源码
├── configs/                   # 配置文件
├── debug_tools/               # 调试工具 (旧)
├── scripts/                   # 脚本 (旧)
└── work_dirs/                 # 训练输出
```

## 快速开始

### 1. 环境配置

```bash
# 安装基础依赖
pip install rich click omegaconf fastapi uvicorn

# 或使用脚本
bash scripts/setup_env.sh
conda activate vad_debug
```

### 2. 初始化项目

```bash
# 创建新项目
python -m ultraad.cli.main init my_project
cd my_project
```

### 3. 数据准备

```bash
# 下载 nuScenes 数据集到 data/nuscenes/
bash scripts/prepare_data.sh data/nuscenes
```

### 4. 单样本调试

```bash
# 检查整个管道是否正常工作
ultraad debug configs/VAD_tiny_debug.py

# 指定样本索引和输出目录
ultraad debug configs/VAD_tiny_debug.py -i 42 -o logs/debug
```

### 5. 启动训练

```bash
# 基础训练
ultraad train configs/VAD_tiny_debug.py

# 带调试信息
ultraad train configs/VAD_tiny_debug.py --debug

# 多 GPU 训练
ultraad train configs/VAD_tiny_debug.py -g 0 1 2 3

# 从检查点恢复
ultraad train configs/VAD_tiny_debug.py --resume work_dirs/exp1/latest.pth
```

### 6. AI 诊断

```bash
# 诊断训练日志
ultraad doctor -l work_dirs/exp1/20240101.log

# 诊断检查点
ultraad doctor -c work_dirs/exp1/latest.pth

# 实时监控模式
ultraad doctor -w

# 自动修复问题
ultraad doctor -l work_dirs/exp1/20240101.log --auto-fix
```

### 7. Web Studio

```bash
# 启动 Web 界面
ultraad studio

# 指定端口
ultraad studio -p 8888

# 开发模式（热重载）
ultraad studio --reload
```

访问 http://localhost:8080 查看 Web 界面。

## CLI 命令参考

```bash
ultraad --help                    # 显示帮助信息
ultraad --version                 # 显示版本信息

ultraad init <project_name>       # 初始化新项目
ultraad train <config> [options]  # 启动训练
ultraad debug <config> [options]  # 单样本调试
ultraad doctor [options]          # AI 诊断
ultraad studio [options]          # Web Studio
ultraad info                      # 显示系统信息
```

## 配置系统

ultraAD 使用基于 Python dataclass 的类型安全配置系统：

```python
from ultraad.core.config import Config, load_config

# 从文件加载
config = Config.from_file('configs/vad_tiny.yaml')

# 使用覆盖参数
config = load_config('configs/vad_tiny.yaml',
                     overrides={'trainer.lr': 1e-4, 'data.batch_size': 4})

# 保存配置
config.save('work_dirs/exp1/config.yaml')
```

### 配置结构

```yaml
name: experiment_name
work_dir: work_dirs/exp1
seed: 0

model:
  name: vad_tiny
  backbone: resnet50
  bev_h: 200
  bev_w: 200
  use_temporal: true

data:
  dataset: nuscenes
  data_root: data/nuscenes
  batch_size: 1
  num_workers: 0

trainer:
  max_epochs: 20
  lr: 2e-4
  optimizer: adamw
  use_amp: true
  grad_clip: 35.0

debug:
  enabled: true
  enable_ai_doctor: true
  check_interval: 50
```

## AI Doctor 诊断系统

AI Doctor 是一个智能诊断系统，可以自动检测训练过程中的问题并提供修复建议。

### 诊断器类型

| 诊断器 | 检测问题 | 严重级别 |
|--------|----------|----------|
| GradientDiagnoser | 梯度消失、爆炸、NaN/Inf | CRITICAL |
| LossDiagnoser | Loss plateau、NaN loss、increasing loss | ERROR |
| DataDiagnoser | 数据加载慢、空 batch、NaN 数据 | WARNING |
| ModelDiagnoser | 大模型、未使用参数、权重异常 | INFO |

### 使用示例

```bash
# 诊断训练日志
ultraad doctor -l work_dirs/exp1/20240101.log

# 生成详细报告
ultraad doctor -l work_dirs/exp1/20240101.log --report

# 交互式诊断
ultraad doctor
```

## SmartTrainer 智能训练引擎

SmartTrainer 是一个功能强大的训练引擎，支持多种高级特性：

### 特性

- **混合精度训练 (AMP)** - 自动混合精度加速训练
- **梯度裁剪** - 防止梯度爆炸
- **学习率调度** - 支持多种调度策略
- **早停机制** - 自动检测过拟合
- **检查点管理** - 自动保存和恢复
- **AI Doctor 集成** - 实时监控训练健康度

### 使用示例

```python
from ultraad.core.config import Config
from ultraad.core.trainer import SmartTrainer

# 加载配置
config = Config.from_file('configs/vad_tiny.yaml')

# 创建训练器
trainer = SmartTrainer(config)

# 开始训练
history = trainer.train(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler
)
```

## 调试工具 API

### BEV 可视化

```python
from ultraad import BEVVisualizer

vis = BEVVisualizer(pc_range=[-15, -30, -2, 15, 30, 2])

# 可视化检测结果
vis.visualize_sample(img_metas, bbox_results, gt_bboxes_3d, gt_labels_3d,
                     save_path='output.png')

# 可视化 BEV 特征
vis.visualize_bev_feature(bev_embedding, save_path='bev.png')
```

### 数据管道检查

```python
from ultraad import PipelineInspector

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
from ultraad import LossAnalyzer

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
from ultraad import ModelInspector

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

## 常见问题

### Q: 如何调试训练过程？

A: 使用 `ultraad doctor` 命令可以自动诊断训练问题：

```bash
# 诊断日志文件
ultraad doctor -l work_dirs/exp1/latest.log

# 实时监控
ultraad doctor -w
```

### Q: 如何启用 Web 可视化？

A: 使用 `ultraad studio` 命令启动 Web 界面：

```bash
ultraad studio -p 8080
```

然后在浏览器访问 http://localhost:8080

### Q: 如何配置分布式训练？

A: 使用 `-g` 参数指定 GPU：

```bash
ultraad train configs/vad_base.py -g 0 1 2 3
```

## 贡献指南

欢迎提交 Issue 和 PR！请确保：

1. 代码符合 PEP8 规范
2. 添加必要的测试
3. 更新相关文档

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 致谢

- [VAD](https://github.com/hustvl/VAD) - 基础的 VAD 实现
- [OpenMMLab](https://github.com/open-mmlab) - 优秀的深度学习工具链

---

**ultraAD** - 让自动驾驶算法开发更高效、更智能！