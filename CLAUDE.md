# CLAUDE.md - ultraAD 项目指南

## 项目概述

ultraAD 是基于 VAD (Vectorized Scene Representation) 的企业级端到端自动驾驶算法开发调试系统。v0.2.0 版本引入了全新的架构设计，包括统一配置管理、现代化 CLI、AI 智能诊断系统和 Web 可视化平台。

## 核心特性

- **统一配置管理** - 基于 Python dataclass 的类型安全配置系统
- **现代化 CLI** - 统一的命令行工具链 (ultraad train/debug/doctor/init)
- **AI 智能诊断** - 自动检测训练问题并提供修复建议
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

## CLI 命令参考

```bash
ultraad --help                    # 显示帮助信息
ultraad --version                 # 显示版本信息

ultraad init <project_name>       # 初始化新项目
ultraad train <config> [options]  # 启动训练
ultraad debug <config> [options]  # 单样本调试
ultraad doctor [options]          # AI 诊断
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

## 代码规范

### 新增模块规范

1. **核心模块** - 放在 `ultraad/` 目录下
   ```
   ultraad/
   ├── core/       # 核心功能（配置、训练器等）
   ├── cli/        # CLI 命令
   └── plugins/    # 插件系统
   ```

2. **AI Doctor 模块** - 放在 `ai_doctor/` 目录下
   ```
   ai_doctor/
   ├── core.py          # 诊断引擎
   ├── diagnosers.py    # 诊断器实现
   └── knowledge/       # 知识库
   ```

3. **Web 模块** - 放在 `web/` 目录下
   ```
   web/
   ├── backend/    # FastAPI 后端
   └── frontend/   # React 前端
   ```

### 配置规范

1. **新配置项** - 添加到相应的 dataclass
   ```python
   @dataclass
   class TrainerConfig:
       max_epochs: int = 20
       lr: float = 2e-4
       # 新增配置项...
   ```

2. **配置验证** - 在 `__post_init__` 中添加验证逻辑

### CLI 命令规范

1. **新命令** - 在 `ultraad/cli/` 下创建新文件
   ```python
   # ultraad/cli/new_cmd.py
   import click

   @click.command('new-cmd')
   def new_cmd():
       """新命令的帮助信息."""
       pass
   ```

2. **注册命令** - 在 `ultraad/cli/main.py` 中注册
   ```python
   from ultraad.cli.new_cmd import new_cmd
   app.add_command(new_cmd, name='new-cmd')
   ```

## 迁移指南

### 从 v0.1.x 迁移到 v0.2.0

1. **CLI 命令更新**
   - 旧: `python scripts/train_debug.py configs/VAD_tiny_debug.py`
   - 新: `ultraad train configs/VAD_tiny_debug.py`

2. **配置格式更新**
   - 旧: Python 配置文件
   - 新: YAML 配置文件（支持继承和覆盖）

3. **调试工具更新**
   - 旧: `from debug_tools import BEVVisualizer`
   - 新: `from ultraad import BEVVisualizer`

4. **训练脚本更新**
   - 旧: `python scripts/train_debug.py`
   - 新: `ultraad train --debug`

## 注意事项

- VAD 原始依赖 Python 3.7 / PyTorch 1.9，调试配置已适配到 Python 3.8 / PyTorch 1.12
- 调试配置默认 `workers_per_gpu=0`（禁用多进程），方便断点调试
- 如需断点调试，在代码中加 `import pdb; pdb.set_trace()`
- nuPlan 适配将在后续阶段添加

## 故障排除

### CLI 命令未找到

确保 ultraad 包已正确安装：

```bash
# 从项目根目录安装
pip install -e .

# 或直接使用 Python 模块运行
python -m ultraad.cli.main --help
```

### AI Doctor 无法加载

检查依赖是否安装：

```bash
pip install rich omegaconf
```

### Web Studio 无法启动

检查 FastAPI 和 uvicorn 是否安装：

```bash
pip install fastapi uvicorn
```

## 开发计划

- [x] v0.2.0 核心架构重构
- [x] AI Doctor 诊断系统
- [x] Web Studio 基础版本
- [ ] v0.3.0 多数据集支持 (nuPlan, Waymo)
- [ ] v0.4.0 分布式训练优化
- [ ] v0.5.0 模型压缩与部署工具

---

**ultraAD** - 让自动驾驶算法开发更高效、更智能！