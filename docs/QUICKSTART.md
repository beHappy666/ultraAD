# ultraAD 快速上手指南

本指南将帮助您在 10 分钟内上手 ultraAD，完成从安装到训练第一个模型的完整流程。

## 目录

1. [环境准备](#环境准备)
2. [安装 ultraAD](#安装-ultraad)
3. [数据准备](#数据准备)
4. [第一个调试](#第一个调试)
5. [第一次训练](#第一次训练)
6. [使用 AI Doctor](#使用-ai-doctor)
7. [启动 Web Studio](#启动-web-studio)

## 环境准备

### 系统要求

- **操作系统**: Linux (推荐 Ubuntu 18.04+), Windows 10/11, macOS (仅限 CPU)
- **Python**: 3.8 或更高版本
- **CUDA**: 11.3 或更高版本 (推荐使用 GPU)
- **内存**: 至少 16GB RAM
- **存储**: 至少 100GB 可用空间 (用于数据集)

### 检查环境

```bash
# 检查 Python 版本
python --version  # 应该 >= 3.8

# 检查 CUDA 版本 (如果有 GPU)
nvidia-smi

# 检查 PyTorch 是否安装
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"
```

## 安装 ultraAD

### 方式一: 从源码安装 (推荐)

```bash
# 克隆仓库
git clone https://github.com/your-org/ultraAD.git
cd ultraAD

# 安装依赖
pip install -r requirements.txt

# 安装 ultraAD (开发模式)
pip install -e .
```

### 方式二: 使用 pip 安装

```bash
pip install ultraad
```

### 验证安装

```bash
# 检查 CLI 是否可用
ultraad --version

# 应该显示类似: ultraAD version 0.2.0

# 显示帮助
ultraad --help
```

## 数据准备

### 下载 nuScenes 数据集

```bash
# 创建数据目录
mkdir -p data/nuscenes

# 下载 nuScenes 数据集 (v1.0-trainval)
# 访问 https://www.nuscenes.org/nuscenes#download 获取下载链接

# 解压数据
cd data/nuscenes
tar -xf v1.0-trainval_meta.tgz
tar -xf v1.0-trainval01_blobs.tgz
# ... 解压所有相关文件
```

### 准备数据

```bash
# 运行数据准备脚本
bash scripts/prepare_data.sh data/nuscenes
```

这将生成数据缓存和元数据文件。

## 第一个调试

在正式训练之前，建议先运行单样本调试来验证整个流程：

```bash
# 基本调试
ultraad debug configs/VAD_tiny_debug.py

# 指定样本索引
ultraad debug configs/VAD_tiny_debug.py -i 42

# 指定输出目录
ultraad debug configs/VAD_tiny_debug.py -o logs/debug

# 只调试特定阶段 (data/model/train/all)
ultraad debug configs/VAD_tiny_debug.py -s data
```

调试完成后，检查输出目录中的可视化结果，确保：
- 数据加载正常
- 模型前向传播正常
- 损失计算正常

## 第一次训练

### 快速训练测试

```bash
# 使用 Tiny 配置进行快速测试
ultraad train configs/VAD_tiny_debug.py
```

### 标准训练

```bash
# 使用 Base 配置进行正式训练
ultraad train configs/VAD_base_debug.py

# 指定工作目录
ultraad train configs/VAD_base_debug.py -w work_dirs/exp1

# 多 GPU 训练
ultraad train configs/VAD_base_debug.py -g 0 1 2 3
```

### 从检查点恢复

```bash
# 从最新检查点恢复
ultraad train configs/VAD_base_debug.py --resume work_dirs/exp1/latest.pth

# 从特定 epoch 恢复
ultraad train configs/VAD_base_debug.py --resume work_dirs/exp1/epoch_10.pth
```

### 监控训练

训练过程中，您可以在另一个终端使用 AI Doctor 监控训练健康度：

```bash
# 实时监控
ultraad doctor -w

# 或定期检查日志
ultraad doctor -l work_dirs/exp1/latest.log
```

## 使用 AI Doctor

AI Doctor 可以自动诊断训练过程中的问题。

### 诊断日志文件

```bash
# 诊断训练日志
ultraad doctor -l work_dirs/exp1/20240101.log

# 生成详细报告
ultraad doctor -l work_dirs/exp1/20240101.log --report
```

### 诊断检查点

```bash
# 诊断模型检查点
ultraad doctor -c work_dirs/exp1/latest.pth
```

### 实时监控

```bash
# 实时监控模式
ultraad doctor -w

# 按 Ctrl+C 停止监控
```

### 自动修复

```bash
# 尝试自动修复检测到的问题
ultraad doctor -l work_dirs/exp1/20240101.log --auto-fix
```

### 交互式诊断

```bash
# 启动交互式诊断
ultraad doctor

# 然后输入问题描述
```

## 启动 Web Studio

Web Studio 提供基于 Web 的交互式调试和监控界面。

### 启动 Web 服务

```bash
# 默认端口 8080
ultraad studio

# 指定端口
ultraad studio -p 8888

# 开发模式（热重载）
ultraad studio --reload

# 不自动打开浏览器
ultraad studio --no-browser
```

### 访问 Web 界面

启动后，在浏览器中访问：

```
http://localhost:8080
```

### Web 界面功能

- **实时监控** - 查看训练进度、损失曲线、学习率等
- **3D 可视化** - 交互式查看 BEV 特征、检测结果、规划轨迹
- **数据检查** - 查看数据管道各阶段的输出
- **模型分析** - 查看模型架构、参数量、计算量等
- **日志查看** - 实时查看训练日志

## 下一步

完成快速上手后，建议阅读以下文档深入了解 ultraAD：

- [用户指南](USER_GUIDE.md) - 详细的使用说明
- [API 文档](API.md) - 完整的 API 参考
- [配置指南](CONFIG.md) - 配置系统详解
- [开发指南](DEVELOPMENT.md) - 如何贡献代码

## 获取帮助

- **GitHub Issues**: https://github.com/your-org/ultraAD/issues
- **文档**: https://ultraad.readthedocs.io
- **邮件**: support@ultraad.ai

---

**恭喜！** 您已经完成了 ultraAD 的快速上手。现在您可以开始使用 ultraAD 进行自动驾驶算法开发了！