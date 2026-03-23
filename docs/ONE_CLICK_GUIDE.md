# ultraAD 一键式使用指南

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 一键式运行

只需要一个命令，从论文到完整报告：

```bash
# 从 arXiv 论文
ultraad paper-to-report arxiv:2301.xxxxx

# 从本地 PDF
ultraad paper-to-report path/to/paper.pdf

# 指定基线配置
ultraad paper-to-report paper.pdf --baseline configs/baselines/vad_tiny_nuscenes.yaml

# 指定训练轮数和 GPU
ultraad paper-to-report paper.pdf --epochs 30 --gpu-id 0
```

## 工作流程

```
[论文源] → [解析] → [提取创新点] → [生成代码]
→ [集成到 VAD] → [运行基线] → [运行实验]
→ [对比性能] → [生成报告]
```

## 输出内容

运行完成后，在 `auto_output/` 目录下会生成：

```
auto_output/
├── cache/                    # 论文缓存
├── work_dirs/                # 训练输出
│   ├── baseline/             # 基线实验
│   └── experiment_xxx/      # 创新点实验
├── reports/                  # 报告文件
│   ├── report_xxx.html       # HTML 报告（推荐）
│   ├── report_xxx.md         # Markdown 报告
│   └── report_xxx.json       # JSON 数据
└── third_party/VAD/...      # 生成的代码
```

## 报告内容

HTML 报告包含：

- 📝 **创新点摘要** - 从论文中提取的关键创新点
- 📊 **性能对比** - 基线 vs 实验的各项指标对比
- 📈 **关键指标** - mAP、NDS、FPS 的提升百分比
- 📋 **总结** - 总体评价和显著性判断

## 高级用法

### 使用特定基线

```bash
ultraad paper-to-report paper.pdf \\
    --baseline configs/baselines/vad_tiny_nuscenes.yaml \\
    --epochs 20
```

### 多 GPU 训练

```bash
# 修改实验运行器以支持多 GPU
ultraad paper-to-report paper.pdf --gpu-id 0
```

### 仅生成代码，不运行实验

编辑 `ultraad/pipeline/auto_pipeline.py`，跳过实验步骤：

```python
# 注释掉步骤 5
# with self._step_progress("运行实验", 5, 6):
#     ...
```

## 自定义配置

### 创新点提取

编辑 `ultraad/pipeline/innovation_extractor.py`，集成真实的 LLM API：

```python
# 替换 _call_llm_for_innovations 方法
import anthropic

client = anthropic.Anthropic(api_key="your-key")
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=4000,
    messages=[{"role": "user", "content": prompt}]
)
```

### 训练集成

编辑 `ultraad/pipeline/experiment_runner.py`，集成真实训练：

```python
# 替换 _run_training_mock 方法
subprocess.run([
    "ultraad", "train",
    config_path,
    "--work-dir", str(output_dir),
    f"--gpu-ids", str(self.gpu_id)
])
```

## 故障排除

### 论文解析失败

确保已安装 `pymupdf`：

```bash
pip install pymupdf
```

### 找不到 VAD 配置

手动指定基线配置：

```bash
ultraad paper-to-report paper.pdf \\
    --baseline third_party/VAD/projects/configs/VAD/VAD_tiny_stage_1.py
```

### 训练失败

检查 GPU 可用性和数据集路径：

```bash
ultraad info
```

## 开发路线图

- [x] 基础流水线框架
- [x] 论文解析
- [x] 模拟创新点提取
- [ ] 真实 LLM 集成
- [x] 代码生成模板
- [ ] VAD 自动集成
- [x] 模拟实验运行
- [ ] 真实训练集成
- [x] 报告生成
- [ ] 性能图表

## 示例

### 示例 1: VAD 论文

```bash
ultraad paper-to-report arxiv:2302.05462
```

输出报告包含 VAD 的原始结果和模拟改进。

### 示例 2: 本地 PDF

```bash
ultraad paper-to-report ~/papers/my_idea.pdf \\
    --baseline configs/baselines/vad_tiny_nuscenes.yaml \\
    --epochs 10
```

## 性能指标说明

- **mAP**: Mean Average Precision - 检测精度
- **NDS**: NuScenes Detection Score - 综合评估
- **FPS**: Frames Per Second - 推理速度
- **mATE**: Mean Average Translation Error
- **mASE**: Mean Average Scale Error
- **mAOE**: Mean Average Orientation Error
- **mAVE**: Mean Average Velocity Error
- **mAAE**: Mean Average Attribute Error
