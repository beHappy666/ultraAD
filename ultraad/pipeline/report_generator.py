"""报告生成器 - 生成实验对比报告"""

import os
import json
from datetime import datetime
from pathlib import Path
from rich.console import Console

from .types import PaperContent, Innovation, ComparisonResult, Report

console = Console()


class ReportGenerator:
    """报告生成器"""

    def __init__(self, output_dir: str = None):
        """
        初始化报告生成器

        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir) if output_dir else Path("reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, paper: PaperContent,
               innovations: List[Innovation],
               comparison: ComparisonResult) -> Report:
        """
        生成报告

        Args:
            paper: 论文内容
            innovations: 创新点列表
            comparison: 对比结果

        Returns:
            生成的报告
        """
        report_id = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        console.print(f"[dim]正在生成报告...[/]")

        # 生成 HTML 报告
        html_path = self._generate_html(report_id, paper, innovations, comparison)

        # 生成 JSON 数据
        json_path = self._generate_json(report_id, paper, innovations, comparison)

        # 生成 Markdown 报告
        md_path = self._generate_markdown(report_id, paper, innovations, comparison)

        report = Report(
            report_id=report_id,
            paper=paper,
            innovations=innovations,
            comparison=comparison,
            generated_at=datetime.now(),
            output_path=str(html_path),
            summary=self._generate_summary(paper, innovations, comparison)
        )

        console.print(f"[green]✓ 报告生成完成[/]")
        console.print(f"  HTML: {html_path}")
        console.print(f"  JSON: {json_path}")
        console.print(f"  Markdown: {md_path}")

        return report

    def _generate_html(self, report_id: str, paper: PaperContent,
                     innovations: List[Innovation],
                     comparison: ComparisonResult) -> Path:
        """生成 HTML 报告"""
        html_content = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{paper.title} - 实验报告</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .metric-card {{
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
        }}
        .metric-change {{
            font-size: 1.2em;
        }}
        .positive {{ color: #27ae60; }}
        .negative {{ color: #e74c3c; }}
        .neutral {{ color: #95a5a6; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #34495e;
            color: white;
        }}
        .innovation-item {{
            background-color: #fff9e6;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #f39c12;
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{paper.title}</h1>
        <p><strong>论文ID:</strong> {paper.paper_id}</p>
        <p><strong>作者:</strong> {', '.join(paper.authors[:5])}{'...' if len(paper.authors) > 5 else ''}</p>

        <h2>📝 创新点摘要</h2>
        {self._html_innovations(innovations)}

        <h2>📊 性能对比</h2>
        {self._html_metrics_table(comparison)}

        <h2>📈 关键指标</h2>
        {self._html_metric_cards(comparison)}

        {self._html_summary(comparison)}
    </div>
</body>
</html>'''

        output_path = self.output_dir / f"{report_id}.html"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return output_path

    def _html_innovations(self, innovations: List[Innovation]) -> str:
        """生成创新点 HTML"""
        html = ""
        for i, innovation in enumerate(innovations, 1):
            html += f'''
            <div class="innovation-item">
                <h3>{i}. {innovation.name}</h3>
                <p>{innovation.description}</p>
                <p><strong>类别:</strong> {innovation.category.value}</p>
                <p><strong>可行性:</strong> {innovation.feasibility_score:.2f} | <strong>影响:</strong> {innovation.impact_score:.2f}</p>
            </div>
            '''
        return html

    def _html_metrics_table(self, comparison: ComparisonResult) -> str:
        """生成指标表格 HTML"""
        diff = comparison.diff

        rows = []
        if comparison.baseline.metrics.mAP and comparison.experiment.metrics.mAP:
            rows.append(('mAP', comparison.baseline.metrics.mAP,
                        comparison.experiment.metrics.mAP, diff.mAP_delta, diff.mAP_pct))

        if comparison.baseline.metrics.NDS and comparison.experiment.metrics.NDS:
            rows.append(('NDS', comparison.baseline.metrics.NDS,
                        comparison.experiment.metrics.NDS, diff.NDS_delta, diff.NDS_pct))

        if comparison.baseline.metrics.fps and comparison.experiment.metrics.fps:
            rows.append(('FPS', comparison.baseline.metrics.fps,
                        comparison.experiment.metrics.fps, diff.fps_delta, diff.fps_pct))

        html = '<table><tr><th>指标</th><th>基线</th><th>实验</th><th>变化</th></tr>'

        for name, baseline, exp, delta, pct in rows:
            change_class = 'positive' if delta > 0 else ('negative' if delta < 0 else 'neutral')
            change_str = f"{delta:+.4f} ({pct:+.1f}%)" if pct is not None else f"{delta:+.4f}"

            html += f'''
            <tr>
                <td><strong>{name}</strong></td>
                <td>{baseline:.4f}</td>
                <td>{exp:.4f}</td>
                <td class="{change_class}">{change_str}</td>
            </tr>
            '''

        html += '</table>'
        return html

    def _html_metric_cards(self, comparison: ComparisonResult) -> str:
        """生成指标卡片 HTML"""
        diff = comparison.diff

        cards = []

        if diff.mAP_pct is not None:
            cards.append(('mAP 提升', f"{diff.mAP_pct:+.1f}%"))

        if diff.NDS_pct is not None:
            cards.append(('NDS 提升', f"{diff.NDS_pct:+.1f}%"))

        if diff.fps_pct is not None:
            cards.append(('FPS 变化', f"{diff.fps_pct:+.1f}%"))

        html = '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-top: 20px;">'

        for title, value in cards:
            is_positive = '+' in value
            color_class = 'positive' if is_positive else ('negative' if '-' in value else 'neutral')

            html += f'''
            <div class="metric-card">
                <div class="metric-value {color_class}">{value}</div>
                <div class="metric-label">{title}</div>
            </div>
            '''

        html += '</div>'
        return html

    def _html_summary(self, comparison: ComparisonResult) -> str:
        """生成总结 HTML"""
        improvement_map = {
            'positive': '✅ 实验结果优于基线',
            'negative': '❌ 实验结果不如基线',
            'neutral': '⚖️ 实验结果与基线相当'
        }

        significance_map = {
            'high': '显著',
            'medium': '中等',
            'low': '较小'
        }

        return f'''
        <h2>📋 总结</h2>
        <div class="metric-card">
            <p><strong>总体评价:</strong> {improvement_map.get(comparison.overall_improvement, '未知')}</p>
            <p><strong>显著性:</strong> {significance_map.get(comparison.significance, '未知')}</p>
            <p><strong>训练时间:</strong>
                基线: {comparison.baseline.metrics.training_time_hours:.1f}h |
                实验: {comparison.experiment.metrics.training_time_hours:.1f}h
            </p>
            <p><strong>GPU 显存:</strong>
                基线: {comparison.baseline.metrics.gpu_memory_gb:.1f}GB |
                实验: {comparison.experiment.metrics.gpu_memory_gb:.1f}GB
            </p>
        </div>
        '''

    def _generate_json(self, report_id: str, paper: PaperContent,
                     innovations: List[Innovation],
                     comparison: ComparisonResult) -> Path:
        """生成 JSON 报告"""
        data = {
            'report_id': report_id,
            'paper': {
                'id': paper.paper_id,
                'title': paper.title,
                'authors': paper.authors,
                'abstract': paper.abstract
            },
            'innovations': [
                {
                    'id': i.id,
                    'name': i.name,
                    'description': i.description,
                    'category': i.category.value,
                    'feasibility_score': i.feasibility_score,
                    'impact_score': i.impact_score
                }
                for i in innovations
            ],
            'comparison': {
                'baseline': comparison.baseline.metrics.to_dict(),
                'experiment': comparison.experiment.metrics.to_dict(),
                'diff': comparison.diff.to_dict(),
                'overall_improvement': comparison.overall_improvement,
                'significance': comparison.significance
            }
        }

        output_path = self.output_dir / f"{report_id}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return output_path

    def _generate_markdown(self, report_id: str, paper: PaperContent,
                         innovations: List[Innovation],
                         comparison: ComparisonResult) -> Path:
        """生成 Markdown 报告"""
        md_content = f'''# {paper.title} - 实验报告

**论文ID:** {paper.paper_id}
**生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📝 创新点摘要

'''

        for i, innovation in enumerate(innovations, 1):
            md_content += f'''
### {i}. {innovation.name}

{innovation.description}

- **类别:** {innovation.category.value}
- **可行性:** {innovation.feasibility_score:.2f}
- **影响:** {innovation.impact_score:.2f}

'''

        md_content += f'''
## 📊 性能对比

| 指标 | 基线 | 实验 | 变化 |
|------|------|------|------|
'''

        diff = comparison.diff

        if comparison.baseline.metrics.mAP:
            md_content += f"| mAP | {comparison.baseline.metrics.mAP:.4f} | {comparison.experiment.metrics.mAP:.4f} | {diff.mAP_delta:+.4f} ({diff.mAP_pct:+.1f}%) |\n"

        if comparison.baseline.metrics.NDS:
            md_content += f"| NDS | {comparison.baseline.metrics.NDS:.4f} | {comparison.experiment.metrics.NDS:.4f} | {diff.NDS_delta:+.4f} ({diff.NDS_pct:+.1f}%) |\n"

        if comparison.baseline.metrics.fps:
            md_content += f"| FPS | {comparison.baseline.metrics.fps:.2f} | {comparison.experiment.metrics.fps:.2f} | {diff.fps_delta:+.2f} ({diff.fps_pct:+.1f}%) |\n"

        md_content += f'''
## 📋 总结

- **总体评价:** {comparison.overall_improvement}
- **显著性:** {comparison.significance}
- **基线训练时间:** {comparison.baseline.metrics.training_time_hours:.1f}h
- **实验训练时间:** {comparison.experiment.metrics.training_time_hours:.1f}h
- **基线显存:** {comparison.baseline.metrics.gpu_memory_gb:.1f}GB
- **实验显存:** {comparison.experiment.metrics.gpu_memory_gb:.1f}GB
'''

        output_path = self.output_dir / f"{report_id}.md"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

        return output_path

    def _generate_summary(self, paper: PaperContent,
                       innovations: List[Innovation],
                       comparison: ComparisonResult) -> str:
        """生成摘要"""
        diff = comparison.diff

        summary_parts = [
            f"论文: {paper.title}",
            f"创新点: {', '.join([i.name for i in innovations])}",
        ]

        if diff.mAP_pct is not None:
            summary_parts.append(f"mAP变化: {diff.mAP_pct:+.1f}%")

        if diff.NDS_pct is not None:
            summary_parts.append(f"NDS变化: {diff.NDS_pct:+.1f}%")

        summary_parts.append(f"总体评价: {comparison.overall_improvement}")

        return ' | '.join(summary_parts)
