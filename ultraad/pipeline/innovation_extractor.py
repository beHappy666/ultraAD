"""创新点提取器 - 使用LLM提取论文创新点"""

import json
import uuid
import os
from typing import List
from rich.console import Console

from .types import PaperContent, Innovation, InnovationCategory

console = Console()


class Innovation_Extractor:
    """创新点提取器"""

    def __init__(self, model_name: str = None, api_key: str = None):
        """
        初始化提取器

        Args:
            model_name: 使用的LLM模型
            api_key: API密钥（自动从环境变量读取）
        """
        self.model_name = model_name or os.getenv("LLM_MODEL", "claude-3-5-sonnet-20241022")

        # 检查并获取 API 密钥
        if api_key:
            self.api_key = api_key
        else:
            # 优先尝试 Anthropic，然后 OpenAI
            self.api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")

        # 强制要求 API 密钥
        if not self.api_key:
            raise RuntimeError(
                "需要设置 API 密钥才能运行。\n"
                "请设置环境变量：\n"
                "  export ANTHROPIC_API_KEY=sk-xxx  # 或\n"
                "  export OPENAI_API_KEY=sk-xxx"
            )

        # 检测模型类型
        if "claude" in self.model_name.lower() or "anthropic" in self.model_name.lower():
            self.provider = "anthropic"
        elif "gpt" in self.model_name.lower() or "openai" in self.model_name.lower():
            self.provider = "openai"
        else:
            self.provider = "anthropic"  # 默认

        console.print(f"[dim]使用 {self.provider} API[/]")

    def extract(self, paper: PaperContent) -> List[Innovation]:
        """
        提取创新点

        Args:
            paper: 论文内容

        Returns:
            创新点列表
        """
        console.print(f"[dim]正在提取创新点...[/]")

        # 构建提示
        prompt = self._build_extraction_prompt(paper)

        # 调用 LLM
        innovations = self._call_llm_for_innovations(prompt)

        console.print(f"[green]✓ 提取到 {len(innovations)} 个创新点[/]")
        for i, innovation in enumerate(innovations, 1):
            console.print(f"  {i}. {innovation.name} (评分: {innovation.impact_score:.2f})")

        return innovations

    def _build_extraction_prompt(self, paper: PaperContent) -> str:
        """构建提取提示"""
        prompt = f"""请分析以下论文，提取主要的技术创新点。

论文标题: {paper.title}
论文摘要: {paper.abstract}

请提取3-5个最重要的创新点，以JSON格式输出：

{{
  "innovations": [
    {{
      "name": "创新点名称（简短）",
      "description": "详细描述（1-2句话）",
      "category": "architecture|attention|temporal|planning|efficiency|data",
      "feasibility_score": 0.9,
      "complexity_score": 0.5,
      "impact_score": 0.8
    }}
  ]
}}

评分说明：
- feasibility_score: 实现可行性（0-1，1为最可行）
- complexity_score: 实现复杂度（0-1，0为最简单）
- impact_score: 预期影响（0-1，1为影响最大）

只输出JSON，不要其他内容。
"""
        return prompt

    def _call_llm_for_innovations(self, prompt: str) -> List[Innovation]:
        """调用LLM提取创新点"""
        if self.provider == "anthropic":
            return self._call_anthropic(prompt)
        elif self.provider == "openai":
            return self._call_openai(prompt)
        else:
            raise ValueError(f"不支持的提供商: {self.provider}")

    def _call_anthropic(self, prompt: str) -> List[Innovation]:
        """调用Anthropic Claude"""
        try:
            from anthropic import Anthropic

            client = Anthropic(api_key=self.api_key)
            message = client.messages.create(
                model=self.model_name,
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = message.content[0].text
            return self._parse_json_response(response_text)

        except ImportError:
            raise RuntimeError("需要安装: pip install anthropic")
        except Exception as e:
            raise RuntimeError(f"Anthropic API 调用失败: {e}")

    def _call_openai(self, prompt: str) -> List[Innovation]:
        """调用OpenAI GPT"""
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4000
            )

            response_text = response.choices[0].message.content
            return self._parse_json_response(response_text)

        except ImportError:
            raise RuntimeError("需要安装: pip install openai")
        except Exception as e:
            raise RuntimeError(f"OpenAI API 调用失败: {e}")

    def _parse_json_response(self, response_text: str) -> List[Innovation]:
        """解析JSON响应"""
        try:
            # 提取JSON部分
            import re
            json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(0)

            data = json.loads(response_text)
            innovations = []

            for item in data.get("innovations", []):
                innovation = Innovation(
                    id=str(uuid.uuid4())[:8],
                    name=item.get("name", "Unknown"),
                    description=item.get("description", ""),
                    category=InnovationCategory(item.get("category", "architecture")),
                    feasibility_score=float(item.get("feasibility_score", 0.5)),
                    complexity_score=float(item.get("complexity_score", 0.5)),
                    impact_score=float(item.get("impact_score", 0.5))
                )
                innovations.append(innovation)

            if not innovations:
                raise ValueError("LLM 未返回有效的创新点")

            return innovations

        except json.JSONDecodeError as e:
            raise RuntimeError(f"JSON 解析失败: {e}\n响应: {response_text[:500]}")
        except Exception as e:
            raise RuntimeError(f"解析响应失败: {e}")

    def select_best(self, innovations: List[Innovation]) -> Innovation:
        """
        选择最佳创新点

        Args:
            innovations: 创新点列表

        Returns:
            评分最高的创新点
        """
        if not innovations:
            raise ValueError("没有可用的创新点")

        # 综合评分：可行性 + 影响力 - 复杂度
        def score(innovation):
            return (
                innovation.feasibility_score * 0.4 +
                innovation.impact_score * 0.5 -
                innovation.complexity_score * 0.1
            )

        best = max(innovations, key=score)
        console.print(f"[dim]选择最佳创新点: {best.name}[/]")
        return best


# 向后兼容
InnovationExtractor = Innovation_Extractor
