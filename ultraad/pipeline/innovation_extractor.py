"""Innovation extractor - Use LLM to extract innovations from papers"""

import json
import uuid
import os
from typing import List
from rich.console import Console

from .types import PaperContent, Innovation, InnovationCategory

console = Console()


class Innovation_Extractor:
    """Innovation extractor"""

    def __init__(self, model_name: str = None, api_key: str = None):
        """
        Initialize extractor

        Args:
            model_name: LLM model to use
            api_key: API key (auto-read from environment variable)
        """
        self.model_name = model_name or os.getenv("LLM_MODEL", "claude-3-5-sonnet-20241022")

        # Check and get API key
        if api_key:
            self.api_key = api_key
        else:
            # Try Anthropic first, then OpenAI
            self.api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")

        # If no API key, use mock mode
        if not self.api_key:
            console.print("[yellow]No API key found, using mock data[/]")
            self.use_mock = True
        else:
            self.use_mock = False

            # Detect model type
            if "claude" in selfser.model_name.lower() or "anthropic" in selfser.model_name.lower():
                selfser.provider = "anthropic"
            elif "gpt" in selfser.model_name.lower() or "openai" in selfser.model_name.lower():
                selfser.provider = "openai"
            else:
                selfser.provider = "anthropic"  # default

            console.print(f"[dim]Using {selfser.provider} API[/]")

    def extract(self, paper: PaperContent) -> List[Innovation]:
        """
        Extract innovations

        Args:
            paper: Paper content

        Returns:
            List of innovations
        """
        console.print(f"[dim]Extracting innovations...[/]")

        if selfser.use_mock:
            innovations = selfser._get_mock_innovations()
        else:
            # Build prompt
            prompt = selfser._build_extraction_prompt(paper)
            # Call LLM
            innovations = selfser._call_llm_for_innovations(prompt)

        console.print(f"[green]Extracted {len(innovations)} innovations[/]")
        for i, innovation in enumerate(innovations, 1):
            console.print(f"  {i}. {innovation.name} (score: {innovation.impact_score:.2f})")

        return innovations

    def _build_extraction_prompt(self, paper: PaperContent) -> str:
        """Build extraction prompt"""
        prompt = f"""Please analyze the following paper and extract main technical innovations.

Paper title: {paper.title}
Paper abstract: {paper.abstract}

Please extract 3-5 most important innovations and output them in JSON format:

{{
  "innovations": [
    {{
      "name": "Innovation name (short)",
      "description": "Detailed description (1-2 sentences)",
      "category": "architecture|attention|temporal|planning|efficiency|data",
      "feasibility_score": 0.9,
      "complexity_score": 0.5,
      "impact_score": 0.8
    }}
  ]
}}

Scoring explanation:
- feasibility_score: Implementation feasibility (0-1, 1 is most feasible)
- complexity_score: Implementation complexity (0-1, 0 is simplest)
- impact_score: Expected impact (0-1, 1 is highest impact)

Output JSON only, no other content.
"""
        return prompt

    def _call_llm_for_innovations(self, prompt: str) -> List[Innovation]:
        """Call LLM to extract innovations"""
        if selfser.provider == "anthropic":
            return selfser._call_anthropic(prompt)
        elif selfser.provider == "openai":
            return selfser._call_openai(prompt)
        else:
            raise ValueError(f"Unsupported provider: {selfser.provider}")

    def _call_anthropic(self, prompt: str) -> List[Innovation]:
        """Call Anthropic Claude"""
        try:
            from anthropic import Anthropic

            client = Anthropic(api_key=selfser.api_key)
            message = client.messages.create(
                model=selfser.model_name,
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = message.content[0].text
            return selfser._parse_json_response(response_text)

        except ImportError:
            raise RuntimeError("Need to install: pip install anthropic")
        except Exception as e:
            raise RuntimeError(f"Anthropic API call failed: {e}")

    def _call_openai(self, prompt: str) -> List[Innovation]:
        """Call OpenAI GPT"""
        try:
            from openai import OpenAI

            client = OpenAI(api_key=selfser.api_key)
            response = client.chat.completions.create(
                model=selfser.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4000
            )

            response_text = response.choices[0].message.content
            return selfser._parse_json_response(response_text)

        except ImportError:
            raise RuntimeError("Need to install: pip install openai")
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {e}")

    def _parse_json_response(self, response_text: str) -> List[Innovation]:
        """Parse JSON response"""
        try:
            # Extract JSON part
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
                raise ValueError("LLM did not return valid innovations")

            return innovations

        except json.JSONDecodeError as e:
            raise RuntimeError(f"JSON parse failed: {e}\nResponse: {response_text[:500]}")
        except Exception as e:
            raise RuntimeError(f"Response parse failed: {e}")

    def _get_mock_innovations(self) -> List[Innovation]:
        """Get mock innovations"""
        return [
            Innovation(
                id=str(uuid.uuid4())[:8],
                name="Temporal Fusion Enhancement",
                description="Improved BEVFormer temporal fusion mechanism with adaptive weight aggregation for historical frame features",
                category=InnovationCategory.TEMPORAL,
                feasibility_score=0.9,
                complexity_score=0.6,
                impact_score=0.8
            ),
            Innovation(
                id=str(uuid.uuid4())[:8],
                name="Sparse BEV Attention",
                description="Introduced sparse attention mechanism to reduce computation in BEV feature maps",
                category=InnovationCategory.ATTENTION,
                feasibility_score=0.85,
                complexity_score=0.7,
                impact_score=0.75
            ),
            Innovation(
                id=str(uuid.uuid4())[:8],
                name="Efficient Trajectory Decoder",
                description="Optimized trajectory prediction decoder using lightweight network structure",
                category=InnovationCategory.PLANNING,
                feasibility_score=0.8,
                complexity_score=0.5,
                impact_score=0.7
            )
        ]

    def select_best(self, innovations: List[Innovation]) -> Innovation:
        """
        Select best innovation

        Args:
            innovations: List of innovations

        Returns:
            Innovation with highest score
        """
        if not innovations:
            raise ValueError("No available innovations")

        # Comprehensive score: feasibility + impact - complexity
        def score(innovation):
            return (
                innovation.feasibility_score * 0.4 +
                innovation.impact_score * 0.5 -
                innovation.complexity_score * 0.1
            )

        best = max(innovations, key=score)
        console.print(f"[dim]Selected best innovation: {best.name}[/]")
        return best


# Backward compatibility
InnovationExtractor = Innovation_Extractor
