"""论文解析器 - 解析PDF/URL提取论文内容"""

import os
import re
import hashlib
import requests
import fitz  # PyMuPDF
from typing import Optional, Dict, List
from pathlib import Path
from rich.console import Console

from .types import PaperContent, PaperSourceType

console = Console()


class PaperAnalyzer:
    """论文解析器"""

    def __init__(self, cache_dir: str = None):
        """
        初始化解析器

        Args:
            cache_dir: 缓存目录
        """
        self.cache_dir = cache_dir or "cache/papers"
        os.makedirs(self.cache_dir, exist_ok=True)

    def parse(self, source: str) -> PaperContent:
        """
        解析论文

        Args:
            source: 论文路径或URL

        Returns:
            PaperContent
        """
        # 检测源类型
        source_type = self._detect_source_type(source)

        if source_type == PaperSourceType.ARXIV:
            return self._parse_arxiv(source)
        elif source_type == PaperSourceType.PDF:
            return self._parse_pdf(source)
        else:
            raise ValueError(f"不支持的源类型: {source_type}")

    def _detect_source_type(self, source: str) -> PaperSourceType:
        """检测源类型"""
        if source.startswith("arxiv:"):
            return PaperSourceType.ARXIV
        elif source.startswith("http://") or source.startswith("https://"):
            if "arxiv.org" in source:
                return PaperSourceType.ARXIV
            return PaperSourceType.URL
        elif source.endswith(".pdf"):
            return PaperSourceType.PDF
        elif os.path.exists(source):
            return PaperSourceType.PDF
        else:
            raise ValueError(f"无法识别的源: {source}")

    def _parse_arxiv(self, arxiv_id: str) -> PaperContent:
        """解析 arXiv 论文"""
        arxiv_id = arxiv_id.replace("arxiv:", "").strip()

        console.print(f"[dim]正在从 arXiv 下载论文: {arxiv_id}[/]")

        # 下载 PDF
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        pdf_path = os.path.join(self.cache_dir, f"{arxiv_id}.pdf")

        if not os.path.exists(pdf_path):
            response = requests.get(pdf_url, stream=True)
            response.raise_for_status()

            with open(pdf_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            console.print(f"[green]✓ PDF 下载完成[/]")
        else:
            console.print(f"[dim]使用缓存的 PDF[/]")

        return self._parse_pdf(pdf_path, arxiv_id=arxiv_id)

    def _parse_pdf(self, pdf_path: str, arxiv_id: str = None) -> PaperContent:
        """解析 PDF 文件"""
        console.print(f"[dim]正在解析 PDF: {pdf_path}[/]")

        doc = fitz.open(pdf_path)

        # 提取文本
        full_text = ""
        for page in doc:
            full_text += page.get_text()

        # 提取元数据
        metadata = self._extract_metadata(doc, full_text)

        # 提取章节
        sections = self._extract_sections(full_text)

        # 生成 paper_id
        if arxiv_id:
            paper_id = arxiv_id
        else:
            paper_id = hashlib.md5(pdf_path.encode()).hexdigest()[:12]

        # 提取作者
        authors = metadata.get('authors', [])
        if not authors:
            authors = self._extract_authors(full_text)

        # 提取摘要
        abstract = metadata.get('abstract', '')
        if not abstract:
            abstract = self._extract_abstract(full_text)

        # 提取标题
        title = metadata.get('title', '')
        if not title:
            title = self._extract_title(full_text)

        paper_content = PaperContent(
            paper_id=paper_id,
            title=title,
            authors=authors,
            abstract=abstract,
            full_text=full_text,
            sections=sections,
            metadata=metadata
        )

        console.print(f"[green]✓ 论文解析完成[/]")
        console.print(f"  标题: {title[:60]}...")
        console.print(f"  作者: {len(authors)} 人")
        console.print(f"  章节: {len(sections)} 个")

        return paper_content

    def _extract_metadata(self, doc, full_text: str) -> Dict:
        """提取元数据"""
        metadata = {}

        # 从文档信息中提取
        metadata.update(doc.metadata)

        # 从文本中提取
        metadata['abstract'] = self._extract_abstract(full_text)
        metadata['title'] = self._extract_title(full_text)

        return metadata

    def _extract_sections(self, full_text: str) -> Dict[str, str]:
        """提取章节"""
        sections = {}

        # 常见章节标题模式
        section_patterns = [
            r'\n(\d+\.\s+([^\n]+))\n',           # 1. Introduction
            r'\n([A-Z][A-Za-z\s]+)\n\n',         # Abstract
            r'##\s+([^\n]+)\n',                  # Markdown style
        ]

        current_section = "Introduction"
        section_content = []

        lines = full_text.split('\n')

        for line in lines:
            # 检测新章节
            new_section = None
            for pattern in section_patterns:
                match = re.search(pattern, line)
                if match:
                    new_section = match.group(1).strip()
                    break

            if new_section:
                # 保存上一章节
                if section_content:
                    sections[current_section] = '\n'.join(section_content)
                current_section = new_section
                section_content = []
            else:
                section_content.append(line)

        # 保存最后一章
        if section_content:
            sections[current_section] = '\n'.join(section_content)

        return sections

    def _extract_abstract(self, full_text: str) -> str:
        """提取摘要"""
        # 查找 Abstract 部分
        abstract_match = re.search(
            r'(?:Abstract|ABSTRACT)\s*\n(.*?)(?:\n\s*(?:Introduction|1\.|Keywords|Key words)|$)',
            full_text,
            re.DOTALL
        )

        if abstract_match:
            abstract = abstract_match.group(1).strip()
            # 清理多余的空白
            abstract = re.sub(r'\s+', ' ', abstract)
            return abstract[:500]  # 限制长度

        return ""

    def _extract_title(self, full_text: str) -> str:
        """提取标题"""
        lines = full_text.split('\n')

        # 标题通常在前几行，且较长
        for i, line in enumerate(lines[:10]):
            line = line.strip()
            if len(line) > 20 and len(line) < 200:
                # 检查是否像标题（首字母大写，避免常见词）
                if line[0].isupper() and line not in ['Abstract', 'Introduction']:
                    return line

        return "Unknown Title"

    def _extract_authors(self, full_text: str) -> List[str]:
        """提取作者"""
        authors = []

        # 常见作者模式
        author_patterns = [
            r'(?:Author(?:s)?|作者)\s*[:：]\s*([^\n]+)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+(?:,\s*|\s+and\s+)){2,})',
        ]

        for pattern in author_patterns:
            match = re.search(pattern, full_text)
            if match:
                author_text = match.group(1)
                # 分割作者
                authors = re.split(r',\s*|\s+and\s+', author_text)
                authors = [a.strip() for a in authors if a.strip()]
                if authors:
                    return authors[:10]  # 限制数量

        return authors
