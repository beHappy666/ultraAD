"""
Paper Analyzer Module

Responsible for parsing papers from various sources (arXiv, PDF, URL, text)
and extracting structured content including sections, figures, tables, and references.
"""

import os
import re
import json
from typing import Optional, List, Dict, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import tempfile

# Optional imports - handle gracefully if not available
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    from PyPDF2 import PdfReader
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

from .types import PaperSource, PaperMetadata


@dataclass
class Section:
    """Represents a section in the paper"""
    title: str
    content: str
    level: int = 1  # Heading level (1 for main sections, 2 for subsections, etc.)
    start_pos: int = 0
    end_pos: int = 0
    subsections: List['Section'] = field(default_factory=list)

    def get_full_text(self, include_subsections: bool = True) -> str:
        """Get full text of section including subsections"""
        text = self.content
        if include_subsections:
            for sub in self.subsections:
                text += "\n\n" + sub.get_full_text()
        return text


@dataclass
class Figure:
    """Represents a figure in the paper"""
    number: int
    caption: str
    description: str = ""
    page: int = 0
    image_path: Optional[str] = None

    def __str__(self) -> str:
        return f"Figure {self.number}: {self.caption[:100]}..."


@dataclass
class Table:
    """Represents a table in the paper"""
    number: int
    caption: str
    data: List[List[str]] = field(default_factory=list)
    headers: List[str] = field(default_factory=list)
    page: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert table to dictionary format"""
        return {
            'number': self.number,
            'caption': self.caption,
            'headers': self.headers,
            'data': self.data
        }


@dataclass
class PaperContent:
    """Complete structured content of a paper"""
    metadata: PaperMetadata
    source: PaperSource
    source_path: str

    # Content
    abstract: str = ""
    sections: List[Section] = field(default_factory=list)
    figures: List[Figure] = field(default_factory=list)
    tables: List[Table] = field(default_factory=list)
    references: List[str] = field(default_factory=list)

    # Raw content for fallback
    raw_text: str = ""

    def get_section(self, title: str) -> Optional[Section]:
        """Get section by title"""
        for section in self.sections:
            if section.title.lower() == title.lower():
                return section
            # Check subsections
            for sub in section.subsections:
                if sub.title.lower() == title.lower():
                    return sub
        return None

    def get_method_section(self) -> Optional[Section]:
        """Get method/methodology section"""
        method_titles = ['method', 'methodology', 'approach', 'proposed method',
                        'model architecture', 'system design']
        for section in self.sections:
            if any(title in section.title.lower() for title in method_titles):
                return section
        return None

    def get_experiments_section(self) -> Optional[Section]:
        """Get experiments/evaluation section"""
        exp_titles = ['experiment', 'evaluation', 'results', 'empirical study',
                     'performance analysis']
        for section in self.sections:
            if any(title in section.title.lower() for title in exp_titles):
                return section
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'metadata': {
                'title': self.metadata.title,
                'authors': self.metadata.authors,
                'abstract': self.metadata.abstract,
            },
            'abstract': self.abstract,
            'sections': [
                {
                    'title': s.title,
                    'level': s.level,
                    'content_length': len(s.content)
                }
                for s in self.sections
            ],
            'figures_count': len(self.figures),
            'tables_count': len(self.tables),
            'references_count': len(self.references)
        }


class PaperAnalyzer:
    """
    Main analyzer class for parsing papers from various sources.

    Supports:
    - arXiv papers (via API)
    - PDF files (local or downloaded)
    - URLs (web pages with paper content)
    - Raw text input
    """

    def __init__(self, cache_dir: Optional[str] = None,
                 llm_client=None):
        """
        Initialize the analyzer.

        Args:
            cache_dir: Directory for caching downloaded papers
            llm_client: Optional LLM client for enhanced analysis
        """
        self.cache_dir = cache_dir or tempfile.gettempdir()
        self.llm = llm_client
        self.source_handlers = {
            PaperSource.ARXIV: self._handle_arxiv,
            PaperSource.PDF: self._handle_pdf,
            PaperSource.URL: self._handle_url,
            PaperSource.TEXT: self._handle_text,
        }

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

    def analyze(self, source: str,
                  source_type: Optional[PaperSource] = None) -> PaperContent:
        """
        Analyze a paper from the given source.

        Args:
            source: Paper source (URL, path, or text)
            source_type: Type of source (auto-detected if None)

        Returns:
            Structured paper content
        """
        # Auto-detect source type if not provided
        if source_type is None or source_type == PaperSource.AUTO:
            source_type = self._detect_source_type(source)

        # Get appropriate handler
        handler = self.source_handlers.get(source_type)
        if not handler:
            raise ValueError(f"Unsupported source type: {source_type}")

        # Process source
        raw_content = handler(source)

        # Parse and structure content
        paper_content = self._parse_content(raw_content, source, source_type)

        return paper_content

    def _detect_source_type(self, source: str) -> PaperSource:
        """Detect the type of paper source"""
        source = source.strip()

        # Check for arXiv
        if 'arxiv.org' in source or source.startswith('arXiv:'):
            return PaperSource.ARXIV

        # Check for PDF file
        if source.lower().endswith('.pdf') or source.startswith('file://'):
            return PaperSource.PDF

        # Check for URL
        if source.startswith(('http://', 'https://')):
            # Try to determine if it's a PDF URL
            if '.pdf' in source.lower():
                return PaperSource.PDF
            return PaperSource.URL

        # Assume it's text content
        return PaperSource.TEXT

    def _handle_arxiv(self, source: str) -> Dict[str, Any]:
        """Handle arXiv paper source"""
        if not HAS_REQUESTS:
            raise ImportError(
                "requests is required for arXiv support. "
                "Install with: pip install requests"
            )

        # Extract arXiv ID
        arxiv_id = self._extract_arxiv_id(source)

        # Download from arXiv API
        api_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"

        print(f"Fetching arXiv paper: {arxiv_id}")
        response = requests.get(api_url, timeout=30)
        response.raise_for_status()

        # Parse XML response
        metadata = self._parse_arxiv_response(response.text)

        # Download PDF
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        pdf_path = os.path.join(self.cache_dir, f"{arxiv_id}.pdf")

        if not os.path.exists(pdf_path):
            print(f"Downloading PDF: {pdf_url}")
            response = requests.get(pdf_url, timeout=60)
            response.raise_for_status()
            with open(pdf_path, 'wb') as f:
                f.write(response.content)

        return {
            'source_type': 'arxiv',
            'arxiv_id': arxiv_id,
            'metadata': metadata,
            'pdf_path': pdf_path,
            'api_response': response.text if 'response' in dir() else None
        }

    def _extract_arxiv_id(self, source: str) -> str:
        """Extract arXiv ID from source string"""
        # Handle various arXiv URL formats
        patterns = [
            r'arxiv\.org/abs/([\d\.]+)',
            r'arxiv\.org/pdf/([\d\.]+)',
            r'arXiv:([\d\.]+)',
            r'([\d\.]+)$'  # Just the ID
        ]

        for pattern in patterns:
            match = re.search(pattern, source)
            if match:
                return match.group(1)

        raise ValueError(f"Could not extract arXiv ID from: {source}")

    def _parse_arxiv_response(self, xml_content: str) -> Dict[str, Any]:
        """Parse arXiv API XML response"""
        try:
            import xml.etree.ElementTree as ET

            root = ET.fromstring(xml_content)

            # Define namespaces
            ns = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }

            entry = root.find('atom:entry', ns)
            if entry is None:
                return {}

            metadata = {
                'title': entry.findtext('atom:title', '', ns).strip(),
                'authors': [
                    author.findtext('atom:name', '', ns)
                    for author in entry.findall('atom:author', ns)
                ],
                'abstract': entry.findtext('atom:summary', '', ns).strip(),
                'published': entry.findtext('atom:published', '', ns),
                'updated': entry.findtext('atom:updated', '', ns),
                'categories': [
                    cat.get('term', '')
                    for cat in entry.findall('atom:category', ns)
                ],
                'primary_category': entry.findtext('arxiv:primary_category', '', ns),
                'journal_ref': entry.findtext('arxiv:journal_ref', '', ns),
                'comment': entry.findtext('arxiv:comment', '', ns),
                'doi': entry.findtext('arxiv:doi', '', ns)
            }

            return metadata

        except Exception as e:
            print(f"Warning: Could not parse arXiv response: {e}")
            return {}

    def _handle_pdf(self, source: str) -> Dict[str, Any]:
        """Handle PDF file source"""
        # Determine PDF path
        if source.startswith('http'):
            # Download PDF
            if not HAS_REQUESTS:
                raise ImportError("requests is required for URL PDF support")

            pdf_path = os.path.join(self.cache_dir, f"download_{timestamp()}.pdf")
            print(f"Downloading PDF from: {source}")
            response = requests.get(source, timeout=60)
            response.raise_for_status()
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
        else:
            # Local file path
            pdf_path = source.replace('file://', '')
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF not found: {pdf_path}")

        return {
            'source_type': 'pdf',
            'pdf_path': pdf_path,
            'is_local': not source.startswith('http')
        }

    def _handle_url(self, source: str) -> Dict[str, Any]:
        """Handle URL source (web page)"""
        if not HAS_REQUESTS:
            raise ImportError("requests is required for URL support")

        print(f"Fetching content from: {source}")
        response = requests.get(source, timeout=30)
        response.raise_for_status()

        # Try to extract main content
        content = response.text

        return {
            'source_type': 'url',
            'url': source,
            'content': content,
            'headers': dict(response.headers)
        }

    def _handle_text(self, source: str) -> Dict[str, Any]:
        """Handle raw text source"""
        return {
            'source_type': 'text',
            'content': source
        }

    def _parse_content(self, raw_content: Dict[str, Any],
                       source: str,
                       source_type: PaperSource) -> PaperContent:
        """
        Parse raw content into structured PaperContent.

        This is the main parsing logic that extracts sections, figures, tables, etc.
        """
        content_type = raw_content.get('source_type', 'unknown')

        # Initialize paper content
        paper = PaperContent(
            metadata=PaperMetadata(
                title="",
                authors=[],
                abstract=""
            ),
            source=source_type,
            source_path=source
        )

        # Parse based on content type
        if content_type == 'arxiv':
            self._parse_arxiv_content(paper, raw_content)
        elif content_type == 'pdf':
            self._parse_pdf_content(paper, raw_content)
        elif content_type == 'url':
            self._parse_url_content(paper, raw_content)
        elif content_type == 'text':
            self._parse_text_content(paper, raw_content)

        return paper

    def _parse_arxiv_content(self, paper: PaperContent,
                             raw_content: Dict[str, Any]) -> None:
        """Parse arXiv content"""
        metadata = raw_content.get('metadata', {})

        paper.metadata.title = metadata.get('title', '')
        paper.metadata.authors = metadata.get('authors', [])
        paper.metadata.abstract = metadata.get('abstract', '')
        paper.metadata.published = metadata.get('published', '')
        paper.metadata.arxiv_id = raw_content.get('arxiv_id', '')
        paper.metadata.categories = metadata.get('categories', [])
        paper.metadata.doi = metadata.get('doi', '')

        paper.abstract = paper.metadata.abstract

        # Parse PDF content for full text
        pdf_path = raw_content.get('pdf_path', '')
        if pdf_path and os.path.exists(pdf_path):
            self._extract_pdf_sections(paper, pdf_path)

    def _parse_pdf_content(self, paper: PaperContent,
                           raw_content: Dict[str, Any]) -> None:
        """Parse PDF content"""
        pdf_path = raw_content.get('pdf_path', '')
        if not pdf_path or not os.path.exists(pdf_path):
            return

        # Extract text from PDF
        self._extract_pdf_sections(paper, pdf_path)

    def _extract_pdf_sections(self, paper: PaperContent,
                               pdf_path: str) -> None:
        """Extract sections from PDF"""
        try:
            if HAS_PDFPLUMBER:
                self._extract_with_pdfplumber(paper, pdf_path)
            elif HAS_PYPDF2:
                self._extract_with_pypdf2(paper, pdf_path)
            else:
                print("Warning: No PDF library available. Install pdfplumber or PyPDF2.")
        except Exception as e:
            print(f"Warning: Error extracting PDF content: {e}")

    def _extract_with_pdfplumber(self, paper: PaperContent,
                                  pdf_path: str) -> None:
        """Extract content using pdfplumber"""
        with pdfplumber.open(pdf_path) as pdf:
            full_text = ""
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n\n"

            paper.raw_text = full_text

            # Extract sections from text
            self._parse_sections_from_text(paper, full_text)

    def _extract_with_pypdf2(self, paper: PaperContent,
                              pdf_path: str) -> None:
        """Extract content using PyPDF2"""
        reader = PdfReader(pdf_path)
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n\n"

        paper.raw_text = full_text
        self._parse_sections_from_text(paper, full_text)

    def _parse_sections_from_text(self, paper: PaperContent,
                                   text: str) -> None:
        """Parse sections from text content"""
        # Common section headers in academic papers
        section_patterns = [
            r'\n\s*(\d+)\.\s*([A-Z][A-Za-z\s]+)\n',  # 1. Introduction
            r'\n\s*([A-Z][A-Z\s]{2,})\n',  # INTRODUCTION
            r'\n\s*([\d\.]+)\s+([A-Z][a-z]+)\n',  # 2.1 Method
        ]

        sections_found = []

        for pattern in section_patterns:
            matches = list(re.finditer(pattern, text))
            for match in matches:
                start = match.start()
                title = match.group(0).strip()
                sections_found.append((start, title))

        # Sort by position and create sections
        sections_found.sort(key=lambda x: x[0])

        for i, (start, title) in enumerate(sections_found):
            end = sections_found[i + 1][0] if i + 1 < len(sections_found) else len(text)
            content = text[start:end].strip()

            section = Section(
                title=title,
                content=content,
                level=1,
                start_pos=start,
                end_pos=end
            )

            paper.sections.append(section)

        # Extract abstract if not already set
        if not paper.abstract:
            self._extract_abstract(paper, text)

    def _extract_abstract(self, paper: PaperContent, text: str) -> None:
        """Extract abstract from text"""
        # Look for abstract section
        abstract_patterns = [
            r'(?i)abstract\s*[:\n]\s*([^\n]+(?:\n(?!\d+\.|introduction|related)[^\n]+)*)',
            r'(?i)abstract\s*-?\s*\n\s*([^\n]+(?:\n(?!\d+\.|introduction|related)[^\n]+)*)',
        ]

        for pattern in abstract_patterns:
            match = re.search(pattern, text[:5000])  # Check first 5000 chars
            if match:
                paper.abstract = match.group(1).strip()
                break

    def _parse_url_content(self, paper: PaperContent,
                           raw_content: Dict[str, Any]) -> None:
        """Parse URL/web page content"""
        content = raw_content.get('content', '')
        paper.raw_text = content
        self._parse_sections_from_text(paper, content)

    def _parse_text_content(self, paper: PaperContent,
                            raw_content: Dict[str, Any]) -> None:
        """Parse raw text content"""
        content = raw_content.get('content', '')
        paper.raw_text = content
        self._parse_sections_from_text(paper, content)


def timestamp() -> str:
    """Generate timestamp string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# Convenience function for direct usage
def analyze_paper(source: str,
                  source_type: Optional[PaperSource] = None,
                  cache_dir: Optional[str] = None) -> PaperContent:
    """
    Convenience function to analyze a paper.

    Args:
        source: Paper source (URL, path, or text)
        source_type: Type of source (auto-detected if None)
        cache_dir: Directory for caching

    Returns:
        Structured paper content
    """
    analyzer = PaperAnalyzer(cache_dir=cache_dir)
    return analyzer.analyze(source, source_type)
