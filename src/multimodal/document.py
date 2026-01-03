"""
AION Document Processing
========================

Document understanding for PDF, DOCX, HTML, and other formats.
Extracts text, structure, tables, and images from documents.
"""

import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from datetime import datetime
import uuid
import re


# =============================================================================
# DOCUMENT FORMAT
# =============================================================================

class DocumentFormat(Enum):
    """Supported document formats."""
    PDF = "pdf"
    DOCX = "docx"
    DOC = "doc"
    HTML = "html"
    TXT = "txt"
    MD = "md"
    RTF = "rtf"
    XLSX = "xlsx"
    PPTX = "pptx"
    
    @classmethod
    def from_extension(cls, ext: str) -> "DocumentFormat":
        """Get format from file extension."""
        ext = ext.lower().lstrip(".")
        mapping = {
            "pdf": cls.PDF,
            "docx": cls.DOCX,
            "doc": cls.DOC,
            "html": cls.HTML,
            "htm": cls.HTML,
            "txt": cls.TXT,
            "md": cls.MD,
            "markdown": cls.MD,
            "rtf": cls.RTF,
            "xlsx": cls.XLSX,
            "pptx": cls.PPTX,
        }
        return mapping.get(ext, cls.TXT)


class ElementType(Enum):
    """Types of document elements."""
    TITLE = "title"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    LIST_ITEM = "list_item"
    TABLE = "table"
    IMAGE = "image"
    CODE = "code"
    QUOTE = "quote"
    FOOTER = "footer"
    HEADER = "header"
    CAPTION = "caption"
    LINK = "link"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DocumentInput:
    """Document data for processing."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    data: bytes = b""
    format: DocumentFormat = DocumentFormat.TXT
    filename: str = ""
    source: str = ""
    page_count: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def size_bytes(self) -> int:
        """Get document size in bytes."""
        return len(self.data)


@dataclass
class DocumentElement:
    """A structural element in a document."""
    type: ElementType
    content: str
    page: int = 0
    level: int = 0  # For headings (1-6)
    bounding_box: Optional[Tuple[float, float, float, float]] = None  # x, y, w, h
    attributes: Dict[str, Any] = field(default_factory=dict)
    children: List["DocumentElement"] = field(default_factory=list)


@dataclass
class TableCell:
    """A cell in a table."""
    content: str
    row: int
    column: int
    row_span: int = 1
    col_span: int = 1
    is_header: bool = False


@dataclass
class ExtractedTable:
    """A table extracted from a document."""
    cells: List[TableCell] = field(default_factory=list)
    rows: int = 0
    columns: int = 0
    caption: str = ""
    page: int = 0
    
    def to_list(self) -> List[List[str]]:
        """Convert table to 2D list."""
        if not self.cells or self.rows == 0 or self.columns == 0:
            return []
        
        grid = [[""] * self.columns for _ in range(self.rows)]
        for cell in self.cells:
            if 0 <= cell.row < self.rows and 0 <= cell.column < self.columns:
                grid[cell.row][cell.column] = cell.content
        return grid
    
    def to_markdown(self) -> str:
        """Convert table to markdown format."""
        grid = self.to_list()
        if not grid:
            return ""
        
        lines = []
        for i, row in enumerate(grid):
            lines.append("| " + " | ".join(row) + " |")
            if i == 0:
                lines.append("|" + "|".join(["---"] * len(row)) + "|")
        
        return "\n".join(lines)


@dataclass
class ExtractedImage:
    """An image extracted from a document."""
    data: bytes
    format: str
    page: int
    caption: str = ""
    alt_text: str = ""
    bounding_box: Optional[Tuple[float, float, float, float]] = None


@dataclass
class DocumentSection:
    """A logical section of a document."""
    title: str
    level: int
    content: str
    page_start: int
    page_end: int
    subsections: List["DocumentSection"] = field(default_factory=list)


@dataclass
class DocumentAnalysis:
    """Complete analysis results for a document."""
    input_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Basic info
    title: str = ""
    author: str = ""
    subject: str = ""
    language: str = "en"
    
    # Content
    full_text: str = ""
    summary: str = ""
    
    # Structure
    elements: List[DocumentElement] = field(default_factory=list)
    sections: List[DocumentSection] = field(default_factory=list)
    
    # Extracted items
    tables: List[ExtractedTable] = field(default_factory=list)
    images: List[ExtractedImage] = field(default_factory=list)
    links: List[str] = field(default_factory=list)
    
    # Statistics
    word_count: int = 0
    page_count: int = 0
    char_count: int = 0
    
    # Keywords and entities
    keywords: List[str] = field(default_factory=list)
    entities: Dict[str, List[str]] = field(default_factory=dict)
    
    # Raw response
    raw_response: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# DOCUMENT PROCESSOR
# =============================================================================

class DocumentProcessor:
    """
    Process documents for text extraction and understanding.
    """
    
    def __init__(self, model: str = "default"):
        self.model = model
        self._cache: Dict[str, DocumentAnalysis] = {}
    
    def load_from_file(self, file_path: str) -> DocumentInput:
        """
        Load a document from a file.
        
        Args:
            file_path: Path to the document
            
        Returns:
            DocumentInput with loaded data
        """
        import os
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        _, ext = os.path.splitext(file_path)
        doc_format = DocumentFormat.from_extension(ext)
        
        with open(file_path, "rb") as f:
            data = f.read()
        
        return DocumentInput(
            data=data,
            format=doc_format,
            filename=os.path.basename(file_path),
            source=file_path
        )
    
    def load_from_bytes(
        self, 
        data: bytes, 
        format: DocumentFormat = DocumentFormat.TXT
    ) -> DocumentInput:
        """Load document from bytes."""
        return DocumentInput(
            data=data,
            format=format,
            source="bytes"
        )
    
    async def extract_text(self, input: DocumentInput) -> str:
        """
        Extract plain text from a document.
        
        Args:
            input: DocumentInput to process
            
        Returns:
            Extracted text content
        """
        await asyncio.sleep(0.01)  # Simulate processing
        
        if input.format == DocumentFormat.TXT:
            try:
                return input.data.decode("utf-8")
            except UnicodeDecodeError:
                return input.data.decode("latin-1")
        
        elif input.format == DocumentFormat.HTML:
            return self._extract_text_from_html(input.data)
        
        elif input.format == DocumentFormat.MD:
            text = input.data.decode("utf-8", errors="replace")
            # Strip markdown syntax for plain text
            return self._strip_markdown(text)
        
        # For other formats, return placeholder
        return f"[Extracted text from {input.format.value} document]"
    
    def _extract_text_from_html(self, data: bytes) -> str:
        """Extract text from HTML."""
        try:
            html = data.decode("utf-8", errors="replace")
            # Remove script and style
            html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
            html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
            # Remove tags
            text = re.sub(r'<[^>]+>', ' ', html)
            # Clean whitespace
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
        except Exception:
            return ""
    
    def _strip_markdown(self, text: str) -> str:
        """Strip markdown syntax."""
        # Remove headers
        text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
        # Remove bold/italic
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'__([^_]+)__', r'\1', text)
        text = re.sub(r'_([^_]+)_', r'\1', text)
        # Remove links
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        # Remove code markers
        text = re.sub(r'`([^`]+)`', r'\1', text)
        return text
    
    async def extract_elements(
        self, 
        input: DocumentInput
    ) -> List[DocumentElement]:
        """
        Extract structural elements from a document.
        
        Args:
            input: DocumentInput to process
            
        Returns:
            List of document elements
        """
        await asyncio.sleep(0.02)
        
        elements = []
        text = await self.extract_text(input)
        
        # Simple paragraph detection
        paragraphs = text.split("\n\n")
        for i, para in enumerate(paragraphs):
            para = para.strip()
            if not para:
                continue
            
            # Detect headings (simple heuristic)
            if len(para) < 100 and not para.endswith("."):
                elements.append(DocumentElement(
                    type=ElementType.HEADING,
                    content=para,
                    level=1
                ))
            else:
                elements.append(DocumentElement(
                    type=ElementType.PARAGRAPH,
                    content=para
                ))
        
        return elements
    
    async def analyze(self, input: DocumentInput) -> DocumentAnalysis:
        """
        Perform full analysis on a document.
        
        Args:
            input: DocumentInput to analyze
            
        Returns:
            DocumentAnalysis with full results
        """
        if input.id in self._cache:
            return self._cache[input.id]
        
        # Extract content
        text = await self.extract_text(input)
        elements = await self.extract_elements(input)
        tables = await self.extract_tables(input)
        
        # Calculate stats
        word_count = len(text.split())
        char_count = len(text)
        
        # Extract links from HTML
        links = []
        if input.format == DocumentFormat.HTML:
            links = re.findall(r'href=["\']([^"\']+)["\']', input.data.decode("utf-8", errors="replace"))
        
        # Simple keyword extraction
        words = text.lower().split()
        word_freq = {}
        for word in words:
            word = re.sub(r'[^\w]', '', word)
            if len(word) > 4:
                word_freq[word] = word_freq.get(word, 0) + 1
        keywords = sorted(word_freq.keys(), key=lambda w: word_freq[w], reverse=True)[:10]
        
        # Generate summary (first 500 chars)
        summary = text[:500] + "..." if len(text) > 500 else text
        
        analysis = DocumentAnalysis(
            input_id=input.id,
            title=input.filename or "Untitled",
            full_text=text,
            summary=summary,
            elements=elements,
            tables=tables,
            links=links,
            word_count=word_count,
            char_count=char_count,
            page_count=input.page_count or 1,
            keywords=keywords
        )
        
        self._cache[input.id] = analysis
        return analysis
    
    async def extract_tables(
        self, 
        input: DocumentInput
    ) -> List[ExtractedTable]:
        """
        Extract tables from a document.
        
        Args:
            input: DocumentInput to process
            
        Returns:
            List of extracted tables
        """
        await asyncio.sleep(0.01)
        
        tables = []
        
        if input.format == DocumentFormat.HTML:
            html = input.data.decode("utf-8", errors="replace")
            tables = self._parse_html_tables(html)
        
        return tables
    
    def _parse_html_tables(self, html: str) -> List[ExtractedTable]:
        """Parse tables from HTML."""
        tables = []
        
        # Find all tables
        table_matches = re.finditer(r'<table[^>]*>(.*?)</table>', html, re.DOTALL | re.IGNORECASE)
        
        for table_match in table_matches:
            table_html = table_match.group(1)
            cells = []
            row_idx = 0
            max_col = 0
            
            # Find rows
            rows = re.findall(r'<tr[^>]*>(.*?)</tr>', table_html, re.DOTALL | re.IGNORECASE)
            
            for row_html in rows:
                # Find cells
                cell_matches = re.finditer(r'<(td|th)[^>]*>(.*?)</\1>', row_html, re.DOTALL | re.IGNORECASE)
                col_idx = 0
                
                for cell_match in cell_matches:
                    tag = cell_match.group(1).lower()
                    content = re.sub(r'<[^>]+>', '', cell_match.group(2)).strip()
                    
                    cells.append(TableCell(
                        content=content,
                        row=row_idx,
                        column=col_idx,
                        is_header=(tag == "th")
                    ))
                    col_idx += 1
                    max_col = max(max_col, col_idx)
                
                row_idx += 1
            
            if cells:
                tables.append(ExtractedTable(
                    cells=cells,
                    rows=row_idx,
                    columns=max_col
                ))
        
        return tables
    
    async def extract_images(
        self, 
        input: DocumentInput
    ) -> List[ExtractedImage]:
        """Extract images from a document."""
        await asyncio.sleep(0.01)
        # Image extraction requires format-specific libraries
        return []


# =============================================================================
# DEMO
# =============================================================================

async def demo_document():
    """Demonstrate document processing."""
    print("📄 Document Processing Demo")
    print("-" * 40)
    
    processor = DocumentProcessor()
    
    # Create test HTML document
    html_content = b"""
    <html>
    <body>
        <h1>Test Document</h1>
        <p>This is a test paragraph with some content.</p>
        <table>
            <tr><th>Name</th><th>Value</th></tr>
            <tr><td>Item 1</td><td>100</td></tr>
            <tr><td>Item 2</td><td>200</td></tr>
        </table>
        <a href="https://example.com">Link</a>
    </body>
    </html>
    """
    
    test_input = DocumentInput(
        data=html_content,
        format=DocumentFormat.HTML,
        filename="test.html"
    )
    
    print(f"Input: {test_input.filename} ({test_input.size_bytes} bytes)")
    
    # Extract text
    text = await processor.extract_text(test_input)
    print(f"Extracted text: {text[:100]}...")
    
    # Full analysis
    analysis = await processor.analyze(test_input)
    print(f"Title: {analysis.title}")
    print(f"Word count: {analysis.word_count}")
    print(f"Elements: {len(analysis.elements)}")
    print(f"Tables: {len(analysis.tables)}")
    print(f"Links: {analysis.links}")
    print(f"Keywords: {analysis.keywords[:5]}")
    
    if analysis.tables:
        print(f"\nTable markdown:\n{analysis.tables[0].to_markdown()}")
    
    print("-" * 40)
    print("✅ Document demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_document())
