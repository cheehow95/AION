"""
AION Extended Context System - Context Chunking
================================================

Efficient chunking strategies for long documents:
- Semantic chunking (by meaning boundaries)
- Overlapping chunks (for context continuity)
- Adaptive chunking (based on content type)
- Token-aware chunking

Optimized for 256K context window utilization.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import re


@dataclass
class Chunk:
    """A chunk of content."""
    id: str = ""
    content: str = ""
    token_count: int = 0
    start_position: int = 0
    end_position: int = 0
    overlap_previous: int = 0
    overlap_next: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkingResult:
    """Result of chunking operation."""
    chunks: List[Chunk] = field(default_factory=list)
    total_tokens: int = 0
    chunk_count: int = 0
    average_chunk_size: int = 0
    
    def get_chunk(self, index: int) -> Optional[Chunk]:
        if 0 <= index < len(self.chunks):
            return self.chunks[index]
        return None


class ChunkingStrategy(ABC):
    """Base class for chunking strategies."""
    
    @abstractmethod
    def chunk(self, text: str, max_chunk_tokens: int = 1000) -> ChunkingResult:
        """Chunk text into smaller pieces."""
        pass
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count."""
        return len(text) // 4


class SemanticChunker(ChunkingStrategy):
    """Chunks by semantic boundaries (paragraphs, sections)."""
    
    BOUNDARY_PATTERNS = [
        r'\n\n+',          # Paragraph breaks
        r'\n#+\s',         # Markdown headers
        r'\n---+\n',       # Horizontal rules
        r'(?<=[.!?])\s+(?=[A-Z])',  # Sentence boundaries
    ]
    
    def chunk(self, text: str, max_chunk_tokens: int = 1000) -> ChunkingResult:
        """Chunk by semantic boundaries."""
        chunks = []
        
        # First, try paragraph-level chunking
        paragraphs = re.split(r'\n\n+', text)
        
        current_chunk = []
        current_tokens = 0
        position = 0
        
        for para in paragraphs:
            para_tokens = self.estimate_tokens(para)
            
            if current_tokens + para_tokens > max_chunk_tokens and current_chunk:
                # Save current chunk
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append(Chunk(
                    id=f"chunk_{len(chunks)}",
                    content=chunk_text,
                    token_count=current_tokens,
                    start_position=position - len(chunk_text),
                    end_position=position
                ))
                current_chunk = []
                current_tokens = 0
            
            current_chunk.append(para)
            current_tokens += para_tokens
            position += len(para) + 2  # +2 for \n\n
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append(Chunk(
                id=f"chunk_{len(chunks)}",
                content=chunk_text,
                token_count=current_tokens,
                start_position=position - len(chunk_text),
                end_position=position
            ))
        
        total_tokens = sum(c.token_count for c in chunks)
        
        return ChunkingResult(
            chunks=chunks,
            total_tokens=total_tokens,
            chunk_count=len(chunks),
            average_chunk_size=total_tokens // len(chunks) if chunks else 0
        )


class OverlappingChunker(ChunkingStrategy):
    """Creates overlapping chunks for context continuity."""
    
    def __init__(self, overlap_tokens: int = 100):
        self.overlap_tokens = overlap_tokens
    
    def chunk(self, text: str, max_chunk_tokens: int = 1000) -> ChunkingResult:
        """Create overlapping chunks."""
        chunks = []
        
        # Calculate character positions based on token estimates
        chars_per_token = 4
        max_chunk_chars = max_chunk_tokens * chars_per_token
        overlap_chars = self.overlap_tokens * chars_per_token
        
        step = max_chunk_chars - overlap_chars
        position = 0
        
        while position < len(text):
            end = min(position + max_chunk_chars, len(text))
            
            # Try to find a good break point
            if end < len(text):
                # Look for sentence boundary
                break_point = self._find_break_point(text, end - 50, end)
                if break_point > position:
                    end = break_point
            
            chunk_text = text[position:end]
            chunk_tokens = self.estimate_tokens(chunk_text)
            
            chunks.append(Chunk(
                id=f"chunk_{len(chunks)}",
                content=chunk_text,
                token_count=chunk_tokens,
                start_position=position,
                end_position=end,
                overlap_previous=overlap_chars if position > 0 else 0,
                overlap_next=overlap_chars if end < len(text) else 0
            ))
            
            position += step
            if position >= end:
                position = end
        
        total_tokens = sum(c.token_count for c in chunks)
        
        return ChunkingResult(
            chunks=chunks,
            total_tokens=total_tokens,
            chunk_count=len(chunks),
            average_chunk_size=total_tokens // len(chunks) if chunks else 0
        )
    
    def _find_break_point(self, text: str, start: int, end: int) -> int:
        """Find a good break point in the text."""
        # Look for sentence boundaries
        for i in range(end, start, -1):
            if text[i-1] in '.!?' and i < len(text) and text[i] == ' ':
                return i
        
        # Look for paragraph breaks
        for i in range(end, start, -1):
            if text[i-1] == '\n':
                return i
        
        # Look for word boundaries
        for i in range(end, start, -1):
            if text[i-1] == ' ':
                return i
        
        return end


class AdaptiveChunker(ChunkingStrategy):
    """Adapts chunking strategy based on content type."""
    
    def __init__(self):
        self.semantic = SemanticChunker()
        self.overlapping = OverlappingChunker()
    
    def chunk(self, text: str, max_chunk_tokens: int = 1000) -> ChunkingResult:
        """Adaptively chunk based on content analysis."""
        content_type = self._analyze_content(text)
        
        if content_type == 'structured':
            # Use section-based chunking
            return self._chunk_by_sections(text, max_chunk_tokens)
        elif content_type == 'code':
            # Use function/class-based chunking
            return self._chunk_code(text, max_chunk_tokens)
        elif content_type == 'conversation':
            # Use turn-based chunking
            return self._chunk_conversation(text, max_chunk_tokens)
        else:
            # Default to overlapping chunks
            return self.overlapping.chunk(text, max_chunk_tokens)
    
    def _analyze_content(self, text: str) -> str:
        """Analyze content type."""
        lines = text.split('\n')
        
        # Check for code
        code_indicators = sum(1 for l in lines if l.strip().startswith(('def ', 'class ', 'import ', 'from ', '```')))
        if code_indicators > len(lines) * 0.1:
            return 'code'
        
        # Check for conversation
        conversation_indicators = sum(1 for l in lines if re.match(r'^(User|Assistant|Human|AI):', l))
        if conversation_indicators > 2:
            return 'conversation'
        
        # Check for structured content
        header_count = sum(1 for l in lines if l.strip().startswith('#'))
        if header_count > 3:
            return 'structured'
        
        return 'general'
    
    def _chunk_by_sections(self, text: str, max_chunk_tokens: int) -> ChunkingResult:
        """Chunk by document sections."""
        chunks = []
        
        # Split by headers
        sections = re.split(r'(^#+\s.*$)', text, flags=re.MULTILINE)
        
        current_chunk = []
        current_tokens = 0
        
        for section in sections:
            section_tokens = self.estimate_tokens(section)
            
            if section.strip().startswith('#'):
                # Start new chunk at headers
                if current_chunk:
                    chunk_text = ''.join(current_chunk)
                    chunks.append(Chunk(
                        id=f"section_{len(chunks)}",
                        content=chunk_text,
                        token_count=current_tokens,
                        metadata={'type': 'section'}
                    ))
                current_chunk = [section]
                current_tokens = section_tokens
            else:
                if current_tokens + section_tokens > max_chunk_tokens and current_chunk:
                    chunk_text = ''.join(current_chunk)
                    chunks.append(Chunk(
                        id=f"section_{len(chunks)}",
                        content=chunk_text,
                        token_count=current_tokens,
                        metadata={'type': 'section'}
                    ))
                    current_chunk = []
                    current_tokens = 0
                
                current_chunk.append(section)
                current_tokens += section_tokens
        
        if current_chunk:
            chunk_text = ''.join(current_chunk)
            chunks.append(Chunk(
                id=f"section_{len(chunks)}",
                content=chunk_text,
                token_count=current_tokens,
                metadata={'type': 'section'}
            ))
        
        total_tokens = sum(c.token_count for c in chunks)
        return ChunkingResult(
            chunks=chunks,
            total_tokens=total_tokens,
            chunk_count=len(chunks),
            average_chunk_size=total_tokens // len(chunks) if chunks else 0
        )
    
    def _chunk_code(self, text: str, max_chunk_tokens: int) -> ChunkingResult:
        """Chunk code by functions/classes."""
        chunks = []
        
        # Split by function/class definitions
        pattern = r'((?:^(?:def |class |async def ).*?(?=^(?:def |class |async def )|\Z)))'
        matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
        
        if not matches:
            return self.overlapping.chunk(text, max_chunk_tokens)
        
        for i, match in enumerate(matches):
            if match.strip():
                chunks.append(Chunk(
                    id=f"code_{i}",
                    content=match.strip(),
                    token_count=self.estimate_tokens(match),
                    metadata={'type': 'code'}
                ))
        
        total_tokens = sum(c.token_count for c in chunks)
        return ChunkingResult(
            chunks=chunks,
            total_tokens=total_tokens,
            chunk_count=len(chunks),
            average_chunk_size=total_tokens // len(chunks) if chunks else 0
        )
    
    def _chunk_conversation(self, text: str, max_chunk_tokens: int) -> ChunkingResult:
        """Chunk conversation by turns."""
        chunks = []
        
        # Split by speaker turns
        turns = re.split(r'(?=^(?:User|Assistant|Human|AI):)', text, flags=re.MULTILINE)
        
        current_chunk = []
        current_tokens = 0
        
        for turn in turns:
            turn_tokens = self.estimate_tokens(turn)
            
            if current_tokens + turn_tokens > max_chunk_tokens and current_chunk:
                chunk_text = '\n'.join(current_chunk)
                chunks.append(Chunk(
                    id=f"conv_{len(chunks)}",
                    content=chunk_text,
                    token_count=current_tokens,
                    metadata={'type': 'conversation'}
                ))
                current_chunk = []
                current_tokens = 0
            
            if turn.strip():
                current_chunk.append(turn.strip())
                current_tokens += turn_tokens
        
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            chunks.append(Chunk(
                id=f"conv_{len(chunks)}",
                content=chunk_text,
                token_count=current_tokens,
                metadata={'type': 'conversation'}
            ))
        
        total_tokens = sum(c.token_count for c in chunks)
        return ChunkingResult(
            chunks=chunks,
            total_tokens=total_tokens,
            chunk_count=len(chunks),
            average_chunk_size=total_tokens // len(chunks) if chunks else 0
        )


class ChunkRetriever:
    """Retrieves relevant chunks based on query."""
    
    def __init__(self):
        self.chunks: List[Chunk] = []
    
    def index(self, result: ChunkingResult):
        """Index chunks for retrieval."""
        self.chunks = result.chunks
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Chunk]:
        """Retrieve top-k relevant chunks."""
        if not self.chunks:
            return []
        
        # Simple keyword-based relevance (would use embeddings in production)
        query_terms = set(query.lower().split())
        
        scored_chunks = []
        for chunk in self.chunks:
            chunk_terms = set(chunk.content.lower().split())
            overlap = len(query_terms & chunk_terms)
            score = overlap / len(query_terms) if query_terms else 0
            scored_chunks.append((chunk, score))
        
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return [c[0] for c in scored_chunks[:top_k]]


async def demo_chunking():
    """Demonstrate chunking strategies."""
    print("✂️ Context Chunking Demo")
    print("=" * 50)
    
    # Sample long document
    long_document = """
# Introduction

This is a comprehensive guide to context chunking strategies for large language models.
Context chunking is essential when dealing with documents that exceed the model's context window.

## Semantic Chunking

Semantic chunking divides text at natural boundaries like paragraphs and sections.
This preserves the logical structure of the document and maintains coherent units of meaning.

The key advantage is that each chunk represents a complete thought or concept.

## Overlapping Chunking

Overlapping chunking creates chunks that share some content at their boundaries.
This technique ensures that context is not lost at chunk boundaries.

User: How does chunking work?
Assistant: Chunking divides large texts into smaller, manageable pieces.

User: What are the different strategies?
Assistant: There are semantic, overlapping, and adaptive strategies.

## Conclusion

Choosing the right chunking strategy depends on your content type and use case.
Structured documents benefit from semantic chunking, while general text works well with overlapping chunks.
"""
    
    # Semantic chunking
    semantic = SemanticChunker()
    result = semantic.chunk(long_document, max_chunk_tokens=200)
    
    print(f"\n📝 Semantic Chunking:")
    print(f"  Chunks: {result.chunk_count}")
    print(f"  Average size: {result.average_chunk_size} tokens")
    for i, chunk in enumerate(result.chunks[:3]):
        print(f"  Chunk {i}: {chunk.token_count} tokens - {chunk.content[:50]}...")
    
    # Overlapping chunking
    overlapping = OverlappingChunker(overlap_tokens=50)
    result = overlapping.chunk(long_document, max_chunk_tokens=200)
    
    print(f"\n🔗 Overlapping Chunking:")
    print(f"  Chunks: {result.chunk_count}")
    print(f"  Average size: {result.average_chunk_size} tokens")
    
    # Adaptive chunking
    adaptive = AdaptiveChunker()
    result = adaptive.chunk(long_document, max_chunk_tokens=200)
    
    print(f"\n🔄 Adaptive Chunking:")
    print(f"  Chunks: {result.chunk_count}")
    print(f"  Content type detected: {adaptive._analyze_content(long_document)}")
    
    # Chunk retrieval
    retriever = ChunkRetriever()
    retriever.index(result)
    
    relevant = retriever.retrieve("overlapping strategy", top_k=2)
    print(f"\n🔍 Query: 'overlapping strategy'")
    print(f"  Found {len(relevant)} relevant chunks")
    
    print("\n✅ Chunking demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_chunking())
