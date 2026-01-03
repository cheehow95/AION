"""
AION Extended Context System - Context Compression
===================================================

Smart compression for long contexts:
- Semantic compression preserving meaning
- Summary-based reduction
- Hierarchical compression for nested content
- Key information extraction

Enables efficient use of 256K context window.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import re


@dataclass
class CompressionResult:
    """Result of compression operation."""
    original_text: str = ""
    compressed_text: str = ""
    original_tokens: int = 0
    compressed_tokens: int = 0
    compression_ratio: float = 0.0
    preserved_entities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def tokens_saved(self) -> int:
        return self.original_tokens - self.compressed_tokens


class CompressionStrategy(ABC):
    """Base class for compression strategies."""
    
    @abstractmethod
    async def compress(self, text: str, target_ratio: float = 0.5) -> CompressionResult:
        """Compress text to target ratio."""
        pass
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count."""
        return len(text) // 4


class SemanticCompressor(CompressionStrategy):
    """Compresses text while preserving semantic meaning."""
    
    # Words that are often removable without losing meaning
    FILLER_PATTERNS = [
        r'\b(very|really|quite|rather|somewhat|fairly|pretty)\b',
        r'\b(just|simply|merely|only|basically|essentially)\b',
        r'\b(actually|literally|definitely|certainly|obviously)\b',
        r'\b(kind of|sort of|type of)\b',
        r'\b(in order to)\b',
        r'\b(the fact that)\b',
        r'\b(it is|there is|there are)\b',
    ]
    
    # Entities to preserve
    ENTITY_PATTERNS = [
        r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Proper nouns
        r'\b\d+(?:\.\d+)?%?\b',  # Numbers
        r'\b\d{4}-\d{2}-\d{2}\b',  # Dates
        r'"[^"]*"',  # Quoted text
        r'`[^`]*`',  # Code
    ]
    
    async def compress(self, text: str, target_ratio: float = 0.5) -> CompressionResult:
        """Compress using semantic preservation."""
        original_tokens = self.estimate_tokens(text)
        
        # Extract entities to preserve
        preserved = []
        for pattern in self.ENTITY_PATTERNS:
            matches = re.findall(pattern, text)
            preserved.extend(matches)
        
        compressed = text
        
        # Remove filler words
        for pattern in self.FILLER_PATTERNS:
            compressed = re.sub(pattern, '', compressed, flags=re.IGNORECASE)
        
        # Remove redundant whitespace
        compressed = re.sub(r'\s+', ' ', compressed).strip()
        
        # Remove redundant punctuation
        compressed = re.sub(r'\.{2,}', '.', compressed)
        compressed = re.sub(r',\s*,', ',', compressed)
        
        compressed_tokens = self.estimate_tokens(compressed)
        
        return CompressionResult(
            original_text=text,
            compressed_text=compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compressed_tokens / original_tokens if original_tokens > 0 else 1.0,
            preserved_entities=list(set(preserved))[:20]
        )


class SummaryCompressor(CompressionStrategy):
    """Compresses text by extractive summarization."""
    
    async def compress(self, text: str, target_ratio: float = 0.3) -> CompressionResult:
        """Compress using extractive summary."""
        original_tokens = self.estimate_tokens(text)
        
        # Split into sentences
        sentences = self._split_sentences(text)
        
        if len(sentences) <= 3:
            return CompressionResult(
                original_text=text,
                compressed_text=text,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                compression_ratio=1.0
            )
        
        # Score sentences by importance
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            score = self._score_sentence(sentence, i, len(sentences))
            scored_sentences.append((sentence, score))
        
        # Sort by score and select top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        target_count = max(1, int(len(sentences) * target_ratio))
        
        # Get top sentences but maintain original order
        top_sentences = scored_sentences[:target_count]
        top_sentences.sort(key=lambda x: sentences.index(x[0]))
        
        compressed = ' '.join(s[0] for s in top_sentences)
        compressed_tokens = self.estimate_tokens(compressed)
        
        return CompressionResult(
            original_text=text,
            compressed_text=compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compressed_tokens / original_tokens if original_tokens > 0 else 1.0
        )
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _score_sentence(self, sentence: str, position: int, total: int) -> float:
        """Score sentence importance."""
        score = 0.0
        
        # Position bonus (first and last sentences are important)
        if position == 0:
            score += 2.0
        elif position == total - 1:
            score += 1.5
        elif position < 3:
            score += 1.0
        
        # Length bonus (medium length sentences are best)
        word_count = len(sentence.split())
        if 10 <= word_count <= 25:
            score += 1.0
        elif word_count < 5:
            score -= 0.5
        
        # Contains important indicators
        important_words = ['important', 'key', 'main', 'critical', 'essential',
                          'conclusion', 'result', 'therefore', 'however', 'because']
        for word in important_words:
            if word in sentence.lower():
                score += 0.5
        
        # Contains numbers (often factual)
        if re.search(r'\d+', sentence):
            score += 0.3
        
        return score


class HierarchicalCompressor(CompressionStrategy):
    """Compresses hierarchical/structured content."""
    
    async def compress(self, text: str, target_ratio: float = 0.5) -> CompressionResult:
        """Compress hierarchical content."""
        original_tokens = self.estimate_tokens(text)
        
        # Detect structure
        lines = text.split('\n')
        structure = self._analyze_structure(lines)
        
        # Compress based on structure
        compressed_lines = []
        for line, level in zip(lines, structure):
            if level == 'header':
                compressed_lines.append(line)  # Keep headers
            elif level == 'list_item':
                # Compress list items
                compressed = self._compress_line(line)
                compressed_lines.append(compressed)
            elif level == 'code':
                compressed_lines.append(line)  # Keep code as-is
            else:
                # Regular text - apply compression
                if len(line) > 100:
                    compressed_lines.append(line[:100] + '...')
                else:
                    compressed_lines.append(line)
        
        compressed = '\n'.join(compressed_lines)
        compressed_tokens = self.estimate_tokens(compressed)
        
        return CompressionResult(
            original_text=text,
            compressed_text=compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compressed_tokens / original_tokens if original_tokens > 0 else 1.0,
            metadata={'structure': structure[:10]}
        )
    
    def _analyze_structure(self, lines: List[str]) -> List[str]:
        """Analyze line structure."""
        structure = []
        in_code = False
        
        for line in lines:
            stripped = line.strip()
            
            if stripped.startswith('```'):
                in_code = not in_code
                structure.append('code')
            elif in_code:
                structure.append('code')
            elif stripped.startswith('#'):
                structure.append('header')
            elif re.match(r'^[-*•]\s', stripped) or re.match(r'^\d+\.\s', stripped):
                structure.append('list_item')
            elif not stripped:
                structure.append('empty')
            else:
                structure.append('text')
        
        return structure
    
    def _compress_line(self, line: str) -> str:
        """Compress a single line."""
        # Remove excessive whitespace
        compressed = re.sub(r'\s+', ' ', line)
        return compressed.strip()


class AdaptiveCompressor:
    """Adaptively selects and applies compression strategies."""
    
    def __init__(self):
        self.semantic = SemanticCompressor()
        self.summary = SummaryCompressor()
        self.hierarchical = HierarchicalCompressor()
    
    async def compress(self, text: str, target_tokens: int = None,
                       target_ratio: float = 0.5) -> CompressionResult:
        """Adaptively compress text."""
        # Detect content type
        content_type = self._detect_content_type(text)
        
        # Choose strategy based on content type
        if content_type == 'structured':
            result = await self.hierarchical.compress(text, target_ratio)
        elif content_type == 'narrative':
            result = await self.summary.compress(text, target_ratio)
        else:
            result = await self.semantic.compress(text, target_ratio)
        
        # If still too long and target specified, apply additional compression
        if target_tokens and result.compressed_tokens > target_tokens:
            additional_ratio = target_tokens / result.compressed_tokens
            result = await self.summary.compress(result.compressed_text, additional_ratio)
        
        return result
    
    def _detect_content_type(self, text: str) -> str:
        """Detect content type."""
        lines = text.split('\n')
        
        # Check for structured content
        header_count = sum(1 for l in lines if l.strip().startswith('#'))
        list_count = sum(1 for l in lines if re.match(r'^[-*•]\s|^\d+\.\s', l.strip()))
        
        if header_count > 2 or list_count > 5:
            return 'structured'
        
        # Check for narrative (long paragraphs)
        avg_line_length = sum(len(l) for l in lines) / len(lines) if lines else 0
        if avg_line_length > 100:
            return 'narrative'
        
        return 'general'


async def demo_compression():
    """Demonstrate context compression."""
    print("🗜️ Context Compression Demo")
    print("=" * 50)
    
    long_text = """
    This is a very important document that contains really essential information about 
    the actual implementation of the context compression system. It is basically designed 
    to reduce token usage while preserving the key semantic meaning of the text.
    
    The system uses multiple compression strategies:
    1. Semantic compression removes filler words
    2. Summary compression extracts key sentences
    3. Hierarchical compression handles structured content
    
    In conclusion, this compression system is definitely essential for working with 
    large context windows like the 256K tokens in GPT-5.2. The compression ratio 
    achieved is typically between 30% and 50% reduction.
    """
    
    # Semantic compression
    semantic = SemanticCompressor()
    result = await semantic.compress(long_text)
    
    print(f"\n📝 Semantic Compression:")
    print(f"  Original: {result.original_tokens} tokens")
    print(f"  Compressed: {result.compressed_tokens} tokens")
    print(f"  Ratio: {result.compression_ratio:.1%}")
    print(f"  Saved: {result.tokens_saved} tokens")
    
    # Summary compression
    summary = SummaryCompressor()
    result = await summary.compress(long_text, target_ratio=0.3)
    
    print(f"\n📋 Summary Compression:")
    print(f"  Original: {result.original_tokens} tokens")
    print(f"  Compressed: {result.compressed_tokens} tokens")
    print(f"  Ratio: {result.compression_ratio:.1%}")
    
    # Adaptive compression
    adaptive = AdaptiveCompressor()
    result = await adaptive.compress(long_text, target_tokens=50)
    
    print(f"\n🔄 Adaptive Compression (target: 50 tokens):")
    print(f"  Compressed: {result.compressed_tokens} tokens")
    print(f"  Preview: {result.compressed_text[:100]}...")
    
    print("\n✅ Compression demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_compression())
