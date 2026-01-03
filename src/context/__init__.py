"""
AION Extended Context System - Package Initialization
======================================================

256K token context window management to match GPT-5.2.
"""

from .context_manager import (
    ContextWindow,
    ContextSegment,
    ContextManager,
    ContextPriority
)

from .context_compression import (
    CompressionStrategy,
    SemanticCompressor,
    SummaryCompressor,
    HierarchicalCompressor
)

from .context_chunking import (
    ChunkingStrategy,
    SemanticChunker,
    OverlappingChunker,
    AdaptiveChunker
)

from .hyper_context import (
    HyperContextManager,
    ContextPager,
    ExpertAttention,
    HyperPage
)

__all__ = [
    # Context Manager
    'ContextManager',
    'ContextWindow',
    'ContextSegment',
    'ContextPriority',
    'TokenCounter',
    # Compression
    'AdaptiveCompressor',
    'SemanticCompressor',
    'SummaryCompressor',
    'HierarchicalCompressor',
    # Chunking
    'AdaptiveChunker',
    'SemanticChunker',
    'OverlappingChunker',
    'Chunk',
    'ChunkingResult',
    # Hyper-Context (Gemini 3)
    'HyperContextManager',
    'ContextPager',
    'ExpertAttention',
    'HyperPage',
]
