"""
AION Extended Context System - Context Manager
===============================================

256K token context window management:
- Sliding window with priority retention
- Context importance scoring
- Automatic overflow handling
- Token budget allocation

Matches GPT-5.2's 256K context capability.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
from enum import Enum
import hashlib


class ContextPriority(Enum):
    """Priority levels for context segments."""
    CRITICAL = 5    # System prompts, safety instructions
    HIGH = 4        # Recent user messages, current task
    MEDIUM = 3      # Relevant history, referenced docs
    LOW = 2         # Background context
    EPHEMERAL = 1   # Temporary, can be dropped first


@dataclass
class ContextSegment:
    """A segment of context with metadata."""
    id: str = ""
    content: str = ""
    token_count: int = 0
    priority: ContextPriority = ContextPriority.MEDIUM
    source: str = ""  # user, system, tool, memory
    timestamp: datetime = field(default_factory=datetime.now)
    relevance_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def effective_priority(self) -> float:
        """Calculate effective priority with recency decay."""
        age_seconds = (datetime.now() - self.timestamp).total_seconds()
        recency_factor = max(0.1, 1.0 - (age_seconds / 3600))  # Decay over 1 hour
        return self.priority.value * self.relevance_score * recency_factor
    
    def __hash__(self):
        return hash(self.id)


@dataclass
class ContextWindow:
    """A context window with token budget."""
    max_tokens: int = 256000  # 256K like GPT-5.2
    segments: List[ContextSegment] = field(default_factory=list)
    reserved_tokens: int = 4000  # Reserved for response
    
    @property
    def used_tokens(self) -> int:
        return sum(s.token_count for s in self.segments)
    
    @property
    def available_tokens(self) -> int:
        return self.max_tokens - self.used_tokens - self.reserved_tokens
    
    @property
    def utilization(self) -> float:
        return self.used_tokens / self.max_tokens


class TokenCounter:
    """Token counting utility."""
    
    def __init__(self, chars_per_token: float = 4.0):
        self.chars_per_token = chars_per_token
        self._cache: Dict[str, int] = {}
    
    def count(self, text: str) -> int:
        """Count tokens in text (approximate)."""
        # Use cache for repeated text
        text_hash = hashlib.md5(text.encode()).hexdigest()[:16]
        if text_hash in self._cache:
            return self._cache[text_hash]
        
        # Approximate token count
        count = int(len(text) / self.chars_per_token)
        self._cache[text_hash] = count
        
        # Limit cache size
        if len(self._cache) > 10000:
            self._cache.clear()
        
        return count


class ContextManager:
    """Manages 256K context window with intelligent overflow handling."""
    
    def __init__(self, max_tokens: int = 256000):
        self.window = ContextWindow(max_tokens=max_tokens)
        self.token_counter = TokenCounter()
        self.overflow_history: List[ContextSegment] = []
        self._segment_counter = 0
    
    def add(self, content: str, priority: ContextPriority = ContextPriority.MEDIUM,
            source: str = "user", metadata: Dict[str, Any] = None) -> ContextSegment:
        """Add content to context window."""
        self._segment_counter += 1
        
        segment = ContextSegment(
            id=f"seg_{self._segment_counter}",
            content=content,
            token_count=self.token_counter.count(content),
            priority=priority,
            source=source,
            metadata=metadata or {}
        )
        
        # Check if we need to make room
        if segment.token_count > self.window.available_tokens:
            self._handle_overflow(segment.token_count)
        
        self.window.segments.append(segment)
        return segment
    
    def add_system(self, content: str) -> ContextSegment:
        """Add system-level context (highest priority)."""
        return self.add(content, ContextPriority.CRITICAL, "system")
    
    def add_user(self, content: str) -> ContextSegment:
        """Add user message."""
        return self.add(content, ContextPriority.HIGH, "user")
    
    def add_tool(self, content: str, tool_name: str = "") -> ContextSegment:
        """Add tool output."""
        return self.add(content, ContextPriority.MEDIUM, "tool", {"tool": tool_name})
    
    def add_memory(self, content: str, relevance: float = 1.0) -> ContextSegment:
        """Add retrieved memory."""
        segment = self.add(content, ContextPriority.MEDIUM, "memory")
        segment.relevance_score = relevance
        return segment
    
    def _handle_overflow(self, needed_tokens: int):
        """Handle context overflow by removing lowest priority segments."""
        # Sort by effective priority (lowest first)
        self.window.segments.sort(key=lambda s: s.effective_priority)
        
        tokens_freed = 0
        segments_to_remove = []
        
        for segment in self.window.segments:
            if segment.priority == ContextPriority.CRITICAL:
                continue  # Never remove critical segments
            
            if tokens_freed >= needed_tokens:
                break
            
            segments_to_remove.append(segment)
            tokens_freed += segment.token_count
        
        for segment in segments_to_remove:
            self.window.segments.remove(segment)
            self.overflow_history.append(segment)
        
        # Keep overflow history limited
        if len(self.overflow_history) > 100:
            self.overflow_history = self.overflow_history[-100:]
    
    def get_context(self, max_tokens: int = None) -> str:
        """Get the full context as a string."""
        max_tokens = max_tokens or self.window.max_tokens
        
        # Sort by timestamp to maintain order
        sorted_segments = sorted(self.window.segments, key=lambda s: s.timestamp)
        
        result = []
        tokens_used = 0
        
        for segment in sorted_segments:
            if tokens_used + segment.token_count <= max_tokens:
                result.append(segment.content)
                tokens_used += segment.token_count
        
        return "\n\n".join(result)
    
    def get_segments_by_source(self, source: str) -> List[ContextSegment]:
        """Get segments from a specific source."""
        return [s for s in self.window.segments if s.source == source]
    
    def update_relevance(self, segment_id: str, relevance: float):
        """Update the relevance score of a segment."""
        for segment in self.window.segments:
            if segment.id == segment_id:
                segment.relevance_score = max(0, min(1, relevance))
                break
    
    def remove_segment(self, segment_id: str) -> bool:
        """Remove a specific segment."""
        for segment in self.window.segments:
            if segment.id == segment_id:
                self.window.segments.remove(segment)
                return True
        return False
    
    def clear(self, keep_critical: bool = True):
        """Clear the context window."""
        if keep_critical:
            self.window.segments = [s for s in self.window.segments 
                                   if s.priority == ContextPriority.CRITICAL]
        else:
            self.window.segments = []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get context statistics."""
        by_source = {}
        by_priority = {}
        
        for segment in self.window.segments:
            by_source[segment.source] = by_source.get(segment.source, 0) + segment.token_count
            by_priority[segment.priority.name] = by_priority.get(segment.priority.name, 0) + segment.token_count
        
        return {
            'total_tokens': self.window.used_tokens,
            'max_tokens': self.window.max_tokens,
            'available_tokens': self.window.available_tokens,
            'utilization': self.window.utilization,
            'segment_count': len(self.window.segments),
            'by_source': by_source,
            'by_priority': by_priority,
            'overflow_count': len(self.overflow_history)
        }


class ContextBudgetAllocator:
    """Allocates token budget across different context categories."""
    
    DEFAULT_ALLOCATION = {
        'system': 0.10,    # 10% for system prompts
        'user': 0.30,      # 30% for user messages
        'memory': 0.25,    # 25% for retrieved memories
        'tool': 0.25,      # 25% for tool outputs
        'response': 0.10   # 10% reserved for response
    }
    
    def __init__(self, total_tokens: int = 256000,
                 allocation: Dict[str, float] = None):
        self.total_tokens = total_tokens
        self.allocation = allocation or self.DEFAULT_ALLOCATION
    
    def get_budget(self, category: str) -> int:
        """Get token budget for a category."""
        ratio = self.allocation.get(category, 0.1)
        return int(self.total_tokens * ratio)
    
    def adjust_allocation(self, category: str, ratio: float):
        """Adjust allocation for a category."""
        if 0 <= ratio <= 1:
            self.allocation[category] = ratio
            self._normalize()
    
    def _normalize(self):
        """Normalize allocations to sum to 1."""
        total = sum(self.allocation.values())
        if total > 0:
            for key in self.allocation:
                self.allocation[key] /= total


async def demo_context_manager():
    """Demonstrate context manager."""
    print("📚 Context Manager Demo (256K Window)")
    print("=" * 50)
    
    manager = ContextManager(max_tokens=256000)
    
    # Add system prompt
    manager.add_system("You are a helpful AI assistant.")
    
    # Add conversation
    for i in range(10):
        manager.add_user(f"User message {i+1}: " + "Hello, how are you? " * 50)
        manager.add(f"Assistant response {i+1}: " + "I'm doing great! " * 50, 
                   priority=ContextPriority.HIGH, source="assistant")
    
    # Add tool outputs
    manager.add_tool("Tool result: " + "Data " * 100, tool_name="search")
    
    # Add memories
    manager.add_memory("Previous conversation about AI safety.", relevance=0.8)
    
    stats = manager.get_stats()
    print(f"\n📊 Context Statistics:")
    print(f"  Total Tokens: {stats['total_tokens']:,} / {stats['max_tokens']:,}")
    print(f"  Utilization: {stats['utilization']:.1%}")
    print(f"  Segments: {stats['segment_count']}")
    print(f"\n  By Source:")
    for source, tokens in stats['by_source'].items():
        print(f"    {source}: {tokens:,} tokens")
    
    # Budget allocation
    allocator = ContextBudgetAllocator()
    print(f"\n💰 Token Budget Allocation:")
    for category, ratio in allocator.allocation.items():
        budget = allocator.get_budget(category)
        print(f"  {category}: {budget:,} tokens ({ratio:.0%})")
    
    print("\n✅ Context manager demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_context_manager())
