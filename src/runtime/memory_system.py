"""
AION Memory System
Implements different memory types: working, episodic, long-term, semantic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
from datetime import datetime
import json


class MemoryError(Exception):
    """Raised when a memory operation fails."""
    pass


@dataclass
class MemoryEntry:
    """A single entry in memory."""
    content: Any
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)
    importance: float = 0.5  # 0.0 to 1.0
    access_count: int = 0
    last_accessed: Optional[datetime] = None


class BaseMemory(ABC):
    """Abstract base class for memory implementations."""
    
    def __init__(self, name: str, config: dict = None):
        self.name = name
        self.config = config or {}
        self.entries: list[MemoryEntry] = []
    
    @abstractmethod
    def store(self, content: Any, **metadata) -> None:
        """Store content in memory."""
        pass
    
    @abstractmethod
    def recall(self, query: Any = None, limit: int = 10) -> list[MemoryEntry]:
        """Recall content from memory."""
        pass
    
    @abstractmethod
    def forget(self, entry: MemoryEntry) -> None:
        """Remove an entry from memory."""
        pass
    
    def clear(self) -> None:
        """Clear all memory entries."""
        self.entries.clear()
    
    def size(self) -> int:
        """Return number of entries in memory."""
        return len(self.entries)


class WorkingMemory(BaseMemory):
    """
    Working memory - short-term, limited capacity.
    Used for current task context.
    """
    
    def __init__(self, name: str = "working", config: dict = None):
        super().__init__(name, config)
        self.capacity = self.config.get('capacity', 7)  # Miller's law
    
    def store(self, content: Any, **metadata) -> None:
        """Store in working memory, evicting oldest if at capacity."""
        entry = MemoryEntry(
            content=content,
            metadata=metadata,
            importance=metadata.get('importance', 0.5)
        )
        
        # Evict oldest if at capacity
        while len(self.entries) >= self.capacity:
            self._evict_oldest()
        
        self.entries.append(entry)
    
    def recall(self, query: Any = None, limit: int = None) -> list[MemoryEntry]:
        """Recall all current working memory contents."""
        for entry in self.entries:
            entry.access_count += 1
            entry.last_accessed = datetime.now()
        
        if limit:
            return self.entries[-limit:]
        return self.entries.copy()
    
    def forget(self, entry: MemoryEntry) -> None:
        """Remove a specific entry."""
        if entry in self.entries:
            self.entries.remove(entry)
    
    def _evict_oldest(self) -> None:
        """Evict the oldest entry with lowest importance."""
        if self.entries:
            # Sort by importance, then by timestamp
            sorted_entries = sorted(
                self.entries,
                key=lambda e: (e.importance, e.timestamp)
            )
            self.entries.remove(sorted_entries[0])


class EpisodicMemory(BaseMemory):
    """
    Episodic memory - stores experiences and events.
    Organized by time and context.
    """
    
    def __init__(self, name: str = "episodic", config: dict = None):
        super().__init__(name, config)
        self.episodes: list[dict] = []  # Groups of related entries
    
    def store(self, content: Any, **metadata) -> None:
        """Store an episodic memory entry."""
        episode_id = metadata.get('episode_id')
        
        entry = MemoryEntry(
            content=content,
            metadata={
                'episode_id': episode_id,
                'context': metadata.get('context', {}),
                **metadata
            },
            importance=metadata.get('importance', 0.5)
        )
        
        self.entries.append(entry)
    
    def recall(self, query: Any = None, limit: int = 10) -> list[MemoryEntry]:
        """Recall episodes, optionally filtered by query."""
        results = self.entries.copy()
        
        # Filter by query if provided (simple substring match)
        if query:
            query_str = str(query).lower()
            results = [
                e for e in results
                if query_str in str(e.content).lower()
            ]
        
        # Update access metadata
        for entry in results[:limit]:
            entry.access_count += 1
            entry.last_accessed = datetime.now()
        
        return sorted(results, key=lambda e: e.timestamp, reverse=True)[:limit]
    
    def forget(self, entry: MemoryEntry) -> None:
        """Forget an episode entry."""
        if entry in self.entries:
            self.entries.remove(entry)
    
    def get_episode(self, episode_id: str) -> list[MemoryEntry]:
        """Get all entries for a specific episode."""
        return [
            e for e in self.entries
            if e.metadata.get('episode_id') == episode_id
        ]


class LongTermMemory(BaseMemory):
    """
    Long-term memory - persistent storage.
    Supports consolidation from working/episodic memory.
    """
    
    def __init__(self, name: str = "long_term", config: dict = None):
        super().__init__(name, config)
        self.retention = config.get('retention', 'persistent') if config else 'persistent'
    
    def store(self, content: Any, **metadata) -> None:
        """Store in long-term memory with high importance threshold."""
        importance = metadata.get('importance', 0.5)
        
        # Only store items above importance threshold
        threshold = self.config.get('importance_threshold', 0.3)
        if importance < threshold:
            return
        
        entry = MemoryEntry(
            content=content,
            metadata=metadata,
            importance=importance
        )
        
        self.entries.append(entry)
    
    def recall(self, query: Any = None, limit: int = 10) -> list[MemoryEntry]:
        """Recall from long-term memory."""
        results = self.entries.copy()
        
        if query:
            query_str = str(query).lower()
            results = [
                e for e in results
                if query_str in str(e.content).lower()
            ]
        
        # Sort by relevance (access count + importance)
        results.sort(
            key=lambda e: (e.importance * 2 + e.access_count * 0.1),
            reverse=True
        )
        
        for entry in results[:limit]:
            entry.access_count += 1
            entry.last_accessed = datetime.now()
        
        return results[:limit]
    
    def forget(self, entry: MemoryEntry) -> None:
        """Forget from long-term memory (requires explicit action)."""
        if entry in self.entries:
            self.entries.remove(entry)
    
    def consolidate(self, source_memory: BaseMemory, threshold: float = 0.7) -> int:
        """Consolidate important memories from another memory source."""
        consolidated = 0
        
        for entry in source_memory.entries:
            if entry.importance >= threshold:
                self.store(
                    entry.content,
                    importance=entry.importance,
                    source=source_memory.name,
                    **entry.metadata
                )
                consolidated += 1
        
        return consolidated


class SemanticMemory(BaseMemory):
    """
    Semantic memory - factual knowledge and concepts.
    Organized by relationships and categories.
    """
    
    def __init__(self, name: str = "semantic", config: dict = None):
        super().__init__(name, config)
        self.concepts: dict[str, list[MemoryEntry]] = {}  # Category -> entries
    
    def store(self, content: Any, **metadata) -> None:
        """Store a semantic fact or concept."""
        category = metadata.get('category', 'general')
        
        entry = MemoryEntry(
            content=content,
            metadata={
                'category': category,
                'tags': metadata.get('tags', []),
                'relations': metadata.get('relations', {}),
                **metadata
            },
            importance=metadata.get('importance', 0.5)
        )
        
        self.entries.append(entry)
        
        # Index by category
        if category not in self.concepts:
            self.concepts[category] = []
        self.concepts[category].append(entry)
    
    def recall(self, query: Any = None, limit: int = 10) -> list[MemoryEntry]:
        """Recall semantic knowledge, optionally by category or query."""
        if query and isinstance(query, dict):
            # Query by category
            category = query.get('category')
            if category and category in self.concepts:
                return self.concepts[category][:limit]
        
        if query:
            query_str = str(query).lower()
            results = [
                e for e in self.entries
                if query_str in str(e.content).lower()
                or any(query_str in tag.lower() for tag in e.metadata.get('tags', []))
            ]
            return results[:limit]
        
        return self.entries[:limit]
    
    def forget(self, entry: MemoryEntry) -> None:
        """Remove a semantic memory entry."""
        if entry in self.entries:
            self.entries.remove(entry)
            category = entry.metadata.get('category')
            if category and category in self.concepts:
                if entry in self.concepts[category]:
                    self.concepts[category].remove(entry)
    
    def get_category(self, category: str) -> list[MemoryEntry]:
        """Get all entries in a category."""
        return self.concepts.get(category, [])
    
    def get_related(self, entry: MemoryEntry) -> list[MemoryEntry]:
        """Get entries related to a given entry."""
        relations = entry.metadata.get('relations', {})
        related = []
        
        for relation_type, targets in relations.items():
            for target in targets:
                for e in self.entries:
                    if str(e.content) == target or e.metadata.get('id') == target:
                        related.append(e)
        
        return related


# Memory factory
def create_memory(memory_type: str, name: str = None, config: dict = None) -> BaseMemory:
    """Factory function to create memory instances."""
    memory_classes = {
        'working': WorkingMemory,
        'episodic': EpisodicMemory,
        'long_term': LongTermMemory,
        'semantic': SemanticMemory,
    }
    
    if memory_type not in memory_classes:
        raise MemoryError(f"Unknown memory type: {memory_type}")
    
    name = name or memory_type
    return memory_classes[memory_type](name, config)
