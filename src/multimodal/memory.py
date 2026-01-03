"""
AION Multimodal Memory
======================

Unified memory system that stores and retrieves information
across all modalities (visual, auditory, textual) with
cross-modal associations.
"""

import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Tuple
from enum import Enum
from datetime import datetime
import uuid
import math


# =============================================================================
# MEMORY TYPES
# =============================================================================

class MemoryType(Enum):
    """Types of memory content."""
    VISUAL = "visual"
    AUDITORY = "auditory"
    TEXTUAL = "textual"
    MULTIMODAL = "multimodal"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class RetrievalStrategy(Enum):
    """Strategies for memory retrieval."""
    SIMILARITY = "similarity"
    RECENCY = "recency"
    IMPORTANCE = "importance"
    CONTEXT = "context"
    HYBRID = "hybrid"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class MemoryEmbedding:
    """Vector embedding for a memory."""
    vector: List[float] = field(default_factory=list)
    model: str = "default"
    dimension: int = 0
    
    def __post_init__(self):
        self.dimension = len(self.vector)
    
    def cosine_similarity(self, other: "MemoryEmbedding") -> float:
        """Calculate cosine similarity with another embedding."""
        if not self.vector or not other.vector:
            return 0.0
        if len(self.vector) != len(other.vector):
            return 0.0
        
        dot = sum(a * b for a, b in zip(self.vector, other.vector))
        norm_a = math.sqrt(sum(a * a for a in self.vector))
        norm_b = math.sqrt(sum(b * b for b in other.vector))
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot / (norm_a * norm_b)


@dataclass
class MemoryAssociation:
    """Association between two memories."""
    source_id: str
    target_id: str
    association_type: str  # "related", "caused_by", "part_of", etc.
    strength: float = 0.5  # 0.0 to 1.0
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultimodalMemoryEntry:
    """A single memory entry with multimodal content."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MemoryType = MemoryType.TEXTUAL
    
    # Content (one or more modalities)
    text_content: str = ""
    visual_content: bytes = b""
    audio_content: bytes = b""
    
    # Embeddings for each modality
    text_embedding: Optional[MemoryEmbedding] = None
    visual_embedding: Optional[MemoryEmbedding] = None
    audio_embedding: Optional[MemoryEmbedding] = None
    
    # Unified embedding (combined/cross-modal)
    unified_embedding: Optional[MemoryEmbedding] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    importance: float = 0.5  # 0.0 to 1.0
    
    # Context
    context: str = ""
    tags: List[str] = field(default_factory=list)
    source: str = ""
    
    # Associations
    associations: List[str] = field(default_factory=list)  # IDs of related memories
    
    # Additional data
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def modalities(self) -> List[str]:
        """Get list of modalities present in this memory."""
        mods = []
        if self.text_content:
            mods.append("text")
        if self.visual_content:
            mods.append("visual")
        if self.audio_content:
            mods.append("audio")
        return mods
    
    @property
    def age_seconds(self) -> float:
        """Get age of memory in seconds."""
        return (datetime.now() - self.created_at).total_seconds()
    
    def update_access(self):
        """Update access time and count."""
        self.accessed_at = datetime.now()
        self.access_count += 1


@dataclass
class RetrievalResult:
    """Result of a memory retrieval operation."""
    entry: MultimodalMemoryEntry
    similarity: float
    relevance_score: float
    retrieval_reason: str = ""


@dataclass
class MemoryStats:
    """Statistics about the memory store."""
    total_entries: int = 0
    by_type: Dict[str, int] = field(default_factory=dict)
    total_associations: int = 0
    avg_importance: float = 0.0
    oldest_entry: Optional[datetime] = None
    newest_entry: Optional[datetime] = None


# =============================================================================
# MULTIMODAL MEMORY
# =============================================================================

class MultimodalMemory:
    """
    Unified multimodal memory system.
    
    Stores memories across modalities and enables cross-modal
    retrieval and association.
    """
    
    def __init__(
        self,
        embedding_dimension: int = 384,
        max_entries: int = 10000
    ):
        self.embedding_dimension = embedding_dimension
        self.max_entries = max_entries
        
        # Storage
        self._entries: Dict[str, MultimodalMemoryEntry] = {}
        self._associations: List[MemoryAssociation] = []
        
        # Indices
        self._by_type: Dict[MemoryType, List[str]] = {}
        self._by_tag: Dict[str, List[str]] = {}
    
    async def store(
        self,
        content: Union[str, bytes, Dict[str, Any]],
        memory_type: MemoryType = MemoryType.TEXTUAL,
        context: str = "",
        tags: List[str] = None,
        importance: float = 0.5,
        metadata: Dict[str, Any] = None
    ) -> MultimodalMemoryEntry:
        """
        Store a new memory entry.
        
        Args:
            content: Content to store (text, bytes, or dict with multiple modalities)
            memory_type: Type of memory
            context: Context in which memory was created
            tags: Tags for categorization
            importance: Importance score (0-1)
            metadata: Additional metadata
            
        Returns:
            Created memory entry
        """
        entry = MultimodalMemoryEntry(
            type=memory_type,
            context=context,
            tags=tags or [],
            importance=importance,
            metadata=metadata or {}
        )
        
        # Set content based on type
        if isinstance(content, str):
            entry.text_content = content
            entry.text_embedding = await self._create_embedding(content)
        elif isinstance(content, bytes):
            if memory_type == MemoryType.VISUAL:
                entry.visual_content = content
                entry.visual_embedding = await self._create_visual_embedding(content)
            elif memory_type == MemoryType.AUDITORY:
                entry.audio_content = content
                entry.audio_embedding = await self._create_audio_embedding(content)
        elif isinstance(content, dict):
            # Multimodal content
            if "text" in content:
                entry.text_content = content["text"]
                entry.text_embedding = await self._create_embedding(content["text"])
            if "visual" in content:
                entry.visual_content = content["visual"]
                entry.visual_embedding = await self._create_visual_embedding(content["visual"])
            if "audio" in content:
                entry.audio_content = content["audio"]
                entry.audio_embedding = await self._create_audio_embedding(content["audio"])
            entry.type = MemoryType.MULTIMODAL
        
        # Create unified embedding
        entry.unified_embedding = await self._create_unified_embedding(entry)
        
        # Store
        self._entries[entry.id] = entry
        
        # Update indices
        if entry.type not in self._by_type:
            self._by_type[entry.type] = []
        self._by_type[entry.type].append(entry.id)
        
        for tag in entry.tags:
            if tag not in self._by_tag:
                self._by_tag[tag] = []
            self._by_tag[tag].append(entry.id)
        
        # Check capacity
        await self._enforce_capacity()
        
        return entry
    
    async def retrieve(
        self,
        query: Union[str, bytes],
        limit: int = 5,
        strategy: RetrievalStrategy = RetrievalStrategy.SIMILARITY,
        memory_type: MemoryType = None,
        tags: List[str] = None,
        min_importance: float = 0.0
    ) -> List[RetrievalResult]:
        """
        Retrieve memories matching a query.
        
        Args:
            query: Query text or embedding
            limit: Maximum number of results
            strategy: Retrieval strategy to use
            memory_type: Filter by memory type
            tags: Filter by tags
            min_importance: Minimum importance threshold
            
        Returns:
            List of matching memories with scores
        """
        # Get query embedding
        if isinstance(query, str):
            query_embedding = await self._create_embedding(query)
        else:
            query_embedding = await self._create_visual_embedding(query)
        
        # Filter candidates
        candidates = list(self._entries.values())
        
        if memory_type:
            candidates = [e for e in candidates if e.type == memory_type]
        
        if tags:
            candidates = [e for e in candidates if any(t in e.tags for t in tags)]
        
        if min_importance > 0:
            candidates = [e for e in candidates if e.importance >= min_importance]
        
        # Score candidates
        results = []
        for entry in candidates:
            similarity = self._calculate_similarity(query_embedding, entry)
            
            if strategy == RetrievalStrategy.SIMILARITY:
                score = similarity
            elif strategy == RetrievalStrategy.RECENCY:
                recency = 1.0 / (1.0 + entry.age_seconds / 3600)  # Decay over hours
                score = 0.7 * similarity + 0.3 * recency
            elif strategy == RetrievalStrategy.IMPORTANCE:
                score = 0.5 * similarity + 0.5 * entry.importance
            elif strategy == RetrievalStrategy.HYBRID:
                recency = 1.0 / (1.0 + entry.age_seconds / 3600)
                score = 0.5 * similarity + 0.25 * entry.importance + 0.25 * recency
            else:
                score = similarity
            
            results.append(RetrievalResult(
                entry=entry,
                similarity=similarity,
                relevance_score=score,
                retrieval_reason=strategy.value
            ))
            
            # Update access
            entry.update_access()
        
        # Sort and limit
        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results[:limit]
    
    async def search(
        self,
        text: str,
        limit: int = 10
    ) -> List[RetrievalResult]:
        """Simple text search across memories."""
        return await self.retrieve(text, limit=limit)
    
    def get(self, memory_id: str) -> Optional[MultimodalMemoryEntry]:
        """Get a specific memory by ID."""
        entry = self._entries.get(memory_id)
        if entry:
            entry.update_access()
        return entry
    
    def delete(self, memory_id: str) -> bool:
        """Delete a memory."""
        if memory_id not in self._entries:
            return False
        
        entry = self._entries[memory_id]
        
        # Remove from indices
        if entry.type in self._by_type:
            self._by_type[entry.type] = [
                id for id in self._by_type[entry.type] if id != memory_id
            ]
        
        for tag in entry.tags:
            if tag in self._by_tag:
                self._by_tag[tag] = [
                    id for id in self._by_tag[tag] if id != memory_id
                ]
        
        # Remove associations
        self._associations = [
            a for a in self._associations 
            if a.source_id != memory_id and a.target_id != memory_id
        ]
        
        del self._entries[memory_id]
        return True
    
    async def create_association(
        self,
        source_id: str,
        target_id: str,
        association_type: str = "related",
        strength: float = 0.5
    ) -> Optional[MemoryAssociation]:
        """
        Create an association between two memories.
        
        Args:
            source_id: ID of source memory
            target_id: ID of target memory
            association_type: Type of association
            strength: Association strength (0-1)
            
        Returns:
            Created association or None if memories don't exist
        """
        if source_id not in self._entries or target_id not in self._entries:
            return None
        
        association = MemoryAssociation(
            source_id=source_id,
            target_id=target_id,
            association_type=association_type,
            strength=strength
        )
        
        self._associations.append(association)
        
        # Update memory entries
        self._entries[source_id].associations.append(target_id)
        self._entries[target_id].associations.append(source_id)
        
        return association
    
    def get_associations(
        self, 
        memory_id: str
    ) -> List[Tuple[MultimodalMemoryEntry, MemoryAssociation]]:
        """Get all memories associated with a given memory."""
        results = []
        
        for assoc in self._associations:
            if assoc.source_id == memory_id:
                target = self._entries.get(assoc.target_id)
                if target:
                    results.append((target, assoc))
            elif assoc.target_id == memory_id:
                source = self._entries.get(assoc.source_id)
                if source:
                    results.append((source, assoc))
        
        return results
    
    async def consolidate(self) -> Dict[str, Any]:
        """
        Consolidate memories - merge similar, decay old, strengthen important.
        
        Returns:
            Consolidation statistics
        """
        stats = {
            "merged": 0,
            "decayed": 0,
            "strengthened": 0
        }
        
        # Decay old, unaccessed memories
        for entry in self._entries.values():
            if entry.age_seconds > 86400 and entry.access_count < 2:
                entry.importance *= 0.95  # Decay importance
                stats["decayed"] += 1
            elif entry.access_count > 5:
                entry.importance = min(1.0, entry.importance * 1.1)
                stats["strengthened"] += 1
        
        # Could implement memory merging here for similar entries
        
        return stats
    
    def get_stats(self) -> MemoryStats:
        """Get memory statistics."""
        entries = list(self._entries.values())
        
        by_type = {}
        for t in MemoryType:
            by_type[t.value] = len(self._by_type.get(t, []))
        
        avg_importance = 0.0
        if entries:
            avg_importance = sum(e.importance for e in entries) / len(entries)
        
        oldest = min((e.created_at for e in entries), default=None)
        newest = max((e.created_at for e in entries), default=None)
        
        return MemoryStats(
            total_entries=len(entries),
            by_type=by_type,
            total_associations=len(self._associations),
            avg_importance=avg_importance,
            oldest_entry=oldest,
            newest_entry=newest
        )
    
    # -------------------------------------------------------------------------
    # Internal methods
    # -------------------------------------------------------------------------
    
    async def _create_embedding(self, text: str) -> MemoryEmbedding:
        """Create text embedding."""
        # Simplified embedding - in production, use sentence-transformers etc.
        import hashlib
        
        hash_bytes = hashlib.sha384(text.encode()).digest()
        vector = [b / 255.0 for b in hash_bytes]
        
        # Pad or truncate to dimension
        if len(vector) < self.embedding_dimension:
            vector.extend([0.0] * (self.embedding_dimension - len(vector)))
        else:
            vector = vector[:self.embedding_dimension]
        
        return MemoryEmbedding(vector=vector, model="hash")
    
    async def _create_visual_embedding(self, data: bytes) -> MemoryEmbedding:
        """Create visual embedding."""
        import hashlib
        
        hash_bytes = hashlib.sha384(data).digest()
        vector = [b / 255.0 for b in hash_bytes]
        
        if len(vector) < self.embedding_dimension:
            vector.extend([0.0] * (self.embedding_dimension - len(vector)))
        else:
            vector = vector[:self.embedding_dimension]
        
        return MemoryEmbedding(vector=vector, model="visual_hash")
    
    async def _create_audio_embedding(self, data: bytes) -> MemoryEmbedding:
        """Create audio embedding."""
        import hashlib
        
        hash_bytes = hashlib.sha384(data).digest()
        vector = [b / 255.0 for b in hash_bytes]
        
        if len(vector) < self.embedding_dimension:
            vector.extend([0.0] * (self.embedding_dimension - len(vector)))
        else:
            vector = vector[:self.embedding_dimension]
        
        return MemoryEmbedding(vector=vector, model="audio_hash")
    
    async def _create_unified_embedding(
        self, 
        entry: MultimodalMemoryEntry
    ) -> MemoryEmbedding:
        """Create unified cross-modal embedding."""
        vectors = []
        
        if entry.text_embedding:
            vectors.append(entry.text_embedding.vector)
        if entry.visual_embedding:
            vectors.append(entry.visual_embedding.vector)
        if entry.audio_embedding:
            vectors.append(entry.audio_embedding.vector)
        
        if not vectors:
            return MemoryEmbedding(vector=[0.0] * self.embedding_dimension)
        
        # Average embeddings
        unified = [0.0] * self.embedding_dimension
        for vec in vectors:
            for i, v in enumerate(vec[:self.embedding_dimension]):
                unified[i] += v
        
        unified = [v / len(vectors) for v in unified]
        
        return MemoryEmbedding(vector=unified, model="unified")
    
    def _calculate_similarity(
        self,
        query_embedding: MemoryEmbedding,
        entry: MultimodalMemoryEntry
    ) -> float:
        """Calculate similarity between query and memory entry."""
        # Use unified embedding if available
        if entry.unified_embedding:
            return query_embedding.cosine_similarity(entry.unified_embedding)
        
        # Fallback to text embedding
        if entry.text_embedding:
            return query_embedding.cosine_similarity(entry.text_embedding)
        
        return 0.0
    
    async def _enforce_capacity(self):
        """Remove old/unimportant memories if over capacity."""
        if len(self._entries) <= self.max_entries:
            return
        
        # Sort by importance * recency
        entries = list(self._entries.values())
        entries.sort(
            key=lambda e: e.importance * (1.0 / (1.0 + e.age_seconds / 3600))
        )
        
        # Remove lowest scoring entries
        to_remove = len(entries) - self.max_entries
        for entry in entries[:to_remove]:
            self.delete(entry.id)


# =============================================================================
# DEMO
# =============================================================================

async def demo_memory():
    """Demonstrate multimodal memory."""
    print("🧠 Multimodal Memory Demo")
    print("-" * 40)
    
    memory = MultimodalMemory()
    
    # Store text memories
    m1 = await memory.store(
        "AION is an advanced AI programming language.",
        memory_type=MemoryType.TEXTUAL,
        tags=["aion", "programming"],
        importance=0.8
    )
    print(f"Stored: {m1.text_content[:50]}...")
    
    m2 = await memory.store(
        "Python is a popular programming language.",
        memory_type=MemoryType.TEXTUAL,
        tags=["python", "programming"]
    )
    
    m3 = await memory.store(
        "Machine learning models can process images and text.",
        memory_type=MemoryType.TEXTUAL,
        tags=["ml", "multimodal"]
    )
    
    # Create association
    await memory.create_association(m1.id, m2.id, "related", 0.7)
    print(f"Created association between memories")
    
    # Retrieve
    results = await memory.retrieve("programming language", limit=2)
    print(f"\nSearch 'programming language':")
    for result in results:
        print(f"  - {result.entry.text_content[:40]}... (score: {result.relevance_score:.2f})")
    
    # Get stats
    stats = memory.get_stats()
    print(f"\nMemory stats:")
    print(f"  Total entries: {stats.total_entries}")
    print(f"  By type: {stats.by_type}")
    print(f"  Associations: {stats.total_associations}")
    print(f"  Avg importance: {stats.avg_importance:.2f}")
    
    # Consolidate
    consolidation = await memory.consolidate()
    print(f"\nConsolidation: {consolidation}")
    
    print("-" * 40)
    print("✅ Memory demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_memory())
