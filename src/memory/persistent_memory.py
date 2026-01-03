"""
AION Enhanced Memory System - Persistent Memory
================================================

Cross-session memory persistence:
- User preference tracking
- Project context retention
- Long-term knowledge storage
- Memory retrieval and forgetting

Matches GPT-5.2's enhanced Memory feature.
"""

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from datetime import datetime, timedelta
from enum import Enum
import hashlib


class MemoryType(Enum):
    """Types of memories."""
    FACT = "fact"              # Factual information
    PREFERENCE = "preference"   # User preferences
    PROJECT = "project"        # Project context
    CONVERSATION = "conversation"  # Past conversations
    INSTRUCTION = "instruction"    # User instructions
    ENTITY = "entity"          # Named entities


@dataclass
class Memory:
    """A single memory unit."""
    id: str = ""
    content: str = ""
    memory_type: MemoryType = MemoryType.FACT
    source: str = ""           # Where the memory came from
    confidence: float = 1.0
    importance: float = 0.5
    access_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def age_days(self) -> float:
        return (datetime.now() - self.created_at).total_seconds() / 86400
    
    @property
    def is_expired(self) -> bool:
        if self.expires_at:
            return datetime.now() > self.expires_at
        return False
    
    @property
    def retrieval_score(self) -> float:
        """Score for retrieval ranking."""
        recency = 1.0 / (1.0 + self.age_days)
        access_factor = min(self.access_count / 10.0, 1.0)
        return (self.importance * 0.4 + recency * 0.3 + 
                self.confidence * 0.2 + access_factor * 0.1)
    
    def access(self):
        """Record access."""
        self.access_count += 1
        self.last_accessed = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            'id': self.id,
            'content': self.content,
            'memory_type': self.memory_type.value,
            'source': self.source,
            'confidence': self.confidence,
            'importance': self.importance,
            'access_count': self.access_count,
            'created_at': self.created_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'tags': list(self.tags),
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Memory':
        """Deserialize from dict."""
        data['memory_type'] = MemoryType(data['memory_type'])
        data['tags'] = set(data.get('tags', []))
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['last_accessed'] = datetime.fromisoformat(data['last_accessed'])
        if data.get('expires_at'):
            data['expires_at'] = datetime.fromisoformat(data['expires_at'])
        return cls(**data)


class MemoryStore:
    """Storage backend for memories."""
    
    def __init__(self, storage_path: str = None):
        self.storage_path = storage_path
        self.memories: Dict[str, Memory] = {}
        self._id_counter = 0
    
    def _generate_id(self) -> str:
        self._id_counter += 1
        return f"mem_{self._id_counter}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    def store(self, memory: Memory) -> str:
        """Store a memory."""
        if not memory.id:
            memory.id = self._generate_id()
        self.memories[memory.id] = memory
        return memory.id
    
    def retrieve(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a memory by ID."""
        memory = self.memories.get(memory_id)
        if memory:
            memory.access()
        return memory
    
    def search(self, query: str, 
               memory_type: MemoryType = None,
               tags: Set[str] = None,
               limit: int = 10) -> List[Memory]:
        """Search memories."""
        results = []
        query_terms = set(query.lower().split())
        
        for memory in self.memories.values():
            # Skip expired
            if memory.is_expired:
                continue
            
            # Filter by type
            if memory_type and memory.memory_type != memory_type:
                continue
            
            # Filter by tags
            if tags and not tags.intersection(memory.tags):
                continue
            
            # Simple relevance scoring with both substring and word matching
            content_lower = memory.content.lower()
            query_lower = query.lower() if query else ""
            
            # Check for direct substring match (for preferences like "theme: dark")
            if query_lower and query_lower in content_lower:
                score = 1.0
                results.append((memory, score))
            else:
                # Word-based matching as fallback
                content_terms = set(content_lower.split())
                overlap = len(query_terms.intersection(content_terms))
                
                if overlap > 0 or not query:
                    score = (overlap / len(query_terms) if query_terms else 0.5)
                    results.append((memory, score))
        
        # Sort by combined score
        results.sort(key=lambda x: x[1] * x[0].retrieval_score, reverse=True)
        
        # Mark as accessed
        for memory, _ in results[:limit]:
            memory.access()
        
        return [m for m, _ in results[:limit]]
    
    def delete(self, memory_id: str) -> bool:
        """Delete a memory."""
        if memory_id in self.memories:
            del self.memories[memory_id]
            return True
        return False
    
    def save(self):
        """Save to storage."""
        if not self.storage_path:
            return
        
        data = {
            'memories': [m.to_dict() for m in self.memories.values()],
            'id_counter': self._id_counter
        }
        with open(self.storage_path, 'w') as f:
            json.dump(data, f)
    
    def load(self):
        """Load from storage."""
        if not self.storage_path:
            return
        
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            self.memories = {}
            for m_data in data.get('memories', []):
                memory = Memory.from_dict(m_data)
                self.memories[memory.id] = memory
            self._id_counter = data.get('id_counter', 0)
        except FileNotFoundError:
            pass


class PersistentMemoryManager:
    """Manages persistent cross-session memory."""
    
    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.store = MemoryStore()
        self.active_context: List[Memory] = []
        self.max_context_memories = 20
    
    async def remember(self, content: str,
                       memory_type: MemoryType = MemoryType.FACT,
                       importance: float = 0.5,
                       tags: Set[str] = None,
                       expires_in_days: int = None) -> Memory:
        """Remember new information."""
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now() + timedelta(days=expires_in_days)
        
        memory = Memory(
            content=content,
            memory_type=memory_type,
            source=f"user:{self.user_id}",
            importance=importance,
            tags=tags or set(),
            expires_at=expires_at
        )
        
        self.store.store(memory)
        return memory
    
    async def recall(self, query: str = "",
                     memory_type: MemoryType = None,
                     limit: int = 5) -> List[Memory]:
        """Recall relevant memories."""
        return self.store.search(query, memory_type, limit=limit)
    
    async def forget(self, memory_id: str) -> bool:
        """Explicitly forget a memory."""
        return self.store.delete(memory_id)
    
    async def set_preference(self, key: str, value: Any):
        """Set a user preference."""
        # Check for existing preference
        existing = self.store.search(
            key, 
            memory_type=MemoryType.PREFERENCE,
            limit=1
        )
        
        if existing:
            existing[0].content = f"{key}: {value}"
            existing[0].metadata['value'] = value
        else:
            # Create memory with value in metadata
            memory = Memory(
                content=f"{key}: {value}",
                memory_type=MemoryType.PREFERENCE,
                source=f"user:{self.user_id}",
                importance=0.8,
                tags={key.lower()},
                metadata={'value': value}
            )
            self.store.store(memory)
    
    async def get_preference(self, key: str) -> Optional[Any]:
        """Get a user preference."""
        memories = self.store.search(
            key,
            memory_type=MemoryType.PREFERENCE,
            limit=1
        )
        
        if memories:
            return memories[0].metadata.get('value', memories[0].content)
        return None
    
    def update_context(self, query: str):
        """Update active context with relevant memories."""
        relevant = self.store.search(query, limit=self.max_context_memories)
        self.active_context = relevant
    
    def get_context_summary(self) -> str:
        """Get summary of active context."""
        if not self.active_context:
            return "No relevant memories in context."
        
        lines = []
        for memory in self.active_context[:10]:
            lines.append(f"- [{memory.memory_type.value}] {memory.content[:100]}")
        
        return "Relevant memories:\n" + "\n".join(lines)
    
    def cleanup_expired(self) -> int:
        """Remove expired memories."""
        expired = [m.id for m in self.store.memories.values() if m.is_expired]
        for mem_id in expired:
            self.store.delete(mem_id)
        return len(expired)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        by_type = {}
        for memory in self.store.memories.values():
            t = memory.memory_type.value
            by_type[t] = by_type.get(t, 0) + 1
        
        return {
            'total_memories': len(self.store.memories),
            'by_type': by_type,
            'active_context': len(self.active_context),
            'user_id': self.user_id
        }


async def demo_persistent_memory():
    """Demonstrate persistent memory."""
    print("🧠 Persistent Memory Demo")
    print("=" * 50)
    
    manager = PersistentMemoryManager(user_id="demo_user")
    
    # Remember facts
    await manager.remember(
        "User prefers dark mode for all interfaces",
        memory_type=MemoryType.PREFERENCE,
        importance=0.9,
        tags={"ui", "preference"}
    )
    
    await manager.remember(
        "Working on AION project - an AI agent framework",
        memory_type=MemoryType.PROJECT,
        importance=0.8,
        tags={"project", "aion"}
    )
    
    await manager.remember(
        "User is proficient in Python and TypeScript",
        memory_type=MemoryType.FACT,
        importance=0.7,
        tags={"skill", "programming"}
    )
    
    await manager.remember(
        "Temporary note: meeting at 3pm",
        memory_type=MemoryType.FACT,
        importance=0.3,
        expires_in_days=1
    )
    
    print(f"\n📊 Stats: {manager.get_stats()}")
    
    # Recall memories
    print("\n🔍 Recall: 'project'")
    memories = await manager.recall("project", limit=3)
    for mem in memories:
        print(f"   • {mem.content[:60]}... (importance: {mem.importance})")
    
    # Set preference
    await manager.set_preference("response_length", "detailed")
    pref = await manager.get_preference("response_length")
    print(f"\n⚙️ Preference 'response_length': {pref}")
    
    # Update context
    manager.update_context("working on the AION project")
    print(f"\n📝 Context Summary:\n{manager.get_context_summary()}")
    
    print("\n✅ Persistent memory demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_persistent_memory())
