"""
AION Vector Memory System (RAG)
Provides semantic long-term memory using vector embeddings.
"""

from typing import Any, Optional, List, Dict
from dataclasses import dataclass, field
from datetime import datetime
import json
import math
import hashlib

@dataclass
class VectorEntry:
    content: str
    vector: List[float]
    metadata: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    id: str = field(default_factory=lambda: "")

    def __post_init__(self):
        if not self.id:
            # Generate ID based on content
            self.id = hashlib.md5(self.content.encode()).hexdigest()

class VectorMemory:
    """
    RAG-enabled memory system using vector embeddings.
    """
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.entries: List[VectorEntry] = []
        # In a real implementation, this would connect to:
        # - ChromaDB
        # - Pinecone
        # - Weaviate
        # - pgvector
        
    def add(self, content: str, metadata: Dict[str, Any] = None):
        """Add content to vector memory."""
        vector = self._embed(content)
        entry = VectorEntry(
            content=content,
            vector=vector,
            metadata=metadata or {}
        )
        self.entries.append(entry)
        return entry.id
        
    def search(self, query: str, limit: int = 5, threshold: float = 0.0) -> List[VectorEntry]:
        """Semantic search for relevant entries."""
        query_vector = self._embed(query)
        
        # Calculate cosine similarity
        scored_entries = []
        for entry in self.entries:
            score = self._cosine_similarity(query_vector, entry.vector)
            if score >= threshold:
                scored_entries.append((score, entry))
        
        # Sort by score descending
        scored_entries.sort(key=lambda x: x[0], reverse=True)
        
        return [entry for score, entry in scored_entries[:limit]]
    
    def recall(self, query: str) -> str:
        """Recall relevant context as a formatted string."""
        results = self.search(query)
        if not results:
            return ""
            
        context = []
        for entry in results:
            timestamp = entry.timestamp.split('T')[0]
            context.append(f"[{entry.id[:8]}] ({timestamp}) {entry.content}")
            
        return "\n".join(context)

    def _embed(self, text: str) -> List[float]:
        """
        Generate embedding vector for text.
        
        SIMULATION MODE: Uses a "bag of hashed words" approach.
        Each word in the text contributes to specific dimensions in the vector.
        This allows approximate keyword matching to work like semantic search.
        """
        vector = [0.0] * self.dimension
        words = text.lower().replace('.', '').replace('?', '').split()
        
        if not words:
            return vector
            
        for word in words:
            # Hash word to a dimension index [0, dimension-1]
            word_hash = int(hashlib.md5(word.encode()).hexdigest(), 16)
            idx = word_hash % self.dimension
            vector[idx] += 1.0
            
        # Normalize vector
        magnitude = math.sqrt(sum(x*x for x in vector))
        if magnitude > 0:
            return [x / magnitude for x in vector]
        return vector

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(v1, v2))
        # Vectors are already normalized in _embed, so magnitude is 1.0
        return dot_product

    def save(self, filepath: str):
        """Save memory to disk."""
        data = [
            {
                'content': e.content,
                'vector': e.vector,
                'metadata': e.metadata,
                'timestamp': e.timestamp,
                'id': e.id
            }
            for e in self.entries
        ]
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str):
        """Load memory from disk."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.entries = [
                VectorEntry(
                    content=d['content'],
                    vector=d['vector'],
                    metadata=d['metadata'],
                    timestamp=d['timestamp'],
                    id=d['id']
                )
                for d in data
            ]
        except FileNotFoundError:
            pass
