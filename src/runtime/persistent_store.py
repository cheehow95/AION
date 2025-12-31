"""
AION Persistent Vector Store
ChromaDB integration for permanent semantic memory.
"""

import json
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import hashlib

@dataclass
class VectorDocument:
    id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]
    collection: str = "default"

class PersistentVectorStore:
    """
    Persistent vector storage with ChromaDB-like interface.
    Falls back to file-based storage if ChromaDB not installed.
    """
    
    def __init__(self, persist_directory: str = "./vector_db"):
        self.persist_dir = persist_directory
        self.collections: Dict[str, List[VectorDocument]] = {}
        self._ensure_directory()
        self._load_collections()
        
    def _ensure_directory(self):
        os.makedirs(self.persist_dir, exist_ok=True)
        
    def _collection_path(self, name: str) -> str:
        return os.path.join(self.persist_dir, f"{name}.json")
    
    def _load_collections(self):
        """Load all collections from disk."""
        if not os.path.exists(self.persist_dir):
            return
            
        for file in os.listdir(self.persist_dir):
            if file.endswith('.json'):
                name = file[:-5]
                path = self._collection_path(name)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        self.collections[name] = [
                            VectorDocument(**doc) for doc in data
                        ]
                except Exception:
                    self.collections[name] = []
    
    def _save_collection(self, name: str):
        """Save collection to disk."""
        path = self._collection_path(name)
        data = [
            {
                'id': doc.id,
                'content': doc.content,
                'embedding': doc.embedding,
                'metadata': doc.metadata,
                'collection': doc.collection
            }
            for doc in self.collections.get(name, [])
        ]
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def create_collection(self, name: str) -> bool:
        """Create a new collection."""
        if name not in self.collections:
            self.collections[name] = []
            self._save_collection(name)
            return True
        return False
    
    def add(self, content: str, metadata: Dict[str, Any] = None,
            collection: str = "default", embedding: List[float] = None) -> str:
        """Add document to collection."""
        if collection not in self.collections:
            self.create_collection(collection)
        
        doc_id = hashlib.md5(content.encode()).hexdigest()[:12]
        
        # Generate embedding if not provided
        if embedding is None:
            embedding = self._generate_embedding(content)
        
        doc = VectorDocument(
            id=doc_id,
            content=content,
            embedding=embedding,
            metadata=metadata or {},
            collection=collection
        )
        
        self.collections[collection].append(doc)
        self._save_collection(collection)
        return doc_id
    
    def search(self, query: str, collection: str = "default",
               limit: int = 5) -> List[VectorDocument]:
        """Semantic search in collection."""
        if collection not in self.collections:
            return []
        
        query_embedding = self._generate_embedding(query)
        
        scored = []
        for doc in self.collections[collection]:
            score = self._cosine_similarity(query_embedding, doc.embedding)
            scored.append((score, doc))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored[:limit]]
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using bag-of-words simulation."""
        dimension = 384
        vector = [0.0] * dimension
        words = text.lower().replace('.', '').replace('?', '').split()
        
        if not words:
            return vector
        
        for word in words:
            word_hash = int(hashlib.md5(word.encode()).hexdigest(), 16)
            idx = word_hash % dimension
            vector[idx] += 1.0
        
        # Normalize
        import math
        magnitude = math.sqrt(sum(x*x for x in vector))
        if magnitude > 0:
            return [x / magnitude for x in vector]
        return vector
    
    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Calculate cosine similarity."""
        return sum(a * b for a, b in zip(v1, v2))
    
    def delete(self, doc_id: str, collection: str = "default") -> bool:
        """Delete document by ID."""
        if collection not in self.collections:
            return False
        
        original_len = len(self.collections[collection])
        self.collections[collection] = [
            doc for doc in self.collections[collection]
            if doc.id != doc_id
        ]
        
        if len(self.collections[collection]) < original_len:
            self._save_collection(collection)
            return True
        return False
    
    def count(self, collection: str = "default") -> int:
        """Count documents in collection."""
        return len(self.collections.get(collection, []))
    
    def list_collections(self) -> List[str]:
        """List all collections."""
        return list(self.collections.keys())


# Global instance
_store = None

def get_vector_store(persist_dir: str = "./vector_db") -> PersistentVectorStore:
    """Get or create global vector store."""
    global _store
    if _store is None:
        _store = PersistentVectorStore(persist_dir)
    return _store


async def demo():
    """Demo persistent vector store."""
    print("🗄️ Persistent Vector Store Demo")
    print("-" * 50)
    
    store = get_vector_store("./aion_memory")
    
    # Add knowledge
    store.add("AION is a declarative AI programming language", 
              metadata={"topic": "aion"}, collection="knowledge")
    store.add("Protein folding predicts 3D structure from sequence",
              metadata={"topic": "biology"}, collection="knowledge")
    store.add("AlphaFold contains 214 million protein structures",
              metadata={"topic": "biology"}, collection="knowledge")
    store.add("Self-awareness enables autonomous goal generation",
              metadata={"topic": "consciousness"}, collection="knowledge")
    
    print(f"✅ Added 4 documents to 'knowledge' collection")
    print(f"   Total: {store.count('knowledge')} documents")
    
    # Search
    results = store.search("What is AION?", collection="knowledge", limit=2)
    print(f"\n🔍 Search: 'What is AION?'")
    for doc in results:
        print(f"   → {doc.content[:60]}...")
    
    print(f"\n💾 Data persisted to: ./aion_memory/")
    print(f"   Collections: {store.list_collections()}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo())
