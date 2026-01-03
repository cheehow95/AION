"""
AION Hyper-Context System
==========================

Gemini 3-style massive context support (1M+ tokens):
- Disk-backed context paging
- Sparse attention simulation (Expert Attention)
- Hierarchical memory retrieval
- Context virtualization

Enables handling entire codebases or long session histories.
"""

import asyncio
import os
import pickle
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime
from enum import Enum

# Import basic context structures from context_manager if needed
# For this implementation, we'll define specialized structures

@dataclass
class HyperPage:
    """A page of context tokens stored on disk."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content_hash: str = ""
    token_count: int = 0
    start_index: int = 0
    end_index: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    file_path: str = ""
    is_dirty: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # In-memory data (cleared when paged out)
    _content: Optional[str] = None
    
    @property
    def is_loaded(self) -> bool:
        return self._content is not None
    
    def load(self):
        """Load content from disk."""
        if not self.is_loaded and os.path.exists(self.file_path):
            with open(self.file_path, 'r', encoding='utf-8') as f:
                self._content = f.read()
    
    def unload(self):
        """Unload content to free memory (save if dirty)."""
        if self.is_loaded:
            if self.is_dirty:
                self.save()
            self._content = None
    
    def save(self):
        """Save content to disk."""
        if self.is_loaded:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                f.write(self._content)
            self.is_dirty = False


class ContextPager:
    """
    Manages paging of context to/from disk.
    Implements a robust LRU-like policy for memory management.
    """
    
    def __init__(self, storage_dir: str, max_memory_tokens: int = 100000):
        self.storage_dir = storage_dir
        self.max_memory_tokens = max_memory_tokens
        self.pages: Dict[str, HyperPage] = {}
        self.loaded_pages: Set[str] = set()
        
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)
            
    def create_page(self, content: str, metadata: Dict = None) -> HyperPage:
        """Create a new page with content."""
        page_id = str(uuid.uuid4())
        file_path = os.path.join(self.storage_dir, f"{page_id}.txt")
        token_count = len(content.split()) # Approximation
        
        page = HyperPage(
            id=page_id,
            token_count=token_count,
            file_path=file_path,
            metadata=metadata or {}
        )
        page._content = content
        page.is_dirty = True
        page.save()
        
        self.pages[page_id] = page
        self.loaded_pages.add(page_id)
        
        self._enforce_memory_limit()
        return page
    
    def get_page(self, page_id: str) -> Optional[str]:
        """Get content of a page, loading if necessary."""
        if page_id not in self.pages:
            return None
        
        page = self.pages[page_id]
        page.last_accessed = datetime.now()
        
        if not page.is_loaded:
            page.load()
            self.loaded_pages.add(page_id)
            self._enforce_memory_limit()
            
        return page._content

    def _enforce_memory_limit(self):
        """Unload pages if memory limit exceeded."""
        current_tokens = sum(self.pages[pid].token_count for pid in self.loaded_pages)
        
        if current_tokens > self.max_memory_tokens:
            # Sort loaded pages by last_accessed (LRU)
            sorted_pages = sorted(
                [self.pages[pid] for pid in self.loaded_pages],
                key=lambda p: p.last_accessed
            )
            
            for page in sorted_pages:
                if current_tokens <= self.max_memory_tokens:
                    break
                
                page.unload()
                self.loaded_pages.remove(page.id)
                current_tokens -= page.token_count


class ExpertAttention:
    """
    Simulates sparse 'Expert Attention' to retrieve relevant pages.
    Instead of full self-attention, we select top-k relevant pages.
    """
    
    def __init__(self, pager: ContextPager):
        self.pager = pager
    
    async def retrieve_relevant(self, query: str, top_k: int = 5) -> List[HyperPage]:
        """
        Retrieve top-k relevant pages for a query.
        In a real LLM, this would use attention heads.
        Here we use semantic similarity simulation.
        """
        # Simulation: Score pages based on simple overlap/keywords
        # Real impl would use vector embeddings
        query_terms = set(query.lower().split())
        scored_pages = []
        
        for page in self.pager.pages.values():
            # Metadata based scoring (faster than loading content)
            score = 0
            if page.metadata:
                tags = page.metadata.get('tags', [])
                score += sum(1 for tag in tags if tag in query_terms) * 10
            
            # If loaded, refine score with content
            if page.is_loaded:
                content_terms = set(page._content.lower().split())
                score += len(query_terms.intersection(content_terms))
            
            scored_pages.append((score, page))
            
        # Sort by score desc, creation time desc
        scored_pages.sort(key=lambda x: (x[0], x[1].created_at), reverse=True)
        
        return [p for s, p in scored_pages[:top_k]]


class HyperContextManager:
    """
    Manages a massive context window (1M+ tokens).
    Virtualizes the context window using the pager and attention mechanism.
    """
    
    def __init__(self, storage_dir: str = "data/hyper_context"):
        self.pager = ContextPager(storage_dir)
        self.attention = ExpertAttention(self.pager)
        self.total_tokens = 0
        self.page_index: List[str] = [] # Ordered list of page IDs representing full stream
        
    async def add_context(self, content: str, metadata: Dict = None):
        """Add content to the hyper context."""
        # Chunk content if too large for a single page
        # For simplicity, assume content fits in one page or we just create large page
        # In production, would chunk to ~4k-8k tokens per page
        
        page = self.pager.create_page(content, metadata)
        self.page_index.append(page.id)
        self.total_tokens += page.token_count
        
    async def get_full_context_stream(self) -> Any:
        """
        Get an iterator over the full context.
        WARNING: This can be huge.
        """
        for page_id in self.page_index:
            yield self.pager.get_page(page_id)

    async def get_relevant_context(self, query: str, budget: int = 128000) -> str:
        """
        Construct a context window relevant to the query within budget.
        Uses Expert Attention to select pages.
        """
        # Always include most recent context (Recency Bias)
        recent_tokens = 0
        selected_pages = []
        recent_page_ids = set()
        
        # Take last N pages until ~10% of budget
        recency_budget = budget * 0.1
        for page_id in reversed(self.page_index):
            page = self.pager.pages[page_id]
            if recent_tokens + page.token_count > recency_budget:
                break
            selected_pages.append(page)
            recent_page_ids.add(page.id)
            recent_tokens += page.token_count
            
        # Use remaining budget for retrieval
        remaining_budget = budget - recent_tokens
        
        # Retrieve relevant pages not already selected
        retrieved = await self.attention.retrieve_relevant(query, top_k=50)
        
        for page in retrieved:
            if remaining_budget <= 0:
                break
            if page.id not in recent_page_ids:
                selected_pages.append(page)
                remaining_budget -= page.token_count
                
        # Sort selected pages by their original order in the stream
        # This preserves causal consistency
        selected_ids = {p.id for p in selected_pages}
        ordered_context_parts = []
        
        for page_id in self.page_index:
            if page_id in selected_ids:
                content = self.pager.get_page(page_id)
                if content:
                    ordered_context_parts.append(content)
                    
        return "\n\n".join(ordered_context_parts)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get hyper context stats."""
        return {
            'total_pages': len(self.pager.pages),
            'loaded_pages': len(self.pager.loaded_pages),
            'total_tokens': self.total_tokens,
            'virtual_tokens': self.total_tokens, # Same for now, but virtualization implies capacity > loaded
            'storage_path': self.pager.storage_dir
        }

async def demo_hyper_context():
    """Demonstrate the Hyper-Context system."""
    print("🚀 Hyper-Context System Demo (1M+ Token Simulation)")
    print("=" * 50)
    
    # Use a temp dir
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        manager = HyperContextManager(storage_dir=temp_dir)
        
        # 1. Simulate adding massive context (Codebase files)
        print("\n📚 Ingesting context segments...")
        files = [
            ("main.py", "def main():\n    print('Hello World')\n    run_app()", {"tags": ["python", "entry"]}),
            ("utils.py", "def helper():\n    return True\n\ndef calc(x): return x*2", {"tags": ["python", "utils"]}),
            ("schema.sql", "CREATE TABLE users (id INT, name TEXT);", {"tags": ["sql", "db"]}),
            ("readme.md", "# Project AION\nDocumentation...", {"tags": ["docs"]}),
        ]
        
        # Add many dummy files to simulate scale
        for i in range(50):
            files.append((f"dummy_{i}.txt", f"Some content for file {i} related to AI...", {"tags": ["data"]}))
            
        for name, content, meta in files:
            await manager.add_context(f"--- File: {name} ---\n{content}", meta)
            
        print(f"✅ Context Loaded. Stats: {manager.get_stats()}")
        
        # 2. Query for specific context
        query = "Show me the python entry point and database schema"
        print(f"\n🔍 Query: '{query}'")
        
        context = await manager.get_relevant_context(query, budget=1000)
        
        print("\n📄 Retrieved Context Window:")
        print("-" * 30)
        print(context)
        print("-" * 30)
        
        # Verify paging logic (check loaded pages)
        print(f"\n💾 Memory Logic Check:")
        print(f"   Loaded pages: {len(manager.pager.loaded_pages)}")
        print(f"   Total pages: {len(manager.pager.pages)}")
        
    finally:
        shutil.rmtree(temp_dir)
        print("\n✅ Cleaned up storage.")

if __name__ == "__main__":
    asyncio.run(demo_hyper_context())
