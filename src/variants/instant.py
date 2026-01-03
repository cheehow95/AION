"""
AION Tiered Agent Variants - Instant
=====================================

Fast, optimized agent for daily tasks:
- Quick responses for simple queries
- Information seeking and lookup
- Translation and summarization
- Technical writing assistance

Matches GPT-5.2 Instant tier.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, AsyncIterator
from datetime import datetime
from enum import Enum


class TaskType(Enum):
    """Task types optimized for Instant."""
    INFORMATION_SEEKING = "information_seeking"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    WRITING = "writing"
    QUESTION_ANSWER = "question_answer"
    LOOKUP = "lookup"
    FORMATTING = "formatting"


@dataclass
class QuickResponse:
    """A quick response from Instant agent."""
    content: str = ""
    task_type: TaskType = TaskType.QUESTION_ANSWER
    latency_ms: float = 0.0
    tokens_used: int = 0
    cached: bool = False
    timestamp: datetime = field(default_factory=datetime.now)


class StreamingGenerator:
    """Generates streaming responses for low latency."""
    
    def __init__(self, chunk_size: int = 10):
        self.chunk_size = chunk_size
    
    async def stream(self, content: str) -> AsyncIterator[str]:
        """Stream content in chunks."""
        words = content.split()
        
        for i in range(0, len(words), self.chunk_size):
            chunk = ' '.join(words[i:i + self.chunk_size])
            yield chunk + ' '
            await asyncio.sleep(0.01)  # Simulate streaming delay


class ResponseCache:
    """Cache for frequent queries."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, QuickResponse] = {}
        self.hits = 0
        self.misses = 0
    
    def get(self, query: str) -> Optional[QuickResponse]:
        """Get cached response."""
        # Simple normalization
        key = query.lower().strip()
        if key in self.cache:
            self.hits += 1
            response = self.cache[key]
            response.cached = True
            return response
        self.misses += 1
        return None
    
    def put(self, query: str, response: QuickResponse):
        """Cache a response."""
        if len(self.cache) >= self.max_size:
            # Remove oldest entries
            oldest = sorted(self.cache.items(), 
                          key=lambda x: x[1].timestamp)[:100]
            for key, _ in oldest:
                del self.cache[key]
        
        key = query.lower().strip()
        self.cache[key] = response
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class InstantAgent:
    """Fast agent optimized for daily tasks."""
    
    def __init__(self, agent_id: str = "instant-agent"):
        self.agent_id = agent_id
        self.cache = ResponseCache()
        self.streamer = StreamingGenerator()
        self.max_response_tokens = 2000
        self.target_latency_ms = 500
    
    def classify_task(self, query: str) -> TaskType:
        """Classify the task type."""
        query_lower = query.lower()
        
        if any(w in query_lower for w in ['translate', 'translation', 'in spanish', 'in french']):
            return TaskType.TRANSLATION
        elif any(w in query_lower for w in ['summarize', 'summary', 'tldr', 'brief']):
            return TaskType.SUMMARIZATION
        elif any(w in query_lower for w in ['write', 'draft', 'compose', 'create a']):
            return TaskType.WRITING
        elif any(w in query_lower for w in ['what is', 'who is', 'when', 'where', 'how to']):
            return TaskType.INFORMATION_SEEKING
        elif any(w in query_lower for w in ['format', 'convert', 'reformat']):
            return TaskType.FORMATTING
        elif any(w in query_lower for w in ['look up', 'find', 'search']):
            return TaskType.LOOKUP
        else:
            return TaskType.QUESTION_ANSWER
    
    async def respond(self, query: str, 
                      stream: bool = False) -> QuickResponse:
        """Generate a quick response."""
        start_time = datetime.now()
        
        # Check cache
        cached = self.cache.get(query)
        if cached:
            return cached
        
        task_type = self.classify_task(query)
        
        # Generate response based on task type
        content = await self._generate(query, task_type)
        
        latency = (datetime.now() - start_time).total_seconds() * 1000
        
        response = QuickResponse(
            content=content,
            task_type=task_type,
            latency_ms=latency,
            tokens_used=len(content) // 4
        )
        
        # Cache the response
        self.cache.put(query, response)
        
        return response
    
    async def stream_respond(self, query: str) -> AsyncIterator[str]:
        """Stream a response for minimum latency."""
        response = await self.respond(query)
        async for chunk in self.streamer.stream(response.content):
            yield chunk
    
    async def _generate(self, query: str, task_type: TaskType) -> str:
        """Generate response based on task type."""
        # Simulated responses for demo
        templates = {
            TaskType.INFORMATION_SEEKING: "Based on my knowledge, {}...",
            TaskType.TRANSLATION: "Translation: {}",
            TaskType.SUMMARIZATION: "Summary: {}",
            TaskType.WRITING: "Here's a draft: {}",
            TaskType.QUESTION_ANSWER: "The answer is: {}",
            TaskType.LOOKUP: "I found: {}",
            TaskType.FORMATTING: "Formatted: {}"
        }
        
        # Simulate quick processing
        await asyncio.sleep(0.05)
        
        template = templates.get(task_type, "{}")
        return template.format(f"Response to '{query[:50]}...'")
    
    async def batch_respond(self, queries: List[str]) -> List[QuickResponse]:
        """Process multiple queries efficiently."""
        tasks = [self.respond(q) for q in queries]
        return await asyncio.gather(*tasks)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            'agent_id': self.agent_id,
            'tier': 'instant',
            'cache_hit_rate': self.cache.hit_rate,
            'cache_size': len(self.cache.cache),
            'target_latency_ms': self.target_latency_ms
        }


async def demo_instant():
    """Demonstrate Instant agent."""
    print("⚡ Instant Agent Demo")
    print("=" * 50)
    
    agent = InstantAgent()
    
    queries = [
        "What is machine learning?",
        "Translate 'hello' to Spanish",
        "Summarize the benefits of AI",
        "Write a short poem about coding",
        "How to make coffee?"
    ]
    
    print("\n🚀 Processing queries...")
    for query in queries:
        response = await agent.respond(query)
        print(f"\n📝 Query: {query}")
        print(f"   Type: {response.task_type.value}")
        print(f"   Latency: {response.latency_ms:.1f}ms")
        print(f"   Response: {response.content[:60]}...")
    
    # Test caching
    print("\n💾 Testing cache...")
    response = await agent.respond(queries[0])
    print(f"   Cached: {response.cached}")
    print(f"   Cache hit rate: {agent.cache.hit_rate:.1%}")
    
    # Streaming
    print("\n📡 Streaming response...")
    async for chunk in agent.stream_respond("Explain quantum computing"):
        print(chunk, end='', flush=True)
    print()
    
    print(f"\n📊 Stats: {agent.get_stats()}")
    print("\n✅ Instant demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_instant())
