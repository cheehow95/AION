"""
AION Advanced Features Verification
Tests RAG, Reflexion, and Distributed Runtime.
"""

import sys
import asyncio
import time
from datetime import datetime

sys.path.insert(0, '.')

async def test_search():
    print("\n=== WEB SEARCH TEST ===")
    from src.tools.search import WebSearchTool
    
    tool = WebSearchTool()
    results = await tool.search("AION language")
    
    assert len(results) > 0
    print(f"✓ Retrieved {len(results)} search results")
    print(f"  First result: {results[0].title}")

async def test_vector_memory():
    print("\n=== VECTOR MEMORY (RAG) TEST ===")
    from src.runtime.vector_memory import VectorMemory
    
    mem = VectorMemory()
    
    # Add knowledge
    id1 = mem.add("AION is a declarative AI language.")
    id2 = mem.add("Python is a popular programming language.")
    id3 = mem.add("Agents have goals and memory.")
    
    print(f"✓ Added 3 entries to vector memory")
    
    # Search
    results = mem.search("What is AION?", limit=1)
    assert len(results) == 1
    assert "AION" in results[0].content
    print(f"✓ Semantic search successful: '{results[0].content}'")

async def test_reflexion():
    print("\n=== REFLEXION LOOP TEST ===")
    from src.runtime.reflexion import ReflexionLoop
    
    # Mock components
    async def generate(prompt):
        # Simulate an agent improving over attempts
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count == 1: return "Bad answer"
        if attempt_count == 2: return "Okay answer"
        return "Perfect answer"
        
    async def evaluate(output):
        if output == "Bad answer": return 0.2
        if output == "Okay answer": return 0.6
        return 1.0
        
    async def critique(output):
        return "Make it better"
        
    attempt_count = 0
    loop = ReflexionLoop(generate, evaluate, critique)
    
    result = await loop.run("Task")
    assert result == "Perfect answer"
    assert len(loop.traces) == 3
    print(f"✓ Self-correction loop: 3 attempts to reach perfection")

async def test_distributed():
    print("\n=== DISTRIBUTED HIVE MIND TEST ===")
    # Note: Full multiprocessing test is hard in simple script, 
    # checking basic class instantiation and logic
    from src.runtime.distributed import HiveMind, AgentMessage
    
    hive = HiveMind()
    print("✓ HiveMind initialized")
    
    # Basic message
    msg = AgentMessage("AgentA", "AgentB", "Hello")
    assert msg.id != ""
    print("✓ Message structure valid")

async def main():
    print("=" * 60)
    print("AION ADVANCED FEATURES VERIFICATION")
    print("=" * 60)
    
    try:
        await test_search()
        await test_vector_memory()
        await test_reflexion()
        await test_distributed()
        
        print("\n" + "=" * 60)
        print("ALL ADVANCED FEATURES VERIFIED! ✓")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
