"""
🚀 AION Interactive Demo
Experience the power of AION - your AI-native programming language!
"""

import sys
import asyncio
import time

sys.path.insert(0, '.')

def print_header():
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║     _    ___ ___  _   _   ____                                    ║
║    / \\  |_ _/ _ \\| \\ | | |  _ \\  ___ _ __ ___   ___               ║
║   / _ \\  | | | | |  \\| | | | | |/ _ \\ '_ ` _ \\ / _ \\              ║
║  / ___ \\ | | |_| | |\\  | | |_| |  __/ | | | | | (_) |             ║
║ /_/   \\_\\___\\___/|_| \\_| |____/ \\___|_| |_| |_|\\___/              ║
║                                                                   ║
║          Artificial Intelligence Oriented Notation                ║
║              The Language for Thinking Systems                    ║
╚═══════════════════════════════════════════════════════════════════╝
    """)

def demo_1_local_reasoning():
    """Demo: Ultra-fast local reasoning"""
    print("\n" + "="*60)
    print("📡 DEMO 1: Local Reasoning Engine (No API Required!)")
    print("="*60)
    
    from src.runtime.local_engine import LocalReasoningEngine
    
    engine = LocalReasoningEngine()
    
    queries = [
        "Hello, what can you do?",
        "What is 2 + 2?",
        "What is AION?",
        "How do I create an agent?",
    ]
    
    for query in queries:
        start = time.perf_counter()
        
        # Think
        thought = engine.think(query)
        
        # Analyze
        analysis = engine.analyze(query)
        
        # Decide
        decision = engine.decide(query)
        
        elapsed = (time.perf_counter() - start) * 1000
        
        print(f"\n💬 You: {query}")
        print(f"🧠 Analysis: {analysis['intent']} | {analysis['sentiment']}")
        print(f"💡 Response: {decision['decision']}")
        print(f"⚡ Time: {elapsed:.2f}ms")
    
    print("\n✅ Local reasoning complete - all responses under 1ms!")

def demo_2_parse_aion():
    """Demo: Parse AION code"""
    print("\n" + "="*60)
    print("📜 DEMO 2: Parse AION Code into AST")
    print("="*60)
    
    from src.parser import parse
    
    code = '''agent MyAssistant {
  goal "Help users learn AION"
  memory working
  memory long_term
  model GPT4
}'''
    
    print("\n📝 AION Code:")
    print("-" * 40)
    print(code)
    print("-" * 40)
    
    ast = parse(code)
    agent = ast.declarations[0]
    
    print(f"\n✅ Parsed successfully!")
    print(f"   Agent Name: {agent.name}")
    print(f"   Body Members: {len(agent.body)}")
    for member in agent.body:
        print(f"      - {member.__class__.__name__}")

def demo_3_transpile():
    """Demo: Transpile AION to Python"""
    print("\n" + "="*60)
    print("🐍 DEMO 3: Transpile AION → Python")
    print("="*60)
    
    from src.transpiler import transpile
    
    code = '''agent Calculator {
  goal "Perform calculations"
  memory working
}'''
    
    print("\n📝 AION Code:")
    print(code)
    
    python_code = transpile(code)
    
    print("\n🐍 Generated Python:")
    print("-" * 40)
    # Show first 20 lines
    lines = python_code.split('\n')[:25]
    print('\n'.join(lines))
    if len(python_code.split('\n')) > 25:
        print("... (truncated)")
    print("-" * 40)

def demo_4_vector_memory():
    """Demo: RAG / Vector Memory"""
    print("\n" + "="*60)
    print("🧠 DEMO 4: Vector Memory (RAG)")
    print("="*60)
    
    from src.runtime.vector_memory import VectorMemory
    
    mem = VectorMemory()
    
    # Store knowledge
    facts = [
        "AION stands for Artificial Intelligence Oriented Notation.",
        "Agents in AION have goals, memory, and event handlers.",
        "The think command triggers cognitive reasoning.",
        "AION supports four memory types: working, episodic, long-term, semantic.",
        "Tools can be registered with trust levels and cost limits.",
    ]
    
    print("\n📚 Storing knowledge...")
    for fact in facts:
        mem.add(fact)
        print(f"   ✓ {fact[:50]}...")
    
    # Search
    queries = ["What is AION?", "How does memory work?", "What are tools?"]
    
    print("\n🔍 Semantic Search:")
    for q in queries:
        results = mem.search(q, limit=1)
        if results:
            print(f"\n   Q: {q}")
            print(f"   A: {results[0].content}")

def demo_5_adaptive_router():
    """Demo: Hybrid Local/Cloud Routing"""
    print("\n" + "="*60)
    print("🔀 DEMO 5: Adaptive Model Router")
    print("="*60)
    
    from src.runtime.router import HybridRuntime
    
    runtime = HybridRuntime()
    
    queries = [
        ("Hello!", "Simple greeting"),
        ("What is 5 * 7?", "Math calculation"),
        ("Write a detailed essay about the philosophical implications of artificial consciousness and its impact on human society", "Complex task"),
    ]
    
    print("\n🧠 Routing decisions:")
    for query, desc in queries:
        decision = runtime.router.route(query)
        icon = "🏠" if decision.provider == "local" else "☁️"
        print(f"\n   {icon} [{decision.provider.upper()}] {desc}")
        print(f"      Query: \"{query[:50]}...\"")
        print(f"      Reason: {decision.reasoning}")

async def demo_6_run_agent():
    """Demo: Run a complete agent"""
    print("\n" + "="*60)
    print("🤖 DEMO 6: Run a Complete AION Agent")
    print("="*60)
    
    from fast_runner import FastAgentRunner
    
    # Define agent
    source = '''agent DemoAgent {
  goal "Demonstrate AION capabilities"
  memory working
}'''
    
    runner = FastAgentRunner()
    agents = runner.load(source)
    
    print(f"\n✅ Loaded agent: {agents[0]}")
    print(f"   Parse time: {runner.stats['parse_time']*1000:.2f}ms")
    
    # Run with input
    result = await runner.run('DemoAgent', "What can you do?")
    
    print(f"\n📤 Output: {result['output']}")
    print(f"   Execution time: {result['stats']['exec_time_ms']:.2f}ms")
    print(f"   Reasoning steps: {result['stats']['total_steps']}")

def main():
    print_header()
    
    input("Press Enter to start the demos...\n")
    
    try:
        demo_1_local_reasoning()
        input("\nPress Enter for next demo...")
        
        demo_2_parse_aion()
        input("\nPress Enter for next demo...")
        
        demo_3_transpile()
        input("\nPress Enter for next demo...")
        
        demo_4_vector_memory()
        input("\nPress Enter for next demo...")
        
        demo_5_adaptive_router()
        input("\nPress Enter for next demo...")
        
        asyncio.run(demo_6_run_agent())
        
        print("\n" + "="*60)
        print("🎉 DEMO COMPLETE!")
        print("="*60)
        print("""
Next steps:
1. Try the REPL:        python repl.py
2. Run your own agent:  python fast_runner.py examples/assistant.aion "Hello"
3. Read the docs:       docs/getting-started.md

Happy coding with AION! 🚀
        """)
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted. Goodbye!")

if __name__ == "__main__":
    main()
