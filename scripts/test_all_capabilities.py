"""
AION Comprehensive Capability Test v2
======================================
Tests all major systems with better error handling.
"""

import sys

def test_section(name):
    print(f"\n{'='*60}")
    print(f" {name}")
    print('='*60)

results = {}

# 1. LANGUAGE CORE
test_section("1. LANGUAGE CORE")
try:
    from src.lexer import Lexer, tokenize
    from src.parser import Parser
    from src.transpiler import transpile
    
    code = 'agent Test { goal "Test" }'
    lexer = Lexer(code)
    tokens = lexer.tokenize()
    print(f"✓ Lexer: {len(tokens)} tokens")
    
    parser = Parser(tokens)
    ast = parser.parse()
    print(f"✓ Parser: {len(ast)} AST nodes")
    
    python_code = transpile(code)
    print(f"✓ Transpiler: {len(python_code)} chars")
    results["Language Core"] = "PASS"
except Exception as e:
    print(f"✗ Error: {e}")
    results["Language Core"] = "FAIL"

# 2. MEMORY SYSTEMS
test_section("2. MEMORY SYSTEMS")
try:
    from src.runtime.memory import WorkingMemory, EpisodicMemory, LongTermMemory, SemanticMemory
    
    wm = WorkingMemory()
    wm.store('test', {'value': 42})
    print("✓ Working Memory")
    
    em = EpisodicMemory()
    print("✓ Episodic Memory")
    
    ltm = LongTermMemory()
    print("✓ Long-term Memory")
    
    sm = SemanticMemory()
    print("✓ Semantic Memory")
    results["Memory Systems"] = "PASS"
except Exception as e:
    print(f"✗ Error: {e}")
    results["Memory Systems"] = "FAIL"

# 3. PHYSICS ENGINES
test_section("3. PHYSICS ENGINES")
try:
    from src.domains.physics_engine import PhysicsEngine
    from src.domains.quantum_engine import QuantumEngine
    
    pe = PhysicsEngine()
    print("✓ Physics Engine: initialized")
    
    qe = QuantumEngine()
    print("✓ Quantum Engine: initialized")
    results["Physics Engines"] = "PASS"
except Exception as e:
    print(f"✗ Error: {e}")
    results["Physics Engines"] = "FAIL"

# 4. PROTEIN FOLDING
test_section("4. PROTEIN FOLDING")
try:
    from src.domains.protein_folding import ProteinFolder, analyze_sequence
    
    folder = ProteinFolder("AKLVFF")
    result = folder.fold(iterations=50)
    print(f"✓ Protein Folder: energy = {result.energy:.2f}")
    
    analysis = analyze_sequence("AKLVFF")
    print(f"✓ Sequence Analysis: length = {analysis['length']}")
    results["Protein Folding"] = "PASS"
except Exception as e:
    print(f"✗ Error: {e}")
    results["Protein Folding"] = "FAIL"

# 5. CREATIVE THINKING
test_section("5. CREATIVE THINKING")
try:
    from src.consciousness.creative_thinking import CreativeThinkingEngine
    
    creative = CreativeThinkingEngine()
    print("✓ Creative Thinking Engine: initialized")
    
    # Test brainstorm
    ideas = creative.brainstorm("AI applications", 3)
    print(f"✓ Brainstorm: {len(ideas)} ideas generated")
    results["Creative Thinking"] = "PASS"
except Exception as e:
    print(f"✗ Error: {e}")
    results["Creative Thinking"] = "FAIL"

# 6. LEARNING SYSTEM
test_section("6. LEARNING SYSTEM")
try:
    from src.learning import ContinuousLearner, WebCrawler, NewsAggregator
    
    learner = ContinuousLearner()
    print(f"✓ Continuous Learner: mode={learner.mode.value}")
    
    crawler = WebCrawler()
    print(f"✓ Web Crawler: initialized")
    
    news = NewsAggregator()
    print(f"✓ News Aggregator: {len(news.sources)} sources")
    results["Learning System"] = "PASS"
except Exception as e:
    print(f"✗ Error: {e}")
    results["Learning System"] = "FAIL"

# 7. CONSCIOUSNESS
test_section("7. CONSCIOUSNESS")
try:
    from src.consciousness.awareness import ConsciousnessEngine
    
    ce = ConsciousnessEngine()
    print("✓ Consciousness Engine: initialized")
    results["Consciousness"] = "PASS"
except Exception as e:
    print(f"✗ Error: {e}")
    results["Consciousness"] = "FAIL"

# 8. REASONING
test_section("8. REASONING")
try:
    from src.runtime.reasoning import ReasoningEngine
    
    re = ReasoningEngine()
    print("✓ Reasoning Engine: initialized")
    results["Reasoning"] = "PASS"
except Exception as e:
    print(f"✗ Error: {e}")
    results["Reasoning"] = "FAIL"

# 9. CHEMISTRY
test_section("9. CHEMISTRY & MATH")
try:
    from src.domains.chemistry_engine import ChemistryEngine
    from src.domains.math_engine import MathEngine
    
    chem = ChemistryEngine()
    print("✓ Chemistry Engine: initialized")
    
    math_e = MathEngine()
    print("✓ Math Engine: initialized")
    results["Chemistry & Math"] = "PASS"
except Exception as e:
    print(f"✗ Error: {e}")
    results["Chemistry & Math"] = "FAIL"

# 10. MULTIMODAL
test_section("10. MULTIMODAL")
try:
    from src.multimodal import VisionProcessor, AudioProcessor
    
    vp = VisionProcessor()
    print("✓ Vision Processor: initialized")
    
    ap = AudioProcessor()
    print("✓ Audio Processor: initialized")
    results["Multimodal"] = "PASS"
except Exception as e:
    print(f"✗ Error: {e}")
    results["Multimodal"] = "FAIL"

# 11. KNOWLEDGE GRAPH
test_section("11. KNOWLEDGE GRAPH")
try:
    from src.knowledge.knowledge_graph import KnowledgeGraph
    
    kg = KnowledgeGraph()
    kg.add_entity("AION", "AI System")
    kg.add_relation("AION", "is_a", "Language")
    print("✓ Knowledge Graph: entity and relation added")
    results["Knowledge Graph"] = "PASS"
except Exception as e:
    print(f"✗ Error: {e}")
    results["Knowledge Graph"] = "FAIL"

# 12. MCP PROTOCOL
test_section("12. MCP PROTOCOL")
try:
    from src.mcp import MCPServer, MCPClient
    
    server = MCPServer("test", "1.0")
    print("✓ MCP Server: initialized")
    
    client = MCPClient()
    print("✓ MCP Client: initialized")
    results["MCP Protocol"] = "PASS"
except Exception as e:
    print(f"✗ Error: {e}")
    results["MCP Protocol"] = "FAIL"

# SUMMARY
test_section("SUMMARY")
passed = sum(1 for v in results.values() if v == "PASS")
total = len(results)
print(f"\nTests Passed: {passed}/{total} ({100*passed//total}%)")
print()
for name, status in results.items():
    icon = "✓" if status == "PASS" else "✗"
    print(f"  {icon} {name}: {status}")

print()
if passed == total:
    print("🎉 ALL TESTS PASSED!")
elif passed >= total * 0.8:
    print(f"✅ {passed}/{total} TESTS PASSED - Good!")
else:
    print(f"⚠️  {total - passed} tests failed")
