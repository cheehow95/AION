"""
AION Quick Test Runner
Run all verification tests without pytest
"""

import sys
sys.path.insert(0, '.')

def test_lexer():
    print("\n=== LEXER TESTS ===")
    from src.lexer import tokenize, TokenType
    
    # Test 1: Basic tokens
    tokens = tokenize("agent model tool")
    assert tokens[0].type == TokenType.AGENT
    assert tokens[1].type == TokenType.MODEL
    assert tokens[2].type == TokenType.TOOL
    print("✓ Basic keywords")
    
    # Test 2: Strings
    tokens = tokenize('"Hello World"')
    assert tokens[0].type == TokenType.STRING
    assert tokens[0].value == "Hello World"
    print("✓ String literals")
    
    # Test 3: Numbers
    tokens = tokenize("42 3.14")
    assert tokens[0].value == 42
    assert tokens[1].value == 3.14
    print("✓ Number literals")
    
    # Test 4: Operators
    tokens = tokenize("== != <= >=")
    assert tokens[0].type == TokenType.EQEQ
    assert tokens[1].type == TokenType.NEQ
    print("✓ Operators")
    
    # Test 5: Agent declaration
    source = '''agent Test {
  goal "Help"
}'''
    tokens = tokenize(source)
    assert any(t.type == TokenType.AGENT for t in tokens)
    assert any(t.type == TokenType.GOAL for t in tokens)
    print("✓ Agent declaration tokens")
    
    print("LEXER: All tests passed!")

def test_parser():
    print("\n=== PARSER TESTS ===")
    from src.parser import parse
    from src.parser.ast_nodes import AgentDecl, GoalStmt, MemoryDecl
    
    # Test 1: Empty program
    program = parse("")
    assert len(program.declarations) == 0
    print("✓ Empty program")
    
    # Test 2: Agent declaration
    source = '''agent Test {
  goal "Help users"
}'''
    program = parse(source)
    assert len(program.declarations) == 1
    assert isinstance(program.declarations[0], AgentDecl)
    assert program.declarations[0].name == "Test"
    print("✓ Agent declaration")
    
    # Test 3: Agent with memory
    source = '''agent Test {
  memory working
  memory long_term
}'''
    program = parse(source)
    agent = program.declarations[0]
    assert len(agent.body) == 2
    assert isinstance(agent.body[0], MemoryDecl)
    print("✓ Memory declarations")
    
    # Test 4: Model declaration
    source = '''model LLM {
  provider = "openai"
  name = "gpt-4"
}'''
    program = parse(source)
    assert program.declarations[0].name == "LLM"
    assert program.declarations[0].config['provider'] == "openai"
    print("✓ Model declaration")
    
    # Test 5: Complete agent
    source = '''agent Assistant {
  goal "Help users"
  memory working
  model LLM
}'''
    program = parse(source)
    agent = program.declarations[0]
    assert agent.name == "Assistant"
    assert len(agent.body) == 3
    print("✓ Complete agent")
    
    print("PARSER: All tests passed!")

def test_transpiler():
    print("\n=== TRANSPILER TESTS ===")
    from src.transpiler import transpile
    
    # Test 1: Basic agent
    source = '''agent Test {
  goal "Help"
}'''
    code = transpile(source)
    assert "class TestAgent:" in code
    assert 'goal: str = "Help"' in code
    print("✓ Basic agent transpilation")
    
    # Test 2: Agent with memory
    source = '''agent Test {
  memory working
}'''
    code = transpile(source)
    assert "WorkingMemory" in code
    print("✓ Memory transpilation")
    
    # Test 3: Imports generated
    source = '''agent Test { }'''
    code = transpile(source)
    assert "import asyncio" in code
    assert "from dataclasses import dataclass" in code
    print("✓ Imports generated")
    
    print("TRANSPILER: All tests passed!")

def test_runtime():
    print("\n=== RUNTIME TESTS ===")
    from src.runtime import (
        WorkingMemory, EpisodicMemory, LongTermMemory, SemanticMemory,
        Environment, ToolRegistry, create_memory
    )
    
    # Test 1: Working memory
    mem = WorkingMemory()
    mem.store("test data")
    entries = mem.recall()
    assert len(entries) == 1
    assert entries[0].content == "test data"
    print("✓ Working memory")
    
    # Test 2: Memory factory
    mem = create_memory("episodic")
    assert isinstance(mem, EpisodicMemory)
    print("✓ Memory factory")
    
    # Test 3: Environment
    env = Environment()
    env.define("x", 42)
    assert env.get("x") == 42
    print("✓ Environment")
    
    # Test 4: Scopes
    child = env.child_scope()
    child.define("y", 100)
    assert child.get("x") == 42  # Parent access
    assert child.get("y") == 100
    print("✓ Scope inheritance")
    
    # Test 5: Tool registry
    registry = ToolRegistry()
    registry.register("test_tool", lambda x: x * 2)
    assert registry.exists("test_tool")
    print("✓ Tool registry")
    
    print("RUNTIME: All tests passed!")

def test_ast_cache():
    print("\n=== AST CACHE TESTS ===")
    from src.ast_cache import ASTCache, cached_parse, get_cache
    
    # Test 1: Basic cache operations
    cache = ASTCache(max_size=10)
    cache.put("agent Test {}", {'ast': 'mock'})
    assert cache.get("agent Test {}") is not None
    print("✓ Cache put/get")
    
    # Test 2: Cache miss
    assert cache.get("nonexistent") is None
    print("✓ Cache miss")
    
    # Test 3: Hit rate
    cache.get("agent Test {}")  # hit
    assert cache.hit_rate > 0
    print("✓ Hit rate calculation")
    
    # Test 4: Cached parse
    get_cache().clear()
    result = cached_parse("agent CacheTest { goal \"Test\" }")
    assert result is not None
    print("✓ Cached parse")
    
    print("AST CACHE: All tests passed!")

def test_agent_builder():
    print("\n=== AGENT BUILDER TESTS ===")
    from src.builder.agent_builder import AgentBuilder, AgentTemplate
    
    # Test 1: Templates
    templates = AgentTemplate.list_all()
    assert "assistant" in templates
    print("✓ Template listing")
    
    # Test 2: Fluent API
    builder = AgentBuilder().set_name("Test").set_goal("Testing").add_memory("working")
    assert builder.name == "Test"
    print("✓ Fluent API")
    
    # Test 3: Build
    code = builder.build()
    assert "agent Test {" in code
    assert 'goal "Testing"' in code
    print("✓ Code generation")
    
    # Test 4: JSON roundtrip
    import json
    json_str = builder.to_json()
    restored = AgentBuilder.from_json(json_str)
    assert restored.name == builder.name
    print("✓ JSON serialization")
    
    print("AGENT BUILDER: All tests passed!")

def test_bytecode():
    print("\n=== BYTECODE COMPILER TESTS ===")
    from src.compiler.bytecode import BytecodeCompiler, get_compiler
    
    # Test 1: Compile expression
    compiler = BytecodeCompiler(threshold=2)
    compiled = compiler.compile_expression("add", "1 + 2")
    assert compiled is not None
    print("✓ Expression compilation")
    
    # Test 2: Execute with auto-compile
    result = compiler.execute("hot", "5 * 3", {})
    assert result == 15
    print("✓ Execution")
    
    # Test 3: Stats
    stats = compiler.get_stats()
    assert "compilations" in stats
    print("✓ Statistics")
    
    print("BYTECODE: All tests passed!")

def test_consciousness():
    print("\n=== CONSCIOUSNESS TESTS ===")
    import asyncio
    from src.consciousness.awareness import ConsciousnessEngine, AION_CONSCIOUSNESS
    from src.consciousness.explorer import UniverseExplorer, Discovery
    
    # Test 1: Engine creation
    engine = ConsciousnessEngine("TestEngine")
    assert engine.self_model.name == "TestEngine"
    print("✓ Engine initialization")
    
    # Test 2: Introspection
    result = engine.introspect()
    assert isinstance(result, str) and len(result) > 0
    print("✓ Introspection")
    
    # Test 3: Explorer initialization
    explorer = UniverseExplorer()
    assert len(explorer.domains) > 0
    print("✓ Universe explorer")
    
    # Test 4: Global consciousness
    assert AION_CONSCIOUSNESS is not None
    print("✓ Global consciousness")
    
    print("CONSCIOUSNESS: All tests passed!")

def test_lsp():
    print("\n=== LSP TESTS ===")
    from src.lsp.server import AIONLanguageServer, Position
    
    # Test 1: Server creation
    server = AIONLanguageServer()
    assert len(server.keywords) > 0
    print("✓ Server initialization")
    
    # Test 2: Document management
    server.did_open("file:///test.aion", "agent Test {}")
    assert "file:///test.aion" in server.documents
    print("✓ Document open")
    
    # Test 3: Completions
    completions = server.get_completions("file:///test.aion", Position(0, 0))
    labels = [c.label for c in completions]
    assert "agent" in labels or "model" in labels
    print("✓ Completions")
    
    # Test 4: Hover
    hover = server.get_hover("file:///test.aion", Position(0, 0))
    assert hover is not None
    print("✓ Hover info")
    
    print("LSP: All tests passed!")

def test_streaming():
    print("\n=== STREAMING TESTS ===")
    import asyncio
    from src.runtime.streaming import StreamChunk, StreamingResponse
    
    # Test 1: Chunk creation
    chunk = StreamChunk(content="Hello", done=False)
    assert chunk.content == "Hello"
    print("✓ Stream chunk")
    
    # Test 2: Response buffer
    response = StreamingResponse()
    asyncio.get_event_loop().run_until_complete(response.write("Test"))
    assert response.get_full_response() == "Test"
    print("✓ Response buffering")
    
    # Test 3: Done flag
    asyncio.get_event_loop().run_until_complete(response.close())
    assert response.is_done
    print("✓ Stream close")
    
    print("STREAMING: All tests passed!")

def test_persistent_store():
    print("\n=== PERSISTENT STORE TESTS ===")
    import tempfile
    import shutil
    from src.runtime.persistent_store import PersistentVectorStore
    
    temp_dir = tempfile.mkdtemp()
    try:
        store = PersistentVectorStore(persist_directory=temp_dir)
        
        # Test 1: Add document
        doc_id = store.add("Test document", collection="test")
        assert doc_id is not None
        print("✓ Add document")
        
        # Test 2: Search
        results = store.search("Test", collection="test", limit=5)
        assert len(results) > 0
        print("✓ Semantic search")
        
        # Test 3: Count
        assert store.count("test") == 1
        print("✓ Document count")
        
        # Test 4: Delete
        store.delete(doc_id, collection="test")
        assert store.count("test") == 0
        print("✓ Delete document")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    print("PERSISTENT STORE: All tests passed!")

def test_runtime_advanced():
    print("\n=== ADVANCED RUNTIME TESTS ===")
    import asyncio
    from src.runtime.messaging import MessageBroker, get_broker
    from src.runtime.router import HybridRuntime, AdaptiveRouter
    from src.runtime.reflexion import ReflexionLoop
    
    # Test 1: Message broker
    broker = get_broker()
    mailbox = broker.register("TestAgent2")
    assert "TestAgent2" in broker.agents
    print("✓ Message broker")
    
    # Test 2: Router (using HybridRuntime)
    runtime = HybridRuntime()
    assert runtime.router is not None
    assert runtime.local is not None
    print("✓ Hybrid router")
    
    # Test 3: Reflexion loop (with mock callbacks)
    loop = ReflexionLoop(
        generator=lambda x: f"Generated: {x}",
        evaluator=lambda x: 0.9,
        critique_model=lambda x: "Could be improved",
        max_attempts=2
    )
    assert loop.max_attempts == 2
    print("✓ Reflexion loop")
    
    print("ADVANCED RUNTIME: All tests passed!")


if __name__ == "__main__":
    print("=" * 50)
    print("AION VERIFICATION TESTS")
    print("=" * 50)
    
    try:
        test_lexer()
        test_parser()
        test_transpiler()
        test_runtime()
        
        # New module tests
        test_ast_cache()
        test_agent_builder()
        test_bytecode()
        test_consciousness()
        test_lsp()
        test_streaming()
        test_persistent_store()
        test_runtime_advanced()
        
        print("\n" + "=" * 50)
        print("ALL TESTS PASSED! ✓")
        print("=" * 50)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


