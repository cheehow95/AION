"""Tests for AION Transpiler"""

import pytest
import sys
sys.path.insert(0, '.')

from src.transpiler import transpile


class TestTranspiler:
    """Test cases for the AION transpiler."""
    
    def test_simple_agent(self):
        """Test transpiling a simple agent."""
        source = '''agent Assistant {
  goal "Help users"
}'''
        python_code = transpile(source)
        
        assert "class AssistantAgent:" in python_code
        assert 'goal: str = "Help users"' in python_code
    
    def test_agent_with_memory(self):
        """Test transpiling agent with memory."""
        source = '''agent Test {
  memory working
  memory long_term
}'''
        python_code = transpile(source)
        
        assert "WorkingMemory" in python_code
        assert "LongTermMemory" in python_code
    
    def test_agent_with_model(self):
        """Test transpiling agent with model reference."""
        source = '''agent Test {
  model LLM
}'''
        python_code = transpile(source)
        
        assert 'model_name: str = "LLM"' in python_code
    
    def test_event_handler(self):
        """Test transpiling event handler."""
        source = '''agent Test {
  on input(msg):
    respond msg
}'''
        python_code = transpile(source)
        
        assert "async def on_input(self, msg)" in python_code
    
    def test_reasoning_statements(self):
        """Test transpiling reasoning statements."""
        source = '''agent Test {
  on input(x):
    think "Consider options"
    analyze x
    reflect
    decide answer
}'''
        python_code = transpile(source)
        
        assert "reasoning_engine.think" in python_code
        assert "reasoning_engine.analyze" in python_code
        assert "reasoning_engine.reflect" in python_code
        assert "reasoning_engine.decide" in python_code
    
    def test_control_flow(self):
        """Test transpiling control flow statements."""
        source = '''agent Test {
  on input(x):
    if x > 0:
      respond
    repeat 3 times:
      think
}'''
        python_code = transpile(source)
        
        assert "if (x > 0):" in python_code
        assert "for _ in range" in python_code
    
    def test_tool_usage(self):
        """Test transpiling tool usage."""
        source = '''agent Test {
  on input(x):
    use calculator(x)
}'''
        python_code = transpile(source)
        
        assert 'tool_registry.execute("calculator"' in python_code
    
    def test_memory_operations(self):
        """Test transpiling memory operations."""
        source = '''agent Test {
  memory working
  on input(x):
    store x in my_mem
}'''
        python_code = transpile(source)
        
        assert "store" in python_code.lower() or "memory" in python_code.lower()
    
    def test_imports_generated(self):
        """Test that necessary imports are generated."""
        source = '''agent Test {
  goal "Test"
}'''
        python_code = transpile(source)
        
        assert "import asyncio" in python_code
        assert "from dataclasses import dataclass" in python_code
        assert "from aion.runtime import" in python_code


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
