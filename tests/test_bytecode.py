"""Tests for AION Bytecode Compiler"""

import pytest
import sys
sys.path.insert(0, 'd:/Time/aion')

from src.compiler.bytecode import BytecodeCompiler, CompiledFunction, get_compiler


class TestCompiledFunction:
    """Test cases for CompiledFunction dataclass."""
    
    def test_compiled_function_creation(self):
        """Test creating a CompiledFunction."""
        code = compile("1 + 2", "<test>", "eval")
        func = CompiledFunction(
            name="test",
            bytecode=code,
            source_hash="abc123"
        )
        
        assert func.name == "test"
        assert func.bytecode == code
        assert func.source_hash == "abc123"
        assert func.hits == 0


class TestBytecodeCompiler:
    """Test cases for BytecodeCompiler."""
    
    def test_init(self):
        """Test compiler initialization."""
        compiler = BytecodeCompiler(threshold=5)
        
        assert compiler.threshold == 5
        assert len(compiler.call_counts) == 0
        assert len(compiler.compiled) == 0
    
    def test_should_compile_threshold(self):
        """Test compilation threshold behavior."""
        compiler = BytecodeCompiler(threshold=3)
        
        # Initially should not compile
        assert not compiler.should_compile("test")
        
        # Increment call count
        compiler.call_counts["test"] = 3
        
        # Now should compile
        assert compiler.should_compile("test")
    
    def test_should_compile_already_compiled(self):
        """Test that already compiled functions don't recompile."""
        compiler = BytecodeCompiler(threshold=1)
        compiler.call_counts["test"] = 10
        
        # Compile it
        compiler.compile_expression("test", "1 + 2")
        
        # Should not compile again
        assert not compiler.should_compile("test")
    
    def test_compile_expression(self):
        """Test expression compilation."""
        compiler = BytecodeCompiler()
        
        result = compiler.compile_expression("add", "x + y")
        
        assert result.name == "add"
        assert result.bytecode is not None
        assert "add" in compiler.compiled
        assert compiler.stats["compilations"] == 1
    
    def test_compile_expression_invalid(self):
        """Test compilation of invalid expression."""
        compiler = BytecodeCompiler()
        
        with pytest.raises(ValueError):
            compiler.compile_expression("bad", "if x then")
    
    def test_compile_function(self):
        """Test function compilation."""
        compiler = BytecodeCompiler()
        
        result = compiler.compile_function(
            "multiply", 
            "return x * y",
            params=["x", "y"]
        )
        
        assert result.name == "multiply"
        assert result.bytecode is not None
    
    def test_execute_interpretation(self):
        """Test execution in interpretation mode."""
        compiler = BytecodeCompiler(threshold=100)
        
        result = compiler.execute("expr", "10 + 5", {})
        
        assert result == 15
        assert compiler.stats["interpretations"] >= 1
    
    def test_execute_with_context(self):
        """Test execution with context variables."""
        compiler = BytecodeCompiler(threshold=100)
        
        result = compiler.execute("expr", "x * 2", {"x": 21})
        
        assert result == 42
    
    def test_execute_auto_compile(self):
        """Test automatic compilation after threshold."""
        compiler = BytecodeCompiler(threshold=3)
        
        # Execute below threshold
        for i in range(3):
            result = compiler.execute("hot", "5 + 5", {})
        
        # Should now be compiled
        assert "hot" in compiler.compiled
        
        # Next execution uses bytecode
        result = compiler.execute("hot", "5 + 5", {})
        assert result == 10
        assert compiler.stats["bytecode_executions"] >= 1
    
    def test_get_disassembly_compiled(self):
        """Test getting disassembly of compiled function."""
        compiler = BytecodeCompiler()
        compiler.compile_expression("test", "1 + 2")
        
        disasm = compiler.get_disassembly("test")
        
        assert isinstance(disasm, str)
        assert len(disasm) > 0
    
    def test_get_disassembly_not_compiled(self):
        """Test disassembly of non-existent function."""
        compiler = BytecodeCompiler()
        
        disasm = compiler.get_disassembly("nonexistent")
        
        assert "not compiled" in disasm
    
    def test_get_stats(self):
        """Test statistics retrieval."""
        compiler = BytecodeCompiler(threshold=2)
        
        # Do some operations
        compiler.execute("a", "1 + 1", {})
        compiler.execute("a", "1 + 1", {})
        compiler.execute("a", "1 + 1", {})
        
        stats = compiler.get_stats()
        
        assert "interpretations" in stats
        assert "compilations" in stats
        assert "bytecode_executions" in stats
        assert "compiled_functions" in stats
        assert "total_bytecode_hits" in stats
    
    def test_hash_source(self):
        """Test source hashing."""
        compiler = BytecodeCompiler()
        
        hash1 = compiler._hash_source("x + y")
        hash2 = compiler._hash_source("x + y")
        hash3 = compiler._hash_source("x - y")
        
        assert hash1 == hash2
        assert hash1 != hash3
        assert len(hash1) == 8  # MD5 truncated to 8 chars


class TestGlobalCompiler:
    """Test global compiler instance."""
    
    def test_get_compiler(self):
        """Test global compiler access."""
        compiler = get_compiler()
        
        assert isinstance(compiler, BytecodeCompiler)
    
    def test_get_compiler_same_instance(self):
        """Test that global compiler is singleton."""
        c1 = get_compiler()
        c2 = get_compiler()
        
        assert c1 is c2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
