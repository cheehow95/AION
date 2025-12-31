"""
AION Bytecode Compiler
Compiles hot paths to bytecode for faster execution.
"""

import dis
import types
from typing import Dict, Any, Callable, List
from dataclasses import dataclass
import hashlib

@dataclass
class CompiledFunction:
    """A compiled AION function."""
    name: str
    bytecode: types.CodeType
    source_hash: str
    hits: int = 0

class BytecodeCompiler:
    """
    Compiles AION operations to Python bytecode.
    Hot paths are automatically cached for faster execution.
    """
    
    def __init__(self, threshold: int = 10):
        self.threshold = threshold  # Compile after N executions
        self.call_counts: Dict[str, int] = {}
        self.compiled: Dict[str, CompiledFunction] = {}
        self.stats = {
            "interpretations": 0,
            "compilations": 0,
            "bytecode_executions": 0
        }
    
    def _hash_source(self, source: str) -> str:
        return hashlib.md5(source.encode()).hexdigest()[:8]
    
    def should_compile(self, name: str) -> bool:
        """Check if function should be compiled."""
        count = self.call_counts.get(name, 0)
        return count >= self.threshold and name not in self.compiled
    
    def compile_expression(self, name: str, expression: str) -> CompiledFunction:
        """Compile a simple expression to bytecode."""
        try:
            code = compile(expression, f"<aion:{name}>", "eval")
            compiled = CompiledFunction(
                name=name,
                bytecode=code,
                source_hash=self._hash_source(expression)
            )
            self.compiled[name] = compiled
            self.stats["compilations"] += 1
            return compiled
        except SyntaxError as e:
            raise ValueError(f"Cannot compile '{expression}': {e}")
    
    def compile_function(self, name: str, body: str, 
                         params: List[str] = None) -> CompiledFunction:
        """Compile a function body to bytecode."""
        params = params or []
        param_str = ", ".join(params)
        
        # Wrap in function
        func_source = f"def {name}({param_str}):\n"
        for line in body.split('\n'):
            func_source += f"    {line}\n"
        
        # Compile and extract
        namespace = {}
        exec(compile(func_source, f"<aion:{name}>", "exec"), namespace)
        code = namespace[name].__code__
        
        compiled = CompiledFunction(
            name=name,
            bytecode=code,
            source_hash=self._hash_source(body)
        )
        self.compiled[name] = compiled
        self.stats["compilations"] += 1
        return compiled
    
    def execute(self, name: str, source: str, 
                context: Dict[str, Any] = None) -> Any:
        """Execute with automatic compilation."""
        context = context or {}
        
        # Track calls
        self.call_counts[name] = self.call_counts.get(name, 0) + 1
        
        # Check if already compiled
        if name in self.compiled:
            result = eval(self.compiled[name].bytecode, context)
            self.compiled[name].hits += 1
            self.stats["bytecode_executions"] += 1
            return result
        
        # Check if should compile
        if self.should_compile(name):
            self.compile_expression(name, source)
            return self.execute(name, source, context)
        
        # Interpret
        self.stats["interpretations"] += 1
        return eval(source, context)
    
    def get_disassembly(self, name: str) -> str:
        """Get bytecode disassembly for a compiled function."""
        if name not in self.compiled:
            return f"Function '{name}' not compiled"
        
        import io
        output = io.StringIO()
        dis.dis(self.compiled[name].bytecode, file=output)
        return output.getvalue()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get compiler statistics."""
        return {
            **self.stats,
            "compiled_functions": len(self.compiled),
            "total_bytecode_hits": sum(c.hits for c in self.compiled.values())
        }


# Global compiler
_compiler = BytecodeCompiler()

def get_compiler() -> BytecodeCompiler:
    return _compiler


def demo():
    """Demo bytecode compiler."""
    print("⚡ AION Bytecode Compiler Demo")
    print("-" * 50)
    
    compiler = BytecodeCompiler(threshold=5)
    
    # Execute hot path multiple times
    expr = "x * 2 + y"
    context = {"x": 10, "y": 5}
    
    print(f"Expression: {expr}")
    print(f"Threshold: {compiler.threshold} executions\n")
    
    for i in range(15):
        result = compiler.execute("hot_path", expr, context)
        mode = "BYTECODE" if "hot_path" in compiler.compiled else "INTERPRET"
        print(f"   Execution {i+1}: {result} [{mode}]")
    
    print(f"\n📊 Statistics:")
    stats = compiler.get_stats()
    for k, v in stats.items():
        print(f"   {k}: {v}")
    
    print(f"\n🔍 Bytecode Disassembly:")
    print(compiler.get_disassembly("hot_path"))


if __name__ == "__main__":
    demo()
