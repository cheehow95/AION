"""
AION REPL (Read-Eval-Print Loop)
Interactive shell for experimenting with AION code.
"""

import sys
import asyncio
from typing import Optional

sys.path.insert(0, '.')


class AionREPL:
    """Interactive REPL for AION language."""
    
    def __init__(self):
        from src.lexer import Lexer, LexerError
        from src.parser import Parser, ParserError, parse
        from src.interpreter import Interpreter
        from src.transpiler import transpile
        from src.runtime import Environment
        
        self.interpreter = Interpreter()
        self.buffer = []
        self.mode = 'run'  # run, transpile, parse, tokens
        
    def print_banner(self):
        print("""
╔═══════════════════════════════════════════════════════╗
║     _    ___ ___  _   _                               ║
║    / \\  |_ _/ _ \\| \\ | |                              ║
║   / _ \\  | | | | |  \\| |                              ║
║  / ___ \\ | | |_| | |\\  |                              ║
║ /_/   \\_\\___\\___/|_| \\_|  REPL v0.1.0                 ║
║                                                       ║
║ Artificial Intelligence Oriented Notation             ║
╚═══════════════════════════════════════════════════════╝

Commands:
  :mode [run|transpile|parse|tokens]  - Switch mode
  :clear                              - Clear buffer
  :run                                - Execute buffer
  :help                               - Show this help
  :quit                               - Exit REPL
  
Enter AION code (end multi-line input with blank line):
""")
    
    def run(self):
        self.print_banner()
        
        while True:
            try:
                line = input(f"aion[{self.mode}]> ")
                
                # Handle commands
                if line.startswith(':'):
                    if not self.handle_command(line):
                        break
                    continue
                
                # Collect multi-line input
                if line.strip():
                    self.buffer.append(line)
                    # Check if we should continue collecting
                    if line.rstrip().endswith('{') or line.rstrip().endswith(':'):
                        continue
                
                # Execute when buffer has content and line is empty or complete
                if self.buffer and (not line.strip() or not line.rstrip().endswith(('{', ':'))):
                    self.execute()
                    self.buffer = []
                    
            except KeyboardInterrupt:
                print("\n(Use :quit to exit)")
            except EOFError:
                break
        
        print("\nGoodbye!")
    
    def handle_command(self, cmd: str) -> bool:
        """Handle REPL commands. Returns False to quit."""
        parts = cmd.split()
        command = parts[0].lower()
        
        if command == ':quit' or command == ':q':
            return False
        elif command == ':clear':
            self.buffer = []
            print("Buffer cleared.")
        elif command == ':run':
            if self.buffer:
                self.execute()
                self.buffer = []
            else:
                print("Buffer is empty.")
        elif command == ':mode':
            if len(parts) > 1:
                mode = parts[1].lower()
                if mode in ('run', 'transpile', 'parse', 'tokens'):
                    self.mode = mode
                    print(f"Mode set to: {mode}")
                else:
                    print("Invalid mode. Use: run, transpile, parse, tokens")
            else:
                print(f"Current mode: {self.mode}")
        elif command == ':help':
            self.print_banner()
        elif command == ':buffer':
            if self.buffer:
                print("Current buffer:")
                for i, line in enumerate(self.buffer, 1):
                    print(f"  {i}: {line}")
            else:
                print("Buffer is empty.")
        else:
            print(f"Unknown command: {command}")
        
        return True
    
    def execute(self):
        """Execute the current buffer."""
        source = '\n'.join(self.buffer)
        
        try:
            if self.mode == 'tokens':
                self.show_tokens(source)
            elif self.mode == 'parse':
                self.show_ast(source)
            elif self.mode == 'transpile':
                self.show_transpiled(source)
            else:  # run
                asyncio.run(self.run_code(source))
        except Exception as e:
            print(f"Error: {e}")
    
    def show_tokens(self, source: str):
        """Display tokens from source."""
        from src.lexer import tokenize
        
        tokens = tokenize(source)
        print("\nTokens:")
        for token in tokens:
            print(f"  {token.type.name:15} {token.value!r:20} (line {token.line})")
    
    def show_ast(self, source: str):
        """Display AST from source."""
        from src.parser import parse
        
        program = parse(source)
        print("\nAST:")
        self.print_node(program, 0)
    
    def print_node(self, node, indent: int):
        """Pretty print an AST node."""
        prefix = "  " * indent
        name = node.__class__.__name__
        
        # Get key attributes
        attrs = []
        for attr in ['name', 'value', 'goal', 'memory_type', 'operator', 'event_type']:
            if hasattr(node, attr) and getattr(node, attr) is not None:
                val = getattr(node, attr)
                if isinstance(val, str) and len(val) > 30:
                    val = val[:30] + "..."
                attrs.append(f"{attr}={val!r}")
        
        attrs_str = ", ".join(attrs)
        print(f"{prefix}{name}({attrs_str})")
        
        # Print children
        for attr in ['declarations', 'body', 'then_body', 'else_body', 'elements']:
            if hasattr(node, attr):
                children = getattr(node, attr)
                if isinstance(children, list):
                    for child in children:
                        if hasattr(child, '__class__'):
                            self.print_node(child, indent + 1)
    
    def show_transpiled(self, source: str):
        """Display transpiled Python code."""
        from src.transpiler import transpile
        
        code = transpile(source)
        print("\nGenerated Python:")
        print("-" * 40)
        print(code)
        print("-" * 40)
    
    async def run_code(self, source: str):
        """Execute AION code."""
        from src.parser import parse
        
        program = parse(source)
        await self.interpreter.interpret(program)
        
        # Show results
        if self.interpreter.output_buffer:
            print("\nOutput:")
            for line in self.interpreter.output_buffer:
                print(f"  {line}")
            self.interpreter.clear_output()
        
        # Show registered entities
        if self.interpreter.env.agents:
            print("\nRegistered agents:", list(self.interpreter.env.agents.keys()))


def main():
    repl = AionREPL()
    repl.run()


if __name__ == "__main__":
    main()
