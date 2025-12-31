"""
AION - Artificial Intelligence Oriented Notation
A declarative, AI-native programming language for building thinking systems.
"""

__version__ = "0.1.0"
__author__ = "AION Contributors"

from .lexer import Lexer, Token, TokenType, tokenize, LexerError
from .parser import Parser, parse, ParserError
from .parser.ast_nodes import *
from .interpreter import Interpreter, run_aion, InterpreterError
from .transpiler import transpile, CodeGenerator
from .runtime import (
    Environment,
    WorkingMemory, EpisodicMemory, LongTermMemory, SemanticMemory, create_memory,
    ModelRegistry, OpenAIProvider, AnthropicProvider, OllamaProvider,
    ToolRegistry, Tool, get_global_registry,
    ReasoningEngine, ReasoningTrace
)

__all__ = [
    # Version
    '__version__',
    
    # Lexer
    'Lexer', 'Token', 'TokenType', 'tokenize', 'LexerError',
    
    # Parser
    'Parser', 'parse', 'ParserError',
    
    # Interpreter
    'Interpreter', 'run_aion', 'InterpreterError',
    
    # Transpiler
    'transpile', 'CodeGenerator',
    
    # Runtime
    'Environment',
    'WorkingMemory', 'EpisodicMemory', 'LongTermMemory', 'SemanticMemory', 'create_memory',
    'ModelRegistry', 'OpenAIProvider', 'AnthropicProvider', 'OllamaProvider',
    'ToolRegistry', 'Tool', 'get_global_registry',
    'ReasoningEngine', 'ReasoningTrace',
]
