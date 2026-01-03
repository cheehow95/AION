"""
AION - Artificial Intelligence Oriented Notation
A declarative, AI-native programming language for building thinking systems.
"""

__version__ = "0.2.0"  # Phase 3: Perception
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

# Phase 3: Perception modules
from .multimodal import (
    VisionProcessor, AudioProcessor, DocumentProcessor, ScreenProcessor,
    MultimodalMemory, VisionInput, AudioInput, DocumentInput, ScreenCapture
)
from .embodied import (
    SensorStream, SensorFusion, ActuatorController, ROS2Bridge,
    SimulationEnvironment, GenericSimulator
)
from .enterprise import (
    PromptRegistry, AuditLogger, PIIDetector, PIIMasker, QuotaManager
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
    
    # Multimodal (Phase 3)
    'VisionProcessor', 'AudioProcessor', 'DocumentProcessor', 'ScreenProcessor',
    'MultimodalMemory', 'VisionInput', 'AudioInput', 'DocumentInput', 'ScreenCapture',
    
    # Embodied AI (Phase 3)
    'SensorStream', 'SensorFusion', 'ActuatorController', 'ROS2Bridge',
    'SimulationEnvironment', 'GenericSimulator',
    
    # Enterprise (Phase 3)
    'PromptRegistry', 'AuditLogger', 'PIIDetector', 'PIIMasker', 'QuotaManager',
]

