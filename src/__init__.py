"""
AION - Artificial Intelligence Oriented Notation
=================================================

A declarative, AI-native programming language for building thinking systems.

Features:
- Domain-specific language for AI agents
- 26 scientific domain engines
- Internet knowledge learning
- Creative thinking capabilities
- Self-awareness and meta-cognition
"""

__version__ = "4.0.0"  # Phase 9: Advanced Intelligence
__author__ = "AION Contributors"

# Core Language
from .lexer import Lexer, Token, TokenType, tokenize, LexerError
from .parser import Parser, parse, ParserError
from .parser.ast_nodes import *
from .interpreter import Interpreter, run_aion, InterpreterError
from .transpiler import transpile, CodeGenerator

# Runtime
from .runtime import (
    Environment,
    WorkingMemory, EpisodicMemory, LongTermMemory, SemanticMemory, create_memory,
    ModelRegistry, OpenAIProvider, AnthropicProvider, OllamaProvider,
    ToolRegistry, Tool, get_global_registry,
    ReasoningEngine, ReasoningTrace
)

# Multimodal
from .multimodal import (
    VisionProcessor, AudioProcessor, DocumentProcessor, ScreenProcessor,
    MultimodalMemory, VisionInput, AudioInput, DocumentInput, ScreenCapture
)

# Embodied AI
from .embodied import (
    SensorStream, SensorFusion, ActuatorController, ROS2Bridge,
    SimulationEnvironment, GenericSimulator
)

# Enterprise
from .enterprise import (
    PromptRegistry, AuditLogger, PIIDetector, PIIMasker, QuotaManager
)

# Domains (lazy import for performance)
# from .domains import PhysicsEngine, QuantumEngine, ProteinFoldingEngine, ...

# Learning System
# from .learning import ContinuousLearner, WebCrawler, NewsAggregator, ...

# Consciousness
# from .consciousness import ConsciousnessEngine, CreativeThinkingEngine, ...

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
    
    # Multimodal
    'VisionProcessor', 'AudioProcessor', 'DocumentProcessor', 'ScreenProcessor',
    'MultimodalMemory', 'VisionInput', 'AudioInput', 'DocumentInput', 'ScreenCapture',
    
    # Embodied AI
    'SensorStream', 'SensorFusion', 'ActuatorController', 'ROS2Bridge',
    'SimulationEnvironment', 'GenericSimulator',
    
    # Enterprise
    'PromptRegistry', 'AuditLogger', 'PIIDetector', 'PIIMasker', 'QuotaManager',
]
