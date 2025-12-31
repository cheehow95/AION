"""
AION Token Types
Defines all token types for the AION language lexer.
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Any


class TokenType(Enum):
    # Literals
    STRING = auto()
    NUMBER = auto()
    BOOLEAN = auto()
    NULL = auto()
    IDENTIFIER = auto()
    
    # Keywords - Declarations
    AGENT = auto()
    MODEL = auto()
    TOOL = auto()
    POLICY = auto()
    MEMORY = auto()
    GOAL = auto()
    
    # Keywords - Memory Types
    WORKING = auto()
    EPISODIC = auto()
    LONG_TERM = auto()
    SEMANTIC = auto()
    
    # Keywords - Reasoning
    THINK = auto()
    ANALYZE = auto()
    REFLECT = auto()
    DECIDE = auto()
    
    # Keywords - Events
    ON = auto()
    INPUT = auto()
    ERROR = auto()
    TIMEOUT = auto()
    COMPLETE = auto()
    
    # Keywords - Control Flow
    IF = auto()
    ELSE = auto()
    WHEN = auto()
    REPEAT = auto()
    TIMES = auto()
    
    # Keywords - Actions
    USE = auto()
    RESPOND = auto()
    EMIT = auto()
    STORE = auto()
    RECALL = auto()
    FROM = auto()
    WHERE = auto()
    IN = auto()
    
    # Keywords - Logic
    AND = auto()
    OR = auto()
    NOT = auto()
    TRUE = auto()
    FALSE = auto()
    
    # === NEW FEATURES (v2.0) ===
    
    # Async/Await - For concurrent agent operations
    ASYNC = auto()
    AWAIT = auto()
    
    # Pattern Matching - Match on message types
    MATCH = auto()
    CASE = auto()
    DEFAULT = auto()
    
    # Imports - Module system
    IMPORT = auto()
    EXPORT = auto()
    AS = auto()
    
    # Type Hints - Optional type annotations
    TYPE = auto()
    ARROW = auto()         # ->
    
    # Concurrency
    PARALLEL = auto()
    SPAWN = auto()
    JOIN = auto()
    
    # Advanced Control Flow
    TRY = auto()
    CATCH = auto()
    FINALLY = auto()
    RETURN = auto()
    BREAK = auto()
    CONTINUE = auto()
    FOR = auto()
    EACH = auto()
    
    # Operators
    PLUS = auto()          # +
    MINUS = auto()         # -
    STAR = auto()          # *
    SLASH = auto()         # /
    EQ = auto()            # =
    EQEQ = auto()          # ==
    NEQ = auto()           # !=
    LT = auto()            # <
    GT = auto()            # >
    LTE = auto()           # <=
    GTE = auto()           # >=
    PIPE = auto()          # |> (pipeline operator)
    AT = auto()            # @ (decorator)
    QUESTION = auto()      # ? (optional chaining)
    DOUBLECOLON = auto()   # :: (type annotation)
    
    # Delimiters
    LPAREN = auto()        # (
    RPAREN = auto()        # )
    LBRACE = auto()        # {
    RBRACE = auto()        # }
    LBRACKET = auto()      # [
    RBRACKET = auto()      # ]
    COMMA = auto()         # ,
    COLON = auto()         # :
    DOT = auto()           # .
    
    # Special
    NEWLINE = auto()
    INDENT = auto()
    DEDENT = auto()
    COMMENT = auto()
    EOF = auto()


@dataclass
class Token:
    """Represents a single token in the AION source code."""
    type: TokenType
    value: Any
    line: int
    column: int
    
    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, line={self.line}, col={self.column})"


# Keyword mapping
KEYWORDS = {
    # Declarations
    'agent': TokenType.AGENT,
    'model': TokenType.MODEL,
    'tool': TokenType.TOOL,
    'policy': TokenType.POLICY,
    'memory': TokenType.MEMORY,
    'goal': TokenType.GOAL,
    
    # Memory types
    'working': TokenType.WORKING,
    'episodic': TokenType.EPISODIC,
    'long_term': TokenType.LONG_TERM,
    'semantic': TokenType.SEMANTIC,
    
    # Reasoning
    'think': TokenType.THINK,
    'analyze': TokenType.ANALYZE,
    'reflect': TokenType.REFLECT,
    'decide': TokenType.DECIDE,
    
    # Events
    'on': TokenType.ON,
    'input': TokenType.INPUT,
    'error': TokenType.ERROR,
    'timeout': TokenType.TIMEOUT,
    'complete': TokenType.COMPLETE,
    
    # Control flow
    'if': TokenType.IF,
    'else': TokenType.ELSE,
    'when': TokenType.WHEN,
    'repeat': TokenType.REPEAT,
    'times': TokenType.TIMES,
    
    # Actions
    'use': TokenType.USE,
    'respond': TokenType.RESPOND,
    'emit': TokenType.EMIT,
    'store': TokenType.STORE,
    'recall': TokenType.RECALL,
    'from': TokenType.FROM,
    'where': TokenType.WHERE,
    'in': TokenType.IN,
    
    # Logic
    'and': TokenType.AND,
    'or': TokenType.OR,
    'not': TokenType.NOT,
    'true': TokenType.TRUE,
    'false': TokenType.FALSE,
    'null': TokenType.NULL,
    
    # === NEW v2.0 KEYWORDS ===
    
    # Async/Await
    'async': TokenType.ASYNC,
    'await': TokenType.AWAIT,
    
    # Pattern Matching
    'match': TokenType.MATCH,
    'case': TokenType.CASE,
    'default': TokenType.DEFAULT,
    
    # Imports
    'import': TokenType.IMPORT,
    'export': TokenType.EXPORT,
    'as': TokenType.AS,
    
    # Type system
    'type': TokenType.TYPE,
    
    # Concurrency
    'parallel': TokenType.PARALLEL,
    'spawn': TokenType.SPAWN,
    'join': TokenType.JOIN,
    
    # Advanced control flow
    'try': TokenType.TRY,
    'catch': TokenType.CATCH,
    'finally': TokenType.FINALLY,
    'return': TokenType.RETURN,
    'break': TokenType.BREAK,
    'continue': TokenType.CONTINUE,
    'for': TokenType.FOR,
    'each': TokenType.EACH,
}

# Operator mapping for multi-character operators
OPERATORS = {
    '|>': TokenType.PIPE,
    '->': TokenType.ARROW,
    '::': TokenType.DOUBLECOLON,
    '==': TokenType.EQEQ,
    '!=': TokenType.NEQ,
    '<=': TokenType.LTE,
    '>=': TokenType.GTE,
}
