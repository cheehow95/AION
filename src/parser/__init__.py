"""AION Parser Package"""
from .ast_nodes import (
    # Core
    ASTNode, ASTVisitor, Program,
    # Declarations
    AgentDecl, GoalStmt, MemoryDecl, ModelDecl, ModelRef, ToolDecl, ToolRef, PolicyDecl,
    # Event Handlers
    EventHandler,
    # Reasoning Statements
    ThinkStmt, AnalyzeStmt, ReflectStmt, DecideStmt,
    # Control Flow (v1)
    IfStmt, WhenStmt, RepeatStmt,
    # Action Statements
    UseStmt, RespondStmt, EmitStmt, StoreStmt, RecallStmt, AssignStmt,
    # Expressions (v1)
    Expression, BinaryExpr, UnaryExpr, Literal, Identifier, MemberAccess, ListLiteral,
    # v2.0: Imports/Exports
    ImportStmt, ExportDecl,
    # v2.0: Type System
    TypeDef, TypeAnnotation,
    # v2.0: Async/Await
    AsyncEventHandler, AwaitExpr,
    # v2.0: Pattern Matching
    MatchStmt, CaseClause, PatternExpr,
    # v2.0: Error Handling
    TryStmt, RaiseStmt,
    # v2.0: Function Definitions
    FunctionDef, ParamDef, ReturnStmt, BreakStmt, ContinueStmt,
    # v2.0: Loops
    ForStmt,
    # v2.0: Concurrency
    ParallelBlock, SpawnExpr, JoinExpr,
    # v2.0: Pipeline & Expressions
    PipelineExpr, CallExpr, MapLiteral, WithExpr, IndexExpr, StringInterpolation,
    # v2.0: Decorators
    DecoratorExpr, DecoratedDecl
)
from .parser import Parser, ParserError, parse

__all__ = [
    # Core
    'ASTNode', 'ASTVisitor', 'Program',
    # Declarations
    'AgentDecl', 'GoalStmt', 'MemoryDecl', 'ModelDecl', 'ModelRef', 'ToolDecl', 'ToolRef', 'PolicyDecl',
    # Event Handlers
    'EventHandler',
    # Reasoning Statements
    'ThinkStmt', 'AnalyzeStmt', 'ReflectStmt', 'DecideStmt',
    # Control Flow (v1)
    'IfStmt', 'WhenStmt', 'RepeatStmt',
    # Action Statements
    'UseStmt', 'RespondStmt', 'EmitStmt', 'StoreStmt', 'RecallStmt', 'AssignStmt',
    # Expressions (v1)
    'Expression', 'BinaryExpr', 'UnaryExpr', 'Literal', 'Identifier', 'MemberAccess', 'ListLiteral',
    # v2.0: Imports/Exports
    'ImportStmt', 'ExportDecl',
    # v2.0: Type System
    'TypeDef', 'TypeAnnotation',
    # v2.0: Async/Await
    'AsyncEventHandler', 'AwaitExpr',
    # v2.0: Pattern Matching
    'MatchStmt', 'CaseClause', 'PatternExpr',
    # v2.0: Error Handling
    'TryStmt', 'RaiseStmt',
    # v2.0: Function Definitions
    'FunctionDef', 'ParamDef', 'ReturnStmt', 'BreakStmt', 'ContinueStmt',
    # v2.0: Loops
    'ForStmt',
    # v2.0: Concurrency
    'ParallelBlock', 'SpawnExpr', 'JoinExpr',
    # v2.0: Pipeline & Expressions
    'PipelineExpr', 'CallExpr', 'MapLiteral', 'WithExpr', 'IndexExpr', 'StringInterpolation',
    # v2.0: Decorators
    'DecoratorExpr', 'DecoratedDecl',
    # Parser
    'Parser', 'ParserError', 'parse'
]
