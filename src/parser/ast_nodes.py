"""
AION Abstract Syntax Tree Nodes
Defines all AST node types for the AION language.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


class ASTNode(ABC):
    """Base class for all AST nodes."""
    
    @abstractmethod
    def accept(self, visitor: 'ASTVisitor') -> Any:
        """Accept a visitor for tree traversal."""
        pass


class ASTVisitor(ABC):
    """Base visitor class for AST traversal."""
    pass


# ============ Program Structure ============

@dataclass
class Program(ASTNode):
    """Root node of an AION program."""
    declarations: list[ASTNode] = field(default_factory=list)
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_program(self)


# ============ Declarations ============

@dataclass
class AgentDecl(ASTNode):
    """Agent declaration node."""
    name: str
    body: list[ASTNode] = field(default_factory=list)
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_agent_decl(self)


@dataclass
class GoalStmt(ASTNode):
    """Goal statement within an agent."""
    goal: str
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_goal_stmt(self)


@dataclass
class MemoryDecl(ASTNode):
    """Memory declaration."""
    memory_type: str  # working, episodic, long_term, semantic
    config: dict[str, Any] = field(default_factory=dict)
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_memory_decl(self)


@dataclass
class ModelDecl(ASTNode):
    """Model declaration."""
    name: str
    config: dict[str, Any] = field(default_factory=dict)
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_model_decl(self)


@dataclass
class ModelRef(ASTNode):
    """Reference to a model within an agent."""
    name: str
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_model_ref(self)


@dataclass
class ToolDecl(ASTNode):
    """Tool declaration."""
    name: str
    config: dict[str, Any] = field(default_factory=dict)
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_tool_decl(self)


@dataclass
class ToolRef(ASTNode):
    """Reference to a tool within an agent."""
    name: str
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_tool_ref(self)


@dataclass
class PolicyDecl(ASTNode):
    """Policy declaration."""
    name: Optional[str]
    config: dict[str, Any] = field(default_factory=dict)
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_policy_decl(self)


# ============ Event Handlers ============

@dataclass
class EventHandler(ASTNode):
    """Event handler (on input, on error, etc.)."""
    event_type: str  # input, error, timeout, complete
    params: list[str] = field(default_factory=list)
    body: list[ASTNode] = field(default_factory=list)
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_event_handler(self)


# ============ Reasoning Statements ============

@dataclass
class ThinkStmt(ASTNode):
    """Think statement - initiates reasoning."""
    prompt: Optional[str] = None
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_think_stmt(self)


@dataclass
class AnalyzeStmt(ASTNode):
    """Analyze statement - examines an expression."""
    target: 'Expression'
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_analyze_stmt(self)


@dataclass
class ReflectStmt(ASTNode):
    """Reflect statement - introspection."""
    target: Optional['Expression'] = None
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_reflect_stmt(self)


@dataclass 
class DecideStmt(ASTNode):
    """Decide statement - makes a decision."""
    target: 'Expression'
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_decide_stmt(self)


# ============ Control Flow ============

@dataclass
class IfStmt(ASTNode):
    """If/else statement."""
    condition: 'Expression'
    then_body: list[ASTNode] = field(default_factory=list)
    else_body: list[ASTNode] = field(default_factory=list)
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_if_stmt(self)


@dataclass
class WhenStmt(ASTNode):
    """When statement - conditional execution."""
    condition: 'Expression'
    body: list[ASTNode] = field(default_factory=list)
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_when_stmt(self)


@dataclass
class RepeatStmt(ASTNode):
    """Repeat statement - loop construct."""
    times: Optional['Expression'] = None
    body: list[ASTNode] = field(default_factory=list)
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_repeat_stmt(self)


# ============ Action Statements ============

@dataclass
class UseStmt(ASTNode):
    """Use statement - invoke a tool."""
    tool_name: str
    args: list['Expression'] = field(default_factory=list)
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_use_stmt(self)


@dataclass
class RespondStmt(ASTNode):
    """Respond statement - send output."""
    value: Optional['Expression'] = None
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_respond_stmt(self)


@dataclass
class EmitStmt(ASTNode):
    """Emit statement - trigger an event."""
    event_name: str
    args: list['Expression'] = field(default_factory=list)
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_emit_stmt(self)


@dataclass
class StoreStmt(ASTNode):
    """Store statement - save to memory."""
    value: 'Expression'
    memory_name: Optional[str] = None
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_store_stmt(self)


@dataclass
class RecallStmt(ASTNode):
    """Recall statement - retrieve from memory."""
    memory_name: Optional[str] = None
    condition: Optional['Expression'] = None
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_recall_stmt(self)


@dataclass
class AssignStmt(ASTNode):
    """Assignment statement."""
    name: str
    value: 'Expression'
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_assign_stmt(self)


# ============ Expressions ============

class Expression(ASTNode):
    """Base class for expressions."""
    pass


@dataclass
class BinaryExpr(Expression):
    """Binary expression (a op b)."""
    left: Expression
    operator: str
    right: Expression
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_binary_expr(self)


@dataclass
class UnaryExpr(Expression):
    """Unary expression (op a)."""
    operator: str
    operand: Expression
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_unary_expr(self)


@dataclass
class Literal(Expression):
    """Literal value (string, number, boolean, null)."""
    value: Any
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_literal(self)


@dataclass
class Identifier(Expression):
    """Variable/identifier reference."""
    name: str
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_identifier(self)


@dataclass
class MemberAccess(Expression):
    """Member access (a.b.c)."""
    object: Expression
    member: str
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_member_access(self)


@dataclass
class ListLiteral(Expression):
    """List literal [a, b, c]."""
    elements: list[Expression] = field(default_factory=list)
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_list_literal(self)


# ============ v2.0 Features: Imports/Exports ============

@dataclass
class ImportStmt(ASTNode):
    """Import statement: import path.to.module as alias"""
    module_path: str
    alias: Optional[str] = None
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_import_stmt(self)


@dataclass
class ExportDecl(ASTNode):
    """Export declaration: export agent/tool/model Name"""
    export_type: str  # 'agent', 'tool', 'model'
    target_name: str
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_export_decl(self)


# ============ v2.0 Features: Type System ============

@dataclass
class TypeDef(ASTNode):
    """Type definition: type Name = { field: Type, ... }"""
    name: str
    fields: dict[str, 'TypeAnnotation'] = field(default_factory=dict)
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_type_def(self)


@dataclass
class TypeAnnotation(ASTNode):
    """Type annotation: :: Type or :: Type[T]"""
    type_name: str
    type_params: list[str] = field(default_factory=list)
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_type_annotation(self)


# ============ v2.0 Features: Async/Await ============

@dataclass
class AsyncEventHandler(ASTNode):
    """Async event handler: async on event(params):"""
    event_type: str
    params: list[str] = field(default_factory=list)
    param_types: list[Optional['TypeAnnotation']] = field(default_factory=list)
    body: list[ASTNode] = field(default_factory=list)
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_async_event_handler(self)


@dataclass
class AwaitExpr(Expression):
    """Await expression: await expr"""
    expression: Expression
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_await_expr(self)


# ============ v2.0 Features: Pattern Matching ============

@dataclass
class MatchStmt(ASTNode):
    """Match statement: match expr: case patterns..."""
    target: Expression
    cases: list['CaseClause'] = field(default_factory=list)
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_match_stmt(self)


@dataclass
class CaseClause(ASTNode):
    """Case clause with pattern, guard, and body."""
    patterns: list[Expression] = field(default_factory=list)  # Multiple patterns with |
    guard: Optional[Expression] = None  # where condition
    binding: Optional[str] = None  # as name
    body: list[ASTNode] = field(default_factory=list)
    is_default: bool = False
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_case_clause(self)


@dataclass
class PatternExpr(Expression):
    """Pattern expression: pattern /regex/ as binding"""
    regex: str
    bindings: list[str] = field(default_factory=list)
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_pattern_expr(self)


# ============ v2.0 Features: Error Handling ============

@dataclass
class TryStmt(ASTNode):
    """Try/catch/finally statement."""
    try_body: list[ASTNode] = field(default_factory=list)
    catch_param: Optional[str] = None
    catch_body: list[ASTNode] = field(default_factory=list)
    finally_body: list[ASTNode] = field(default_factory=list)
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_try_stmt(self)


@dataclass
class RaiseStmt(ASTNode):
    """Raise statement: raise expr"""
    expression: Expression
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_raise_stmt(self)


# ============ v2.0 Features: Function Definitions ============

@dataclass
class FunctionDef(ASTNode):
    """Function definition with optional async and type annotations."""
    name: str
    params: list['ParamDef'] = field(default_factory=list)
    return_type: Optional[TypeAnnotation] = None
    body: list[ASTNode] = field(default_factory=list)
    is_async: bool = False
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_function_def(self)


@dataclass
class ParamDef(ASTNode):
    """Parameter definition: name :: Type"""
    name: str
    type_annotation: Optional[TypeAnnotation] = None
    default_value: Optional[Expression] = None
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_param_def(self)


@dataclass
class ReturnStmt(ASTNode):
    """Return statement: return expr"""
    value: Optional[Expression] = None
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_return_stmt(self)


@dataclass
class BreakStmt(ASTNode):
    """Break statement in loops."""
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_break_stmt(self)


@dataclass
class ContinueStmt(ASTNode):
    """Continue statement in loops."""
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_continue_stmt(self)


# ============ v2.0 Features: Loops ============

@dataclass
class ForStmt(ASTNode):
    """For each loop: for each item in collection:"""
    variable: str
    iterable: Expression
    body: list[ASTNode] = field(default_factory=list)
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_for_stmt(self)


# ============ v2.0 Features: Concurrency ============

@dataclass
class ParallelBlock(ASTNode):
    """Parallel block: parallel: spawn tasks..."""
    statements: list[ASTNode] = field(default_factory=list)
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_parallel_block(self)


@dataclass
class SpawnExpr(Expression):
    """Spawn expression: spawn call()"""
    expression: Expression
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_spawn_expr(self)


@dataclass
class JoinExpr(Expression):
    """Join expression: join(task1, task2, ...)"""
    tasks: list[Expression] = field(default_factory=list)
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_join_expr(self)


# ============ v2.0 Features: Pipeline & Expressions ============

@dataclass
class PipelineExpr(Expression):
    """Pipeline expression: expr |> func1 |> func2"""
    initial: Expression
    stages: list[Expression] = field(default_factory=list)
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_pipeline_expr(self)


@dataclass
class CallExpr(Expression):
    """Function/method call: func(args)"""
    callee: Expression
    args: list[Expression] = field(default_factory=list)
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_call_expr(self)


@dataclass
class MapLiteral(Expression):
    """Map/object literal: { key: value, ... }"""
    entries: list[tuple[str, Expression]] = field(default_factory=list)
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_map_literal(self)


@dataclass
class WithExpr(Expression):
    """With expression: obj with { updates }"""
    base: Expression
    updates: 'MapLiteral'
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_with_expr(self)


@dataclass
class IndexExpr(Expression):
    """Index expression: arr[index]"""
    object: Expression
    index: Expression
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_index_expr(self)


@dataclass
class StringInterpolation(Expression):
    """String with interpolation: "Hello {name}"."""
    parts: list[Any] = field(default_factory=list)  # Alternating str and Expression
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_string_interpolation(self)


# ============ v2.0 Features: Decorators ============

@dataclass
class DecoratorExpr(ASTNode):
    """Decorator: @name or @name(args)"""
    name: str
    args: list[Expression] = field(default_factory=list)
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_decorator_expr(self)


@dataclass
class DecoratedDecl(ASTNode):
    """A declaration with decorators applied."""
    decorators: list[DecoratorExpr] = field(default_factory=list)
    declaration: ASTNode = None
    
    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_decorated_decl(self)
