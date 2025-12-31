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
