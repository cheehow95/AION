"""
AION Interpreter
Executes AION AST nodes in the runtime environment.
"""

import asyncio
from typing import Any, Optional
from ..parser import (
    ASTNode, ASTVisitor, Program,
    AgentDecl, GoalStmt, MemoryDecl, ModelDecl, ModelRef, ToolDecl, ToolRef, PolicyDecl,
    EventHandler,
    ThinkStmt, AnalyzeStmt, ReflectStmt, DecideStmt,
    IfStmt, WhenStmt, RepeatStmt,
    UseStmt, RespondStmt, EmitStmt, StoreStmt, RecallStmt, AssignStmt,
    Expression, BinaryExpr, UnaryExpr, Literal, Identifier, MemberAccess, ListLiteral
)
from ..runtime import (
    Environment, AgentInstance, MemoryInstance,
    create_memory, ModelRegistry, ToolRegistry, get_global_registry, create_builtin_tools,
    ReasoningEngine
)


class InterpreterError(Exception):
    """Raised when interpretation fails."""
    pass


class Interpreter(ASTVisitor):
    """
    Interprets AION programs by visiting AST nodes.
    """
    
    def __init__(self, env: Environment = None):
        self.env = env or Environment()
        self.tool_registry = get_global_registry()
        self.model_registry = ModelRegistry()
        self.reasoning_engine = ReasoningEngine()
        self.current_agent: Optional[AgentInstance] = None
        self.output_buffer: list[str] = []
        
        # Register built-in tools
        create_builtin_tools(self.tool_registry)
    
    async def interpret(self, program: Program) -> Any:
        """Interpret a complete program."""
        result = None
        for decl in program.declarations:
            result = await self.visit(decl)
        return result
    
    async def visit(self, node: ASTNode) -> Any:
        """Visit a node and dispatch to appropriate method."""
        method_name = f'visit_{node.__class__.__name__}'
        method = getattr(self, method_name, self.generic_visit)
        return await method(node)
    
    async def generic_visit(self, node: ASTNode) -> Any:
        """Default visit method."""
        raise InterpreterError(f"No visit method for {node.__class__.__name__}")
    
    # ============ Program & Declarations ============
    
    async def visit_Program(self, node: Program) -> Any:
        """Visit program node."""
        return await self.interpret(node)
    
    async def visit_AgentDecl(self, node: AgentDecl) -> AgentInstance:
        """Visit agent declaration."""
        agent = AgentInstance(name=node.name)
        
        # Create agent scope
        agent_env = self.env.child_scope()
        self.env.register_agent(node.name, agent)
        
        # Process agent body
        old_agent = self.current_agent
        self.current_agent = agent
        
        for member in node.body:
            await self.visit_agent_member(member, agent, agent_env)
        
        self.current_agent = old_agent
        return agent
    
    async def visit_agent_member(
        self,
        member: ASTNode,
        agent: AgentInstance,
        env: Environment
    ) -> None:
        """Process an agent body member."""
        if isinstance(member, GoalStmt):
            agent.goal = member.goal
        elif isinstance(member, MemoryDecl):
            memory = create_memory(member.memory_type, config=member.config)
            env.register_memory(member.memory_type, memory)
            agent.memories.append(member.memory_type)
        elif isinstance(member, ModelRef):
            agent.model = member.name
        elif isinstance(member, ToolRef):
            agent.tools.append(member.name)
        elif isinstance(member, PolicyDecl):
            agent.policy.update(member.config)
        elif isinstance(member, EventHandler):
            agent.event_handlers[member.event_type] = member
    
    async def visit_ModelDecl(self, node: ModelDecl) -> Any:
        """Visit model declaration."""
        provider = node.config.get('provider', 'openai')
        model = self.model_registry.create_model(
            node.name,
            provider=provider,
            config=node.config
        )
        self.env.register_model(node.name, model)
        return model
    
    async def visit_ToolDecl(self, node: ToolDecl) -> Any:
        """Visit tool declaration."""
        # Tools are registered at declaration time
        # Actual handler would be provided by the host environment
        return None
    
    async def visit_PolicyDecl(self, node: PolicyDecl) -> dict:
        """Visit policy declaration."""
        # Apply global policies
        for key, value in node.config.items():
            self.tool_registry.set_policy(key, value)
        return node.config
    
    # ============ Reasoning Statements ============
    
    async def visit_ThinkStmt(self, node: ThinkStmt) -> str:
        """Execute think statement."""
        context = self._build_context()
        result = await self.reasoning_engine.think(node.prompt, context)
        return result
    
    async def visit_AnalyzeStmt(self, node: AnalyzeStmt) -> dict:
        """Execute analyze statement."""
        target = await self.evaluate(node.target)
        context = self._build_context()
        result = await self.reasoning_engine.analyze(target, context)
        return result
    
    async def visit_ReflectStmt(self, node: ReflectStmt) -> str:
        """Execute reflect statement."""
        target = None
        if node.target:
            target = await self.evaluate(node.target)
        context = self._build_context()
        result = await self.reasoning_engine.reflect(target, context)
        return result
    
    async def visit_DecideStmt(self, node: DecideStmt) -> dict:
        """Execute decide statement."""
        target = await self.evaluate(node.target)
        context = self._build_context()
        result = await self.reasoning_engine.decide(target, context)
        return result
    
    # ============ Control Flow ============
    
    async def visit_IfStmt(self, node: IfStmt) -> Any:
        """Execute if statement."""
        condition = await self.evaluate(node.condition)
        
        if self._is_truthy(condition):
            return await self.execute_block(node.then_body)
        elif node.else_body:
            return await self.execute_block(node.else_body)
        
        return None
    
    async def visit_WhenStmt(self, node: WhenStmt) -> Any:
        """Execute when statement."""
        condition = await self.evaluate(node.condition)
        
        if self._is_truthy(condition):
            return await self.execute_block(node.body)
        
        return None
    
    async def visit_RepeatStmt(self, node: RepeatStmt) -> Any:
        """Execute repeat statement."""
        times = 1
        if node.times:
            times = int(await self.evaluate(node.times))
        
        result = None
        for _ in range(times):
            result = await self.execute_block(node.body)
        
        return result
    
    # ============ Action Statements ============
    
    async def visit_UseStmt(self, node: UseStmt) -> Any:
        """Execute use statement (tool invocation)."""
        args = []
        for arg in node.args:
            args.append(await self.evaluate(arg))
        
        context = {'agent': self.current_agent.name} if self.current_agent else {}
        result = await self.tool_registry.execute(
            node.tool_name,
            *args,
            agent_context=context
        )
        
        if result.success:
            return result.data
        else:
            raise InterpreterError(f"Tool {node.tool_name} failed: {result.error}")
    
    async def visit_RespondStmt(self, node: RespondStmt) -> str:
        """Execute respond statement."""
        value = ""
        if node.value:
            value = str(await self.evaluate(node.value))
        
        self.output_buffer.append(value)
        return value
    
    async def visit_EmitStmt(self, node: EmitStmt) -> None:
        """Execute emit statement (event emission)."""
        args = []
        for arg in node.args:
            args.append(await self.evaluate(arg))
        
        # In a full implementation, this would trigger event handlers
        pass
    
    async def visit_StoreStmt(self, node: StoreStmt) -> None:
        """Execute store statement."""
        value = await self.evaluate(node.value)
        
        memory_name = node.memory_name or 'working'
        try:
            memory = self.env.get_memory(memory_name)
            memory.store(value)
        except Exception:
            # Create memory if it doesn't exist
            memory = create_memory('working', memory_name)
            self.env.register_memory(memory_name, memory)
            memory.store(value)
    
    async def visit_RecallStmt(self, node: RecallStmt) -> list:
        """Execute recall statement."""
        memory_name = node.memory_name or 'working'
        
        try:
            memory = self.env.get_memory(memory_name)
        except Exception:
            return []
        
        query = None
        if node.condition:
            query = await self.evaluate(node.condition)
        
        return memory.recall(query)
    
    async def visit_AssignStmt(self, node: AssignStmt) -> Any:
        """Execute assignment statement."""
        value = await self.evaluate(node.value)
        
        if self.env.exists(node.name):
            self.env.set(node.name, value)
        else:
            self.env.define(node.name, value)
        
        return value
    
    # ============ Expression Evaluation ============
    
    async def evaluate(self, expr: Expression) -> Any:
        """Evaluate an expression."""
        if isinstance(expr, Literal):
            return expr.value
        elif isinstance(expr, Identifier):
            return self.env.get(expr.name)
        elif isinstance(expr, BinaryExpr):
            return await self.eval_binary(expr)
        elif isinstance(expr, UnaryExpr):
            return await self.eval_unary(expr)
        elif isinstance(expr, MemberAccess):
            return await self.eval_member_access(expr)
        elif isinstance(expr, ListLiteral):
            return [await self.evaluate(e) for e in expr.elements]
        else:
            raise InterpreterError(f"Cannot evaluate: {type(expr)}")
    
    async def eval_binary(self, expr: BinaryExpr) -> Any:
        """Evaluate binary expression."""
        left = await self.evaluate(expr.left)
        
        # Short-circuit evaluation for logical operators
        if expr.operator == 'and':
            if not self._is_truthy(left):
                return False
            return self._is_truthy(await self.evaluate(expr.right))
        elif expr.operator == 'or':
            if self._is_truthy(left):
                return True
            return self._is_truthy(await self.evaluate(expr.right))
        
        right = await self.evaluate(expr.right)
        
        ops = {
            '+': lambda a, b: a + b,
            '-': lambda a, b: a - b,
            '*': lambda a, b: a * b,
            '/': lambda a, b: a / b,
            '==': lambda a, b: a == b,
            '!=': lambda a, b: a != b,
            '<': lambda a, b: a < b,
            '>': lambda a, b: a > b,
            '<=': lambda a, b: a <= b,
            '>=': lambda a, b: a >= b,
        }
        
        if expr.operator not in ops:
            raise InterpreterError(f"Unknown operator: {expr.operator}")
        
        return ops[expr.operator](left, right)
    
    async def eval_unary(self, expr: UnaryExpr) -> Any:
        """Evaluate unary expression."""
        operand = await self.evaluate(expr.operand)
        
        if expr.operator == 'not':
            return not self._is_truthy(operand)
        elif expr.operator == '-':
            return -operand
        
        raise InterpreterError(f"Unknown unary operator: {expr.operator}")
    
    async def eval_member_access(self, expr: MemberAccess) -> Any:
        """Evaluate member access expression."""
        obj = await self.evaluate(expr.object)
        
        if isinstance(obj, dict):
            return obj.get(expr.member)
        elif hasattr(obj, expr.member):
            return getattr(obj, expr.member)
        
        raise InterpreterError(f"Cannot access member {expr.member}")
    
    # ============ Helper Methods ============
    
    async def execute_block(self, statements: list[ASTNode]) -> Any:
        """Execute a block of statements."""
        result = None
        for stmt in statements:
            result = await self.visit(stmt)
        return result
    
    def _is_truthy(self, value: Any) -> bool:
        """Check if a value is truthy."""
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return len(value) > 0
        if isinstance(value, (list, dict)):
            return len(value) > 0
        return True
    
    def _build_context(self) -> dict:
        """Build context for reasoning operations."""
        context = {}
        
        if self.current_agent:
            context['agent_name'] = self.current_agent.name
            context['goal'] = self.current_agent.goal
        
        return context
    
    async def run_agent(self, agent_name: str, input_data: Any = None) -> Any:
        """Run an agent with input."""
        agent = self.env.get_agent(agent_name)
        self.current_agent = agent
        
        # Start reasoning trace
        self.reasoning_engine.start_trace()
        
        # Execute input handler if exists
        if 'input' in agent.event_handlers:
            handler = agent.event_handlers['input']
            
            # Create handler scope with parameters
            handler_env = self.env.child_scope()
            if handler.params and input_data is not None:
                handler_env.define(handler.params[0], input_data)
            
            old_env = self.env
            self.env = handler_env
            
            try:
                await self.execute_block(handler.body)
            finally:
                self.env = old_env
        
        # End trace and get result
        trace = self.reasoning_engine.end_trace()
        
        self.current_agent = None
        
        return {
            'output': self.output_buffer.copy(),
            'trace': trace.to_dict() if trace else None
        }
    
    def get_output(self) -> list[str]:
        """Get accumulated output."""
        return self.output_buffer.copy()
    
    def clear_output(self) -> None:
        """Clear output buffer."""
        self.output_buffer.clear()


async def run_aion(source: str, input_data: Any = None) -> dict:
    """
    Convenience function to run AION source code.
    
    Args:
        source: AION source code
        input_data: Optional input to pass to agents
    
    Returns:
        Execution results
    """
    from ..parser import parse
    
    program = parse(source)
    interpreter = Interpreter()
    await interpreter.interpret(program)
    
    # Find and run first agent
    for name in interpreter.env.agents:
        return await interpreter.run_agent(name, input_data)
    
    return {'output': interpreter.get_output()}
