"""
AION Interpreter v2.0
Executes AION AST nodes in the runtime environment.
Supports all v2.0 language features.
"""

import asyncio
import re
from typing import Any, Optional, Callable
from ..parser import (
    # Core
    ASTNode, ASTVisitor, Program,
    # Declarations
    AgentDecl, GoalStmt, MemoryDecl, ModelDecl, ModelRef, ToolDecl, ToolRef, PolicyDecl,
    EventHandler,
    # Reasoning
    ThinkStmt, AnalyzeStmt, ReflectStmt, DecideStmt,
    # Control Flow v1
    IfStmt, WhenStmt, RepeatStmt,
    # Actions
    UseStmt, RespondStmt, EmitStmt, StoreStmt, RecallStmt, AssignStmt,
    # Expressions v1
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
from ..runtime import (
    Environment, AgentInstance, MemoryInstance,
    create_memory, ModelRegistry, ToolRegistry, get_global_registry, create_builtin_tools,
    ReasoningEngine
)


class InterpreterError(Exception):
    """Raised when interpretation fails."""
    pass


class ReturnValue(Exception):
    """Used to propagate return values through the call stack."""
    def __init__(self, value: Any = None):
        self.value = value


class BreakLoop(Exception):
    """Used to break out of loops."""
    pass


class ContinueLoop(Exception):
    """Used to continue to next iteration."""
    pass


class Interpreter(ASTVisitor):
    """
    Interprets AION programs by visiting AST nodes.
    Supports all v2.0 language features.
    """
    
    def __init__(self, env: Environment = None):
        self.env = env or Environment()
        self.tool_registry = get_global_registry()
        self.model_registry = ModelRegistry()
        self.reasoning_engine = ReasoningEngine()
        self.current_agent: Optional[AgentInstance] = None
        self.output_buffer: list[str] = []
        
        # v2.0: Module system
        self.modules: dict[str, Any] = {}
        self.exports: dict[str, Any] = {}
        
        # v2.0: Type definitions
        self.types: dict[str, TypeDef] = {}
        
        # v2.0: Function definitions
        self.functions: dict[str, FunctionDef] = {}
        
        # v2.0: Decorators registry
        self.decorators: dict[str, Callable] = self._create_builtin_decorators()
        
        # Register built-in tools
        create_builtin_tools(self.tool_registry)
    
    def _create_builtin_decorators(self) -> dict[str, Callable]:
        """Create built-in decorators."""
        return {
            'logged': self._decorator_logged,
            'cached': self._decorator_cached,
            'rate_limited': self._decorator_rate_limited,
        }
    
    def _decorator_logged(self, func: Callable, args: list) -> Callable:
        """Logging decorator."""
        async def wrapper(*call_args, **call_kwargs):
            print(f"[LOG] Calling {func.__name__ if hasattr(func, '__name__') else 'function'}")
            result = await func(*call_args, **call_kwargs)
            print(f"[LOG] Result: {result}")
            return result
        return wrapper
    
    def _decorator_cached(self, func: Callable, args: list) -> Callable:
        """Caching decorator."""
        cache = {}
        async def wrapper(*call_args, **call_kwargs):
            key = str(call_args)
            if key not in cache:
                cache[key] = await func(*call_args, **call_kwargs)
            return cache[key]
        return wrapper
    
    def _decorator_rate_limited(self, func: Callable, args: list) -> Callable:
        """Rate limiting decorator."""
        # Extract rate limit config from args
        max_calls = 100
        per_minute = 1
        for arg in args:
            if hasattr(arg, 'left') and hasattr(arg.left, 'name'):
                if arg.left.name == 'max_calls':
                    max_calls = arg.right.value if hasattr(arg.right, 'value') else 100
        return func  # Simplified - actual rate limiting would track calls
    
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
        elif isinstance(member, ModelDecl):
            # Inline model declaration with config
            agent.model = member.name
            await self.visit_ModelDecl(member)
        elif isinstance(member, ToolRef):
            agent.tools.append(member.name)
        elif isinstance(member, PolicyDecl):
            agent.policy.update(member.config)
        elif isinstance(member, EventHandler):
            agent.event_handlers[member.event_type] = member
        elif isinstance(member, AsyncEventHandler):
            agent.event_handlers[member.event_type] = member
        elif isinstance(member, FunctionDef):
            # Register function in agent scope
            self.functions[member.name] = member
    
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
        return None
    
    async def visit_PolicyDecl(self, node: PolicyDecl) -> dict:
        """Visit policy declaration."""
        for key, value in node.config.items():
            self.tool_registry.set_policy(key, value)
        return node.config
    
    # ============ v2.0: Imports/Exports ============
    
    async def visit_ImportStmt(self, node: ImportStmt) -> Any:
        """Execute import statement."""
        # Simplified module loading
        alias = node.alias or node.module_path.split('.')[-1]
        self.modules[alias] = {'path': node.module_path}
        self.env.define(alias, self.modules[alias])
        return self.modules[alias]
    
    async def visit_ExportDecl(self, node: ExportDecl) -> Any:
        """Execute export declaration."""
        self.exports[node.target_name] = {
            'type': node.export_type,
            'name': node.target_name
        }
        return self.exports[node.target_name]
    
    # ============ v2.0: Type System ============
    
    async def visit_TypeDef(self, node: TypeDef) -> Any:
        """Register type definition."""
        self.types[node.name] = node
        return node
    
    async def visit_TypeAnnotation(self, node: TypeAnnotation) -> Any:
        """Process type annotation (validation)."""
        # Type annotations are metadata, return the type info
        return {'type': node.type_name, 'params': node.type_params}
    
    # ============ v2.0: Decorators ============
    
    async def visit_DecoratedDecl(self, node: DecoratedDecl) -> Any:
        """Process decorated declaration."""
        # First process the underlying declaration
        result = await self.visit(node.declaration)
        
        # Apply decorators in reverse order (innermost first)
        for decorator in reversed(node.decorators):
            if decorator.name in self.decorators:
                decorator_func = self.decorators[decorator.name]
                result = decorator_func(result, decorator.args)
        
        return result
    
    # ============ v2.0: Async Event Handler ============
    
    async def visit_AsyncEventHandler(self, node: AsyncEventHandler) -> Any:
        """Register async event handler."""
        # Handlers are stored, not executed during declaration
        return node
    
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
            try:
                result = await self.execute_block(node.body)
            except BreakLoop:
                break
            except ContinueLoop:
                continue
        
        return result
    
    # ============ v2.0: Pattern Matching ============
    
    async def visit_MatchStmt(self, node: MatchStmt) -> Any:
        """Execute match statement."""
        target_value = await self.evaluate(node.target)
        
        for case in node.cases:
            if case.is_default:
                return await self.execute_block(case.body)
            
            for pattern in case.patterns:
                matched, bindings = await self._match_pattern(target_value, pattern)
                
                if matched:
                    # Check guard condition if present
                    if case.guard:
                        # Create scope with bindings
                        old_env = self.env
                        self.env = self.env.child_scope()
                        for name, value in bindings.items():
                            self.env.define(name, value)
                        
                        guard_result = await self.evaluate(case.guard)
                        self.env = old_env
                        
                        if not self._is_truthy(guard_result):
                            continue
                    
                    # Execute case body with bindings
                    old_env = self.env
                    self.env = self.env.child_scope()
                    
                    for name, value in bindings.items():
                        self.env.define(name, value)
                    
                    if case.binding:
                        if isinstance(case.binding, tuple):
                            for i, name in enumerate(case.binding):
                                if i < len(bindings):
                                    self.env.define(name, list(bindings.values())[i])
                        else:
                            self.env.define(case.binding, target_value)
                    
                    try:
                        result = await self.execute_block(case.body)
                        return result
                    finally:
                        self.env = old_env
        
        return None
    
    async def _match_pattern(self, value: Any, pattern: Expression) -> tuple[bool, dict]:
        """Match a value against a pattern. Returns (matched, bindings)."""
        bindings = {}
        
        if isinstance(pattern, Literal):
            return (value == pattern.value, bindings)
        
        if isinstance(pattern, Identifier):
            if pattern.name == '_':
                return (True, {})  # Wildcard
            # Identifier patterns bind the value
            return (True, {pattern.name: value})
        
        if isinstance(pattern, PatternExpr):
            # Regex pattern matching
            match = re.match(pattern.regex, str(value))
            if match:
                for i, binding in enumerate(pattern.bindings):
                    if i < len(match.groups()):
                        bindings[binding] = match.group(i + 1)
                return (True, bindings)
            return (False, {})
        
        # For other patterns, evaluate and compare
        pattern_value = await self.evaluate(pattern)
        return (value == pattern_value, bindings)
    
    # ============ v2.0: Error Handling ============
    
    async def visit_TryStmt(self, node: TryStmt) -> Any:
        """Execute try/catch/finally statement."""
        try:
            result = await self.execute_block(node.try_body)
        except Exception as e:
            if node.catch_body:
                old_env = self.env
                self.env = self.env.child_scope()
                
                if node.catch_param:
                    self.env.define(node.catch_param, str(e))
                
                try:
                    result = await self.execute_block(node.catch_body)
                finally:
                    self.env = old_env
            else:
                raise
        finally:
            if node.finally_body:
                await self.execute_block(node.finally_body)
        
        return result
    
    async def visit_RaiseStmt(self, node: RaiseStmt) -> None:
        """Execute raise statement."""
        error = await self.evaluate(node.expression)
        raise InterpreterError(str(error))
    
    # ============ v2.0: Loops ============
    
    async def visit_ForStmt(self, node: ForStmt) -> Any:
        """Execute for each loop."""
        iterable = await self.evaluate(node.iterable)
        
        result = None
        for item in iterable:
            old_env = self.env
            self.env = self.env.child_scope()
            self.env.define(node.variable, item)
            
            try:
                result = await self.execute_block(node.body)
            except BreakLoop:
                self.env = old_env
                break
            except ContinueLoop:
                pass
            finally:
                self.env = old_env
        
        return result
    
    async def visit_ReturnStmt(self, node: ReturnStmt) -> None:
        """Execute return statement."""
        value = None
        if node.value:
            value = await self.evaluate(node.value)
        raise ReturnValue(value)
    
    async def visit_BreakStmt(self, node: BreakStmt) -> None:
        """Execute break statement."""
        raise BreakLoop()
    
    async def visit_ContinueStmt(self, node: ContinueStmt) -> None:
        """Execute continue statement."""
        raise ContinueLoop()
    
    # ============ v2.0: Concurrency ============
    
    async def visit_ParallelBlock(self, node: ParallelBlock) -> list:
        """Execute parallel block - run all statements concurrently."""
        tasks = []
        for stmt in node.statements:
            if isinstance(stmt, AssignStmt) and isinstance(stmt.value, SpawnExpr):
                # Create task for spawn expression
                task = asyncio.create_task(self.evaluate(stmt.value.expression))
                self.env.define(stmt.name, task)
                tasks.append(task)
            else:
                # Execute other statements immediately
                await self.visit(stmt)
        
        return tasks
    
    async def visit_SpawnExpr(self, node: SpawnExpr) -> asyncio.Task:
        """Evaluate spawn expression - create async task."""
        return asyncio.create_task(self.evaluate(node.expression))
    
    async def visit_JoinExpr(self, node: JoinExpr) -> list:
        """Evaluate join expression - wait for all tasks."""
        tasks = []
        for task_expr in node.tasks:
            task = await self.evaluate(task_expr)
            if isinstance(task, asyncio.Task):
                tasks.append(task)
        
        if tasks:
            results = await asyncio.gather(*tasks)
            return list(results)
        return []
    
    async def visit_AwaitExpr(self, node: AwaitExpr) -> Any:
        """Evaluate await expression."""
        value = await self.evaluate(node.expression)
        if asyncio.iscoroutine(value) or isinstance(value, asyncio.Task):
            return await value
        return value
    
    # ============ v2.0: Function Definitions ============
    
    async def visit_FunctionDef(self, node: FunctionDef) -> Callable:
        """Register function definition."""
        self.functions[node.name] = node
        
        async def function_wrapper(*args):
            return await self._call_function(node, list(args))
        
        self.env.define(node.name, function_wrapper)
        return function_wrapper
    
    async def _call_function(self, func: FunctionDef, args: list) -> Any:
        """Call a function with arguments."""
        old_env = self.env
        self.env = self.env.child_scope()
        
        # Bind parameters
        for i, param in enumerate(func.params):
            if i < len(args):
                self.env.define(param.name, args[i])
            elif param.default_value:
                default = await self.evaluate(param.default_value)
                self.env.define(param.name, default)
        
        try:
            await self.execute_block(func.body)
            return None
        except ReturnValue as rv:
            return rv.value
        finally:
            self.env = old_env
    
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
        # v2.0 expressions
        elif isinstance(expr, CallExpr):
            return await self.eval_call(expr)
        elif isinstance(expr, MapLiteral):
            return await self.eval_map_literal(expr)
        elif isinstance(expr, IndexExpr):
            return await self.eval_index(expr)
        elif isinstance(expr, PipelineExpr):
            return await self.eval_pipeline(expr)
        elif isinstance(expr, AwaitExpr):
            return await self.visit_AwaitExpr(expr)
        elif isinstance(expr, SpawnExpr):
            return await self.visit_SpawnExpr(expr)
        elif isinstance(expr, JoinExpr):
            return await self.visit_JoinExpr(expr)
        elif isinstance(expr, WithExpr):
            return await self.eval_with(expr)
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
    
    async def eval_call(self, expr: CallExpr) -> Any:
        """Evaluate function call."""
        callee = await self.evaluate(expr.callee)
        args = [await self.evaluate(arg) for arg in expr.args]
        
        if callable(callee):
            result = callee(*args)
            if asyncio.iscoroutine(result):
                return await result
            return result
        elif isinstance(callee, FunctionDef):
            return await self._call_function(callee, args)
        
        raise InterpreterError(f"Cannot call non-function: {type(callee)}")
    
    async def eval_map_literal(self, expr: MapLiteral) -> dict:
        """Evaluate map literal."""
        result = {}
        for key, value_expr in expr.entries:
            result[key] = await self.evaluate(value_expr)
        return result
    
    async def eval_index(self, expr: IndexExpr) -> Any:
        """Evaluate index expression."""
        obj = await self.evaluate(expr.object)
        index = await self.evaluate(expr.index)
        
        if isinstance(obj, (list, tuple, str)):
            return obj[int(index)]
        elif isinstance(obj, dict):
            return obj.get(index)
        
        raise InterpreterError(f"Cannot index into {type(obj)}")
    
    async def eval_pipeline(self, expr: PipelineExpr) -> Any:
        """Evaluate pipeline expression: value |> func1 |> func2"""
        value = await self.evaluate(expr.initial)
        
        for stage in expr.stages:
            if isinstance(stage, Identifier):
                # Simple function call
                func = self.env.get(stage.name)
                if callable(func):
                    result = func(value)
                    if asyncio.iscoroutine(result):
                        value = await result
                    else:
                        value = result
                elif isinstance(func, FunctionDef):
                    value = await self._call_function(func, [value])
            elif isinstance(stage, CallExpr):
                # Function with additional args
                callee = await self.evaluate(stage.callee)
                args = [value] + [await self.evaluate(arg) for arg in stage.args]
                if callable(callee):
                    result = callee(*args)
                    if asyncio.iscoroutine(result):
                        value = await result
                    else:
                        value = result
            else:
                # Evaluate stage and use as function
                func = await self.evaluate(stage)
                if callable(func):
                    result = func(value)
                    if asyncio.iscoroutine(result):
                        value = await result
                    else:
                        value = result
        
        return value
    
    async def eval_with(self, expr: WithExpr) -> Any:
        """Evaluate with expression: obj with { updates }"""
        base = await self.evaluate(expr.base)
        updates = await self.eval_map_literal(expr.updates)
        
        if isinstance(base, dict):
            result = base.copy()
            result.update(updates)
            return result
        
        raise InterpreterError(f"Cannot use 'with' on {type(base)}")
    
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
