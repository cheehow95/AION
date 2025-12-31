"""
AION Tool Registry
Manages tool registration, trust levels, and execution.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Awaitable
from enum import Enum
import asyncio
import functools


class ToolError(Exception):
    """Raised when a tool operation fails."""
    pass


class TrustLevel(Enum):
    """Trust levels for tools."""
    LOW = "low"         # Sandboxed, limited access
    MEDIUM = "medium"   # Standard access with logging
    HIGH = "high"       # Full access, minimal restrictions


class CostLevel(Enum):
    """Cost levels for tool usage."""
    LOW = "low"         # Free or negligible cost
    MEDIUM = "medium"   # Moderate cost per use
    HIGH = "high"       # Expensive, use sparingly


@dataclass
class ToolConfig:
    """Configuration for a tool."""
    trust: TrustLevel = TrustLevel.MEDIUM
    cost: CostLevel = CostLevel.LOW
    rate_limit: Optional[int] = None  # Max calls per minute
    requires_approval: bool = False
    timeout: float = 30.0
    retry_count: int = 3
    metadata: dict = field(default_factory=dict)


@dataclass
class ToolResult:
    """Result from a tool execution."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class Tool:
    """Represents a registered tool."""
    name: str
    handler: Callable
    description: str = ""
    config: ToolConfig = field(default_factory=ToolConfig)
    usage_count: int = 0
    
    async def execute(self, *args, **kwargs) -> ToolResult:
        """Execute the tool with the given arguments."""
        import time
        start_time = time.time()
        
        try:
            # Apply timeout
            if asyncio.iscoroutinefunction(self.handler):
                result = await asyncio.wait_for(
                    self.handler(*args, **kwargs),
                    timeout=self.config.timeout
                )
            else:
                # Run sync handler in executor
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        functools.partial(self.handler, *args, **kwargs)
                    ),
                    timeout=self.config.timeout
                )
            
            self.usage_count += 1
            execution_time = time.time() - start_time
            
            return ToolResult(
                success=True,
                data=result,
                execution_time=execution_time
            )
        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                error=f"Tool {self.name} timed out after {self.config.timeout}s",
                execution_time=time.time() - start_time
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )


class ToolRegistry:
    """
    Registry for tools with trust and cost management.
    """
    
    def __init__(self):
        self.tools: dict[str, Tool] = {}
        self.usage_log: list[dict] = []
        self.policies: dict[str, Any] = {}
    
    def register(
        self,
        name: str,
        handler: Callable,
        description: str = "",
        trust: str = "medium",
        cost: str = "low",
        **config_kwargs
    ) -> Tool:
        """Register a new tool."""
        config = ToolConfig(
            trust=TrustLevel(trust),
            cost=CostLevel(cost),
            **config_kwargs
        )
        
        tool = Tool(
            name=name,
            handler=handler,
            description=description,
            config=config
        )
        
        self.tools[name] = tool
        return tool
    
    def get(self, name: str) -> Tool:
        """Get a registered tool by name."""
        if name not in self.tools:
            raise ToolError(f"Tool not found: {name}")
        return self.tools[name]
    
    def exists(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self.tools
    
    async def execute(
        self,
        name: str,
        *args,
        agent_context: dict = None,
        **kwargs
    ) -> ToolResult:
        """Execute a tool with policy checks."""
        tool = self.get(name)
        
        # Check policies
        if not self._check_policies(tool, agent_context):
            return ToolResult(
                success=False,
                error=f"Policy violation for tool: {name}"
            )
        
        # Check if approval required
        if tool.config.requires_approval:
            # In a real implementation, this would pause for human approval
            pass
        
        # Execute the tool
        result = await tool.execute(*args, **kwargs)
        
        # Log usage
        self._log_usage(tool, args, kwargs, result)
        
        return result
    
    def _check_policies(self, tool: Tool, context: dict = None) -> bool:
        """Check if tool execution is allowed by policies."""
        context = context or {}
        
        # Check trust level policies
        if 'max_trust' in self.policies:
            max_trust = TrustLevel(self.policies['max_trust'])
            if tool.config.trust.value > max_trust.value:
                return False
        
        # Check cost level policies
        if 'max_cost' in self.policies:
            max_cost = CostLevel(self.policies['max_cost'])
            if tool.config.cost.value > max_cost.value:
                return False
        
        # Check rate limits
        if tool.config.rate_limit:
            recent_uses = sum(
                1 for log in self.usage_log[-100:]
                if log['tool'] == tool.name
            )
            if recent_uses >= tool.config.rate_limit:
                return False
        
        return True
    
    def _log_usage(
        self,
        tool: Tool,
        args: tuple,
        kwargs: dict,
        result: ToolResult
    ) -> None:
        """Log tool usage for auditing."""
        import time
        
        self.usage_log.append({
            'tool': tool.name,
            'timestamp': time.time(),
            'success': result.success,
            'execution_time': result.execution_time,
            'trust': tool.config.trust.value,
            'cost': tool.config.cost.value,
        })
    
    def set_policy(self, key: str, value: Any) -> None:
        """Set a global policy."""
        self.policies[key] = value
    
    def get_usage_stats(self, tool_name: str = None) -> dict:
        """Get usage statistics for tools."""
        if tool_name:
            logs = [l for l in self.usage_log if l['tool'] == tool_name]
        else:
            logs = self.usage_log
        
        if not logs:
            return {'total_calls': 0}
        
        return {
            'total_calls': len(logs),
            'success_rate': sum(1 for l in logs if l['success']) / len(logs),
            'avg_execution_time': sum(l['execution_time'] for l in logs) / len(logs),
        }


# Built-in tools
def create_builtin_tools(registry: ToolRegistry) -> None:
    """Register built-in tools."""
    
    # Calculator tool
    def calculator(expression: str) -> float:
        """Evaluate a mathematical expression."""
        import ast
        import operator
        
        operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
        }
        
        def eval_node(node):
            if isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.BinOp):
                left = eval_node(node.left)
                right = eval_node(node.right)
                return operators[type(node.op)](left, right)
            elif isinstance(node, ast.UnaryOp):
                operand = eval_node(node.operand)
                return operators[type(node.op)](operand)
            else:
                raise ValueError(f"Unsupported operation: {type(node)}")
        
        tree = ast.parse(expression, mode='eval')
        return eval_node(tree.body)
    
    registry.register(
        name="calculator",
        handler=calculator,
        description="Evaluate mathematical expressions",
        trust="high",
        cost="low"
    )
    
    # Web search (placeholder)
    async def web_search(query: str, num_results: int = 5) -> list[dict]:
        """Search the web for information."""
        # This would integrate with a real search API
        return [{"title": "Placeholder", "url": "#", "snippet": f"Results for: {query}"}]
    
    registry.register(
        name="web_search",
        handler=web_search,
        description="Search the web for information",
        trust="medium",
        cost="medium"
    )
    
    # File reader (placeholder)
    def read_file(path: str) -> str:
        """Read contents of a file."""
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    
    registry.register(
        name="read_file",
        handler=read_file,
        description="Read contents of a file",
        trust="low",
        cost="low"
    )


# Global registry instance
_global_registry = ToolRegistry()


def get_global_registry() -> ToolRegistry:
    """Get the global tool registry."""
    return _global_registry
