"""
AION Runtime Environment
Manages symbol tables, scopes, and execution context.
"""

from typing import Any, Optional, Callable
from dataclasses import dataclass, field


class EnvironmentError(Exception):
    """Raised when an environment operation fails."""
    pass


@dataclass
class Symbol:
    """Represents a symbol in the environment."""
    name: str
    value: Any
    symbol_type: str  # 'variable', 'agent', 'model', 'tool', 'memory'
    metadata: dict = field(default_factory=dict)


class Environment:
    """
    Runtime environment with nested scopes.
    Manages variables, agents, models, tools, and memory.
    """
    
    def __init__(self, parent: Optional['Environment'] = None):
        self.parent = parent
        self.symbols: dict[str, Symbol] = {}
        
        # Special registries for first-class constructs
        self.agents: dict[str, 'AgentInstance'] = {}
        self.models: dict[str, 'ModelInstance'] = {}
        self.tools: dict[str, 'ToolInstance'] = {}
        self.memories: dict[str, 'MemoryInstance'] = {}
    
    def define(self, name: str, value: Any, symbol_type: str = 'variable', **metadata) -> None:
        """Define a symbol in the current scope."""
        self.symbols[name] = Symbol(name, value, symbol_type, metadata)
    
    def get(self, name: str) -> Any:
        """Get a symbol value, searching parent scopes."""
        if name in self.symbols:
            return self.symbols[name].value
        if self.parent:
            return self.parent.get(name)
        raise EnvironmentError(f"Undefined symbol: {name}")
    
    def set(self, name: str, value: Any) -> None:
        """Set a symbol value, searching parent scopes."""
        if name in self.symbols:
            self.symbols[name].value = value
            return
        if self.parent:
            self.parent.set(name, value)
            return
        raise EnvironmentError(f"Undefined symbol: {name}")
    
    def exists(self, name: str) -> bool:
        """Check if a symbol exists in any scope."""
        if name in self.symbols:
            return True
        if self.parent:
            return self.parent.exists(name)
        return False
    
    def child_scope(self) -> 'Environment':
        """Create a child scope."""
        return Environment(parent=self)
    
    # Agent management
    def register_agent(self, name: str, agent: 'AgentInstance') -> None:
        """Register an agent instance."""
        self.agents[name] = agent
        self.define(name, agent, 'agent')
    
    def get_agent(self, name: str) -> 'AgentInstance':
        """Get an agent instance."""
        if name in self.agents:
            return self.agents[name]
        if self.parent:
            return self.parent.get_agent(name)
        raise EnvironmentError(f"Undefined agent: {name}")
    
    # Model management
    def register_model(self, name: str, model: 'ModelInstance') -> None:
        """Register a model instance."""
        self.models[name] = model
        self.define(name, model, 'model')
    
    def get_model(self, name: str) -> 'ModelInstance':
        """Get a model instance."""
        if name in self.models:
            return self.models[name]
        if self.parent:
            return self.parent.get_model(name)
        raise EnvironmentError(f"Undefined model: {name}")
    
    # Tool management
    def register_tool(self, name: str, tool: 'ToolInstance') -> None:
        """Register a tool instance."""
        self.tools[name] = tool
        self.define(name, tool, 'tool')
    
    def get_tool(self, name: str) -> 'ToolInstance':
        """Get a tool instance."""
        if name in self.tools:
            return self.tools[name]
        if self.parent:
            return self.parent.get_tool(name)
        raise EnvironmentError(f"Undefined tool: {name}")
    
    # Memory management
    def register_memory(self, name: str, memory: 'MemoryInstance') -> None:
        """Register a memory instance."""
        self.memories[name] = memory
        self.define(name, memory, 'memory')
    
    def get_memory(self, name: str) -> 'MemoryInstance':
        """Get a memory instance."""
        if name in self.memories:
            return self.memories[name]
        if self.parent:
            return self.parent.get_memory(name)
        raise EnvironmentError(f"Undefined memory: {name}")


@dataclass
class AgentInstance:
    """Runtime instance of an agent."""
    name: str
    goal: str = ""
    memories: list[str] = field(default_factory=list)
    model: Optional[str] = None
    tools: list[str] = field(default_factory=list)
    policy: dict = field(default_factory=dict)
    event_handlers: dict = field(default_factory=dict)
    
    # Runtime state
    context: dict = field(default_factory=dict)
    reasoning_trace: list[str] = field(default_factory=list)


@dataclass
class ModelInstance:
    """Runtime instance of a model."""
    name: str
    provider: str = "openai"
    model_name: str = "gpt-4"
    config: dict = field(default_factory=dict)
    
    async def complete(self, prompt: str, **kwargs) -> str:
        """Generate a completion from the model."""
        # This will be implemented by the model interface
        raise NotImplementedError("Model interface not configured")


@dataclass
class ToolInstance:
    """Runtime instance of a tool."""
    name: str
    trust: str = "medium"  # low, medium, high
    cost: str = "low"  # low, medium, high
    handler: Optional[Callable] = None
    config: dict = field(default_factory=dict)
    
    async def execute(self, *args, **kwargs) -> Any:
        """Execute the tool."""
        if self.handler:
            return await self.handler(*args, **kwargs)
        raise NotImplementedError(f"Tool {self.name} has no handler")


@dataclass
class MemoryInstance:
    """Runtime instance of a memory store."""
    name: str
    memory_type: str  # working, episodic, long_term, semantic
    retention: str = "session"  # session, persistent
    config: dict = field(default_factory=dict)
    
    # Memory storage
    _storage: list = field(default_factory=list)
    
    def store(self, item: Any, metadata: dict = None) -> None:
        """Store an item in memory."""
        entry = {
            'data': item,
            'metadata': metadata or {},
            'timestamp': None  # Will be set by runtime
        }
        self._storage.append(entry)
    
    def recall(self, query: str = None, limit: int = 10) -> list:
        """Recall items from memory."""
        # Simple implementation - semantic search would be added
        return self._storage[-limit:]
    
    def clear(self) -> None:
        """Clear memory contents."""
        self._storage.clear()
