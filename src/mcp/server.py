"""
AION MCP Server
===============

Exposes AION capabilities as an MCP server that other applications can connect to.
Follows the Model Context Protocol specification for tool discovery and invocation.
"""

import json
import asyncio
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import uuid


@dataclass
class MCPTool:
    """Represents a tool exposed via MCP."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable
    annotations: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
            "annotations": self.annotations
        }


@dataclass
class MCPResource:
    """Represents a resource exposed via MCP."""
    uri: str
    name: str
    description: str
    mime_type: str = "application/json"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mimeType": self.mime_type
        }


@dataclass
class MCPPrompt:
    """Represents a prompt template exposed via MCP."""
    name: str
    description: str
    arguments: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "arguments": self.arguments
        }


class MCPServer:
    """
    MCP Server implementation for AION.
    
    Exposes AION's capabilities (tools, resources, prompts) via the
    Model Context Protocol, allowing external clients to discover
    and invoke AION functionality.
    """
    
    PROTOCOL_VERSION = "2024-11-05"
    
    def __init__(
        self,
        name: str = "aion-mcp-server",
        version: str = "1.0.0",
        host: str = "localhost",
        port: int = 8765
    ):
        self.name = name
        self.version = version
        self.host = host
        self.port = port
        
        # Registered capabilities
        self.tools: Dict[str, MCPTool] = {}
        self.resources: Dict[str, MCPResource] = {}
        self.prompts: Dict[str, MCPPrompt] = {}
        
        # Server capabilities
        self.capabilities = {
            "tools": {"listChanged": True},
            "resources": {"subscribe": True, "listChanged": True},
            "prompts": {"listChanged": True},
            "logging": {}
        }
        
        # Active connections
        self.connections: Dict[str, Any] = {}
        
        # Register built-in AION tools
        self._register_builtin_tools()
    
    def _register_builtin_tools(self):
        """Register built-in AION tools."""
        # Reasoning tools
        self.register_tool(
            name="aion.think",
            description="Initiate reasoning process with optional prompt",
            input_schema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Reasoning prompt"}
                }
            },
            handler=self._handle_think
        )
        
        self.register_tool(
            name="aion.analyze",
            description="Analyze a piece of data or text",
            input_schema={
                "type": "object",
                "properties": {
                    "target": {"type": "string", "description": "Text or data to analyze"}
                },
                "required": ["target"]
            },
            handler=self._handle_analyze
        )
        
        self.register_tool(
            name="aion.reflect",
            description="Reflect on agent actions or reasoning",
            input_schema={
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Topic to reflect on"}
                }
            },
            handler=self._handle_reflect
        )
        
        # Memory tools
        self.register_tool(
            name="aion.store",
            description="Store data in agent memory",
            input_schema={
                "type": "object",
                "properties": {
                    "value": {"description": "Value to store"},
                    "memory": {"type": "string", "description": "Memory type (working/episodic/semantic)"}
                },
                "required": ["value"]
            },
            handler=self._handle_store
        )
        
        self.register_tool(
            name="aion.recall",
            description="Recall data from agent memory",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "memory": {"type": "string", "description": "Memory type"}
                }
            },
            handler=self._handle_recall
        )
        
        # Agent management
        self.register_tool(
            name="aion.run_agent",
            description="Run an AION agent with input",
            input_schema={
                "type": "object",
                "properties": {
                    "agent_name": {"type": "string", "description": "Name of agent to run"},
                    "input": {"description": "Input data for the agent"}
                },
                "required": ["agent_name"]
            },
            handler=self._handle_run_agent
        )
    
    def register_tool(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        handler: Callable,
        annotations: Dict[str, Any] = None
    ):
        """Register a new tool."""
        self.tools[name] = MCPTool(
            name=name,
            description=description,
            input_schema=input_schema,
            handler=handler,
            annotations=annotations or {}
        )
    
    def register_resource(
        self,
        uri: str,
        name: str,
        description: str,
        mime_type: str = "application/json"
    ):
        """Register a new resource."""
        self.resources[uri] = MCPResource(
            uri=uri,
            name=name,
            description=description,
            mime_type=mime_type
        )
    
    def register_prompt(
        self,
        name: str,
        description: str,
        arguments: List[Dict[str, Any]] = None
    ):
        """Register a new prompt template."""
        self.prompts[name] = MCPPrompt(
            name=name,
            description=description,
            arguments=arguments or []
        )
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an incoming MCP request."""
        method = request.get("method", "")
        params = request.get("params", {})
        request_id = request.get("id")
        
        try:
            # Route to appropriate handler
            if method == "initialize":
                result = await self._handle_initialize(params)
            elif method == "tools/list":
                result = await self._handle_list_tools(params)
            elif method == "tools/call":
                result = await self._handle_call_tool(params)
            elif method == "resources/list":
                result = await self._handle_list_resources(params)
            elif method == "resources/read":
                result = await self._handle_read_resource(params)
            elif method == "prompts/list":
                result = await self._handle_list_prompts(params)
            elif method == "prompts/get":
                result = await self._handle_get_prompt(params)
            else:
                return self._error_response(request_id, -32601, f"Method not found: {method}")
            
            return self._success_response(request_id, result)
            
        except Exception as e:
            return self._error_response(request_id, -32000, str(e))
    
    async def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialize request."""
        client_info = params.get("clientInfo", {})
        
        return {
            "protocolVersion": self.PROTOCOL_VERSION,
            "capabilities": self.capabilities,
            "serverInfo": {
                "name": self.name,
                "version": self.version
            }
        }
    
    async def _handle_list_tools(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/list request."""
        return {
            "tools": [tool.to_dict() for tool in self.tools.values()]
        }
    
    async def _handle_call_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/call request."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if tool_name not in self.tools:
            raise ValueError(f"Tool not found: {tool_name}")
        
        tool = self.tools[tool_name]
        result = await tool.handler(arguments)
        
        return {
            "content": [
                {"type": "text", "text": json.dumps(result)}
            ]
        }
    
    async def _handle_list_resources(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resources/list request."""
        return {
            "resources": [res.to_dict() for res in self.resources.values()]
        }
    
    async def _handle_read_resource(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resources/read request."""
        uri = params.get("uri")
        
        if uri not in self.resources:
            raise ValueError(f"Resource not found: {uri}")
        
        # In a real implementation, this would fetch the resource content
        return {
            "contents": [
                {"uri": uri, "mimeType": "application/json", "text": "{}"}
            ]
        }
    
    async def _handle_list_prompts(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prompts/list request."""
        return {
            "prompts": [prompt.to_dict() for prompt in self.prompts.values()]
        }
    
    async def _handle_get_prompt(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prompts/get request."""
        prompt_name = params.get("name")
        
        if prompt_name not in self.prompts:
            raise ValueError(f"Prompt not found: {prompt_name}")
        
        prompt = self.prompts[prompt_name]
        return {
            "description": prompt.description,
            "messages": []
        }
    
    # Built-in tool handlers
    
    async def _handle_think(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle think tool invocation."""
        prompt = args.get("prompt", "")
        return {"result": f"Thinking about: {prompt}", "status": "complete"}
    
    async def _handle_analyze(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle analyze tool invocation."""
        target = args.get("target", "")
        return {"analysis": f"Analysis of: {target}", "confidence": 0.85}
    
    async def _handle_reflect(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle reflect tool invocation."""
        topic = args.get("topic", "")
        return {"reflection": f"Reflecting on: {topic}"}
    
    async def _handle_store(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle store tool invocation."""
        value = args.get("value")
        memory = args.get("memory", "working")
        return {"stored": True, "memory": memory}
    
    async def _handle_recall(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle recall tool invocation."""
        query = args.get("query", "")
        memory = args.get("memory", "working")
        return {"results": [], "query": query, "memory": memory}
    
    async def _handle_run_agent(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle run_agent tool invocation."""
        agent_name = args.get("agent_name")
        input_data = args.get("input")
        return {"agent": agent_name, "status": "executed", "output": []}
    
    def _success_response(self, request_id: Any, result: Any) -> Dict[str, Any]:
        """Create a success response."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result
        }
    
    def _error_response(self, request_id: Any, code: int, message: str) -> Dict[str, Any]:
        """Create an error response."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": code,
                "message": message
            }
        }
    
    async def start(self):
        """Start the MCP server."""
        # In a real implementation, this would start a WebSocket or stdio server
        print(f"[MCP] Server '{self.name}' starting on {self.host}:{self.port}")
        print(f"[MCP] Protocol version: {self.PROTOCOL_VERSION}")
        print(f"[MCP] Tools registered: {len(self.tools)}")
        print(f"[MCP] Resources registered: {len(self.resources)}")
        print(f"[MCP] Prompts registered: {len(self.prompts)}")
    
    async def stop(self):
        """Stop the MCP server."""
        print(f"[MCP] Server '{self.name}' stopping")
        self.connections.clear()
