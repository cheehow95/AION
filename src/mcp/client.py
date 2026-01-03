"""
AION MCP Client
===============

Connects to external MCP servers to consume tools and resources.
Allows AION agents to seamlessly use external capabilities.
"""

import json
import asyncio
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import uuid


@dataclass
class DiscoveredTool:
    """A tool discovered from an MCP server."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    server_uri: str
    annotations: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DiscoveredResource:
    """A resource discovered from an MCP server."""
    uri: str
    name: str
    description: str
    mime_type: str
    server_uri: str


class MCPClient:
    """
    MCP Client for connecting to external MCP servers.
    
    Discovers and invokes tools, reads resources, and uses prompts
    from external MCP-compliant services.
    """
    
    PROTOCOL_VERSION = "2024-11-05"
    
    def __init__(self, client_name: str = "aion-client", client_version: str = "1.0.0"):
        self.client_name = client_name
        self.client_version = client_version
        
        # Connected servers
        self.servers: Dict[str, Dict[str, Any]] = {}
        
        # Cached capabilities from all servers
        self.tools: Dict[str, DiscoveredTool] = {}
        self.resources: Dict[str, DiscoveredResource] = {}
        
        # Request tracking
        self.request_id = 0
        self.pending_requests: Dict[int, asyncio.Future] = {}
    
    async def connect(self, server_uri: str, transport: str = "stdio") -> Dict[str, Any]:
        """
        Connect to an MCP server.
        
        Args:
            server_uri: URI of the MCP server (e.g., "mcp://localhost:8765")
            transport: Connection type ("stdio", "websocket", "http")
        
        Returns:
            Server capabilities
        """
        # Initialize connection
        init_response = await self._send_request(server_uri, "initialize", {
            "protocolVersion": self.PROTOCOL_VERSION,
            "clientInfo": {
                "name": self.client_name,
                "version": self.client_version
            },
            "capabilities": {
                "roots": {"listChanged": True},
                "sampling": {}
            }
        })
        
        server_info = init_response.get("serverInfo", {})
        capabilities = init_response.get("capabilities", {})
        
        # Store server connection
        self.servers[server_uri] = {
            "info": server_info,
            "capabilities": capabilities,
            "transport": transport,
            "connected": True
        }
        
        # Discover tools and resources
        await self._discover_capabilities(server_uri)
        
        return {
            "server": server_info,
            "capabilities": capabilities
        }
    
    async def disconnect(self, server_uri: str):
        """Disconnect from an MCP server."""
        if server_uri in self.servers:
            self.servers[server_uri]["connected"] = False
            
            # Remove cached tools and resources from this server
            self.tools = {k: v for k, v in self.tools.items() if v.server_uri != server_uri}
            self.resources = {k: v for k, v in self.resources.items() if v.server_uri != server_uri}
            
            del self.servers[server_uri]
    
    async def _discover_capabilities(self, server_uri: str):
        """Discover tools and resources from a server."""
        server = self.servers.get(server_uri, {})
        capabilities = server.get("capabilities", {})
        
        # Discover tools
        if capabilities.get("tools"):
            tools_response = await self._send_request(server_uri, "tools/list", {})
            for tool_data in tools_response.get("tools", []):
                tool = DiscoveredTool(
                    name=tool_data["name"],
                    description=tool_data.get("description", ""),
                    input_schema=tool_data.get("inputSchema", {}),
                    server_uri=server_uri,
                    annotations=tool_data.get("annotations", {})
                )
                # Use fully qualified name to avoid conflicts
                fqn = f"{server_uri}:{tool.name}"
                self.tools[fqn] = tool
        
        # Discover resources
        if capabilities.get("resources"):
            resources_response = await self._send_request(server_uri, "resources/list", {})
            for res_data in resources_response.get("resources", []):
                resource = DiscoveredResource(
                    uri=res_data["uri"],
                    name=res_data["name"],
                    description=res_data.get("description", ""),
                    mime_type=res_data.get("mimeType", "application/json"),
                    server_uri=server_uri
                )
                self.resources[resource.uri] = resource
    
    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any] = None,
        server_uri: str = None
    ) -> Dict[str, Any]:
        """
        Call a tool on an MCP server.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool
            server_uri: Specific server to use (optional)
        
        Returns:
            Tool execution result
        """
        # Find the tool
        if server_uri and f"{server_uri}:{tool_name}" in self.tools:
            full_name = f"{server_uri}:{tool_name}"
            tool = self.tools[full_name]
        else:
            # Search all servers
            matching = [t for t in self.tools.values() if t.name == tool_name]
            if not matching:
                raise ValueError(f"Tool not found: {tool_name}")
            tool = matching[0]
        
        # Call the tool
        response = await self._send_request(tool.server_uri, "tools/call", {
            "name": tool.name,
            "arguments": arguments or {}
        })
        
        # Parse response content
        content = response.get("content", [])
        if content and content[0].get("type") == "text":
            try:
                return json.loads(content[0]["text"])
            except json.JSONDecodeError:
                return {"text": content[0]["text"]}
        
        return response
    
    async def read_resource(self, uri: str) -> Dict[str, Any]:
        """
        Read a resource from an MCP server.
        
        Args:
            uri: URI of the resource
        
        Returns:
            Resource content
        """
        if uri not in self.resources:
            raise ValueError(f"Resource not found: {uri}")
        
        resource = self.resources[uri]
        
        response = await self._send_request(resource.server_uri, "resources/read", {
            "uri": uri
        })
        
        return response.get("contents", [])
    
    async def list_tools(self, server_uri: str = None) -> List[DiscoveredTool]:
        """List all available tools, optionally filtered by server."""
        if server_uri:
            return [t for t in self.tools.values() if t.server_uri == server_uri]
        return list(self.tools.values())
    
    async def list_resources(self, server_uri: str = None) -> List[DiscoveredResource]:
        """List all available resources, optionally filtered by server."""
        if server_uri:
            return [r for r in self.resources.values() if r.server_uri == server_uri]
        return list(self.resources.values())
    
    async def _send_request(
        self,
        server_uri: str,
        method: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Send a request to an MCP server."""
        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
            "params": params
        }
        
        # In a real implementation, this would:
        # 1. Look up the connection for server_uri
        # 2. Send the request via the appropriate transport
        # 3. Wait for and return the response
        
        # Simulated response for now
        return await self._simulate_response(method, params)
    
    async def _simulate_response(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate MCP responses for testing."""
        if method == "initialize":
            return {
                "protocolVersion": self.PROTOCOL_VERSION,
                "capabilities": {
                    "tools": {"listChanged": True},
                    "resources": {"subscribe": True}
                },
                "serverInfo": {"name": "simulated", "version": "1.0.0"}
            }
        elif method == "tools/list":
            return {"tools": []}
        elif method == "tools/call":
            return {"content": [{"type": "text", "text": "{}"}]}
        elif method == "resources/list":
            return {"resources": []}
        elif method == "resources/read":
            return {"contents": []}
        return {}
    
    def get_tool_for_aion(self, tool_name: str) -> Callable:
        """
        Create an AION-compatible tool wrapper for an MCP tool.
        
        This allows MCP tools to be used directly in AION agents
        via the 'use' statement.
        """
        async def tool_wrapper(*args, **kwargs):
            # Convert positional args to dict based on schema
            if tool_name not in self.tools and not any(t.name == tool_name for t in self.tools.values()):
                raise ValueError(f"Tool not found: {tool_name}")
            
            return await self.call_tool(tool_name, kwargs or {"args": args})
        
        return tool_wrapper
