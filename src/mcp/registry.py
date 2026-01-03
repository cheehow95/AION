"""
AION MCP Registry
=================

Centralized registry for managing MCP connections and capabilities.
Provides unified access to all MCP tools and resources.
"""

import asyncio
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime

from .client import MCPClient, DiscoveredTool, DiscoveredResource
from .server import MCPServer


@dataclass
class MCPConnection:
    """Represents a connection to an MCP server."""
    uri: str
    name: str
    transport: str
    connected: bool = False
    last_ping: Optional[datetime] = None
    capabilities: Dict[str, Any] = field(default_factory=dict)
    tools_count: int = 0
    resources_count: int = 0


class MCPRegistry:
    """
    Central registry for all MCP connections and capabilities.
    
    Manages both:
    - Outbound connections (as an MCP client)
    - Inbound connections (as an MCP server)
    """
    
    def __init__(self):
        # Client for connecting to external servers
        self.client = MCPClient(
            client_name="aion-agent",
            client_version="2.0.0"
        )
        
        # Server for exposing AION capabilities
        self.server = MCPServer(
            name="aion-mcp-server",
            version="2.0.0"
        )
        
        # Track all connections
        self.connections: Dict[str, MCPConnection] = {}
        
        # Tool aliases for easy access
        self.tool_aliases: Dict[str, str] = {}
        
        # Health monitoring
        self.health_check_interval = 30  # seconds
        self._health_check_task: Optional[asyncio.Task] = None
    
    async def connect(
        self,
        uri: str,
        transport: str = "stdio",
        alias: Optional[str] = None
    ) -> MCPConnection:
        """
        Connect to an MCP server.
        
        Args:
            uri: Server URI (e.g., "mcp://localhost:8765")
            transport: Connection type ("stdio", "websocket", "http")
            alias: Optional short name for the connection
        
        Returns:
            Connection info
        """
        result = await self.client.connect(uri, transport)
        
        connection = MCPConnection(
            uri=uri,
            name=result.get("server", {}).get("name", "unknown"),
            transport=transport,
            connected=True,
            last_ping=datetime.now(),
            capabilities=result.get("capabilities", {}),
            tools_count=len([t for t in self.client.tools.values() if t.server_uri == uri]),
            resources_count=len([r for r in self.client.resources.values() if r.server_uri == uri])
        )
        
        self.connections[uri] = connection
        
        # Set up alias if provided
        if alias:
            for tool in await self.client.list_tools(uri):
                self.tool_aliases[f"{alias}.{tool.name}"] = f"{uri}:{tool.name}"
        
        return connection
    
    async def disconnect(self, uri: str):
        """Disconnect from an MCP server."""
        await self.client.disconnect(uri)
        if uri in self.connections:
            del self.connections[uri]
        
        # Remove aliases for this server
        self.tool_aliases = {
            k: v for k, v in self.tool_aliases.items()
            if not v.startswith(uri)
        }
    
    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any] = None
    ) -> Any:
        """
        Call a tool by name (supports aliases).
        
        Args:
            tool_name: Tool name or alias
            arguments: Arguments to pass
        
        Returns:
            Tool result
        """
        # Check for alias
        if tool_name in self.tool_aliases:
            full_name = self.tool_aliases[tool_name]
            parts = full_name.split(":", 1)
            if len(parts) == 2:
                return await self.client.call_tool(parts[1], arguments, parts[0])
        
        # Direct call
        return await self.client.call_tool(tool_name, arguments)
    
    async def read_resource(self, uri: str) -> Any:
        """Read a resource by URI."""
        return await self.client.read_resource(uri)
    
    def list_connections(self) -> List[MCPConnection]:
        """List all active connections."""
        return list(self.connections.values())
    
    async def list_all_tools(self) -> List[DiscoveredTool]:
        """List tools from all connected servers."""
        return await self.client.list_tools()
    
    async def list_all_resources(self) -> List[DiscoveredResource]:
        """List resources from all connected servers."""
        return await self.client.list_resources()
    
    def get_aion_tool(self, tool_name: str) -> Callable:
        """Get an AION-compatible wrapper for an MCP tool."""
        return self.client.get_tool_for_aion(tool_name)
    
    def register_local_tool(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        handler: Callable
    ):
        """Register a local tool on the MCP server."""
        self.server.register_tool(name, description, input_schema, handler)
    
    def register_local_resource(
        self,
        uri: str,
        name: str,
        description: str,
        mime_type: str = "application/json"
    ):
        """Register a local resource on the MCP server."""
        self.server.register_resource(uri, name, description, mime_type)
    
    async def start_server(self):
        """Start the MCP server."""
        await self.server.start()
    
    async def stop_server(self):
        """Stop the MCP server."""
        await self.server.stop()
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all connections."""
        results = {}
        for uri, conn in self.connections.items():
            try:
                # Ping the server
                await self.client._send_request(uri, "ping", {})
                conn.last_ping = datetime.now()
                conn.connected = True
                results[uri] = True
            except Exception:
                conn.connected = False
                results[uri] = False
        return results
    
    async def auto_discover(self, network: str = "local") -> List[MCPConnection]:
        """
        Auto-discover MCP servers on the network.
        
        Args:
            network: "local" for localhost, "mdns" for network discovery
        
        Returns:
            List of discovered and connected servers
        """
        discovered = []
        
        if network == "local":
            # Check common local ports
            common_ports = [8765, 8766, 8080, 3000]
            for port in common_ports:
                uri = f"mcp://localhost:{port}"
                try:
                    conn = await self.connect(uri)
                    discovered.append(conn)
                except Exception:
                    pass
        
        return discovered
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            "connections": len(self.connections),
            "active_connections": sum(1 for c in self.connections.values() if c.connected),
            "total_tools": len(self.client.tools),
            "total_resources": len(self.client.resources),
            "local_tools": len(self.server.tools),
            "local_resources": len(self.server.resources)
        }


# Global registry instance
_global_mcp_registry: Optional[MCPRegistry] = None


def get_mcp_registry() -> MCPRegistry:
    """Get the global MCP registry."""
    global _global_mcp_registry
    if _global_mcp_registry is None:
        _global_mcp_registry = MCPRegistry()
    return _global_mcp_registry
