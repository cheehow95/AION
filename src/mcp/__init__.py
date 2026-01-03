"""
AION MCP (Model Context Protocol) Integration
==============================================

Provides standardized tool integration using the Model Context Protocol.
Enables AION agents to interact with external services and tools
through a unified, secure interface.
"""

from .server import MCPServer
from .client import MCPClient
from .registry import MCPRegistry
from .security import MCPSecurityManager

__all__ = [
    'MCPServer',
    'MCPClient', 
    'MCPRegistry',
    'MCPSecurityManager'
]
