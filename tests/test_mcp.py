"""
AION MCP Module Tests
=====================

Tests for MCP client, server, and security.
"""

import unittest
import sys
sys.path.insert(0, '.')


class TestMCPServer(unittest.TestCase):
    """Test MCP server."""
    
    def setUp(self):
        from src.mcp.server import MCPServer
        self.server = MCPServer("test-server")
    
    def test_server_creation(self):
        """Test server instantiation."""
        self.assertEqual(self.server.name, "test-server")
    
    def test_register_tool(self):
        """Test tool registration."""
        def test_tool(x: int) -> int:
            return x * 2
        
        self.server.register_tool("double", test_tool, "Doubles input")
        self.assertIn("double", self.server.tools)
    
    def test_list_tools(self):
        """Test tool listing."""
        def tool1(): pass
        def tool2(): pass
        
        self.server.register_tool("t1", tool1)
        self.server.register_tool("t2", tool2)
        
        tools = self.server.list_tools()
        self.assertGreaterEqual(len(tools), 2)


class TestMCPClient(unittest.TestCase):
    """Test MCP client."""
    
    def setUp(self):
        from src.mcp.client import MCPClient
        self.client = MCPClient()
    
    def test_client_creation(self):
        """Test client instantiation."""
        self.assertIsNotNone(self.client)
    
    def test_discover_servers(self):
        """Test server discovery."""
        servers = self.client.discover_servers()
        self.assertIsInstance(servers, list)


class TestMCPSecurity(unittest.TestCase):
    """Test MCP security features."""
    
    def test_security_manager(self):
        """Test security manager."""
        from src.mcp.security import MCPSecurityManager
        manager = MCPSecurityManager()
        self.assertIsNotNone(manager)
    
    def test_rate_limiter(self):
        """Test rate limiting."""
        from src.mcp.security import RateLimiter
        limiter = RateLimiter(max_requests=10, window_seconds=60)
        
        # Should allow first request
        self.assertTrue(limiter.allow("client1"))


class TestMCPRegistry(unittest.TestCase):
    """Test MCP registry."""
    
    def test_registry_creation(self):
        """Test registry instantiation."""
        from src.mcp.registry import MCPRegistry
        registry = MCPRegistry()
        self.assertIsNotNone(registry)
    
    def test_register_connection(self):
        """Test connection registration."""
        from src.mcp.registry import MCPRegistry
        registry = MCPRegistry()
        
        registry.register("test-conn", {"url": "http://test"})
        self.assertIn("test-conn", registry.connections)


if __name__ == '__main__':
    unittest.main(verbosity=2)
