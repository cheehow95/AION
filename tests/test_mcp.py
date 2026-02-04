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
        async def test_tool(args):
            return {"result": args.get("x", 0) * 2}
        
        self.server.register_tool(
            name="double",
            description="Doubles input",
            input_schema={"type": "object", "properties": {"x": {"type": "integer"}}},
            handler=test_tool
        )
        self.assertIn("double", self.server.tools)
    
    def test_list_tools(self):
        """Test tool listing."""
        async def tool1(args): pass
        async def tool2(args): pass
        
        self.server.register_tool("t1", "Tool 1", {}, tool1)
        self.server.register_tool("t2", "Tool 2", {}, tool2)
        
        # Server has builtin tools + our 2
        self.assertIn("t1", self.server.tools)
        self.assertIn("t2", self.server.tools)


class TestMCPClient(unittest.TestCase):
    """Test MCP client."""
    
    def setUp(self):
        from src.mcp.client import MCPClient
        self.client = MCPClient()
    
    def test_client_creation(self):
        """Test client instantiation."""
        self.assertIsNotNone(self.client)
    
    def test_list_tools(self):
        """Test listing tools from client."""
        # Client starts with empty tools dict
        self.assertIsInstance(self.client.tools, dict)


class TestMCPSecurity(unittest.TestCase):
    """Test MCP security features."""
    
    def test_security_manager(self):
        """Test security manager."""
        from src.mcp.security import MCPSecurityManager
        manager = MCPSecurityManager()
        self.assertIsNotNone(manager)
    
    def test_rate_limit_check(self):
        """Test rate limiting via manager."""
        from src.mcp.security import MCPSecurityManager
        manager = MCPSecurityManager()
        
        # Should allow first request
        allowed, _ = manager.check_rate_limit("client1")
        self.assertTrue(allowed)


class TestMCPRegistry(unittest.TestCase):
    """Test MCP registry."""
    
    def test_registry_creation(self):
        """Test registry instantiation."""
        from src.mcp.registry import MCPRegistry
        registry = MCPRegistry()
        self.assertIsNotNone(registry)
    
    def test_connections_dict(self):
        """Test connections dictionary exists."""
        from src.mcp.registry import MCPRegistry
        registry = MCPRegistry()
        
        # Connections dict should exist and be empty initially
        self.assertIsInstance(registry.connections, dict)
        self.assertEqual(len(registry.connections), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
