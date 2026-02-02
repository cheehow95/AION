"""Tests for AION Agent Builder"""

import pytest
import json
import sys
sys.path.insert(0, '.')

from src.builder.agent_builder import AgentBuilder, AgentTemplate


class TestAgentTemplate:
    """Test cases for AgentTemplate."""
    
    def test_list_all_templates(self):
        """Test listing all available templates."""
        templates = AgentTemplate.list_all()
        
        assert "assistant" in templates
        assert "researcher" in templates
        assert "analyzer" in templates
        assert "creative" in templates
    
    def test_get_template(self):
        """Test getting a specific template."""
        template = AgentTemplate.get("researcher")
        
        assert template["name"] == "Researcher"
        assert "Research" in template["goal"]
        assert "semantic" in template["memory"]
        assert "web_search" in template.get("tools", [])
    
    def test_get_unknown_template_returns_default(self):
        """Test that unknown template returns assistant."""
        template = AgentTemplate.get("nonexistent")
        
        assert template["name"] == "Assistant"


class TestAgentBuilder:
    """Test cases for AgentBuilder."""
    
    def test_builder_init(self):
        """Test builder initialization."""
        builder = AgentBuilder()
        
        assert builder.name == "NewAgent"
        assert builder.goal == ""
        assert builder.memories == []
        assert builder.tools == []
    
    def test_fluent_api(self):
        """Test fluent API chain."""
        builder = (AgentBuilder()
            .set_name("TestAgent")
            .set_goal("Test goal")
            .add_memory("working")
            .add_tool("calculator"))
        
        assert builder.name == "TestAgent"
        assert builder.goal == "Test goal"
        assert "working" in builder.memories
        assert "calculator" in builder.tools
    
    def test_add_memory_no_duplicates(self):
        """Test that duplicate memories are not added."""
        builder = (AgentBuilder()
            .add_memory("working")
            .add_memory("working")
            .add_memory("working"))
        
        assert builder.memories.count("working") == 1
    
    def test_add_tool_no_duplicates(self):
        """Test that duplicate tools are not added."""
        builder = (AgentBuilder()
            .add_tool("web_search")
            .add_tool("web_search"))
        
        assert builder.tools.count("web_search") == 1
    
    def test_add_handler(self):
        """Test adding event handler."""
        builder = AgentBuilder()
        builder.add_handler("input", ["think", "respond"])
        
        assert len(builder.handlers) == 1
        assert builder.handlers[0]["event"] == "input"
        assert "think" in builder.handlers[0]["actions"]
    
    def test_add_policy(self):
        """Test adding policy rules."""
        builder = AgentBuilder()
        builder.add_policy("trust_level = high")
        
        assert "trust_level = high" in builder.policies
    
    def test_from_template(self):
        """Test building from template."""
        builder = AgentBuilder().from_template("researcher")
        
        assert builder.name == "Researcher"
        assert "semantic" in builder.memories
        assert "web_search" in builder.tools
    
    def test_build_minimal(self):
        """Test building minimal agent."""
        code = AgentBuilder().set_name("Minimal").build()
        
        assert "agent Minimal {" in code
        assert "}" in code
    
    def test_build_with_goal(self):
        """Test building agent with goal."""
        code = (AgentBuilder()
            .set_name("Helper")
            .set_goal("Help users")
            .build())
        
        assert 'goal "Help users"' in code
    
    def test_build_with_memory(self):
        """Test building agent with memory."""
        code = (AgentBuilder()
            .set_name("MemAgent")
            .add_memory("working")
            .add_memory("long_term")
            .build())
        
        assert "memory working" in code
        assert "memory long_term" in code
    
    def test_build_with_tools(self):
        """Test building agent with tools."""
        code = (AgentBuilder()
            .set_name("ToolAgent")
            .add_tool("calculator")
            .build())
        
        assert "tool calculator" in code
    
    def test_build_with_policy(self):
        """Test building agent with policy."""
        code = (AgentBuilder()
            .set_name("PolicyAgent")
            .add_policy("max_tokens = 1000")
            .build())
        
        assert "policy {" in code
        assert "max_tokens = 1000" in code
    
    def test_build_with_handlers(self):
        """Test building agent with handlers."""
        code = (AgentBuilder()
            .set_name("HandlerAgent")
            .add_handler("input", ["think", "respond data"])
            .build())
        
        assert "on input(data):" in code
        assert "think" in code
        assert "respond data" in code
    
    def test_to_json(self):
        """Test JSON export."""
        builder = (AgentBuilder()
            .set_name("JsonAgent")
            .set_goal("Test JSON")
            .add_memory("working"))
        
        json_str = builder.to_json()
        data = json.loads(json_str)
        
        assert data["name"] == "JsonAgent"
        assert data["goal"] == "Test JSON"
        assert "working" in data["memories"]
    
    def test_from_json(self):
        """Test JSON import."""
        json_str = json.dumps({
            "name": "ImportedAgent",
            "goal": "Imported goal",
            "memories": ["episodic"],
            "tools": ["web_search"],
            "handlers": [],
            "policies": []
        })
        
        builder = AgentBuilder.from_json(json_str)
        
        assert builder.name == "ImportedAgent"
        assert builder.goal == "Imported goal"
        assert "episodic" in builder.memories
        assert "web_search" in builder.tools
    
    def test_round_trip_json(self):
        """Test JSON export/import round trip."""
        original = (AgentBuilder()
            .set_name("RoundTrip")
            .set_goal("Test round trip")
            .add_memory("semantic")
            .add_tool("calculator"))
        
        json_str = original.to_json()
        restored = AgentBuilder.from_json(json_str)
        
        assert restored.name == original.name
        assert restored.goal == original.goal
        assert restored.memories == original.memories
        assert restored.tools == original.tools


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
