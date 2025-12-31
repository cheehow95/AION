"""
AION Visual Agent Builder
Web-based UI for creating agents without code.
"""

import json
from typing import Dict, List, Any

class AgentTemplate:
    """Pre-built agent templates."""
    
    TEMPLATES = {
        "assistant": {
            "name": "Assistant",
            "goal": "Help users with tasks",
            "memory": ["working"],
            "handlers": ["on input"]
        },
        "researcher": {
            "name": "Researcher", 
            "goal": "Research topics and summarize findings",
            "memory": ["working", "semantic"],
            "tools": ["web_search"],
            "handlers": ["on input"]
        },
        "analyzer": {
            "name": "Analyzer",
            "goal": "Analyze data and provide insights",
            "memory": ["working", "episodic"],
            "handlers": ["on input", "on error"]
        },
        "creative": {
            "name": "Creative",
            "goal": "Generate creative content",
            "memory": ["working", "long_term"],
            "handlers": ["on input"]
        }
    }
    
    @classmethod
    def get(cls, template_id: str) -> Dict[str, Any]:
        return cls.TEMPLATES.get(template_id, cls.TEMPLATES["assistant"])
    
    @classmethod
    def list_all(cls) -> List[str]:
        return list(cls.TEMPLATES.keys())


class AgentBuilder:
    """Build AION agents programmatically."""
    
    def __init__(self):
        self.name = "NewAgent"
        self.goal = ""
        self.memories = []
        self.tools = []
        self.handlers = []
        self.policies = []
        
    def set_name(self, name: str) -> 'AgentBuilder':
        self.name = name
        return self
    
    def set_goal(self, goal: str) -> 'AgentBuilder':
        self.goal = goal
        return self
    
    def add_memory(self, memory_type: str) -> 'AgentBuilder':
        if memory_type not in self.memories:
            self.memories.append(memory_type)
        return self
    
    def add_tool(self, tool_name: str) -> 'AgentBuilder':
        if tool_name not in self.tools:
            self.tools.append(tool_name)
        return self
    
    def add_handler(self, event: str, actions: List[str]) -> 'AgentBuilder':
        self.handlers.append({"event": event, "actions": actions})
        return self
    
    def add_policy(self, rule: str) -> 'AgentBuilder':
        self.policies.append(rule)
        return self
    
    def from_template(self, template_id: str) -> 'AgentBuilder':
        template = AgentTemplate.get(template_id)
        self.name = template.get("name", "Agent")
        self.goal = template.get("goal", "")
        self.memories = template.get("memory", [])
        self.tools = template.get("tools", [])
        return self
    
    def build(self) -> str:
        """Generate AION source code."""
        lines = [f'agent {self.name} {{']
        
        if self.goal:
            lines.append(f'  goal "{self.goal}"')
        
        for mem in self.memories:
            lines.append(f'  memory {mem}')
        
        for tool in self.tools:
            lines.append(f'  tool {tool}')
        
        if self.policies:
            lines.append('  policy {')
            for policy in self.policies:
                lines.append(f'    {policy}')
            lines.append('  }')
        
        for handler in self.handlers:
            event = handler["event"]
            lines.append(f'  on {event}(data):')
            for action in handler.get("actions", ["respond data"]):
                lines.append(f'    {action}')
        
        lines.append('}')
        return '\n'.join(lines)
    
    def to_json(self) -> str:
        """Export as JSON."""
        return json.dumps({
            "name": self.name,
            "goal": self.goal,
            "memories": self.memories,
            "tools": self.tools,
            "handlers": self.handlers,
            "policies": self.policies
        }, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'AgentBuilder':
        """Import from JSON."""
        data = json.loads(json_str)
        builder = cls()
        builder.name = data.get("name", "Agent")
        builder.goal = data.get("goal", "")
        builder.memories = data.get("memories", [])
        builder.tools = data.get("tools", [])
        builder.handlers = data.get("handlers", [])
        builder.policies = data.get("policies", [])
        return builder


def demo():
    """Demo agent builder."""
    print("🔧 AION Agent Builder Demo")
    print("-" * 50)
    
    # Build from template
    builder = AgentBuilder().from_template("researcher")
    builder.add_handler("input", ["think", "analyze data", "decide response", "respond response"])
    
    code = builder.build()
    print("\n📝 Generated AION Code:")
    print("-" * 30)
    print(code)
    
    # Custom build
    print("\n" + "=" * 50)
    print("🛠️ Custom Agent:")
    
    custom = (AgentBuilder()
        .set_name("ProteinAnalyzer")
        .set_goal("Analyze protein structures from AlphaFold")
        .add_memory("working")
        .add_memory("semantic")
        .add_tool("alphafold_db")
        .add_policy("trust_level = high")
        .add_handler("input", [
            "think 'Analyzing protein structure'",
            "analyze sequence",
            "decide structure_prediction",
            "respond structure_prediction"
        ]))
    
    print(custom.build())


if __name__ == "__main__":
    demo()
