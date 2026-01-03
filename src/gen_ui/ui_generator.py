"""
AION Generative UI Engine - UI Generator
=========================================

Dynamically generates React/HTML components from user intent.
Simulates Gemini 3's ability to create bespoke interfaces instantly.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import uuid
import json

@dataclass
class ComponentSpec:
    """Specification for a UI component."""
    type: str  # 'button', 'input', 'chart', 'container'
    props: Dict[str, Any] = field(default_factory=dict)
    children: List['ComponentSpec'] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

@dataclass
class UIComponent:
    """A generated UI component ready for rendering."""
    spec: ComponentSpec
    code: str
    dependencies: List[str]

class UIGenerator:
    """
    Translates high-level intent into executable UI code.
    """
    
    async def generate(self, intent: str, context: Dict[str, Any] = None) -> UIComponent:
        """
        Generate a UI component from intent.
        Mocks the LLM generation process.
        """
        # Mock logic based on keywords
        if "calculator" in intent.lower():
            return self._generate_calculator()
        elif "dashboard" in intent.lower():
            return self._generate_dashboard()
        else:
            return self._generate_generic_card(intent)

    def _generate_calculator(self) -> UIComponent:
        spec = ComponentSpec(
            type="container",
            props={"layout": "grid"},
            children=[
                ComponentSpec(type="display", props={"value": "0"}),
                ComponentSpec(type="button", props={"label": "1"}),
                ComponentSpec(type="button", props={"label": "+"}),
            ]
        )
        code = """
        function Calculator() {
            const [val, setVal] = useState(0);
            return (
                <div className="p-4 border rounded">
                    <div className="bg-gray-100 p-2 mb-2">{val}</div>
                    <div className="grid grid-cols-4 gap-2">
                        <button onClick={() => setVal(val + 1)}>1</button>
                        <button onClick={() => setVal(val + 1)}>+</button>
                    </div>
                </div>
            );
        }
        """
        return UIComponent(spec=spec, code=code, dependencies=["react", "lucide-react"])

    def _generate_dashboard(self) -> UIComponent:
        spec = ComponentSpec(
            type="dashboard",
            children=[
                ComponentSpec(type="chart", props={"type": "line"}),
                ComponentSpec(type="stats_card", props={"title": "Users"})
            ]
        )
        code = """
        function Dashboard() {
            return (
                <div className="grid grid-cols-2 gap-4">
                    <Card title="Revenue"><LineChart data={[10, 20, 30]} /></Card>
                    <Card title="Active Users">1,234</Card>
                </div>
            )
        }
        """
        return UIComponent(spec=spec, code=code, dependencies=["recharts"])

    def _generate_generic_card(self, content: str) -> UIComponent:
        spec = ComponentSpec(
            type="card",
            props={"content": content}
        )
        code = f"""
        function InfoCard() {{
            return (
                <div className="p-4 shadow rounded bg-white">
                    <p>{content}</p>
                </div>
            )
        }}
        """
        return UIComponent(spec=spec, code=code, dependencies=[])

async def demo_ui_generator():
    """Demonstrate UI generation."""
    generator = UIGenerator()
    
    # Generate calculator
    calc = await generator.generate("I need a simple calculator")
    print(f"Generated UI: Calculator")
    print(f"Dependencies: {calc.dependencies}")
    print(f"Code Snippet:\n{calc.code[:100]}...")

if __name__ == "__main__":
    asyncio.run(demo_ui_generator())
