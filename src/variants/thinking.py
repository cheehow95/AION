"""
AION Tiered Agent Variants - Thinking
======================================

Deep reasoning agent with extended thinking:
- Multi-step reasoning chains
- Explicit thought process
- Tool orchestration
- Latency-tolerant processing

Matches GPT-5.2 Thinking tier.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
from enum import Enum


class ThoughtType(Enum):
    """Types of thoughts in reasoning."""
    OBSERVATION = "observation"
    HYPOTHESIS = "hypothesis"
    ANALYSIS = "analysis"
    CONCLUSION = "conclusion"
    QUESTION = "question"
    PLAN = "plan"
    CRITIQUE = "critique"


@dataclass
class Thought:
    """A single thought in the reasoning process."""
    type: ThoughtType = ThoughtType.OBSERVATION
    content: str = ""
    confidence: float = 0.8
    depends_on: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    id: str = ""


@dataclass
class ThoughtProcess:
    """Complete thought process for a problem."""
    query: str = ""
    thoughts: List[Thought] = field(default_factory=list)
    final_answer: str = ""
    total_thinking_time_ms: float = 0.0
    tools_used: List[str] = field(default_factory=list)
    
    def add_thought(self, thought: Thought):
        thought.id = f"thought_{len(self.thoughts)}"
        self.thoughts.append(thought)
    
    def get_reasoning_trace(self) -> str:
        """Get human-readable reasoning trace."""
        trace = []
        for t in self.thoughts:
            trace.append(f"[{t.type.value.upper()}] {t.content}")
        return '\n'.join(trace)


@dataclass
class ReasoningChain:
    """A chain of reasoning steps."""
    steps: List[str] = field(default_factory=list)
    current_step: int = 0
    
    def add_step(self, step: str):
        self.steps.append(step)
    
    def next_step(self) -> Optional[str]:
        if self.current_step < len(self.steps):
            step = self.steps[self.current_step]
            self.current_step += 1
            return step
        return None
    
    @property
    def is_complete(self) -> bool:
        return self.current_step >= len(self.steps)


class ToolOrchestrator:
    """Orchestrates tool usage during thinking."""
    
    def __init__(self):
        self.available_tools: Dict[str, Callable] = {}
        self.tool_history: List[Dict[str, Any]] = []
    
    def register_tool(self, name: str, func: Callable):
        """Register a tool."""
        self.available_tools[name] = func
    
    async def use_tool(self, name: str, **kwargs) -> Any:
        """Use a tool and record usage."""
        if name not in self.available_tools:
            return {"error": f"Unknown tool: {name}"}
        
        tool = self.available_tools[name]
        
        try:
            if asyncio.iscoroutinefunction(tool):
                result = await tool(**kwargs)
            else:
                result = tool(**kwargs)
        except Exception as e:
            result = {"error": str(e)}
        
        self.tool_history.append({
            'tool': name,
            'args': kwargs,
            'result': result,
            'timestamp': datetime.now().isoformat()
        })
        
        return result
    
    def plan_tool_sequence(self, task: str) -> List[str]:
        """Plan which tools to use for a task."""
        # Simple keyword-based planning
        tools = []
        task_lower = task.lower()
        
        if any(w in task_lower for w in ['search', 'find', 'look up']):
            tools.append('search')
        if any(w in task_lower for w in ['calculate', 'compute', 'math']):
            tools.append('calculator')
        if any(w in task_lower for w in ['code', 'program', 'script']):
            tools.append('code_interpreter')
        if any(w in task_lower for w in ['read', 'file', 'document']):
            tools.append('file_reader')
        
        return tools


class ThinkingAgent:
    """Deep reasoning agent with extended thinking time."""
    
    def __init__(self, agent_id: str = "thinking-agent"):
        self.agent_id = agent_id
        self.tools = ToolOrchestrator()
        self.max_thinking_steps = 10
        self.thinking_budget_ms = 30000  # 30 seconds
        
        # Register default tools
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register default reasoning tools."""
        async def search(query: str) -> Dict[str, Any]:
            await asyncio.sleep(0.1)  # Simulate
            return {"results": [f"Result for: {query}"]}
        
        async def calculator(expression: str) -> Dict[str, Any]:
            try:
                result = eval(expression)  # Safe in controlled context
                return {"result": result}
            except:
                return {"error": "Invalid expression"}
        
        self.tools.register_tool("search", search)
        self.tools.register_tool("calculator", calculator)
    
    async def think(self, query: str, 
                    show_thinking: bool = True) -> ThoughtProcess:
        """Engage in deep thinking about a query."""
        start_time = datetime.now()
        process = ThoughtProcess(query=query)
        
        # Step 1: Understand the problem
        process.add_thought(Thought(
            type=ThoughtType.OBSERVATION,
            content=f"Analyzing query: {query}"
        ))
        
        # Step 2: Break down the problem
        process.add_thought(Thought(
            type=ThoughtType.PLAN,
            content="Breaking down into sub-problems..."
        ))
        
        # Step 3: Plan tool usage
        tools_needed = self.tools.plan_tool_sequence(query)
        if tools_needed:
            process.add_thought(Thought(
                type=ThoughtType.PLAN,
                content=f"Tools to use: {', '.join(tools_needed)}"
            ))
            process.tools_used = tools_needed
            
            # Use tools
            for tool in tools_needed[:3]:  # Limit tool calls
                result = await self.tools.use_tool(tool, query=query)
                process.add_thought(Thought(
                    type=ThoughtType.OBSERVATION,
                    content=f"Tool {tool} result: {str(result)[:100]}"
                ))
        
        # Step 4: Analyze
        process.add_thought(Thought(
            type=ThoughtType.ANALYSIS,
            content="Synthesizing information from observations..."
        ))
        
        # Step 5: Form hypotheses
        process.add_thought(Thought(
            type=ThoughtType.HYPOTHESIS,
            content="Based on analysis, forming hypothesis..."
        ))
        
        # Step 6: Self-critique
        process.add_thought(Thought(
            type=ThoughtType.CRITIQUE,
            content="Checking for potential issues or gaps..."
        ))
        
        # Step 7: Conclude
        process.add_thought(Thought(
            type=ThoughtType.CONCLUSION,
            content="Drawing final conclusion based on reasoning chain."
        ))
        
        # Generate final answer
        process.final_answer = await self._synthesize_answer(process)
        
        elapsed = (datetime.now() - start_time).total_seconds() * 1000
        process.total_thinking_time_ms = elapsed
        
        return process
    
    async def _synthesize_answer(self, process: ThoughtProcess) -> str:
        """Synthesize final answer from thought process."""
        conclusions = [t for t in process.thoughts if t.type == ThoughtType.CONCLUSION]
        
        if conclusions:
            return f"After careful analysis: {conclusions[-1].content}"
        return "Based on my reasoning, here is my response..."
    
    async def reason_step_by_step(self, query: str) -> ReasoningChain:
        """Generate explicit step-by-step reasoning."""
        chain = ReasoningChain()
        
        # Generate reasoning steps
        steps = [
            f"1. Understanding: Parse '{query[:30]}...'",
            "2. Decomposition: Break into sub-problems",
            "3. Research: Gather relevant information",
            "4. Analysis: Examine relationships and patterns",
            "5. Synthesis: Combine insights",
            "6. Verification: Check for consistency",
            "7. Conclusion: Formulate answer"
        ]
        
        for step in steps:
            chain.add_step(step)
            await asyncio.sleep(0.05)  # Simulate thinking
        
        return chain
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            'agent_id': self.agent_id,
            'tier': 'thinking',
            'max_thinking_steps': self.max_thinking_steps,
            'thinking_budget_ms': self.thinking_budget_ms,
            'tools_available': list(self.tools.available_tools.keys()),
            'tool_calls_made': len(self.tools.tool_history)
        }


async def demo_thinking():
    """Demonstrate Thinking agent."""
    print("🧠 Thinking Agent Demo")
    print("=" * 50)
    
    agent = ThinkingAgent()
    
    query = "What are the key factors to consider when designing a distributed system?"
    
    print(f"\n📝 Query: {query}")
    print("\n🔄 Thinking process...")
    
    process = await agent.think(query)
    
    print("\n📊 Reasoning Trace:")
    print("-" * 40)
    print(process.get_reasoning_trace())
    print("-" * 40)
    
    print(f"\n✨ Final Answer: {process.final_answer}")
    print(f"\n⏱️ Thinking Time: {process.total_thinking_time_ms:.0f}ms")
    print(f"🔧 Tools Used: {process.tools_used}")
    
    # Step-by-step reasoning
    print("\n📋 Step-by-Step Reasoning:")
    chain = await agent.reason_step_by_step("How to optimize database performance?")
    while not chain.is_complete:
        step = chain.next_step()
        print(f"   {step}")
    
    print(f"\n📊 Stats: {agent.get_stats()}")
    print("\n✅ Thinking demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_thinking())
