"""
AION Fast Agent Runner
Execute AION agents instantly without external APIs.
Optimized for speed with local reasoning.
"""

import sys
import asyncio
from typing import Any, Optional
from datetime import datetime
import time

sys.path.insert(0, '.')

from src.parser import parse
from src.runtime import Environment, create_memory
from src.runtime.local_engine import LocalReasoningEngine


class FastAgentRunner:
    """
    Ultra-fast AION agent executor.
    Uses local reasoning engine for instant responses.
    """
    
    def __init__(self):
        self.env = Environment()
        self.engine = LocalReasoningEngine()
        self.agents: dict[str, dict] = {}
        self.output: list[str] = []
        self.stats = {
            'parse_time': 0,
            'exec_time': 0,
            'total_steps': 0,
        }
    
    def load(self, source: str) -> list[str]:
        """Load AION source code and return agent names."""
        start = time.perf_counter()
        
        program = parse(source)
        agent_names = []
        
        for decl in program.declarations:
            class_name = decl.__class__.__name__
            
            if class_name == 'AgentDecl':
                agent = self._process_agent(decl)
                self.agents[decl.name] = agent
                agent_names.append(decl.name)
            elif class_name == 'ModelDecl':
                # Store model config
                self.env.define(decl.name, decl.config, 'model')
            elif class_name == 'ToolDecl':
                # Register tool
                self.env.define(decl.name, decl.config, 'tool')
        
        self.stats['parse_time'] = time.perf_counter() - start
        return agent_names
    
    def _process_agent(self, decl) -> dict:
        """Process agent declaration into executable form."""
        agent = {
            'name': decl.name,
            'goal': '',
            'memories': {},
            'model': None,
            'tools': [],
            'policy': {},
            'handlers': {},
        }
        
        for member in decl.body:
            class_name = member.__class__.__name__
            
            if class_name == 'GoalStmt':
                agent['goal'] = member.goal
            elif class_name == 'MemoryDecl':
                mem = create_memory(member.memory_type, config=member.config)
                agent['memories'][member.memory_type] = mem
            elif class_name == 'ModelRef':
                agent['model'] = member.name
            elif class_name == 'ToolRef':
                agent['tools'].append(member.name)
            elif class_name == 'PolicyDecl':
                agent['policy'].update(member.config)
            elif class_name == 'EventHandler':
                agent['handlers'][member.event_type] = {
                    'params': member.params,
                    'body': member.body,
                }
        
        return agent
    
    async def run(self, agent_name: str, input_data: Any = None) -> dict:
        """Run an agent with input."""
        start = time.perf_counter()
        
        if agent_name not in self.agents:
            return {'error': f"Agent not found: {agent_name}"}
        
        agent = self.agents[agent_name]
        self.output = []
        self.engine.clear_trace()
        
        # Set up context
        context = {
            'agent_name': agent['name'],
            'goal': agent['goal'],
            'input': input_data,
        }
        
        # Execute input handler
        if 'input' in agent['handlers']:
            handler = agent['handlers']['input']
            
            # Bind parameters
            if handler['params'] and input_data is not None:
                context[handler['params'][0]] = input_data
            
            # Execute statements
            await self._execute_block(handler['body'], context, agent)
        
        self.stats['exec_time'] = time.perf_counter() - start
        
        return {
            'output': self.output,
            'trace': self.engine.get_trace(),
            'stats': {
                'parse_time_ms': round(self.stats['parse_time'] * 1000, 2),
                'exec_time_ms': round(self.stats['exec_time'] * 1000, 2),
                'total_steps': self.stats['total_steps'],
            }
        }
    
    async def _execute_block(self, statements: list, context: dict, agent: dict):
        """Execute a block of statements."""
        for stmt in statements:
            await self._execute_statement(stmt, context, agent)
    
    async def _execute_statement(self, stmt, context: dict, agent: dict):
        """Execute a single statement."""
        self.stats['total_steps'] += 1
        class_name = stmt.__class__.__name__
        
        if class_name == 'ThinkStmt':
            result = self.engine.think(stmt.prompt, context)
            context['_last_thought'] = result
            
        elif class_name == 'AnalyzeStmt':
            target = self._eval_expr(stmt.target, context)
            result = self.engine.analyze(target, context)
            context['_analysis'] = result
            
        elif class_name == 'ReflectStmt':
            target = self._eval_expr(stmt.target, context) if stmt.target else None
            result = self.engine.reflect(target, context)
            context['_reflection'] = result
            
        elif class_name == 'DecideStmt':
            target = self._eval_expr(stmt.target, context)
            result = self.engine.decide(target, context)
            context['_decision'] = result
            context[str(stmt.target.name) if hasattr(stmt.target, 'name') else 'answer'] = result['decision']
            
        elif class_name == 'IfStmt':
            condition = self._eval_expr(stmt.condition, context)
            if self._is_truthy(condition):
                await self._execute_block(stmt.then_body, context, agent)
            elif stmt.else_body:
                await self._execute_block(stmt.else_body, context, agent)
                
        elif class_name == 'WhenStmt':
            condition = self._eval_expr(stmt.condition, context)
            if self._is_truthy(condition):
                await self._execute_block(stmt.body, context, agent)
                
        elif class_name == 'RepeatStmt':
            times = int(self._eval_expr(stmt.times, context)) if stmt.times else 1
            for _ in range(times):
                await self._execute_block(stmt.body, context, agent)
                
        elif class_name == 'UseStmt':
            args = [self._eval_expr(a, context) for a in stmt.args]
            result = self._execute_tool(stmt.tool_name, args, agent)
            context['_tool_result'] = result
            
        elif class_name == 'RespondStmt':
            if stmt.value:
                value = self._eval_expr(stmt.value, context)
                self.output.append(str(value))
            else:
                # Respond with decision or last result
                if '_decision' in context:
                    self.output.append(str(context['_decision']['decision']))
                    
        elif class_name == 'StoreStmt':
            value = self._eval_expr(stmt.value, context)
            memory_name = stmt.memory_name or 'working'
            if memory_name in agent['memories']:
                agent['memories'][memory_name].store(value)
                
        elif class_name == 'RecallStmt':
            memory_name = stmt.memory_name or 'working'
            if memory_name in agent['memories']:
                entries = agent['memories'][memory_name].recall()
                context['_recalled'] = entries
                
        elif class_name == 'AssignStmt':
            value = self._eval_expr(stmt.value, context)
            context[stmt.name] = value
    
    def _eval_expr(self, expr, context: dict) -> Any:
        """Evaluate an expression."""
        if expr is None:
            return None
            
        class_name = expr.__class__.__name__
        
        if class_name == 'Literal':
            return expr.value
            
        elif class_name == 'Identifier':
            name = expr.name
            if name in context:
                return context[name]
            return name  # Return as string if not found
            
        elif class_name == 'BinaryExpr':
            left = self._eval_expr(expr.left, context)
            right = self._eval_expr(expr.right, context)
            
            ops = {
                '+': lambda a, b: a + b,
                '-': lambda a, b: a - b,
                '*': lambda a, b: a * b,
                '/': lambda a, b: a / b,
                '==': lambda a, b: a == b,
                '!=': lambda a, b: a != b,
                '<': lambda a, b: a < b,
                '>': lambda a, b: a > b,
                '<=': lambda a, b: a <= b,
                '>=': lambda a, b: a >= b,
                'and': lambda a, b: a and b,
                'or': lambda a, b: a or b,
            }
            
            return ops.get(expr.operator, lambda a, b: None)(left, right)
            
        elif class_name == 'UnaryExpr':
            operand = self._eval_expr(expr.operand, context)
            if expr.operator == 'not':
                return not operand
            elif expr.operator == '-':
                return -operand
                
        elif class_name == 'MemberAccess':
            obj = self._eval_expr(expr.object, context)
            if isinstance(obj, dict):
                return obj.get(expr.member)
            return getattr(obj, expr.member, None)
            
        elif class_name == 'ListLiteral':
            return [self._eval_expr(e, context) for e in expr.elements]
        
        return None
    
    def _execute_tool(self, tool_name: str, args: list, agent: dict) -> Any:
        """Execute a built-in tool."""
        if tool_name == 'calculator':
            if args:
                try:
                    return eval(str(args[0]))
                except:
                    return "Error in calculation"
        elif tool_name == 'time':
            return datetime.now().strftime('%H:%M:%S')
        elif tool_name == 'date':
            return datetime.now().strftime('%Y-%m-%d')
        
        return f"Tool '{tool_name}' executed with args: {args}"
    
    def _is_truthy(self, value: Any) -> bool:
        """Check if value is truthy."""
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return len(value) > 0 and value.lower() not in ('false', 'no', '0')
        if isinstance(value, (list, dict)):
            return len(value) > 0
        return True


async def run_agent_file(filepath: str, input_data: str = None):
    """Run an AION file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()
    
    runner = FastAgentRunner()
    agents = runner.load(source)
    
    if not agents:
        print("No agents found in file.")
        return
    
    print(f"Loaded agents: {', '.join(agents)}")
    print(f"Parse time: {runner.stats['parse_time']*1000:.2f}ms")
    print("-" * 50)
    
    # Run first agent
    result = await runner.run(agents[0], input_data)
    
    print("\nOutput:")
    for line in result['output']:
        print(f"  {line}")
    
    print(f"\nExecution time: {result['stats']['exec_time_ms']:.2f}ms")
    print(f"Total steps: {result['stats']['total_steps']}")
    
    if result['trace']:
        print("\nReasoning trace:")
        for step in result['trace'][:5]:
            print(f"  [{step['type'].upper()}] {str(step.get('output', ''))[:60]}...")


async def interactive_mode():
    """Interactive agent mode."""
    print("""
╔═══════════════════════════════════════════════════════════╗
║   AION Fast Agent Runner - No API Required                ║
╚═══════════════════════════════════════════════════════════╝
""")
    
    # Create a simple interactive agent
    source = '''
agent Assistant {
  goal "Help users quickly and efficiently"
  memory working
  
  on input(message):
    think
    analyze message
    decide response
    respond response
}
'''
    
    runner = FastAgentRunner()
    runner.load(source)
    
    print("Agent loaded. Type your message (or 'quit' to exit):\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ('quit', 'exit', 'q'):
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            start = time.perf_counter()
            result = await runner.run('Assistant', user_input)
            elapsed = (time.perf_counter() - start) * 1000
            
            for line in result['output']:
                print(f"Agent: {line}")
            
            print(f"  [{elapsed:.1f}ms, {result['stats']['total_steps']} steps]\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        input_data = sys.argv[2] if len(sys.argv) > 2 else None
        asyncio.run(run_agent_file(filepath, input_data))
    else:
        asyncio.run(interactive_mode())
