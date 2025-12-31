"""
AION Reflexion Architecture
Implements self-correction and iterative refinement loops.
Allows agents to critique their own output and improve it.
"""

from typing import Any, Callable, List, Dict, Optional
from dataclasses import dataclass, field
import asyncio

@dataclass
class ReflexionTrace:
    attempt: int
    output: Any
    critique: str
    score: float
    timestamp: float

class ReflexionLoop:
    """
    Implements the Reflexion pattern:
    Generate -> Critique -> Curate -> Refine
    """
    
    def __init__(self, 
                 generator: Callable[[Any], Any],
                 evaluator: Callable[[Any], float],
                 critique_model: Callable[[Any], str],
                 max_attempts: int = 3,
                 min_score: float = 0.8):
        self.generator = generator
        self.evaluator = evaluator
        self.critique_model = critique_model
        self.max_attempts = max_attempts
        self.min_score = min_score
        self.traces: List[ReflexionTrace] = []

    async def run(self, input_data: Any) -> Any:
        """Run the self-correction loop."""
        current_input = input_data
        best_output = None
        best_score = -1.0
        
        for attempt in range(1, self.max_attempts + 1):
            # 1. Generate
            output = await self._call_async(self.generator, current_input)
            
            # 2. Evaluate
            score = await self._call_async(self.evaluator, output)
            
            # 3. Critique
            critique = await self._call_async(self.critique_model, output)
            
            # Record trace
            import time
            trace = ReflexionTrace(
                attempt=attempt,
                output=str(output)[:100] + "...",
                critique=critique,
                score=score,
                timestamp=time.time()
            )
            self.traces.append(trace)
            
            # Check if good enough
            if score >= self.min_score:
                return output
                
            # Track best result
            if score > best_score:
                best_score = score
                best_output = output
            
            # 4. Refine input for next attempt (add critique to context)
            if attempt < self.max_attempts:
                current_input = self._RefineInput(input_data, critique, output)
        
        return best_output

    async def _call_async(self, func, *args):
        """Helper to call both sync and async functions."""
        if asyncio.iscoroutinefunction(func):
            return await func(*args)
        return func(*args)
        
    def _RefineInput(self, original_input: Any, critique: str, previous_output: Any) -> str:
        """Create a refined prompt including the critique."""
        return f"""
Original Task: {original_input}

Previous Attempt:
{previous_output}

Critique (Why it was insufficient):
{critique}

Instruction:
Please try again, addressing the critique above to improve the result.
"""

# Helper function to create a standard self-correcting agent
async def create_reflective_agent(name: str, goal: str, engine):
    """
    Creates a ReflexionLoop for an agent using the AION engine.
    """
    async def generate(prompt):
        return engine.think(prompt)
        
    async def evaluate(output):
        # Self-evaluation prompting
        analysis = engine.analyze(output)
        # Simple heuristic: longer and more positive sentiment is better
        score = 0.5
        if analysis['sentiment'] == 'positive': score += 0.2
        if analysis['length'] > 50: score += 0.2
        return min(1.0, score)
        
    async def critique(output):
        return engine.think(f"Critique this response for accuracy and clarity: {output}")
        
    return ReflexionLoop(generate, evaluate, critique)
