"""
AION Reasoning Module
Implements first-class reasoning constructs: think, analyze, reflect, decide.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum
from datetime import datetime


class ReasoningError(Exception):
    """Raised when a reasoning operation fails."""
    pass


class ReasoningType(Enum):
    """Types of reasoning operations."""
    THINK = "think"
    ANALYZE = "analyze"
    REFLECT = "reflect"
    DECIDE = "decide"


@dataclass
class ReasoningStep:
    """Represents a single step in the reasoning process."""
    type: ReasoningType
    input: Any
    output: Any = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'type': self.type.value,
            'input': str(self.input),
            'output': str(self.output) if self.output else None,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


@dataclass
class ReasoningTrace:
    """
    Captures the full reasoning trace of an agent.
    Provides transparency and debuggability.
    """
    steps: list[ReasoningStep] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    
    def add_step(self, step: ReasoningStep) -> None:
        """Add a reasoning step to the trace."""
        self.steps.append(step)
    
    def complete(self) -> None:
        """Mark the reasoning trace as complete."""
        self.end_time = datetime.now()
    
    def to_text(self) -> str:
        """Convert trace to human-readable text."""
        lines = ["=== Reasoning Trace ==="]
        for i, step in enumerate(self.steps, 1):
            lines.append(f"\n[Step {i}] {step.type.value.upper()}")
            lines.append(f"  Input: {step.input}")
            if step.output:
                lines.append(f"  Output: {step.output}")
        return '\n'.join(lines)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'steps': [s.to_dict() for s in self.steps],
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': (
                (self.end_time - self.start_time).total_seconds()
                if self.end_time else None
            )
        }


class ReasoningEngine:
    """
    Executes reasoning operations for agents.
    Integrates with the model interface for LLM-powered reasoning.
    """
    
    def __init__(self, model_provider=None):
        self.model = model_provider
        self.current_trace: Optional[ReasoningTrace] = None
    
    def start_trace(self) -> ReasoningTrace:
        """Start a new reasoning trace."""
        self.current_trace = ReasoningTrace()
        return self.current_trace
    
    def end_trace(self) -> ReasoningTrace:
        """End the current trace and return it."""
        if self.current_trace:
            self.current_trace.complete()
        trace = self.current_trace
        self.current_trace = None
        return trace
    
    async def think(
        self,
        prompt: str = None,
        context: dict = None
    ) -> str:
        """
        Execute a 'think' operation.
        Generates internal reasoning without immediate output.
        
        Args:
            prompt: Optional guiding prompt for thinking
            context: Current context (memories, state, etc.)
        
        Returns:
            The reasoning result
        """
        context = context or {}
        
        system_prompt = """You are an AI assistant engaging in internal reasoning.
Think through the situation carefully, considering:
- What information do you have?
- What are the key aspects to consider?
- What are your preliminary thoughts?

Provide your internal reasoning process."""
        
        user_prompt = prompt or "Think about the current situation and context."
        if context:
            user_prompt += f"\n\nContext: {context}"
        
        result = await self._query_model(system_prompt, user_prompt)
        
        step = ReasoningStep(
            type=ReasoningType.THINK,
            input=prompt or "general thinking",
            output=result,
            metadata={'context_keys': list(context.keys())}
        )
        
        if self.current_trace:
            self.current_trace.add_step(step)
        
        return result
    
    async def analyze(
        self,
        target: Any,
        context: dict = None
    ) -> dict:
        """
        Execute an 'analyze' operation.
        Breaks down and examines the target in detail.
        
        Args:
            target: The subject to analyze
            context: Additional context
        
        Returns:
            Analysis results as a structured dictionary
        """
        context = context or {}
        
        system_prompt = """You are an AI assistant performing detailed analysis.
Analyze the given subject and provide:
1. Key components or aspects
2. Important observations
3. Potential issues or concerns
4. Opportunities or positive aspects
5. Summary assessment

Format your response as structured analysis."""
        
        user_prompt = f"Analyze the following:\n\n{target}"
        if context:
            user_prompt += f"\n\nAdditional context: {context}"
        
        result = await self._query_model(system_prompt, user_prompt)
        
        # Parse result into structured format
        analysis = {
            'target': str(target),
            'raw_analysis': result,
            'key_points': self._extract_key_points(result),
        }
        
        step = ReasoningStep(
            type=ReasoningType.ANALYZE,
            input=target,
            output=analysis,
            metadata={'context': context}
        )
        
        if self.current_trace:
            self.current_trace.add_step(step)
        
        return analysis
    
    async def reflect(
        self,
        target: Any = None,
        context: dict = None
    ) -> str:
        """
        Execute a 'reflect' operation.
        Introspective evaluation of past actions or current state.
        
        Args:
            target: Optional specific subject to reflect on
            context: Current context including past actions
        
        Returns:
            Reflection insights
        """
        context = context or {}
        
        system_prompt = """You are an AI assistant engaging in reflection.
Reflect on the situation, considering:
- What has happened so far?
- What worked well and what didn't?
- What have you learned?
- How might you approach things differently?
- What are the implications of your observations?

Provide thoughtful, introspective insights."""
        
        user_prompt = "Reflect on the current situation."
        if target:
            user_prompt = f"Reflect on the following:\n\n{target}"
        if context:
            user_prompt += f"\n\nContext: {context}"
        
        result = await self._query_model(system_prompt, user_prompt)
        
        step = ReasoningStep(
            type=ReasoningType.REFLECT,
            input=target or "general reflection",
            output=result,
            metadata={'context': context}
        )
        
        if self.current_trace:
            self.current_trace.add_step(step)
        
        return result
    
    async def decide(
        self,
        options: Any,
        context: dict = None
    ) -> dict:
        """
        Execute a 'decide' operation.
        Makes a choice between options with reasoning.
        
        Args:
            options: The options or question to decide on
            context: Context for decision making
        
        Returns:
            Decision with reasoning
        """
        context = context or {}
        
        system_prompt = """You are an AI assistant making a decision.
Evaluate the options carefully and:
1. Consider the pros and cons of each option
2. Apply relevant criteria from context
3. Make a clear decision
4. Explain your reasoning

Provide a clear decision with justification."""
        
        user_prompt = f"Decide on the following:\n\n{options}"
        if context:
            user_prompt += f"\n\nContext and criteria: {context}"
        
        result = await self._query_model(system_prompt, user_prompt)
        
        decision = {
            'options': str(options),
            'decision': self._extract_decision(result),
            'reasoning': result,
            'confidence': self._estimate_confidence(result),
        }
        
        step = ReasoningStep(
            type=ReasoningType.DECIDE,
            input=options,
            output=decision,
            metadata={'context': context}
        )
        
        if self.current_trace:
            self.current_trace.add_step(step)
        
        return decision
    
    async def _query_model(self, system_prompt: str, user_prompt: str) -> str:
        """Query the language model."""
        if self.model is None:
            # Fallback for no model configured
            return f"[Reasoning about: {user_prompt[:100]}...]"
        
        from .model_interface import Message
        
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt)
        ]
        
        result = await self.model.complete(messages)
        return result.content
    
    def _extract_key_points(self, text: str) -> list[str]:
        """Extract key points from analysis text."""
        # Simple extraction - in production, use NLP
        lines = text.split('\n')
        points = []
        for line in lines:
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('•') or 
                        (len(line) > 1 and line[0].isdigit() and line[1] in '.)')):
                points.append(line.lstrip('-•0123456789.) '))
        return points[:10]  # Limit to 10 points
    
    def _extract_decision(self, text: str) -> str:
        """Extract the main decision from reasoning text."""
        # Look for decision indicators
        lower_text = text.lower()
        
        indicators = [
            'i decide', 'my decision is', 'i choose', 'the best option is',
            'i recommend', 'the answer is', 'therefore'
        ]
        
        for indicator in indicators:
            if indicator in lower_text:
                idx = lower_text.index(indicator)
                # Extract sentence containing the decision
                end_idx = text.find('.', idx)
                if end_idx == -1:
                    end_idx = len(text)
                return text[idx:end_idx + 1].strip()
        
        # Fallback: return first sentence
        end = text.find('.')
        return text[:end + 1] if end != -1 else text[:100]
    
    def _estimate_confidence(self, text: str) -> float:
        """Estimate confidence level from reasoning text."""
        lower_text = text.lower()
        
        # High confidence indicators
        high_words = ['definitely', 'certainly', 'clearly', 'obviously', 'must']
        # Low confidence indicators
        low_words = ['maybe', 'perhaps', 'might', 'possibly', 'uncertain']
        
        high_count = sum(1 for w in high_words if w in lower_text)
        low_count = sum(1 for w in low_words if w in lower_text)
        
        # Base confidence
        confidence = 0.7
        confidence += high_count * 0.05
        confidence -= low_count * 0.1
        
        return max(0.1, min(1.0, confidence))
