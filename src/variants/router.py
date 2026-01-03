"""
AION Tiered Agent Variants - Router
====================================

Automatic tier selection based on task complexity:
- Task analysis and classification
- Dynamic escalation/de-escalation
- Load balancing across tiers
- Performance-based routing

Routes to Instant, Thinking, or Pro tiers.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from enum import Enum

from .instant import InstantAgent, QuickResponse
from .thinking import ThinkingAgent, ThoughtProcess
from .pro import ProAgent, DeepAnalysis, LongRunningTask


class TaskComplexity(Enum):
    """Task complexity levels."""
    SIMPLE = 1      # Quick lookup, simple Q&A
    MODERATE = 2    # Requires some reasoning
    COMPLEX = 3     # Multi-step analysis
    ADVANCED = 4    # Long-running, expert-level


@dataclass
class RoutingDecision:
    """Routing decision with reasoning."""
    tier: str = "instant"
    complexity: TaskComplexity = TaskComplexity.SIMPLE
    confidence: float = 1.0
    reasoning: str = ""
    estimated_tokens: int = 0
    estimated_latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class ComplexityAnalyzer:
    """Analyzes task complexity for routing."""
    
    # Complexity indicators
    SIMPLE_INDICATORS = [
        'what is', 'who is', 'when', 'where', 'define', 'translate',
        'quick', 'brief', 'simple', 'short', 'yes or no'
    ]
    
    MODERATE_INDICATORS = [
        'why', 'how does', 'explain', 'compare', 'difference',
        'summarize', 'list', 'steps', 'process'
    ]
    
    COMPLEX_INDICATORS = [
        'analyze', 'design', 'implement', 'architecture', 'optimize',
        'debug', 'evaluate', 'research', 'strategy', 'solve'
    ]
    
    ADVANCED_INDICATORS = [
        'complete project', 'multi-file', 'production', 'scalable',
        'enterprise', 'comprehensive', 'full implementation', 'long-running'
    ]
    
    def analyze(self, query: str) -> TaskComplexity:
        """Analyze query complexity."""
        query_lower = query.lower()
        
        # Count indicators
        simple_score = sum(1 for i in self.SIMPLE_INDICATORS if i in query_lower)
        moderate_score = sum(1 for i in self.MODERATE_INDICATORS if i in query_lower)
        complex_score = sum(1 for i in self.COMPLEX_INDICATORS if i in query_lower)
        advanced_score = sum(1 for i in self.ADVANCED_INDICATORS if i in query_lower)
        
        # Length factor
        word_count = len(query.split())
        if word_count > 100:
            complex_score += 2
        elif word_count > 50:
            moderate_score += 1
        
        # Question count
        question_count = query.count('?')
        if question_count > 3:
            moderate_score += 1
        
        # Determine complexity
        if advanced_score > 0:
            return TaskComplexity.ADVANCED
        elif complex_score >= 2:
            return TaskComplexity.COMPLEX
        elif moderate_score >= 2 or complex_score == 1:
            return TaskComplexity.MODERATE
        else:
            return TaskComplexity.SIMPLE
    
    def estimate_tokens(self, query: str, complexity: TaskComplexity) -> int:
        """Estimate output tokens based on complexity."""
        base = len(query) // 4  # Query tokens
        
        multipliers = {
            TaskComplexity.SIMPLE: 2,
            TaskComplexity.MODERATE: 5,
            TaskComplexity.COMPLEX: 10,
            TaskComplexity.ADVANCED: 20
        }
        
        return base * multipliers.get(complexity, 5)
    
    def estimate_latency(self, complexity: TaskComplexity) -> float:
        """Estimate latency in ms."""
        latencies = {
            TaskComplexity.SIMPLE: 500,
            TaskComplexity.MODERATE: 2000,
            TaskComplexity.COMPLEX: 10000,
            TaskComplexity.ADVANCED: 60000
        }
        return latencies.get(complexity, 1000)


class VariantRouter:
    """Routes tasks to appropriate agent tier."""
    
    def __init__(self):
        self.instant = InstantAgent()
        self.thinking = ThinkingAgent()
        self.pro = ProAgent()
        self.analyzer = ComplexityAnalyzer()
        self.routing_history: List[RoutingDecision] = []
    
    def route(self, query: str, 
              force_tier: str = None) -> RoutingDecision:
        """Decide which tier to route to."""
        if force_tier:
            complexity = self._complexity_for_tier(force_tier)
            return RoutingDecision(
                tier=force_tier,
                complexity=complexity,
                confidence=1.0,
                reasoning=f"Forced to {force_tier} tier"
            )
        
        complexity = self.analyzer.analyze(query)
        
        tier_map = {
            TaskComplexity.SIMPLE: "instant",
            TaskComplexity.MODERATE: "thinking",
            TaskComplexity.COMPLEX: "thinking",
            TaskComplexity.ADVANCED: "pro"
        }
        
        tier = tier_map.get(complexity, "instant")
        
        decision = RoutingDecision(
            tier=tier,
            complexity=complexity,
            confidence=self._calculate_confidence(query, complexity),
            reasoning=self._generate_reasoning(complexity),
            estimated_tokens=self.analyzer.estimate_tokens(query, complexity),
            estimated_latency_ms=self.analyzer.estimate_latency(complexity)
        )
        
        self.routing_history.append(decision)
        return decision
    
    def _complexity_for_tier(self, tier: str) -> TaskComplexity:
        """Get complexity for a given tier."""
        tier_to_complexity = {
            "instant": TaskComplexity.SIMPLE,
            "thinking": TaskComplexity.COMPLEX,
            "pro": TaskComplexity.ADVANCED
        }
        return tier_to_complexity.get(tier, TaskComplexity.SIMPLE)
    
    def _calculate_confidence(self, query: str, complexity: TaskComplexity) -> float:
        """Calculate routing confidence."""
        # Simple queries are high confidence
        if complexity == TaskComplexity.SIMPLE:
            return 0.95
        elif complexity == TaskComplexity.MODERATE:
            return 0.85
        elif complexity == TaskComplexity.COMPLEX:
            return 0.75
        else:
            return 0.70
    
    def _generate_reasoning(self, complexity: TaskComplexity) -> str:
        """Generate routing reasoning."""
        reasons = {
            TaskComplexity.SIMPLE: "Simple query - routed to Instant for fast response",
            TaskComplexity.MODERATE: "Moderate complexity - routed to Thinking for reasoning",
            TaskComplexity.COMPLEX: "Complex analysis needed - routed to Thinking tier",
            TaskComplexity.ADVANCED: "Advanced task - routed to Pro for expert handling"
        }
        return reasons.get(complexity, "Default routing")
    
    async def process(self, query: str, 
                      force_tier: str = None,
                      stream: bool = False) -> Union[QuickResponse, ThoughtProcess, DeepAnalysis]:
        """Route and process a query."""
        decision = self.route(query, force_tier)
        
        if decision.tier == "instant":
            return await self.instant.respond(query)
        elif decision.tier == "thinking":
            return await self.thinking.think(query)
        elif decision.tier == "pro":
            return await self.pro.analyze(query)
        else:
            return await self.instant.respond(query)
    
    def escalate(self, current_tier: str) -> str:
        """Escalate to a higher tier."""
        escalation = {
            "instant": "thinking",
            "thinking": "pro",
            "pro": "pro"  # Can't escalate further
        }
        return escalation.get(current_tier, current_tier)
    
    def deescalate(self, current_tier: str) -> str:
        """De-escalate to a lower tier."""
        deescalation = {
            "instant": "instant",  # Already lowest
            "thinking": "instant",
            "pro": "thinking"
        }
        return deescalation.get(current_tier, current_tier)
    
    def get_tier_stats(self) -> Dict[str, Any]:
        """Get statistics per tier."""
        tier_counts = {"instant": 0, "thinking": 0, "pro": 0}
        
        for decision in self.routing_history:
            tier_counts[decision.tier] = tier_counts.get(decision.tier, 0) + 1
        
        total = len(self.routing_history)
        
        return {
            'total_requests': total,
            'tier_distribution': {
                tier: count / total if total > 0 else 0
                for tier, count in tier_counts.items()
            },
            'tier_counts': tier_counts,
            'average_confidence': sum(d.confidence for d in self.routing_history) / total if total > 0 else 0
        }
    
    def get_recommended_tier(self, task_description: str,
                              constraints: Dict[str, Any] = None) -> str:
        """Get recommended tier based on constraints."""
        constraints = constraints or {}
        
        decision = self.route(task_description)
        tier = decision.tier
        
        # Apply constraints
        if constraints.get('max_latency_ms'):
            max_latency = constraints['max_latency_ms']
            if decision.estimated_latency_ms > max_latency:
                tier = self.deescalate(tier)
        
        if constraints.get('min_quality'):
            # Higher tiers for quality requirements
            if constraints['min_quality'] > 0.9:
                tier = "pro"
            elif constraints['min_quality'] > 0.7:
                tier = max(tier, "thinking", key=lambda x: {"instant": 1, "thinking": 2, "pro": 3}.get(x, 1))
        
        return tier


async def demo_router():
    """Demonstrate variant router."""
    print("🔀 Variant Router Demo")
    print("=" * 50)
    
    router = VariantRouter()
    
    queries = [
        ("What is Python?", None),
        ("Explain the theory of relativity", None),
        ("Design a microservices architecture for high-availability e-commerce", None),
        ("Translate 'hello' to French", None),
        ("Analyze this code and fix all bugs in the multi-file project", None)
    ]
    
    print("\n🔄 Routing queries to appropriate tiers...\n")
    
    for query, force_tier in queries:
        decision = router.route(query, force_tier)
        
        print(f"📝 Query: {query[:50]}...")
        print(f"   → Tier: {decision.tier.upper()}")
        print(f"   → Complexity: {decision.complexity.name}")
        print(f"   → Confidence: {decision.confidence:.0%}")
        print(f"   → Est. Latency: {decision.estimated_latency_ms:.0f}ms")
        print(f"   → Reasoning: {decision.reasoning}")
        print()
    
    # Process a query
    print("⚡ Processing query through router...")
    result = await router.process("What is machine learning?")
    print(f"   Response type: {type(result).__name__}")
    
    # Escalation
    print("\n📈 Escalation paths:")
    print(f"   instant → {router.escalate('instant')}")
    print(f"   thinking → {router.escalate('thinking')}")
    print(f"   pro → {router.escalate('pro')}")
    
    # Stats
    stats = router.get_tier_stats()
    print(f"\n📊 Tier Statistics:")
    print(f"   Total Requests: {stats['total_requests']}")
    print(f"   Distribution: {stats['tier_distribution']}")
    print(f"   Average Confidence: {stats['average_confidence']:.0%}")
    
    print("\n✅ Router demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_router())
