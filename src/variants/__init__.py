"""
AION Tiered Agent Variants - Package Initialization
====================================================

Three-tier agent system matching GPT-5.2:
- Instant: Fast, optimized for daily tasks
- Thinking: Deep reasoning with latency
- Pro: Maximum intelligence for complexity
"""

from .instant import (
    InstantAgent,
    QuickResponse,
    StreamingGenerator
)

from .thinking import (
    ThinkingAgent,
    ReasoningChain,
    ThoughtProcess
)

from .pro import (
    ProAgent,
    DeepAnalysis,
    LongRunningTask
)

from .router import (
    VariantRouter,
    TaskComplexity,
    RoutingDecision
)

__all__ = [
    # Instant
    'InstantAgent',
    'QuickResponse',
    'StreamingGenerator',
    # Thinking
    'ThinkingAgent',
    'ReasoningChain',
    'ThoughtProcess',
    # Pro
    'ProAgent',
    'DeepAnalysis',
    'LongRunningTask',
    # Router
    'VariantRouter',
    'TaskComplexity',
    'RoutingDecision',
]
