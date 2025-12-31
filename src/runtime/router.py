"""
AION Adaptive Model Router
Intelligent routing system that decides whether to use:
1. Local Reasoning Engine (Instant, Free, Privacy-Preserving)
2. External Cloud LLM (Deep Reasoning, Knowledge-Heavy, Costly)

Decision is based on task complexity, constraints, and content analysis.
"""

from typing import Any, Dict, Optional
from dataclasses import dataclass
import re
from .local_engine import LocalReasoningEngine
from .model_interface import ModelRegistry

@dataclass
class RoutingDecision:
    provider: str  # 'local' or 'cloud'
    model: str     # specific model name
    reasoning: str
    confidence: float

class AdaptiveRouter:
    """
    Routes queries between local engine and cloud models.
    """
    
    def __init__(self, local_engine: LocalReasoningEngine, model_registry: ModelRegistry):
        self.local = local_engine
        self.registry = model_registry
        
        # Complexity indicators (regex patterns)
        self.complex_patterns = [
            r"explain.*detailed", 
            r"write.*essay",
            r"code.*complex",
            r"analyze.*implications",
            r"creative.*story",
            r"translate",
            r"summarize.*long",
        ]
        
    def route(self, prompt: str, context: Dict[str, Any] = None) -> RoutingDecision:
        """Decide which brain to use."""
        context = context or {}
        
        # 1. Check for hard constraints
        if context.get('privacy') == 'strict':
            return RoutingDecision('local', 'local-engine', "Strict privacy required", 1.0)
            
        if context.get('latency') == 'realtime':
            return RoutingDecision('local', 'local-engine', "Realtime latency required", 1.0)
            
        # 2. Analyze complexity
        is_complex = self._is_complex_task(prompt)
        
        # 3. Check local capability
        # Can the local engine handle this mathematically or structurally?
        local_analysis = self.local.analyze(prompt)
        
        if local_analysis['type'] in ('math_expression', 'greeting', 'help_request'):
             return RoutingDecision('local', 'local-engine', f"Task type '{local_analysis['type']}' handled locally", 0.95)
             
        # 4. Make decision
        if is_complex:
            return RoutingDecision('cloud', 'best_available', "High complexity detected", 0.8)
        else:
            # Attempt local first pattern match
            return RoutingDecision('local', 'local-engine', "Low complexity, attempting local first", 0.6)
            
    def _is_complex_task(self, text: str) -> bool:
        """Heuristic to determine if task is complex."""
        # Length check
        if len(text.split()) > 50:
            return True
            
        # Pattern check
        for pattern in self.complex_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
                
        # Semantic density (simulated)
        unique_words = set(text.lower().split())
        ratio = len(unique_words) / len(text.split()) if text else 0
        if ratio > 0.8 and len(text) > 20: # Rigid check
            return True
            
        return False

# Integration structure
class HybridRuntime:
    """
    Runtime that uses both engines seamlessly.
    """
    def __init__(self):
        self.local = LocalReasoningEngine()
        self.router = AdaptiveRouter(self.local, None) # Registry would be injected
        
    async def process(self, prompt: str):
        decision = self.router.route(prompt)
        print(f"[Router] Selected: {decision.provider.upper()} ({decision.reasoning})")
        
        if decision.provider == 'local':
            return self.local.think(prompt)
        else:
            # In real impl, call cloud model
            return f"[Cloud Simulation] Deep reasoning about: {prompt}"
