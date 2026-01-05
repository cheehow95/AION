"""
AION Consciousness Package
===========================

Self-awareness, meta-cognition, creative thinking, and goal architecture.
All imports are optional to prevent import failures.
"""

__all__ = []

# Core Consciousness
try:
    from .awareness import (
        ConsciousnessEngine, ConsciousnessState,
        SelfModel, WorldModel, AION_CONSCIOUSNESS, awaken
    )
    __all__.extend([
        'ConsciousnessEngine', 'ConsciousnessState',
        'SelfModel', 'WorldModel', 'AION_CONSCIOUSNESS', 'awaken'
    ])
except ImportError:
    pass

# Meta-Cognition
try:
    from .meta_cognition import MetaCognitionEngine
    __all__.append('MetaCognitionEngine')
except ImportError:
    pass

# Self-Modification
try:
    from .self_modifier import SafeSelfModifier
    __all__.append('SafeSelfModifier')
except ImportError:
    pass

# Goals
try:
    from .goal_architecture import GoalArchitecture
    __all__.append('GoalArchitecture')
except ImportError:
    pass

# Creative Thinking
try:
    from .creative_thinking import (
        CreativeThinkingEngine, DivergentThinker, AnalogicalReasoner,
        ConceptualBlender, ImaginationEngine, InsightGenerator, IntuitionModeler
    )
    __all__.extend([
        'CreativeThinkingEngine', 'DivergentThinker', 'AnalogicalReasoner',
        'ConceptualBlender', 'ImaginationEngine', 'InsightGenerator', 'IntuitionModeler'
    ])
except ImportError:
    pass
