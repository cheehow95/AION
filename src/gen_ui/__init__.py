"""
AION Generative UI Engine - Package Initialization
===================================================

Dynamic UI generation like Gemini 3's generative interface:
- React component generation
- State management
- Interaction handling
"""

from .ui_generator import (
    ComponentSpec,
    UIComponent,
    UIGenerator
)

from .state_manager import (
    UIState,
    InteractionHandler,
    StateTransition
)

__all__ = [
    # Generator
    'ComponentSpec',
    'UIComponent',
    'UIGenerator',
    # State
    'UIState',
    'InteractionHandler',
    'StateTransition',
]
