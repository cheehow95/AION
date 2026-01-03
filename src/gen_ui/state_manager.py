"""
AION Generative UI Engine - State Manager
==========================================

Manages state and interactions for generated UIs.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import uuid

@dataclass
class UIState:
    """State of a specific UI instance."""
    instance_id: str
    component_id: str
    data: Dict[str, Any] = field(default_factory=dict)
    last_updated: float = 0.0

@dataclass
class StateTransition:
    """Transition between states."""
    from_state: Dict[str, Any]
    to_state: Dict[str, Any]
    action: str
    timestamp: float

class InteractionHandler:
    """
    Handles events and updates UI state.
    """
    
    def __init__(self):
        self.states: Dict[str, UIState] = {}
        self.history: Dict[str, List[StateTransition]] = {}
        
    def init_state(self, component_id: str, initial_data: Dict = None) -> str:
        """Initialize state for a component instance."""
        instance_id = str(uuid.uuid4())
        self.states[instance_id] = UIState(
            instance_id=instance_id,
            component_id=component_id,
            data=initial_data or {}
        )
        self.history[instance_id] = []
        return instance_id
        
    async def handle_action(self, instance_id: str, action: str, payload: Dict = None) -> Dict[str, Any]:
        """Handle a user action on the UI."""
        if instance_id not in self.states:
            raise ValueError("Instance not found")
            
        state = self.states[instance_id]
        prev_data = state.data.copy()
        
        # Simple state update logic (would be more complex in real app)
        new_data = prev_data.copy()
        if action == "update":
            new_data.update(payload or {})
        elif action == "reset":
            new_data = {}
            
        state.data = new_data
        
        # Record transition
        self.history[instance_id].append(StateTransition(
            from_state=prev_data,
            to_state=new_data,
            action=action,
            timestamp=0.0
        ))
        
        return state.data

async def demo_state_manager():
    """Demonstrate state management."""
    handler = InteractionHandler()
    
    # Create instance
    instance_id = handler.init_state("calc_1", {"value": 0})
    print(f"Initialized State: {handler.states[instance_id].data}")
    
    # Update state
    new_state = await handler.handle_action(instance_id, "update", {"value": 1})
    print(f"Updated State (Action: update): {new_state}")
    
    new_state = await handler.handle_action(instance_id, "update", {"value": 2})
    print(f"Updated State (Action: update): {new_state}")

if __name__ == "__main__":
    asyncio.run(demo_state_manager())
