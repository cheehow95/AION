"""
AION World Model
================

Internal world representation for predictive reasoning and simulation.
Enables agents to:
- Track state across interactions
- Predict outcomes before taking actions
- Perform mental simulations
- Reason about cause and effect
"""

from .state_graph import StateGraph, Entity, Relation
from .causal_engine import CausalEngine, CausalRule, CausalQuery
from .simulator import WorldSimulator, SimulationResult, Scenario
from .predictor import OutcomePredictor, Prediction

__all__ = [
    'StateGraph',
    'Entity',
    'Relation',
    'CausalEngine',
    'CausalRule',
    'CausalQuery',
    'WorldSimulator',
    'SimulationResult',
    'Scenario',
    'OutcomePredictor',
    'Prediction'
]
