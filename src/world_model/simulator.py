"""
AION World Simulator
====================

Mental simulation of scenarios and actions.
Enables agents to "imagine" outcomes before acting.
"""

from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import copy
import asyncio

from .state_graph import StateGraph, Entity, EntityType
from .causal_engine import CausalEngine, CausalRule


class SimulationMode(Enum):
    """Modes of simulation."""
    DETERMINISTIC = "deterministic"
    PROBABILISTIC = "probabilistic"
    MONTE_CARLO = "monte_carlo"


@dataclass
class Scenario:
    """A scenario to simulate."""
    name: str
    initial_state: Dict[str, Any]
    actions: List[Dict[str, Any]]
    constraints: Dict[str, Any] = field(default_factory=dict)
    max_steps: int = 100
    success_criteria: Optional[Callable[[Dict[str, Any]], bool]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "initial_state": self.initial_state,
            "actions": self.actions,
            "constraints": self.constraints,
            "max_steps": self.max_steps
        }


@dataclass
class SimulationStep:
    """A single step in the simulation."""
    step_number: int
    action: Dict[str, Any]
    state_before: Dict[str, Any]
    state_after: Dict[str, Any]
    effects: List[Dict[str, Any]]
    success: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step_number,
            "action": self.action,
            "state_before": self.state_before,
            "state_after": self.state_after,
            "effects": self.effects,
            "success": self.success
        }


@dataclass
class SimulationResult:
    """Result of a simulation run."""
    scenario_name: str
    mode: SimulationMode
    success: bool
    final_state: Dict[str, Any]
    steps: List[SimulationStep]
    total_steps: int
    duration_ms: float
    metrics: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario_name,
            "mode": self.mode.value,
            "success": self.success,
            "final_state": self.final_state,
            "total_steps": self.total_steps,
            "duration_ms": self.duration_ms,
            "metrics": self.metrics,
            "warnings": self.warnings
        }
    
    def get_state_trajectory(self, key: str) -> List[Any]:
        """Get trajectory of a specific state variable."""
        return [step.state_after.get(key) for step in self.steps]


class WorldSimulator:
    """
    Simulates world state evolution.
    
    Enables:
    - Mental simulation of action sequences
    - Scenario testing
    - Rollback/replay of states
    - Monte Carlo exploration
    """
    
    def __init__(
        self,
        causal_engine: CausalEngine = None,
        state_graph: StateGraph = None
    ):
        self.causal_engine = causal_engine or CausalEngine()
        self.state_graph = state_graph or StateGraph("simulation")
        
        # Simulation state
        self.current_state: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []
        self.checkpoints: Dict[str, Dict[str, Any]] = {}
        
        # Hooks for custom behavior
        self.pre_step_hooks: List[Callable] = []
        self.post_step_hooks: List[Callable] = []
    
    def reset(self, initial_state: Dict[str, Any] = None):
        """Reset simulator to initial state."""
        self.current_state = initial_state.copy() if initial_state else {}
        self.history = [self.current_state.copy()]
        self.checkpoints = {"initial": self.current_state.copy()}
    
    def checkpoint(self, name: str):
        """Save current state as checkpoint."""
        self.checkpoints[name] = copy.deepcopy(self.current_state)
    
    def restore(self, checkpoint_name: str) -> bool:
        """Restore state from checkpoint."""
        if checkpoint_name in self.checkpoints:
            self.current_state = copy.deepcopy(self.checkpoints[checkpoint_name])
            return True
        return False
    
    def step(self, action: Dict[str, Any]) -> SimulationStep:
        """
        Execute one simulation step.
        
        Args:
            action: The action to simulate
        
        Returns:
            SimulationStep with results
        """
        state_before = copy.deepcopy(self.current_state)
        
        # Run pre-step hooks
        for hook in self.pre_step_hooks:
            hook(self.current_state, action)
        
        # Apply action using causal engine
        merged_context = {**self.current_state, **action}
        result = self.causal_engine.predict(action, self.current_state)
        
        # Collect effects and update state
        effects = []
        for rule, effect in result.effects:
            effects.append({
                "rule": rule.name,
                "effect": effect,
                "probability": rule.strength.value
            })
            self.current_state.update(effect)
        
        # Apply action directly to state
        for key, value in action.items():
            if not key.startswith("_"):  # Skip metadata keys
                self.current_state[key] = value
        
        state_after = copy.deepcopy(self.current_state)
        
        # Run post-step hooks
        for hook in self.post_step_hooks:
            hook(state_before, state_after, action)
        
        # Record history
        self.history.append(state_after)
        
        step = SimulationStep(
            step_number=len(self.history) - 1,
            action=action,
            state_before=state_before,
            state_after=state_after,
            effects=effects,
            success=True
        )
        
        return step
    
    async def simulate(
        self,
        scenario: Scenario,
        mode: SimulationMode = SimulationMode.DETERMINISTIC
    ) -> SimulationResult:
        """
        Run a complete simulation scenario.
        
        Args:
            scenario: The scenario to simulate
            mode: Simulation mode
        
        Returns:
            SimulationResult with full trajectory
        """
        start_time = datetime.now()
        
        # Initialize
        self.reset(scenario.initial_state)
        steps = []
        success = True
        warnings = []
        
        for i, action in enumerate(scenario.actions):
            if i >= scenario.max_steps:
                warnings.append(f"Max steps ({scenario.max_steps}) reached")
                break
            
            # Check constraints
            constraint_violated = self._check_constraints(scenario.constraints)
            if constraint_violated:
                warnings.append(f"Constraint violated: {constraint_violated}")
                success = False
                break
            
            # Execute step
            step = self.step(action)
            steps.append(step)
            
            if not step.success:
                success = False
                break
            
            # Check success criteria
            if scenario.success_criteria:
                try:
                    if scenario.success_criteria(self.current_state):
                        break
                except Exception as e:
                    warnings.append(f"Success criteria error: {e}")
        
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        # Calculate metrics
        metrics = self._calculate_metrics(steps)
        
        return SimulationResult(
            scenario_name=scenario.name,
            mode=mode,
            success=success,
            final_state=copy.deepcopy(self.current_state),
            steps=steps,
            total_steps=len(steps),
            duration_ms=duration_ms,
            metrics=metrics,
            warnings=warnings
        )
    
    async def monte_carlo(
        self,
        scenario: Scenario,
        num_runs: int = 100,
        variation_fn: Callable[[Dict[str, Any]], Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation with variations.
        
        Args:
            scenario: Base scenario
            num_runs: Number of runs
            variation_fn: Function to add randomness to actions
        
        Returns:
            Aggregated statistics
        """
        results = []
        success_count = 0
        
        for run in range(num_runs):
            # Create variation of scenario
            varied_actions = scenario.actions
            if variation_fn:
                varied_actions = [variation_fn(a) for a in scenario.actions]
            
            varied_scenario = Scenario(
                name=f"{scenario.name}_run_{run}",
                initial_state=scenario.initial_state,
                actions=varied_actions,
                constraints=scenario.constraints,
                max_steps=scenario.max_steps,
                success_criteria=scenario.success_criteria
            )
            
            result = await self.simulate(varied_scenario, SimulationMode.MONTE_CARLO)
            results.append(result)
            
            if result.success:
                success_count += 1
        
        return {
            "total_runs": num_runs,
            "success_rate": success_count / num_runs,
            "avg_steps": sum(r.total_steps for r in results) / num_runs,
            "avg_duration_ms": sum(r.duration_ms for r in results) / num_runs,
            "all_results": results
        }
    
    def rollback(self, steps: int = 1) -> bool:
        """Rollback simulation by N steps."""
        if steps >= len(self.history):
            return False
        
        target_idx = len(self.history) - 1 - steps
        self.current_state = copy.deepcopy(self.history[target_idx])
        self.history = self.history[:target_idx + 1]
        return True
    
    def replay(self, from_checkpoint: str = "initial") -> List[SimulationStep]:
        """
        Replay simulation from a checkpoint.
        
        Returns all steps taken since checkpoint.
        """
        if from_checkpoint not in self.checkpoints:
            return []
        
        checkpoint_state = self.checkpoints[from_checkpoint]
        replay_steps = []
        
        recording = False
        for i, state in enumerate(self.history):
            if state == checkpoint_state:
                recording = True
                continue
            
            if recording and i > 0:
                replay_steps.append(SimulationStep(
                    step_number=i,
                    action={},  # Action info lost in history
                    state_before=self.history[i-1],
                    state_after=state,
                    effects=[],
                    success=True
                ))
        
        return replay_steps
    
    def _check_constraints(self, constraints: Dict[str, Any]) -> Optional[str]:
        """Check if any constraints are violated."""
        for key, constraint in constraints.items():
            value = self.current_state.get(key)
            
            if isinstance(constraint, dict):
                if "min" in constraint and value < constraint["min"]:
                    return f"{key} below minimum ({value} < {constraint['min']})"
                if "max" in constraint and value > constraint["max"]:
                    return f"{key} above maximum ({value} > {constraint['max']})"
                if "not" in constraint and value == constraint["not"]:
                    return f"{key} equals forbidden value ({value})"
            elif value != constraint:
                return f"{key} must equal {constraint}, got {value}"
        
        return None
    
    def _calculate_metrics(self, steps: List[SimulationStep]) -> Dict[str, float]:
        """Calculate simulation metrics."""
        if not steps:
            return {}
        
        return {
            "total_steps": len(steps),
            "success_rate": sum(1 for s in steps if s.success) / len(steps),
            "avg_effects_per_step": sum(len(s.effects) for s in steps) / len(steps)
        }
    
    def add_pre_step_hook(self, hook: Callable):
        """Add hook to run before each step."""
        self.pre_step_hooks.append(hook)
    
    def add_post_step_hook(self, hook: Callable):
        """Add hook to run after each step."""
        self.post_step_hooks.append(hook)
    
    def get_state_diff(self, step1: int, step2: int) -> Dict[str, Any]:
        """Get difference between two historical states."""
        if step1 >= len(self.history) or step2 >= len(self.history):
            return {}
        
        state1 = self.history[step1]
        state2 = self.history[step2]
        
        diff = {}
        all_keys = set(state1.keys()) | set(state2.keys())
        
        for key in all_keys:
            v1 = state1.get(key)
            v2 = state2.get(key)
            if v1 != v2:
                diff[key] = {"from": v1, "to": v2}
        
        return diff
