"""
AION Simulation Environment Interface
======================================

Generic simulation environment interface for training
and testing embodied AI agents.
"""

import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Protocol, Callable
from enum import Enum
from datetime import datetime
import uuid
import random


# =============================================================================
# SIMULATION TYPES
# =============================================================================

class SimulationType(Enum):
    """Types of simulation environments."""
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    LOCOMOTION = "locomotion"
    MULTI_AGENT = "multi_agent"
    CUSTOM = "custom"


class EpisodeStatus(Enum):
    """Status of a simulation episode."""
    NOT_STARTED = "not_started"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    TERMINATED = "terminated"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ActionSpace:
    """Definition of the action space."""
    type: str  # "discrete", "continuous", "multi_discrete", "multi_binary"
    shape: Tuple[int, ...] = ()
    low: Optional[List[float]] = None
    high: Optional[List[float]] = None
    n: int = 0  # For discrete spaces
    
    @classmethod
    def discrete(cls, n: int) -> "ActionSpace":
        """Create discrete action space."""
        return cls(type="discrete", n=n)
    
    @classmethod
    def continuous(cls, shape: Tuple[int, ...], low: List[float], high: List[float]) -> "ActionSpace":
        """Create continuous action space."""
        return cls(type="continuous", shape=shape, low=low, high=high)


@dataclass
class ObservationSpace:
    """Definition of the observation space."""
    type: str  # "box", "discrete", "dict", "tuple"
    shape: Tuple[int, ...] = ()
    low: Optional[List[float]] = None
    high: Optional[List[float]] = None
    dtype: str = "float32"
    spaces: Dict[str, "ObservationSpace"] = field(default_factory=dict)
    
    @classmethod
    def box(cls, shape: Tuple[int, ...], low: float = -float("inf"), high: float = float("inf")) -> "ObservationSpace":
        """Create box observation space."""
        return cls(type="box", shape=shape, low=[low] * shape[0] if shape else [], high=[high] * shape[0] if shape else [])
    
    @classmethod
    def image(cls, height: int, width: int, channels: int = 3) -> "ObservationSpace":
        """Create image observation space."""
        return cls(type="box", shape=(height, width, channels), low=[0.0], high=[255.0], dtype="uint8")


@dataclass
class SimulationState:
    """Complete simulation state."""
    timestamp: datetime = field(default_factory=datetime.now)
    step: int = 0
    
    # Agent state
    agent_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    agent_orientation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    agent_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    
    # Observation
    observation: Dict[str, Any] = field(default_factory=dict)
    
    # Task state
    goal_position: Optional[Tuple[float, float, float]] = None
    goal_reached: bool = False
    
    # Objects in scene
    objects: List[Dict[str, Any]] = field(default_factory=list)
    
    # Collisions
    collisions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    """Result of taking a step in the simulation."""
    observation: Dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def done(self) -> bool:
        """Check if episode is done (terminated or truncated)."""
        return self.terminated or self.truncated


@dataclass
class Episode:
    """A simulation episode."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: EpisodeStatus = EpisodeStatus.NOT_STARTED
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Episode data
    initial_state: Optional[SimulationState] = None
    final_state: Optional[SimulationState] = None
    
    # Statistics
    total_steps: int = 0
    total_reward: float = 0.0
    
    # History
    states: List[SimulationState] = field(default_factory=list)
    actions: List[Any] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    
    @property
    def duration_seconds(self) -> float:
        """Get episode duration."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    def add_transition(self, state: SimulationState, action: Any, reward: float) -> None:
        """Add a transition to the episode."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.total_steps += 1
        self.total_reward += reward


@dataclass
class SimulationConfig:
    """Configuration for a simulation environment."""
    name: str = "simulation"
    sim_type: SimulationType = SimulationType.NAVIGATION
    
    # Time settings
    time_step: float = 0.02  # seconds
    max_episode_steps: int = 1000
    
    # Physics settings
    gravity: float = 9.81
    friction: float = 0.5
    
    # Rendering
    render_mode: str = "rgb_array"  # "human", "rgb_array", "none"
    render_width: int = 640
    render_height: int = 480
    
    # Random seed
    seed: Optional[int] = None
    
    # Custom params
    params: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# SIMULATION ENVIRONMENT PROTOCOL
# =============================================================================

class SimulationEnvironment(Protocol):
    """Protocol for simulation environments."""
    
    @property
    def action_space(self) -> ActionSpace: ...
    
    @property
    def observation_space(self) -> ObservationSpace: ...
    
    async def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]: ...
    
    async def step(self, action: Any) -> StepResult: ...
    
    async def close(self) -> None: ...
    
    def render(self) -> Optional[bytes]: ...


# =============================================================================
# GENERIC SIMULATOR
# =============================================================================

class GenericSimulator:
    """
    Generic simulation environment implementation.
    
    Provides a flexible simulation framework that can be
    customized for different tasks.
    """
    
    def __init__(self, config: SimulationConfig = None):
        self.config = config or SimulationConfig()
        
        # Spaces
        self._action_space = ActionSpace.continuous((2,), [-1.0, -1.0], [1.0, 1.0])
        self._observation_space = ObservationSpace.box((10,))
        
        # State
        self._state = SimulationState()
        self._episode: Optional[Episode] = None
        self._step_count = 0
        
        # Random generator
        self._rng = random.Random(config.seed if config else None)
    
    @property
    def action_space(self) -> ActionSpace:
        return self._action_space
    
    @property
    def observation_space(self) -> ObservationSpace:
        return self._observation_space
    
    @property
    def current_episode(self) -> Optional[Episode]:
        return self._episode
    
    def set_action_space(self, space: ActionSpace) -> None:
        """Set the action space."""
        self._action_space = space
    
    def set_observation_space(self, space: ObservationSpace) -> None:
        """Set the observation space."""
        self._observation_space = space
    
    async def reset(
        self, 
        seed: Optional[int] = None,
        options: Dict[str, Any] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset the environment.
        
        Args:
            seed: Random seed
            options: Reset options
            
        Returns:
            Tuple of (observation, info)
        """
        if seed is not None:
            self._rng = random.Random(seed)
        
        # Reset state
        self._state = self._generate_initial_state()
        self._step_count = 0
        
        # Create new episode
        self._episode = Episode(
            status=EpisodeStatus.RUNNING,
            start_time=datetime.now(),
            initial_state=self._state
        )
        
        observation = self._get_observation()
        info = {"initial": True}
        
        return observation, info
    
    async def step(self, action: Any) -> StepResult:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            StepResult with observation, reward, done flags
        """
        self._step_count += 1
        
        # Apply action
        self._apply_action(action)
        
        # Update state
        self._update_state()
        
        # Get observation and reward
        observation = self._get_observation()
        reward = self._compute_reward()
        
        # Check termination
        terminated = self._check_termination()
        truncated = self._step_count >= self.config.max_episode_steps
        
        # Update episode
        if self._episode:
            self._episode.add_transition(self._state, action, reward)
            
            if terminated or truncated:
                self._episode.status = EpisodeStatus.SUCCESS if terminated and self._state.goal_reached else EpisodeStatus.FAILURE
                if truncated:
                    self._episode.status = EpisodeStatus.TIMEOUT
                self._episode.end_time = datetime.now()
                self._episode.final_state = self._state
        
        info = {
            "step": self._step_count,
            "goal_reached": self._state.goal_reached
        }
        
        return StepResult(
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info
        )
    
    async def close(self) -> None:
        """Close the environment."""
        if self._episode and self._episode.status == EpisodeStatus.RUNNING:
            self._episode.status = EpisodeStatus.TERMINATED
            self._episode.end_time = datetime.now()
    
    def render(self) -> Optional[bytes]:
        """Render the environment."""
        if self.config.render_mode == "none":
            return None
        
        # Placeholder - would return actual rendered frame
        return b""
    
    def get_state(self) -> SimulationState:
        """Get current simulation state."""
        return self._state
    
    def set_goal(self, position: Tuple[float, float, float]) -> None:
        """Set the goal position."""
        self._state.goal_position = position
    
    def spawn_object(
        self, 
        object_type: str, 
        position: Tuple[float, float, float],
        **kwargs
    ) -> str:
        """Spawn an object in the simulation."""
        obj_id = str(uuid.uuid4())
        
        self._state.objects.append({
            "id": obj_id,
            "type": object_type,
            "position": position,
            **kwargs
        })
        
        return obj_id
    
    def remove_object(self, object_id: str) -> bool:
        """Remove an object from the simulation."""
        for i, obj in enumerate(self._state.objects):
            if obj.get("id") == object_id:
                self._state.objects.pop(i)
                return True
        return False
    
    # -------------------------------------------------------------------------
    # Internal methods
    # -------------------------------------------------------------------------
    
    def _generate_initial_state(self) -> SimulationState:
        """Generate initial state."""
        return SimulationState(
            agent_position=(
                self._rng.uniform(-5, 5),
                self._rng.uniform(-5, 5),
                0.0
            ),
            goal_position=(
                self._rng.uniform(-5, 5),
                self._rng.uniform(-5, 5),
                0.0
            )
        )
    
    def _apply_action(self, action: Any) -> None:
        """Apply action to agent."""
        if isinstance(action, (list, tuple)) and len(action) >= 2:
            # Interpret as velocity command
            vx = float(action[0])
            vy = float(action[1])
            
            # Update position
            x, y, z = self._state.agent_position
            dt = self.config.time_step
            
            self._state.agent_position = (
                x + vx * dt,
                y + vy * dt,
                z
            )
            self._state.agent_velocity = (vx, vy, 0.0)
    
    def _update_state(self) -> None:
        """Update simulation state."""
        self._state.step = self._step_count
        self._state.timestamp = datetime.now()
        
        # Check goal
        if self._state.goal_position:
            goal = self._state.goal_position
            pos = self._state.agent_position
            
            dist = ((goal[0] - pos[0])**2 + (goal[1] - pos[1])**2)**0.5
            self._state.goal_reached = dist < 0.5
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation."""
        pos = self._state.agent_position
        vel = self._state.agent_velocity
        goal = self._state.goal_position or (0, 0, 0)
        
        # Relative goal position
        rel_goal = (goal[0] - pos[0], goal[1] - pos[1])
        
        return {
            "position": pos,
            "velocity": vel,
            "goal_relative": rel_goal,
            "distance_to_goal": (rel_goal[0]**2 + rel_goal[1]**2)**0.5
        }
    
    def _compute_reward(self) -> float:
        """Compute reward for current state."""
        obs = self._get_observation()
        
        # Negative distance to goal
        distance = obs.get("distance_to_goal", 0.0)
        reward = -0.1 * distance
        
        # Bonus for reaching goal
        if self._state.goal_reached:
            reward += 10.0
        
        return reward
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        return self._state.goal_reached


# =============================================================================
# EPISODE MANAGER
# =============================================================================

class EpisodeManager:
    """
    Manages simulation episodes for training/evaluation.
    """
    
    def __init__(self, simulator: GenericSimulator):
        self.simulator = simulator
        self._episodes: List[Episode] = []
    
    async def run_episode(
        self, 
        policy: Callable = None,
        max_steps: int = None,
        render: bool = False
    ) -> Episode:
        """
        Run a single episode.
        
        Args:
            policy: Policy function (observation -> action)
            max_steps: Maximum steps (overrides config)
            render: Whether to render
            
        Returns:
            Completed Episode
        """
        observation, info = await self.simulator.reset()
        
        max_steps = max_steps or self.simulator.config.max_episode_steps
        
        for step in range(max_steps):
            # Get action from policy or random
            if policy:
                action = policy(observation)
            else:
                action = self._random_action()
            
            # Step
            result = await self.simulator.step(action)
            observation = result.observation
            
            # Render
            if render:
                self.simulator.render()
            
            # Check done
            if result.done:
                break
        
        episode = self.simulator.current_episode
        self._episodes.append(episode)
        
        return episode
    
    async def run_episodes(
        self, 
        num_episodes: int,
        policy: Callable = None
    ) -> List[Episode]:
        """Run multiple episodes."""
        episodes = []
        for _ in range(num_episodes):
            episode = await self.run_episode(policy)
            episodes.append(episode)
        return episodes
    
    def _random_action(self) -> Any:
        """Generate random action."""
        space = self.simulator.action_space
        
        if space.type == "discrete":
            return random.randint(0, space.n - 1)
        elif space.type == "continuous":
            return [
                random.uniform(space.low[i], space.high[i])
                for i in range(len(space.low))
            ]
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics over all episodes."""
        if not self._episodes:
            return {}
        
        rewards = [e.total_reward for e in self._episodes]
        steps = [e.total_steps for e in self._episodes]
        success = [e.status == EpisodeStatus.SUCCESS for e in self._episodes]
        
        return {
            "num_episodes": len(self._episodes),
            "avg_reward": sum(rewards) / len(rewards),
            "avg_steps": sum(steps) / len(steps),
            "success_rate": sum(success) / len(success),
            "min_reward": min(rewards),
            "max_reward": max(rewards)
        }


# =============================================================================
# DEMO
# =============================================================================

async def demo_simulation():
    """Demonstrate simulation environment."""
    print("🎮 Simulation Environment Demo")
    print("-" * 40)
    
    # Create simulator
    config = SimulationConfig(
        name="navigation_demo",
        sim_type=SimulationType.NAVIGATION,
        max_episode_steps=100,
        seed=42
    )
    
    simulator = GenericSimulator(config)
    
    # Run episode
    print("Running episode...")
    observation, info = await simulator.reset()
    print(f"Initial distance to goal: {observation['distance_to_goal']:.2f}")
    
    total_reward = 0
    for step in range(50):
        # Simple policy: move towards goal
        goal_rel = observation["goal_relative"]
        action = [goal_rel[0] * 0.5, goal_rel[1] * 0.5]
        
        result = await simulator.step(action)
        observation = result.observation
        total_reward += result.reward
        
        if result.done:
            print(f"Episode done at step {step+1}!")
            break
    
    print(f"Final distance: {observation['distance_to_goal']:.2f}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Goal reached: {simulator.get_state().goal_reached}")
    
    # Run multiple episodes
    manager = EpisodeManager(simulator)
    episodes = await manager.run_episodes(5)
    
    stats = manager.get_statistics()
    print(f"\nEpisode statistics:")
    print(f"  Episodes: {stats['num_episodes']}")
    print(f"  Avg reward: {stats['avg_reward']:.2f}")
    print(f"  Avg steps: {stats['avg_steps']:.1f}")
    
    await simulator.close()
    
    print("-" * 40)
    print("✅ Simulation demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_simulation())
