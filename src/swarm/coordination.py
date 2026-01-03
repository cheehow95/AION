"""
AION Swarm Intelligence 2.0 - Emergent Coordination Protocols
==============================================================

Implements emergent coordination mechanisms for multi-agent collaboration:
- Stigmergy: Indirect coordination through environment modification
- Task Allocation: Market-based task assignment using auctions
- Swarm Signals: Pheromone-inspired message propagation
- Coalition Formation: Dynamic agent grouping for complex tasks

Auto-generated for Phase 4: Autonomy
"""

import asyncio
import uuid
import time
import math
import heapq
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Optional, Callable, Tuple
from datetime import datetime, timedelta
from enum import Enum
import random


# =============================================================================
# COORDINATION PROTOCOL BASE
# =============================================================================

class CoordinationProtocol(ABC):
    """Base class for all coordination protocols."""
    
    @abstractmethod
    async def coordinate(self, agents: List[str], task: Any) -> Dict[str, Any]:
        """Coordinate agents to accomplish a task."""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get current coordination status."""
        pass


# =============================================================================
# STIGMERGY - INDIRECT COORDINATION
# =============================================================================

class PheromoneType(Enum):
    """Types of pheromone signals."""
    ATTRACTION = "attraction"      # Draw agents toward resource
    REPULSION = "repulsion"        # Push agents away from danger
    TRAIL = "trail"                # Mark successful paths
    ALARM = "alarm"                # Signal urgent situations
    RECRUITMENT = "recruitment"    # Call for help


@dataclass
class Pheromone:
    """A pheromone marker in the environment."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: PheromoneType = PheromoneType.TRAIL
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    intensity: float = 1.0
    decay_rate: float = 0.01  # Per second
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""
    
    def get_current_intensity(self) -> float:
        """Get intensity after decay."""
        elapsed = (datetime.now() - self.created_at).total_seconds()
        return self.intensity * math.exp(-self.decay_rate * elapsed)
    
    def is_expired(self, threshold: float = 0.01) -> bool:
        """Check if pheromone has decayed below threshold."""
        return self.get_current_intensity() < threshold


class Stigmergy(CoordinationProtocol):
    """
    Stigmergy-based coordination.
    
    Agents coordinate indirectly by modifying the shared environment
    with pheromone-like signals that influence other agents' behavior.
    """
    
    def __init__(self, decay_interval: float = 1.0, evaporation_rate: float = None):
        self.pheromones: Dict[str, Pheromone] = {}
        self.decay_interval = decay_interval
        self.evaporation_rate = evaporation_rate if evaporation_rate is not None else 0.01
        self._decay_task: Optional[asyncio.Task] = None
        self._running = False
        
    async def start(self):
        """Start the pheromone decay process."""
        self._running = True
        self._decay_task = asyncio.create_task(self._decay_loop())
        
    async def stop(self):
        """Stop the pheromone decay process."""
        self._running = False
        if self._decay_task:
            self._decay_task.cancel()
            try:
                await self._decay_task
            except asyncio.CancelledError:
                pass
    
    async def _decay_loop(self):
        """Background task to decay pheromones."""
        while self._running:
            await asyncio.sleep(self.decay_interval)
            # Remove expired pheromones
            expired = [pid for pid, p in self.pheromones.items() if p.is_expired()]
            for pid in expired:
                del self.pheromones[pid]
    
    def deposit(self, agent_id: str, ptype: PheromoneType, 
                position: Tuple[float, float, float],
                intensity: float = 1.0,
                metadata: Dict[str, Any] = None) -> str:
        """Deposit a pheromone at a position."""
        pheromone = Pheromone(
            type=ptype,
            position=position,
            intensity=intensity,
            metadata=metadata or {},
            created_by=agent_id
        )
        self.pheromones[pheromone.id] = pheromone
        return pheromone.id
    
    def sense(self, position: Tuple[float, float, float], 
              radius: float = 10.0,
              ptype: Optional[PheromoneType] = None) -> List[Pheromone]:
        """Sense pheromones within radius of position."""
        result = []
        for p in self.pheromones.values():
            if p.is_expired():
                continue
            if ptype and p.type != ptype:
                continue
            # Calculate distance
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(position, p.position)))
            if dist <= radius:
                result.append(p)
        return sorted(result, key=lambda x: x.get_current_intensity(), reverse=True)
    
    def get_gradient(self, position: Tuple[float, float, float],
                     ptype: PheromoneType,
                     radius: float = 10.0) -> Tuple[float, float, float]:
        """Get the gradient direction toward strongest pheromone concentration."""
        pheromones = self.sense(position, radius, ptype)
        if not pheromones:
            return (0.0, 0.0, 0.0)
        
        # Calculate weighted direction
        total_weight = 0.0
        gradient = [0.0, 0.0, 0.0]
        
        for p in pheromones:
            weight = p.get_current_intensity()
            total_weight += weight
            for i in range(3):
                gradient[i] += weight * (p.position[i] - position[i])
        
        if total_weight > 0:
            gradient = [g / total_weight for g in gradient]
            # Normalize
            magnitude = math.sqrt(sum(g ** 2 for g in gradient))
            if magnitude > 0:
                gradient = [g / magnitude for g in gradient]
        
        return tuple(gradient)
    
    async def coordinate(self, agents: List[str], task: Any) -> Dict[str, Any]:
        """Coordinate using stigmergic signals."""
        # Deposit recruitment pheromone at task location
        task_pos = task.get('position', (0.0, 0.0, 0.0))
        self.deposit(
            agent_id="system",
            ptype=PheromoneType.RECRUITMENT,
            position=task_pos,
            intensity=2.0,
            metadata={'task': task}
        )
        
        return {
            'protocol': 'stigmergy',
            'task_position': task_pos,
            'active_pheromones': len(self.pheromones),
            'agents_notified': len(agents)
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get stigmergy status."""
        by_type = {}
        for p in self.pheromones.values():
            by_type[p.type.value] = by_type.get(p.type.value, 0) + 1
        
        return {
            'total_pheromones': len(self.pheromones),
            'by_type': by_type,
            'running': self._running
        }


# =============================================================================
# TASK AUCTION - MARKET-BASED ALLOCATION
# =============================================================================

class BidStrategy(Enum):
    """Bidding strategies for agents."""
    TRUTHFUL = "truthful"          # Bid true valuation
    AGGRESSIVE = "aggressive"      # Bid higher than valuation
    CONSERVATIVE = "conservative"  # Bid lower than valuation
    ADAPTIVE = "adaptive"          # Adjust based on history


@dataclass
class TaskBid:
    """A bid on a task auction."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    task_id: str = ""
    bid_value: float = 0.0
    capability_score: float = 1.0  # How capable agent is for this task
    estimated_completion_time: float = 0.0  # Seconds
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskAuctionItem:
    """A task up for auction."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    priority: int = 1
    required_capabilities: Set[str] = field(default_factory=set)
    deadline: Optional[datetime] = None
    reserve_price: float = 0.0  # Minimum acceptable bid
    bids: List[TaskBid] = field(default_factory=list)
    status: str = "open"
    winner: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)


class TaskAuction(CoordinationProtocol):
    """
    Market-based task allocation using auctions.
    
    Tasks are auctioned to agents who bid based on their
    capabilities and availability. Various auction mechanisms
    are supported.
    """
    
    def __init__(self, auction_duration: float = 5.0):
        self.auctions: Dict[str, TaskAuctionItem] = {}
        self.agent_capabilities: Dict[str, Set[str]] = {}
        self.auction_history: List[Dict[str, Any]] = []
        self.auction_duration = auction_duration
    
    def register_agent(self, agent_id: str, capabilities: Set[str]):
        """Register an agent's capabilities."""
        self.agent_capabilities[agent_id] = capabilities
    
    def create_auction(self, task: Dict[str, Any]) -> str:
        """Create a new task auction."""
        auction = TaskAuctionItem(
            name=task.get('name', 'Unnamed Task'),
            description=task.get('description', ''),
            priority=task.get('priority', 1),
            required_capabilities=set(task.get('capabilities', [])),
            deadline=task.get('deadline'),
            reserve_price=task.get('reserve_price', 0.0)
        )
        self.auctions[auction.id] = auction
        return auction.id
    
    def submit_bid(self, auction_id: str, agent_id: str, 
                   bid_value: float, estimated_time: float = 0.0) -> bool:
        """Submit a bid on an auction."""
        if auction_id not in self.auctions:
            return False
        
        auction = self.auctions[auction_id]
        if auction.status != "open":
            return False
        
        # Calculate capability score
        agent_caps = self.agent_capabilities.get(agent_id, set())
        required_caps = auction.required_capabilities
        if required_caps:
            capability_score = len(agent_caps & required_caps) / len(required_caps)
        else:
            capability_score = 1.0
        
        bid = TaskBid(
            agent_id=agent_id,
            task_id=auction_id,
            bid_value=bid_value,
            capability_score=capability_score,
            estimated_completion_time=estimated_time
        )
        auction.bids.append(bid)
        return True
    
    def close_auction(self, auction_id: str) -> Optional[str]:
        """Close an auction and determine winner."""
        if auction_id not in self.auctions:
            return None
        
        auction = self.auctions[auction_id]
        if auction.status != "open":
            return auction.winner
        
        auction.status = "closed"
        
        if not auction.bids:
            auction.status = "no_bids"
            return None
        
        # Score bids: higher is better
        # Score = bid_value * capability_score / (1 + estimated_time/3600)
        def score_bid(bid: TaskBid) -> float:
            time_factor = 1 + bid.estimated_completion_time / 3600
            return bid.bid_value * bid.capability_score / time_factor
        
        # Filter bids meeting reserve price
        valid_bids = [b for b in auction.bids if b.bid_value >= auction.reserve_price]
        
        if not valid_bids:
            auction.status = "reserve_not_met"
            return None
        
        # Find winner
        winner_bid = max(valid_bids, key=score_bid)
        auction.winner = winner_bid.agent_id
        auction.status = "awarded"
        
        # Record history
        self.auction_history.append({
            'auction_id': auction_id,
            'task_name': auction.name,
            'winner': auction.winner,
            'winning_bid': winner_bid.bid_value,
            'num_bids': len(auction.bids),
            'timestamp': datetime.now().isoformat()
        })
        
        return auction.winner
    
    async def coordinate(self, agents: List[str], task: Any) -> Dict[str, Any]:
        """Coordinate using auction mechanism."""
        # Create auction
        auction_id = self.create_auction(task)
        
        # Wait for bids (in real implementation, signal agents)
        await asyncio.sleep(self.auction_duration)
        
        # Close and determine winner
        winner = self.close_auction(auction_id)
        
        return {
            'protocol': 'auction',
            'auction_id': auction_id,
            'winner': winner,
            'num_bids': len(self.auctions[auction_id].bids),
            'status': self.auctions[auction_id].status
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get auction system status."""
        by_status = {}
        for a in self.auctions.values():
            by_status[a.status] = by_status.get(a.status, 0) + 1
        
        return {
            'total_auctions': len(self.auctions),
            'by_status': by_status,
            'registered_agents': len(self.agent_capabilities),
            'completed_auctions': len(self.auction_history)
        }


# =============================================================================
# SWARM SIGNALS - PHEROMONE-INSPIRED MESSAGING
# =============================================================================

@dataclass
class SwarmSignal:
    """A propagating signal in the swarm."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    signal_type: str = "info"
    payload: Any = None
    origin: str = ""
    hop_count: int = 0
    max_hops: int = 10
    ttl: float = 60.0  # Seconds
    created_at: datetime = field(default_factory=datetime.now)
    visited: Set[str] = field(default_factory=set)
    
    def is_expired(self) -> bool:
        """Check if signal has expired."""
        elapsed = (datetime.now() - self.created_at).total_seconds()
        return elapsed > self.ttl or self.hop_count >= self.max_hops
    
    def propagate(self, agent_id: str) -> bool:
        """Propagate signal through an agent."""
        if self.is_expired():
            return False
        if agent_id in self.visited:
            return False
        
        self.visited.add(agent_id)
        self.hop_count += 1
        return True


class SwarmSignalNetwork:
    """
    Network for propagating swarm signals.
    
    Signals spread through the agent network like pheromones,
    with decay and hop limits to prevent flooding.
    """
    
    def __init__(self):
        self.signals: Dict[str, SwarmSignal] = {}
        self.agent_neighbors: Dict[str, Set[str]] = {}
        self.signal_handlers: Dict[str, List[Callable]] = {}
        self.propagation_queue: asyncio.Queue = asyncio.Queue()
    
    def register_agent(self, agent_id: str, neighbors: Set[str] = None):
        """Register an agent in the network."""
        self.agent_neighbors[agent_id] = neighbors or set()
    
    def add_neighbor(self, agent_id: str, neighbor_id: str):
        """Add a neighbor connection."""
        if agent_id not in self.agent_neighbors:
            self.agent_neighbors[agent_id] = set()
        self.agent_neighbors[agent_id].add(neighbor_id)
    
    def register_handler(self, signal_type: str, handler: Callable):
        """Register a handler for a signal type."""
        if signal_type not in self.signal_handlers:
            self.signal_handlers[signal_type] = []
        self.signal_handlers[signal_type].append(handler)
    
    async def emit(self, origin: str, signal_type: str, payload: Any,
                   max_hops: int = 10, ttl: float = 60.0) -> str:
        """Emit a new signal from an agent."""
        signal = SwarmSignal(
            signal_type=signal_type,
            payload=payload,
            origin=origin,
            max_hops=max_hops,
            ttl=ttl
        )
        signal.visited.add(origin)
        self.signals[signal.id] = signal
        
        # Queue propagation to neighbors
        neighbors = self.agent_neighbors.get(origin, set())
        for neighbor in neighbors:
            await self.propagation_queue.put((signal.id, neighbor))
        
        return signal.id
    
    async def process_propagation(self):
        """Process signal propagation queue."""
        while True:
            try:
                signal_id, agent_id = await asyncio.wait_for(
                    self.propagation_queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue
            
            if signal_id not in self.signals:
                continue
            
            signal = self.signals[signal_id]
            if not signal.propagate(agent_id):
                continue
            
            # Call handlers
            for handler in self.signal_handlers.get(signal.signal_type, []):
                try:
                    await handler(signal, agent_id)
                except Exception:
                    pass
            
            # Propagate to neighbors
            neighbors = self.agent_neighbors.get(agent_id, set())
            for neighbor in neighbors:
                if neighbor not in signal.visited:
                    await self.propagation_queue.put((signal_id, neighbor))
    
    def cleanup_expired(self):
        """Remove expired signals."""
        expired = [sid for sid, s in self.signals.items() if s.is_expired()]
        for sid in expired:
            del self.signals[sid]


# =============================================================================
# COALITION FORMATION
# =============================================================================

@dataclass
class Coalition:
    """A coalition of agents working together."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    task_id: str = ""
    members: Set[str] = field(default_factory=set)
    leader: Optional[str] = None
    capabilities: Set[str] = field(default_factory=set)
    formation_time: datetime = field(default_factory=datetime.now)
    status: str = "forming"
    value: float = 0.0  # Coalition value


class CoalitionManager(CoordinationProtocol):
    """
    Manages dynamic coalition formation for complex tasks.
    
    Agents form coalitions to accomplish tasks that require
    combined capabilities or resources.
    """
    
    def __init__(self):
        self.coalitions: Dict[str, Coalition] = {}
        self.agent_coalitions: Dict[str, Set[str]] = {}  # agent -> coalition IDs
        self.agent_capabilities: Dict[str, Set[str]] = {}
        self.agent_values: Dict[str, float] = {}  # Agent contribution values
    
    def register_agent(self, agent_id: str, capabilities: Set[str], value: float = 1.0):
        """Register an agent for coalition formation."""
        self.agent_capabilities[agent_id] = capabilities
        self.agent_values[agent_id] = value
        self.agent_coalitions[agent_id] = set()
    
    def form_coalition(self, task_id: str, required_caps: Set[str],
                       candidate_agents: List[str]) -> Optional[str]:
        """
        Form a coalition for a task using greedy algorithm.
        
        Selects agents that together cover required capabilities
        with maximum value.
        """
        coalition = Coalition(
            name=f"Coalition-{task_id[:8]}",
            task_id=task_id
        )
        
        remaining_caps = set(required_caps)
        available = list(candidate_agents)
        
        # Greedy selection: pick agent that covers most remaining caps
        while remaining_caps and available:
            best_agent = None
            best_score = -1
            
            for agent_id in available:
                agent_caps = self.agent_capabilities.get(agent_id, set())
                covered = len(agent_caps & remaining_caps)
                if covered > 0:
                    # Score = coverage * agent_value
                    score = covered * self.agent_values.get(agent_id, 1.0)
                    if score > best_score:
                        best_score = score
                        best_agent = agent_id
            
            if best_agent is None:
                break
            
            # Add agent to coalition
            coalition.members.add(best_agent)
            coalition.capabilities |= self.agent_capabilities.get(best_agent, set())
            coalition.value += self.agent_values.get(best_agent, 1.0)
            remaining_caps -= self.agent_capabilities.get(best_agent, set())
            available.remove(best_agent)
        
        # Check if we covered all requirements
        if remaining_caps:
            return None  # Cannot form viable coalition
        
        # Elect leader (highest value member)
        if coalition.members:
            coalition.leader = max(
                coalition.members,
                key=lambda a: self.agent_values.get(a, 1.0)
            )
        
        coalition.status = "active"
        self.coalitions[coalition.id] = coalition
        
        # Update agent memberships
        for member in coalition.members:
            self.agent_coalitions.setdefault(member, set()).add(coalition.id)
        
        return coalition.id
    
    def dissolve_coalition(self, coalition_id: str):
        """Dissolve a coalition."""
        if coalition_id not in self.coalitions:
            return
        
        coalition = self.coalitions[coalition_id]
        coalition.status = "dissolved"
        
        # Remove from agent memberships
        for member in coalition.members:
            if member in self.agent_coalitions:
                self.agent_coalitions[member].discard(coalition_id)
    
    def compute_shapley_values(self, coalition_id: str) -> Dict[str, float]:
        """
        Compute Shapley values for fair reward distribution.
        
        Each agent's contribution is their marginal contribution
        averaged over all possible orderings.
        """
        if coalition_id not in self.coalitions:
            return {}
        
        coalition = self.coalitions[coalition_id]
        members = list(coalition.members)
        n = len(members)
        
        if n == 0:
            return {}
        
        shapley = {m: 0.0 for m in members}
        
        # Simple approximation: use marginal contributions
        # In real implementation, would compute over permutations
        
        def coalition_value(subset: Set[str]) -> float:
            """Value of a subset of agents."""
            caps = set()
            for agent in subset:
                caps |= self.agent_capabilities.get(agent, set())
            return len(caps)
        
        # Marginal contribution for each agent
        full_value = coalition_value(coalition.members)
        for agent in members:
            without = coalition.members - {agent}
            marginal = full_value - coalition_value(without)
            shapley[agent] = marginal
        
        # Normalize to sum to coalition value
        total = sum(shapley.values())
        if total > 0:
            for agent in shapley:
                shapley[agent] = (shapley[agent] / total) * coalition.value
        
        return shapley
    
    async def coordinate(self, agents: List[str], task: Any) -> Dict[str, Any]:
        """Coordinate by forming a coalition."""
        task_id = task.get('id', str(uuid.uuid4()))
        required_caps = set(task.get('capabilities', []))
        
        coalition_id = self.form_coalition(task_id, required_caps, agents)
        
        if coalition_id:
            coalition = self.coalitions[coalition_id]
            return {
                'protocol': 'coalition',
                'coalition_id': coalition_id,
                'members': list(coalition.members),
                'leader': coalition.leader,
                'capabilities': list(coalition.capabilities),
                'status': 'success'
            }
        else:
            return {
                'protocol': 'coalition',
                'status': 'failed',
                'reason': 'Could not form viable coalition'
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get coalition system status."""
        active = sum(1 for c in self.coalitions.values() if c.status == "active")
        return {
            'total_coalitions': len(self.coalitions),
            'active_coalitions': active,
            'registered_agents': len(self.agent_capabilities)
        }


# =============================================================================
# EMERGENT COORDINATOR - UNIFIED INTERFACE
# =============================================================================

class EmergentCoordinator:
    """
    Unified coordinator that combines multiple coordination mechanisms.
    
    Selects the appropriate protocol based on task characteristics
    and agent availability.
    """
    
    def __init__(self):
        self.stigmergy = Stigmergy()
        self.auction = TaskAuction()
        self.signals = SwarmSignalNetwork()
        self.coalitions = CoalitionManager()
        
    async def start(self):
        """Start all coordination systems."""
        await self.stigmergy.start()
    
    async def stop(self):
        """Stop all coordination systems."""
        await self.stigmergy.stop()
    
    def register_agent(self, agent_id: str, capabilities: Set[str], value: float = 1.0):
        """Register an agent across all coordination systems."""
        self.auction.register_agent(agent_id, capabilities)
        self.signals.register_agent(agent_id)
        self.coalitions.register_agent(agent_id, capabilities, value)
    
    async def coordinate(self, agents: List[str], task: Dict[str, Any],
                        protocol: Optional[str] = None) -> Dict[str, Any]:
        """
        Coordinate agents for a task.
        
        If protocol is not specified, automatically selects based on task.
        """
        # Auto-select protocol if not specified
        if protocol is None:
            if task.get('requires_coalition'):
                protocol = 'coalition'
            elif task.get('competitive'):
                protocol = 'auction'
            elif task.get('spatial'):
                protocol = 'stigmergy'
            else:
                protocol = 'auction'  # Default
        
        # Dispatch to appropriate coordinator
        if protocol == 'stigmergy':
            return await self.stigmergy.coordinate(agents, task)
        elif protocol == 'auction':
            return await self.auction.coordinate(agents, task)
        elif protocol == 'coalition':
            return await self.coalitions.coordinate(agents, task)
        else:
            return {'error': f'Unknown protocol: {protocol}'}
    
    async def broadcast_signal(self, origin: str, signal_type: str, payload: Any) -> str:
        """Broadcast a signal through the swarm network."""
        return await self.signals.emit(origin, signal_type, payload)
    
    def get_full_status(self) -> Dict[str, Any]:
        """Get status of all coordination systems."""
        return {
            'stigmergy': self.stigmergy.get_status(),
            'auction': self.auction.get_status(),
            'coalitions': self.coalitions.get_status(),
            'signals': {
                'active_signals': len(self.signals.signals),
                'registered_agents': len(self.signals.agent_neighbors)
            }
        }


# =============================================================================
# DEMO
# =============================================================================

async def demo_coordination():
    """Demonstrate emergent coordination protocols."""
    print("🐝 Swarm Intelligence 2.0 - Coordination Demo")
    print("=" * 50)
    
    coordinator = EmergentCoordinator()
    await coordinator.start()
    
    # Register agents
    agents = [
        ("Alpha", {"coding", "analysis"}, 1.0),
        ("Beta", {"design", "testing"}, 0.8),
        ("Gamma", {"coding", "deployment"}, 1.2),
        ("Delta", {"analysis", "documentation"}, 0.9),
    ]
    
    for name, caps, value in agents:
        coordinator.register_agent(name, caps, value)
        print(f"  Registered: {name} with {caps}")
    
    print()
    
    # Demo 1: Coalition Formation
    print("📋 Coalition Formation:")
    task = {
        'id': 'complex-task-1',
        'name': 'Build Feature X',
        'capabilities': ['coding', 'testing', 'deployment'],
        'requires_coalition': True
    }
    
    result = await coordinator.coordinate([a[0] for a in agents], task)
    print(f"  Result: {result}")
    
    if result.get('coalition_id'):
        shapley = coordinator.coalitions.compute_shapley_values(result['coalition_id'])
        print(f"  Shapley Values: {shapley}")
    
    print()
    
    # Demo 2: Task Auction
    print("🏆 Task Auction:")
    task = {
        'name': 'Quick Analysis',
        'capabilities': ['analysis'],
        'reserve_price': 0.5,
        'competitive': True
    }
    
    # Pre-register some bids
    auction_id = coordinator.auction.create_auction(task)
    coordinator.auction.submit_bid(auction_id, "Alpha", 1.0, 100)
    coordinator.auction.submit_bid(auction_id, "Delta", 1.2, 150)
    
    winner = coordinator.auction.close_auction(auction_id)
    print(f"  Winner: {winner}")
    print(f"  Auction Status: {coordinator.auction.auctions[auction_id].status}")
    
    print()
    
    # Demo 3: Stigmergy
    print("🔬 Stigmergy:")
    coordinator.stigmergy.deposit(
        "Alpha", PheromoneType.TRAIL,
        position=(5.0, 0.0, 0.0),
        intensity=1.0
    )
    coordinator.stigmergy.deposit(
        "Beta", PheromoneType.TRAIL,
        position=(3.0, 2.0, 0.0),
        intensity=0.8
    )
    
    # Sense from origin
    pheromones = coordinator.stigmergy.sense((0.0, 0.0, 0.0), radius=10.0)
    print(f"  Sensed {len(pheromones)} pheromones")
    
    gradient = coordinator.stigmergy.get_gradient(
        (0.0, 0.0, 0.0), PheromoneType.TRAIL, radius=10.0
    )
    print(f"  Gradient direction: {gradient}")
    
    print()
    
    # Status
    print("📊 System Status:")
    status = coordinator.get_full_status()
    for system, info in status.items():
        print(f"  {system}: {info}")
    
    await coordinator.stop()
    print()
    print("✅ Coordination demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_coordination())
