"""
AION Swarm Intelligence 2.0 - Distributed Consensus Mechanisms
==============================================================

Implements distributed consensus for multi-agent agreement:
- Raft Consensus: Leader election and log replication
- Byzantine Fault Tolerance: Resistance to malicious agents
- Voting Protocols: Weighted voting based on reputation
- Conflict Resolution: Automatic resolution of agent disagreements

Auto-generated for Phase 4: Autonomy
"""

import asyncio
import uuid
import time
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Optional, Callable, Tuple
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import json


# =============================================================================
# CONSENSUS PROTOCOL BASE
# =============================================================================

class ConsensusState(Enum):
    """States in a consensus protocol."""
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"
    OFFLINE = "offline"


class ConsensusProtocol(ABC):
    """Base class for consensus protocols."""
    
    @abstractmethod
    async def propose(self, value: Any) -> bool:
        """Propose a value for consensus."""
        pass
    
    @abstractmethod
    async def get_consensus(self) -> Optional[Any]:
        """Get the current consensus value."""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get protocol status."""
        pass


# =============================================================================
# RAFT CONSENSUS
# =============================================================================

@dataclass
class LogEntry:
    """Entry in the replicated log."""
    index: int
    term: int
    command: Any
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RaftNode:
    """A node in the Raft cluster."""
    id: str
    state: ConsensusState = ConsensusState.FOLLOWER
    current_term: int = 0
    voted_for: Optional[str] = None
    log: List[LogEntry] = field(default_factory=list)
    commit_index: int = 0
    last_applied: int = 0
    
    # Leader state
    next_index: Dict[str, int] = field(default_factory=dict)
    match_index: Dict[str, int] = field(default_factory=dict)
    
    # Timing
    last_heartbeat: datetime = field(default_factory=datetime.now)
    election_timeout: float = 0.0


class RaftConsensus(ConsensusProtocol):
    """
    Raft consensus implementation.
    
    Provides leader election and log replication for
    consistent distributed state.
    """
    
    def __init__(self, node_id: str, peers: List[str],
                 heartbeat_interval: float = 0.5,
                 election_timeout_range: Tuple[float, float] = (1.5, 3.0)):
        self.node = RaftNode(id=node_id)
        self.peers = set(peers)
        self.heartbeat_interval = heartbeat_interval
        self.election_timeout_range = election_timeout_range
        
        self._reset_election_timeout()
        
        # Message handlers
        self.message_handlers: Dict[str, Callable] = {}
        
        # State machine
        self.state_machine: Dict[str, Any] = {}
        
        # Running flag
        self._running = False
        self._tasks: List[asyncio.Task] = []
    
    def _reset_election_timeout(self):
        """Reset election timeout to random value."""
        self.node.election_timeout = random.uniform(*self.election_timeout_range)
        self.node.last_heartbeat = datetime.now()
    
    async def start(self):
        """Start the Raft node."""
        self._running = True
        self._tasks.append(asyncio.create_task(self._election_timer()))
        if self.node.state == ConsensusState.LEADER:
            self._tasks.append(asyncio.create_task(self._heartbeat_loop()))
    
    async def stop(self):
        """Stop the Raft node."""
        self._running = False
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._tasks.clear()
    
    async def _election_timer(self):
        """Monitor for election timeout."""
        while self._running:
            await asyncio.sleep(0.1)
            
            if self.node.state == ConsensusState.LEADER:
                continue
            
            elapsed = (datetime.now() - self.node.last_heartbeat).total_seconds()
            if elapsed > self.node.election_timeout:
                await self._start_election()
    
    async def _start_election(self):
        """Start a leader election."""
        self.node.state = ConsensusState.CANDIDATE
        self.node.current_term += 1
        self.node.voted_for = self.node.id
        self._reset_election_timeout()
        
        # Request votes from peers
        votes = 1  # Vote for self
        
        for peer in self.peers:
            vote_granted = await self._request_vote(peer)
            if vote_granted:
                votes += 1
        
        # Check if we won
        total_nodes = len(self.peers) + 1
        if votes > total_nodes / 2:
            await self._become_leader()
        else:
            self.node.state = ConsensusState.FOLLOWER
    
    async def _request_vote(self, peer: str) -> bool:
        """Request vote from a peer (simulated)."""
        # In real implementation, would send network request
        # Simulating: random chance of getting vote
        return random.random() > 0.3
    
    async def _become_leader(self):
        """Become the leader."""
        self.node.state = ConsensusState.LEADER
        
        # Initialize leader state
        last_log_index = len(self.node.log)
        for peer in self.peers:
            self.node.next_index[peer] = last_log_index + 1
            self.node.match_index[peer] = 0
        
        # Start heartbeats
        self._tasks.append(asyncio.create_task(self._heartbeat_loop()))
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats as leader."""
        while self._running and self.node.state == ConsensusState.LEADER:
            await self._send_heartbeats()
            await asyncio.sleep(self.heartbeat_interval)
    
    async def _send_heartbeats(self):
        """Send heartbeat (AppendEntries) to all peers."""
        for peer in self.peers:
            await self._append_entries(peer)
    
    async def _append_entries(self, peer: str) -> bool:
        """Send AppendEntries RPC to peer."""
        # In real implementation, would send actual log entries
        # Simulating: always succeeds
        return True
    
    async def propose(self, command: Any) -> bool:
        """Propose a command to be replicated."""
        if self.node.state != ConsensusState.LEADER:
            return False
        
        # Append to local log
        entry = LogEntry(
            index=len(self.node.log) + 1,
            term=self.node.current_term,
            command=command
        )
        self.node.log.append(entry)
        
        # Replicate to peers
        successes = 1  # Self
        for peer in self.peers:
            if await self._append_entries(peer):
                successes += 1
        
        # Check majority
        total = len(self.peers) + 1
        if successes > total / 2:
            # Commit
            self.node.commit_index = entry.index
            await self._apply_committed()
            return True
        
        return False
    
    async def _apply_committed(self):
        """Apply committed log entries to state machine."""
        while self.node.last_applied < self.node.commit_index:
            self.node.last_applied += 1
            entry = self.node.log[self.node.last_applied - 1]
            
            # Apply to state machine
            if isinstance(entry.command, dict):
                key = entry.command.get('key')
                value = entry.command.get('value')
                if key:
                    self.state_machine[key] = value
    
    async def get_consensus(self) -> Optional[Any]:
        """Get the current state machine."""
        return dict(self.state_machine)
    
    def get_leader(self) -> Optional[str]:
        """Get the current leader ID."""
        if self.node.state == ConsensusState.LEADER:
            return self.node.id
        return None  # In real impl, would track leader
    
    def get_status(self) -> Dict[str, Any]:
        """Get Raft node status."""
        return {
            'node_id': self.node.id,
            'state': self.node.state.value,
            'term': self.node.current_term,
            'log_length': len(self.node.log),
            'commit_index': self.node.commit_index,
            'peers': list(self.peers),
            'is_leader': self.node.state == ConsensusState.LEADER
        }


# =============================================================================
# BYZANTINE FAULT TOLERANCE
# =============================================================================

@dataclass
class BFTMessage:
    """Message in BFT protocol."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str = ""  # pre-prepare, prepare, commit
    view: int = 0
    sequence: int = 0
    digest: str = ""
    sender: str = ""
    value: Any = None
    signature: str = ""  # Simulated signature


class ByzantineFaultTolerance(ConsensusProtocol):
    """
    Practical Byzantine Fault Tolerance (PBFT) implementation.
    
    Tolerates up to f byzantine (malicious) nodes in a 3f+1 node system.
    """
    
    def __init__(self, node_id: str, peers: List[str]):
        self.node_id = node_id
        self.peers = set(peers)
        self.total_nodes = len(peers) + 1
        self.f = (self.total_nodes - 1) // 3  # Max faulty nodes
        
        # Protocol state
        self.view = 0
        self.sequence = 0
        self.primary = self._get_primary()
        
        # Message logs
        self.pre_prepare_log: Dict[Tuple[int, int], BFTMessage] = {}
        self.prepare_log: Dict[Tuple[int, int], List[BFTMessage]] = {}
        self.commit_log: Dict[Tuple[int, int], List[BFTMessage]] = {}
        
        # Committed values
        self.committed: Dict[int, Any] = {}
        
        # Reply cache
        self.reply_cache: Dict[str, Any] = {}
    
    def _get_primary(self) -> str:
        """Get the primary for current view."""
        all_nodes = sorted([self.node_id] + list(self.peers))
        return all_nodes[self.view % len(all_nodes)]
    
    def _hash(self, value: Any) -> str:
        """Compute hash of a value."""
        serialized = json.dumps(value, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]
    
    def _sign(self, message: BFTMessage) -> str:
        """Sign a message (simulated)."""
        data = f"{message.type}:{message.view}:{message.sequence}:{message.digest}"
        return hashlib.sha256(f"{data}:{self.node_id}".encode()).hexdigest()[:16]
    
    def _verify_signature(self, message: BFTMessage, sender: str) -> bool:
        """Verify message signature (simulated)."""
        # In real implementation, would use public key crypto
        return len(message.signature) == 16
    
    async def propose(self, value: Any) -> bool:
        """Propose a value through PBFT."""
        if self.node_id != self.primary:
            # Forward to primary
            return False
        
        self.sequence += 1
        digest = self._hash(value)
        
        # Pre-prepare phase
        pre_prepare = BFTMessage(
            type="pre-prepare",
            view=self.view,
            sequence=self.sequence,
            digest=digest,
            sender=self.node_id,
            value=value
        )
        pre_prepare.signature = self._sign(pre_prepare)
        
        key = (self.view, self.sequence)
        self.pre_prepare_log[key] = pre_prepare
        
        # Broadcast pre-prepare
        await self._broadcast(pre_prepare)
        
        # Simulate prepare phase
        prepare_count = await self._collect_prepares(key, digest)
        if prepare_count < 2 * self.f:
            return False
        
        # Simulate commit phase
        commit_count = await self._collect_commits(key, digest)
        if commit_count < 2 * self.f + 1:
            return False
        
        # Commit locally
        self.committed[self.sequence] = value
        return True
    
    async def _broadcast(self, message: BFTMessage):
        """Broadcast message to all peers."""
        # In real implementation, would send network messages
        pass
    
    async def _collect_prepares(self, key: Tuple[int, int], digest: str) -> int:
        """Collect prepare messages."""
        # Simulating: random number of prepares
        return random.randint(self.f, self.total_nodes - 1)
    
    async def _collect_commits(self, key: Tuple[int, int], digest: str) -> int:
        """Collect commit messages."""
        # Simulating: usually get enough commits
        return random.randint(2 * self.f, self.total_nodes)
    
    async def handle_pre_prepare(self, message: BFTMessage) -> bool:
        """Handle incoming pre-prepare message."""
        if message.sender != self.primary:
            return False
        
        if message.view != self.view:
            return False
        
        key = (message.view, message.sequence)
        if key in self.pre_prepare_log:
            return False
        
        # Verify and store
        self.pre_prepare_log[key] = message
        
        # Send prepare
        prepare = BFTMessage(
            type="prepare",
            view=self.view,
            sequence=message.sequence,
            digest=message.digest,
            sender=self.node_id
        )
        prepare.signature = self._sign(prepare)
        await self._broadcast(prepare)
        
        return True
    
    async def view_change(self):
        """Initiate a view change."""
        self.view += 1
        self.primary = self._get_primary()
    
    async def get_consensus(self) -> Optional[Any]:
        """Get committed values."""
        return dict(self.committed)
    
    def get_status(self) -> Dict[str, Any]:
        """Get BFT status."""
        return {
            'node_id': self.node_id,
            'view': self.view,
            'sequence': self.sequence,
            'primary': self.primary,
            'is_primary': self.node_id == self.primary,
            'f': self.f,
            'total_nodes': self.total_nodes,
            'committed_count': len(self.committed)
        }


# =============================================================================
# VOTING PROTOCOLS
# =============================================================================

class VoteType(Enum):
    """Types of votes."""
    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"


@dataclass
class Vote:
    """A vote cast by an agent."""
    voter: str
    vote_type: VoteType
    weight: float = 1.0
    reason: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    signature: str = ""


@dataclass
class Proposal:
    """A proposal to be voted on."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    proposer: str = ""
    value: Any = None
    votes: List[Vote] = field(default_factory=list)
    status: str = "open"  # open, passed, rejected, expired
    created_at: datetime = field(default_factory=datetime.now)
    deadline: Optional[datetime] = None
    quorum: float = 0.5  # Fraction of voters needed
    threshold: float = 0.5  # Fraction of approve votes needed


class VotingProtocol(ConsensusProtocol):
    """
    Weighted voting protocol for agent decisions.
    
    Supports various voting mechanisms with reputation-based weights.
    """
    
    def __init__(self, default_quorum: float = 0.5, default_threshold: float = 0.5):
        self.proposals: Dict[str, Proposal] = {}
        self.voter_weights: Dict[str, float] = {}
        self.default_quorum = default_quorum
        self.default_threshold = default_threshold
        
        # History
        self.voting_history: List[Dict[str, Any]] = []
    
    def register_voter(self, voter_id: str, weight: float = 1.0):
        """Register a voter with a weight."""
        self.voter_weights[voter_id] = weight
    
    def update_weight(self, voter_id: str, weight: float):
        """Update a voter's weight."""
        self.voter_weights[voter_id] = weight
    
    def create_proposal(self, proposer: str, title: str, description: str,
                       value: Any = None, deadline_seconds: float = 3600,
                       quorum: float = None, threshold: float = None) -> str:
        """Create a new proposal."""
        proposal = Proposal(
            title=title,
            description=description,
            proposer=proposer,
            value=value,
            deadline=datetime.now() + timedelta(seconds=deadline_seconds),
            quorum=quorum or self.default_quorum,
            threshold=threshold or self.default_threshold
        )
        self.proposals[proposal.id] = proposal
        return proposal.id
    
    def cast_vote(self, proposal_id: str, voter: str, 
                  vote_type: VoteType, reason: str = "") -> bool:
        """Cast a vote on a proposal."""
        if proposal_id not in self.proposals:
            return False
        
        proposal = self.proposals[proposal_id]
        
        if proposal.status != "open":
            return False
        
        if proposal.deadline and datetime.now() > proposal.deadline:
            proposal.status = "expired"
            return False
        
        # Check if already voted
        if any(v.voter == voter for v in proposal.votes):
            return False
        
        weight = self.voter_weights.get(voter, 1.0)
        vote = Vote(
            voter=voter,
            vote_type=vote_type,
            weight=weight,
            reason=reason
        )
        proposal.votes.append(vote)
        
        return True
    
    def tally_votes(self, proposal_id: str) -> Dict[str, Any]:
        """Tally votes for a proposal."""
        if proposal_id not in self.proposals:
            return {}
        
        proposal = self.proposals[proposal_id]
        
        approve_weight = sum(v.weight for v in proposal.votes if v.vote_type == VoteType.APPROVE)
        reject_weight = sum(v.weight for v in proposal.votes if v.vote_type == VoteType.REJECT)
        abstain_weight = sum(v.weight for v in proposal.votes if v.vote_type == VoteType.ABSTAIN)
        
        total_voted = approve_weight + reject_weight + abstain_weight
        total_possible = sum(self.voter_weights.values())
        
        participation = total_voted / total_possible if total_possible > 0 else 0
        
        decision_votes = approve_weight + reject_weight
        approval_rate = approve_weight / decision_votes if decision_votes > 0 else 0
        
        return {
            'approve': approve_weight,
            'reject': reject_weight,
            'abstain': abstain_weight,
            'total_voted': total_voted,
            'total_possible': total_possible,
            'participation': participation,
            'approval_rate': approval_rate,
            'quorum_met': participation >= proposal.quorum,
            'threshold_met': approval_rate >= proposal.threshold
        }
    
    def finalize_proposal(self, proposal_id: str) -> str:
        """Finalize a proposal based on votes."""
        if proposal_id not in self.proposals:
            return "not_found"
        
        proposal = self.proposals[proposal_id]
        
        if proposal.status != "open":
            return proposal.status
        
        tally = self.tally_votes(proposal_id)
        
        if not tally['quorum_met']:
            proposal.status = "no_quorum"
        elif tally['threshold_met']:
            proposal.status = "passed"
        else:
            proposal.status = "rejected"
        
        # Record history
        self.voting_history.append({
            'proposal_id': proposal_id,
            'title': proposal.title,
            'status': proposal.status,
            'tally': tally,
            'finalized_at': datetime.now().isoformat()
        })
        
        return proposal.status
    
    async def propose(self, value: Any) -> bool:
        """Propose and immediately vote (for ConsensusProtocol interface)."""
        proposal_id = self.create_proposal(
            proposer="system",
            title="Automated Proposal",
            description=str(value),
            value=value,
            deadline_seconds=60
        )
        # Would normally wait for votes
        return proposal_id is not None
    
    async def get_consensus(self) -> Optional[Any]:
        """Get the most recently passed proposal."""
        passed = [p for p in self.proposals.values() if p.status == "passed"]
        if passed:
            return sorted(passed, key=lambda p: p.created_at, reverse=True)[0].value
        return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get voting system status."""
        by_status = {}
        for p in self.proposals.values():
            by_status[p.status] = by_status.get(p.status, 0) + 1
        
        return {
            'total_proposals': len(self.proposals),
            'by_status': by_status,
            'registered_voters': len(self.voter_weights),
            'total_votes_cast': sum(len(p.votes) for p in self.proposals.values())
        }


# =============================================================================
# CONFLICT RESOLUTION
# =============================================================================

class ConflictType(Enum):
    """Types of conflicts between agents."""
    RESOURCE = "resource"          # Competing for same resource
    DECISION = "decision"          # Different preferred actions
    PRIORITY = "priority"          # Task priority disagreement
    VALUE = "value"                # Different value judgments


@dataclass
class Conflict:
    """A conflict between agents."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: ConflictType = ConflictType.DECISION
    parties: Set[str] = field(default_factory=set)
    positions: Dict[str, Any] = field(default_factory=dict)  # party -> position
    context: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"  # pending, resolving, resolved, escalated
    resolution: Optional[Any] = None
    resolution_method: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None


class ConflictResolver:
    """
    Automatic conflict resolution between agents.
    
    Uses various strategies to resolve disagreements fairly.
    """
    
    def __init__(self):
        self.conflicts: Dict[str, Conflict] = {}
        self.agent_priorities: Dict[str, int] = {}  # Higher = more priority
        self.agent_trust: Dict[str, float] = {}  # 0-1 trust score
        self.resolution_history: List[Dict[str, Any]] = []
        
        # Resolution strategies
        self.strategies = {
            'majority': self._resolve_by_majority,
            'priority': self._resolve_by_priority,
            'trust': self._resolve_by_trust,
            'random': self._resolve_by_random,
            'merge': self._resolve_by_merge,
            'negotiate': self._resolve_by_negotiation
        }
    
    def register_agent(self, agent_id: str, priority: int = 0, trust: float = 0.5):
        """Register an agent with priority and trust scores."""
        self.agent_priorities[agent_id] = priority
        self.agent_trust[agent_id] = trust
    
    def report_conflict(self, conflict_type: ConflictType,
                       parties: Set[str], positions: Dict[str, Any],
                       context: Dict[str, Any] = None) -> str:
        """Report a new conflict."""
        conflict = Conflict(
            type=conflict_type,
            parties=parties,
            positions=positions,
            context=context or {}
        )
        self.conflicts[conflict.id] = conflict
        return conflict.id
    
    async def resolve(self, conflict_id: str, 
                     strategy: str = "auto") -> Optional[Any]:
        """Resolve a conflict using specified strategy."""
        if conflict_id not in self.conflicts:
            return None
        
        conflict = self.conflicts[conflict_id]
        
        if conflict.status == "resolved":
            return conflict.resolution
        
        conflict.status = "resolving"
        
        # Auto-select strategy based on conflict type
        if strategy == "auto":
            if conflict.type == ConflictType.RESOURCE:
                strategy = "priority"
            elif conflict.type == ConflictType.DECISION:
                strategy = "majority"
            elif conflict.type == ConflictType.PRIORITY:
                strategy = "trust"
            else:
                strategy = "negotiate"
        
        # Apply strategy
        resolver = self.strategies.get(strategy, self._resolve_by_majority)
        resolution = await resolver(conflict)
        
        conflict.resolution = resolution
        conflict.resolution_method = strategy
        conflict.status = "resolved"
        conflict.resolved_at = datetime.now()
        
        # Record history
        self.resolution_history.append({
            'conflict_id': conflict_id,
            'type': conflict.type.value,
            'parties': list(conflict.parties),
            'strategy': strategy,
            'resolution': resolution,
            'timestamp': datetime.now().isoformat()
        })
        
        return resolution
    
    async def _resolve_by_majority(self, conflict: Conflict) -> Any:
        """Resolve by majority vote."""
        position_counts: Dict[str, int] = {}
        for pos in conflict.positions.values():
            key = str(pos)
            position_counts[key] = position_counts.get(key, 0) + 1
        
        if position_counts:
            winner = max(position_counts.items(), key=lambda x: x[1])
            return winner[0]
        return None
    
    async def _resolve_by_priority(self, conflict: Conflict) -> Any:
        """Resolve by highest priority agent."""
        best_party = None
        best_priority = -1
        
        for party in conflict.parties:
            priority = self.agent_priorities.get(party, 0)
            if priority > best_priority:
                best_priority = priority
                best_party = party
        
        if best_party:
            return conflict.positions.get(best_party)
        return None
    
    async def _resolve_by_trust(self, conflict: Conflict) -> Any:
        """Resolve by most trusted agent."""
        best_party = None
        best_trust = -1
        
        for party in conflict.parties:
            trust = self.agent_trust.get(party, 0.5)
            if trust > best_trust:
                best_trust = trust
                best_party = party
        
        if best_party:
            return conflict.positions.get(best_party)
        return None
    
    async def _resolve_by_random(self, conflict: Conflict) -> Any:
        """Resolve by random selection."""
        if conflict.parties:
            winner = random.choice(list(conflict.parties))
            return conflict.positions.get(winner)
        return None
    
    async def _resolve_by_merge(self, conflict: Conflict) -> Any:
        """Try to merge positions if possible."""
        # For dict positions, try to merge
        merged = {}
        for pos in conflict.positions.values():
            if isinstance(pos, dict):
                merged.update(pos)
        return merged if merged else None
    
    async def _resolve_by_negotiation(self, conflict: Conflict) -> Any:
        """Simulate negotiation process."""
        # In real implementation, would involve back-and-forth
        # Here we use weighted random based on trust
        
        weights = []
        parties = list(conflict.parties)
        
        for party in parties:
            weights.append(self.agent_trust.get(party, 0.5))
        
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
            winner = random.choices(parties, weights=weights)[0]
            return conflict.positions.get(winner)
        
        return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get conflict resolver status."""
        by_status = {}
        by_type = {}
        
        for c in self.conflicts.values():
            by_status[c.status] = by_status.get(c.status, 0) + 1
            by_type[c.type.value] = by_type.get(c.type.value, 0) + 1
        
        return {
            'total_conflicts': len(self.conflicts),
            'by_status': by_status,
            'by_type': by_type,
            'registered_agents': len(self.agent_priorities),
            'resolutions': len(self.resolution_history)
        }


# =============================================================================
# DEMO
# =============================================================================

async def demo_consensus():
    """Demonstrate consensus mechanisms."""
    print("🤝 Swarm Intelligence 2.0 - Consensus Demo")
    print("=" * 50)
    
    # Demo 1: Raft Consensus
    print("\n📋 Raft Consensus:")
    raft = RaftConsensus("node1", ["node2", "node3", "node4", "node5"])
    await raft.start()
    
    # Force leader state for demo
    raft.node.state = ConsensusState.LEADER
    
    success = await raft.propose({'key': 'config', 'value': {'timeout': 30}})
    print(f"  Proposal success: {success}")
    print(f"  Status: {raft.get_status()}")
    
    await raft.stop()
    
    # Demo 2: Voting Protocol
    print("\n🗳️ Voting Protocol:")
    voting = VotingProtocol(default_quorum=0.5, default_threshold=0.6)
    
    for i, weight in enumerate([1.0, 1.2, 0.8, 1.5, 1.0]):
        voting.register_voter(f"agent{i}", weight)
    
    proposal_id = voting.create_proposal(
        proposer="agent0",
        title="Increase Memory Limit",
        description="Proposal to increase agent memory from 1GB to 2GB",
        value={'memory_limit': '2GB'},
        deadline_seconds=60
    )
    
    # Cast votes
    voting.cast_vote(proposal_id, "agent0", VoteType.APPROVE, "Needed for large models")
    voting.cast_vote(proposal_id, "agent1", VoteType.APPROVE, "Agree")
    voting.cast_vote(proposal_id, "agent2", VoteType.REJECT, "Too expensive")
    voting.cast_vote(proposal_id, "agent3", VoteType.APPROVE, "Good improvement")
    voting.cast_vote(proposal_id, "agent4", VoteType.ABSTAIN, "No opinion")
    
    tally = voting.tally_votes(proposal_id)
    print(f"  Tally: {tally}")
    
    result = voting.finalize_proposal(proposal_id)
    print(f"  Result: {result}")
    
    # Demo 3: Conflict Resolution
    print("\n⚖️ Conflict Resolution:")
    resolver = ConflictResolver()
    
    resolver.register_agent("alpha", priority=2, trust=0.8)
    resolver.register_agent("beta", priority=1, trust=0.9)
    resolver.register_agent("gamma", priority=3, trust=0.6)
    
    conflict_id = resolver.report_conflict(
        conflict_type=ConflictType.RESOURCE,
        parties={"alpha", "beta", "gamma"},
        positions={
            "alpha": "use_gpu_0",
            "beta": "use_gpu_1",
            "gamma": "use_gpu_0"
        },
        context={'resource': 'gpu', 'available': ['gpu_0', 'gpu_1']}
    )
    
    resolution = await resolver.resolve(conflict_id, strategy="priority")
    print(f"  Resolution: {resolution}")
    print(f"  Status: {resolver.get_status()}")
    
    print()
    print("✅ Consensus demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_consensus())
