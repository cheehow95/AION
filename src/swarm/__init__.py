"""
AION Swarm Intelligence 2.0 - Package Initialization
=====================================================

Advanced multi-agent coordination with emergent behaviors.
"""

from .coordination import (
    CoordinationProtocol,
    Stigmergy,
    TaskAuction,
    SwarmSignal,
    CoalitionManager,
    EmergentCoordinator
)

from .consensus import (
    ConsensusProtocol,
    RaftConsensus,
    ByzantineFaultTolerance,
    VotingProtocol,
    ConflictResolver
)

from .reputation import (
    ReputationScore,
    ReputationSystem,
    TrustNetwork,
    AntiSybilGuard
)

from .hierarchy import (
    HierarchyNode,
    DynamicHierarchy,
    RoleAssignment,
    HierarchicalRouter
)

__all__ = [
    # Coordination
    'CoordinationProtocol',
    'Stigmergy',
    'TaskAuction',
    'SwarmSignal',
    'CoalitionManager',
    'EmergentCoordinator',
    # Consensus
    'ConsensusProtocol',
    'RaftConsensus',
    'ByzantineFaultTolerance',
    'VotingProtocol',
    'ConflictResolver',
    # Reputation
    'ReputationScore',
    'ReputationSystem',
    'TrustNetwork',
    'AntiSybilGuard',
    # Hierarchy
    'HierarchyNode',
    'DynamicHierarchy',
    'RoleAssignment',
    'HierarchicalRouter',
]
