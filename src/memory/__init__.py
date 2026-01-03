"""
AION Enhanced Memory System - Package Initialization
=====================================================

Cross-session persistent memory like GPT-5.2:
- Persistent memory storage
- Knowledge graph relationships
- Memory consolidation
"""

from .persistent_memory import (
    Memory,
    MemoryStore,
    PersistentMemoryManager,
    MemoryType
)

from .memory_graph import (
    MemoryNode,
    MemoryEdge,
    MemoryGraph,
    EntityType
)

from .personalization import (
    UserPreference,
    PreferenceType,
    PersonalizationEngine
)

__all__ = [
    # Persistent Memory
    'Memory',
    'MemoryStore',
    'PersistentMemoryManager',
    'MemoryType',
    # Memory Graph
    'MemoryNode',
    'MemoryEdge',
    'MemoryGraph',
    'EntityType',
    # Personalization
    'UserPreference',
    'PreferenceType',
    'PersonalizationEngine',
]
