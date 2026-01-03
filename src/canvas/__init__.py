"""
AION Canvas Collaboration - Package Initialization
===================================================

Real-time collaborative editing like GPT-5.2 Canvas:
- Real-time document sync
- Multi-user presence
- Version history
"""

from .real_time import (
    Document,
    Operation,
    OperationType,
    CollaborativeSession,
    CRDTDocument
)

from .sharing import (
    Permission,
    ShareSettings,
    ShareManager,
    CollaborationInvite
)

__all__ = [
    # Real-time
    'Document',
    'Operation',
    'OperationType',
    'CollaborativeSession',
    'CRDTDocument',
    # Sharing
    'Permission',
    'ShareSettings',
    'ShareManager',
    'CollaborationInvite',
]
