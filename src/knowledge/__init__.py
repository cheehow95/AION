"""
AION Knowledge Module
=====================

Knowledge representation and reasoning:
- Knowledge Graph
- Formal Logic
"""

from src.knowledge.knowledge_graph import (
    KnowledgeGraph,
    KnowledgeGraphEngine,
    Entity,
    Relation,
    Triple,
    RelationType,
    EntityType
)

from src.knowledge.formal_logic import (
    FormalLogicEngine,
    Expression,
    Proposition,
    Not,
    And,
    Or,
    Implies,
    Iff,
    TruthTable,
    TheoremProver
)

__all__ = [
    'KnowledgeGraph',
    'KnowledgeGraphEngine',
    'Entity',
    'Relation',
    'Triple',
    'RelationType',
    'EntityType',
    'FormalLogicEngine',
    'Expression',
    'Proposition',
    'Not',
    'And',
    'Or',
    'Implies',
    'Iff',
    'TruthTable',
    'TheoremProver'
]
