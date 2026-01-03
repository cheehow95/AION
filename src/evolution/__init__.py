"""
AION Self-Evolution v2 - Package Initialization
================================================

Enhanced self-improvement capabilities.
"""

from .benchmark_discovery import (
    Benchmark,
    BenchmarkResult,
    BenchmarkDiscovery,
    PerformanceProbe
)

from .architecture_search import (
    ArchitectureSpace,
    Architecture,
    ArchitectureSearch,
    EvolutionStrategy
)

from .knowledge_transfer import (
    Knowledge,
    KnowledgeGraph,
    KnowledgeDistillation,
    ExperienceReplay
)

from .safety_evolution import (
    SafetyConstraint,
    SafetyTest,
    SafetyEvolution,
    ConstraintLearner
)

__all__ = [
    # Benchmark Discovery
    'Benchmark',
    'BenchmarkResult',
    'BenchmarkDiscovery',
    'PerformanceProbe',
    # Architecture Search
    'ArchitectureSpace',
    'Architecture',
    'ArchitectureSearch',
    'EvolutionStrategy',
    # Knowledge Transfer
    'Knowledge',
    'KnowledgeGraph',
    'KnowledgeDistillation',
    'ExperienceReplay',
    # Safety Evolution
    'SafetyConstraint',
    'SafetyTest',
    'SafetyEvolution',
    'ConstraintLearner',
]
