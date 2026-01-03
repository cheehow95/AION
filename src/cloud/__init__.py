"""
AION Cloud-Native Runtime - Package Initialization
===================================================

Enterprise-grade cloud deployment infrastructure.
"""

from .kubernetes_operator import (
    AIOnAgentCRD,
    AIOnSwarmCRD,
    KubernetesOperator,
    ReconciliationLoop
)

from .autoscaling import (
    ScalingPolicy,
    MetricsCollector,
    HorizontalAutoscaler,
    ScaleToZero
)

from .gpu_scheduler import (
    GPUResource,
    GPUScheduler,
    AffinityScheduler,
    VRAMManager
)

from .multi_region import (
    Region,
    RegionManager,
    FailoverController,
    LatencyOptimizer
)

__all__ = [
    # Kubernetes Operator
    'AIOnAgentCRD',
    'AIOnSwarmCRD',
    'KubernetesOperator',
    'ReconciliationLoop',
    # Autoscaling
    'ScalingPolicy',
    'MetricsCollector',
    'HorizontalAutoscaler',
    'ScaleToZero',
    # GPU Scheduler
    'GPUResource',
    'GPUScheduler',
    'AffinityScheduler',
    'VRAMManager',
    # Multi-Region
    'Region',
    'RegionManager',
    'FailoverController',
    'LatencyOptimizer',
]
