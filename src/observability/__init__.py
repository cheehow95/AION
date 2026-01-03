"""
AION Observability Module
=========================

Comprehensive observability for AION agents:
- Distributed tracing (OpenTelemetry-compatible)
- Cost tracking and analytics
- Performance profiling
- Metrics and dashboards
"""

from .tracer import AgentTracer, Span, SpanContext
from .metrics import MetricsCollector, Counter, Gauge, Histogram
from .profiler import AgentProfiler, ProfileResult
from .cost_tracker import CostTracker, CostReport

__all__ = [
    'AgentTracer',
    'Span',
    'SpanContext',
    'MetricsCollector',
    'Counter',
    'Gauge',
    'Histogram',
    'AgentProfiler',
    'ProfileResult',
    'CostTracker',
    'CostReport'
]
