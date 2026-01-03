"""
AION Metrics Collector
======================

Metrics collection for agent monitoring.
Supports counters, gauges, and histograms.
"""

from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict
import threading
import time


@dataclass
class MetricPoint:
    """A single metric data point."""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


class Counter:
    """A monotonically increasing counter metric."""
    
    def __init__(self, name: str, description: str = "", labels: List[str] = None):
        self.name = name
        self.description = description
        self.label_keys = labels or []
        self._values: Dict[tuple, float] = defaultdict(float)
        self._lock = threading.Lock()
    
    def inc(self, value: float = 1, **labels):
        """Increment the counter."""
        label_tuple = tuple(labels.get(k, "") for k in self.label_keys)
        with self._lock:
            self._values[label_tuple] += value
    
    def get(self, **labels) -> float:
        """Get counter value."""
        label_tuple = tuple(labels.get(k, "") for k in self.label_keys)
        return self._values.get(label_tuple, 0)
    
    def get_all(self) -> Dict[tuple, float]:
        """Get all counter values."""
        return dict(self._values)


class Gauge:
    """A metric that can go up and down."""
    
    def __init__(self, name: str, description: str = "", labels: List[str] = None):
        self.name = name
        self.description = description
        self.label_keys = labels or []
        self._values: Dict[tuple, float] = {}
        self._lock = threading.Lock()
    
    def set(self, value: float, **labels):
        """Set gauge value."""
        label_tuple = tuple(labels.get(k, "") for k in self.label_keys)
        with self._lock:
            self._values[label_tuple] = value
    
    def inc(self, value: float = 1, **labels):
        """Increment gauge."""
        label_tuple = tuple(labels.get(k, "") for k in self.label_keys)
        with self._lock:
            self._values[label_tuple] = self._values.get(label_tuple, 0) + value
    
    def dec(self, value: float = 1, **labels):
        """Decrement gauge."""
        self.inc(-value, **labels)
    
    def get(self, **labels) -> float:
        """Get gauge value."""
        label_tuple = tuple(labels.get(k, "") for k in self.label_keys)
        return self._values.get(label_tuple, 0)


class Histogram:
    """A metric for measuring distributions."""
    
    DEFAULT_BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
    
    def __init__(
        self,
        name: str,
        description: str = "",
        labels: List[str] = None,
        buckets: List[float] = None
    ):
        self.name = name
        self.description = description
        self.label_keys = labels or []
        self.buckets = sorted(buckets or self.DEFAULT_BUCKETS)
        
        self._counts: Dict[tuple, Dict[float, int]] = defaultdict(lambda: defaultdict(int))
        self._sums: Dict[tuple, float] = defaultdict(float)
        self._totals: Dict[tuple, int] = defaultdict(int)
        self._lock = threading.Lock()
    
    def observe(self, value: float, **labels):
        """Record an observation."""
        label_tuple = tuple(labels.get(k, "") for k in self.label_keys)
        
        with self._lock:
            self._sums[label_tuple] += value
            self._totals[label_tuple] += 1
            
            for bucket in self.buckets:
                if value <= bucket:
                    self._counts[label_tuple][bucket] += 1
    
    def get_statistics(self, **labels) -> Dict[str, float]:
        """Get histogram statistics."""
        label_tuple = tuple(labels.get(k, "") for k in self.label_keys)
        
        total = self._totals.get(label_tuple, 0)
        if total == 0:
            return {"count": 0, "sum": 0, "avg": 0}
        
        return {
            "count": total,
            "sum": self._sums.get(label_tuple, 0),
            "avg": self._sums.get(label_tuple, 0) / total,
            "buckets": dict(self._counts.get(label_tuple, {}))
        }


class MetricsCollector:
    """
    Central metrics collection system.
    
    Provides:
    - Metric registration and collection
    - Prometheus-compatible export
    - Aggregation and reporting
    """
    
    def __init__(self, prefix: str = "aion"):
        self.prefix = prefix
        self.counters: Dict[str, Counter] = {}
        self.gauges: Dict[str, Gauge] = {}
        self.histograms: Dict[str, Histogram] = {}
        
        # Built-in metrics
        self._register_builtins()
    
    def _register_builtins(self):
        """Register built-in metrics."""
        # Agent metrics
        self.register_counter("agent_requests_total", "Total agent requests", ["agent", "status"])
        self.register_counter("agent_tokens_total", "Total tokens used", ["agent", "direction"])
        self.register_histogram("agent_latency_seconds", "Agent request latency", ["agent"])
        
        # Tool metrics
        self.register_counter("tool_calls_total", "Total tool invocations", ["tool", "status"])
        self.register_histogram("tool_duration_seconds", "Tool call duration", ["tool"])
        
        # Memory metrics
        self.register_gauge("memory_items", "Items in memory", ["type"])
        self.register_counter("memory_operations_total", "Memory operations", ["type", "operation"])
        
        # Model metrics
        self.register_counter("model_requests_total", "Model API requests", ["model", "status"])
        self.register_histogram("model_latency_seconds", "Model request latency", ["model"])
        self.register_counter("model_cost_cents", "Model cost in cents", ["model"])
    
    def register_counter(
        self,
        name: str,
        description: str = "",
        labels: List[str] = None
    ) -> Counter:
        """Register a new counter."""
        full_name = f"{self.prefix}_{name}"
        counter = Counter(full_name, description, labels)
        self.counters[full_name] = counter
        return counter
    
    def register_gauge(
        self,
        name: str,
        description: str = "",
        labels: List[str] = None
    ) -> Gauge:
        """Register a new gauge."""
        full_name = f"{self.prefix}_{name}"
        gauge = Gauge(full_name, description, labels)
        self.gauges[full_name] = gauge
        return gauge
    
    def register_histogram(
        self,
        name: str,
        description: str = "",
        labels: List[str] = None,
        buckets: List[float] = None
    ) -> Histogram:
        """Register a new histogram."""
        full_name = f"{self.prefix}_{name}"
        histogram = Histogram(full_name, description, labels, buckets)
        self.histograms[full_name] = histogram
        return histogram
    
    def get_counter(self, name: str) -> Optional[Counter]:
        """Get a counter by name."""
        full_name = f"{self.prefix}_{name}"
        return self.counters.get(full_name)
    
    def get_gauge(self, name: str) -> Optional[Gauge]:
        """Get a gauge by name."""
        full_name = f"{self.prefix}_{name}"
        return self.gauges.get(full_name)
    
    def get_histogram(self, name: str) -> Optional[Histogram]:
        """Get a histogram by name."""
        full_name = f"{self.prefix}_{name}"
        return self.histograms.get(full_name)
    
    def record_agent_request(
        self,
        agent: str,
        success: bool,
        latency_seconds: float,
        tokens_in: int = 0,
        tokens_out: int = 0
    ):
        """Record an agent request."""
        status = "success" if success else "error"
        
        self.counters[f"{self.prefix}_agent_requests_total"].inc(agent=agent, status=status)
        self.histograms[f"{self.prefix}_agent_latency_seconds"].observe(latency_seconds, agent=agent)
        
        if tokens_in:
            self.counters[f"{self.prefix}_agent_tokens_total"].inc(tokens_in, agent=agent, direction="input")
        if tokens_out:
            self.counters[f"{self.prefix}_agent_tokens_total"].inc(tokens_out, agent=agent, direction="output")
    
    def record_tool_call(
        self,
        tool: str,
        success: bool,
        duration_seconds: float
    ):
        """Record a tool invocation."""
        status = "success" if success else "error"
        
        self.counters[f"{self.prefix}_tool_calls_total"].inc(tool=tool, status=status)
        self.histograms[f"{self.prefix}_tool_duration_seconds"].observe(duration_seconds, tool=tool)
    
    def record_model_request(
        self,
        model: str,
        success: bool,
        latency_seconds: float,
        cost_cents: float = 0
    ):
        """Record a model API request."""
        status = "success" if success else "error"
        
        self.counters[f"{self.prefix}_model_requests_total"].inc(model=model, status=status)
        self.histograms[f"{self.prefix}_model_latency_seconds"].observe(latency_seconds, model=model)
        
        if cost_cents:
            self.counters[f"{self.prefix}_model_cost_cents"].inc(cost_cents, model=model)
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        # Counters
        for name, counter in self.counters.items():
            lines.append(f"# HELP {name} {counter.description}")
            lines.append(f"# TYPE {name} counter")
            for labels, value in counter.get_all().items():
                label_str = self._format_labels(counter.label_keys, labels)
                lines.append(f"{name}{label_str} {value}")
        
        # Gauges
        for name, gauge in self.gauges.items():
            lines.append(f"# HELP {name} {gauge.description}")
            lines.append(f"# TYPE {name} gauge")
            for labels, value in gauge._values.items():
                label_str = self._format_labels(gauge.label_keys, labels)
                lines.append(f"{name}{label_str} {value}")
        
        # Histograms
        for name, histogram in self.histograms.items():
            lines.append(f"# HELP {name} {histogram.description}")
            lines.append(f"# TYPE {name} histogram")
            for labels in histogram._totals.keys():
                label_str = self._format_labels(histogram.label_keys, labels)
                stats = histogram.get_statistics(**dict(zip(histogram.label_keys, labels)))
                lines.append(f"{name}_sum{label_str} {stats['sum']}")
                lines.append(f"{name}_count{label_str} {stats['count']}")
        
        return "\n".join(lines)
    
    def _format_labels(self, keys: List[str], values: tuple) -> str:
        """Format labels for Prometheus output."""
        if not keys:
            return ""
        pairs = [f'{k}="{v}"' for k, v in zip(keys, values)]
        return "{" + ",".join(pairs) + "}"
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics."""
        return {
            "counters": {
                name: sum(c.get_all().values())
                for name, c in self.counters.items()
            },
            "gauges": {
                name: list(g._values.values())
                for name, g in self.gauges.items()
            },
            "histograms": {
                name: h.get_statistics()
                for name, h in self.histograms.items()
            }
        }


# Global collector instance
_global_collector: Optional[MetricsCollector] = None


def get_metrics() -> MetricsCollector:
    """Get the global metrics collector."""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector
