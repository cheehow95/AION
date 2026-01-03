"""
AION Agent Tracer
=================

Distributed tracing for agent operations.
OpenTelemetry-compatible span format.
"""

from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from contextlib import contextmanager
import uuid
import time


class SpanStatus(Enum):
    """Status of a span."""
    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


class SpanKind(Enum):
    """Kind of span."""
    INTERNAL = "internal"
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"


@dataclass
class SpanContext:
    """Context for distributed tracing."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    baggage: Dict[str, str] = field(default_factory=dict)
    
    def to_headers(self) -> Dict[str, str]:
        """Convert to HTTP headers for propagation."""
        return {
            "traceparent": f"00-{self.trace_id}-{self.span_id}-01",
            "x-aion-baggage": ";".join(f"{k}={v}" for k, v in self.baggage.items())
        }
    
    @classmethod
    def from_headers(cls, headers: Dict[str, str]) -> Optional['SpanContext']:
        """Parse from HTTP headers."""
        traceparent = headers.get("traceparent", "")
        if not traceparent:
            return None
        
        parts = traceparent.split("-")
        if len(parts) < 3:
            return None
        
        baggage = {}
        baggage_str = headers.get("x-aion-baggage", "")
        if baggage_str:
            for item in baggage_str.split(";"):
                if "=" in item:
                    k, v = item.split("=", 1)
                    baggage[k] = v
        
        return cls(
            trace_id=parts[1],
            span_id=parts[2],
            baggage=baggage
        )


@dataclass
class SpanEvent:
    """An event within a span."""
    name: str
    timestamp: datetime
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpanLink:
    """A link to another span."""
    context: SpanContext
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    """A traced operation span."""
    name: str
    context: SpanContext
    kind: SpanKind = SpanKind.INTERNAL
    status: SpanStatus = SpanStatus.UNSET
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[SpanEvent] = field(default_factory=list)
    links: List[SpanLink] = field(default_factory=list)
    
    @property
    def duration_ms(self) -> float:
        """Get span duration in milliseconds."""
        if not self.end_time:
            return 0.0
        return (self.end_time - self.start_time).total_seconds() * 1000
    
    def add_event(self, name: str, attributes: Dict[str, Any] = None):
        """Add an event to the span."""
        self.events.append(SpanEvent(
            name=name,
            timestamp=datetime.now(),
            attributes=attributes or {}
        ))
    
    def set_attribute(self, key: str, value: Any):
        """Set a span attribute."""
        self.attributes[key] = value
    
    def set_status(self, status: SpanStatus, message: str = None):
        """Set span status."""
        self.status = status
        if message:
            self.attributes["status.message"] = message
    
    def end(self, status: SpanStatus = None):
        """End the span."""
        self.end_time = datetime.now()
        if status:
            self.status = status
        elif self.status == SpanStatus.UNSET:
            self.status = SpanStatus.OK
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "trace_id": self.context.trace_id,
            "span_id": self.context.span_id,
            "parent_span_id": self.context.parent_span_id,
            "kind": self.kind.value,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
            "events": [
                {"name": e.name, "timestamp": e.timestamp.isoformat(), "attributes": e.attributes}
                for e in self.events
            ]
        }


class AgentTracer:
    """
    Distributed tracer for AION agents.
    
    Provides:
    - Span creation and management
    - Context propagation
    - Export to various backends
    """
    
    def __init__(
        self,
        service_name: str = "aion-agent",
        enabled: bool = True
    ):
        self.service_name = service_name
        self.enabled = enabled
        
        # Active spans stack (per-trace)
        self._active_traces: Dict[str, List[Span]] = {}
        
        # Completed spans for export
        self.completed_spans: List[Span] = []
        self.max_completed = 10000
        
        # Exporters
        self.exporters: List[Callable[[Span], None]] = []
        
        # Sampling
        self.sample_rate = 1.0
    
    def start_span(
        self,
        name: str,
        parent: SpanContext = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Dict[str, Any] = None
    ) -> Span:
        """
        Start a new span.
        
        Args:
            name: Span name
            parent: Parent context for nested spans
            kind: Span kind
            attributes: Initial attributes
        
        Returns:
            New span
        """
        if not self.enabled:
            # Return no-op span
            return Span(
                name=name,
                context=SpanContext(trace_id="", span_id=""),
                kind=kind
            )
        
        # Generate IDs
        trace_id = parent.trace_id if parent else uuid.uuid4().hex[:32]
        span_id = uuid.uuid4().hex[:16]
        parent_span_id = parent.span_id if parent else None
        
        context = SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            baggage=parent.baggage.copy() if parent else {}
        )
        
        span = Span(
            name=name,
            context=context,
            kind=kind,
            attributes=attributes or {}
        )
        
        # Add service name
        span.attributes["service.name"] = self.service_name
        
        # Track active span
        if trace_id not in self._active_traces:
            self._active_traces[trace_id] = []
        self._active_traces[trace_id].append(span)
        
        return span
    
    def end_span(self, span: Span, status: SpanStatus = None):
        """End a span and record it."""
        span.end(status)
        
        # Remove from active
        trace_id = span.context.trace_id
        if trace_id in self._active_traces:
            self._active_traces[trace_id] = [
                s for s in self._active_traces[trace_id]
                if s.context.span_id != span.context.span_id
            ]
            if not self._active_traces[trace_id]:
                del self._active_traces[trace_id]
        
        # Record completed span
        self.completed_spans.append(span)
        if len(self.completed_spans) > self.max_completed:
            self.completed_spans = self.completed_spans[-self.max_completed:]
        
        # Export
        for exporter in self.exporters:
            try:
                exporter(span)
            except Exception:
                pass
    
    @contextmanager
    def trace(
        self,
        name: str,
        parent: SpanContext = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Dict[str, Any] = None
    ):
        """
        Context manager for tracing.
        
        Usage:
            with tracer.trace("operation") as span:
                # do work
                span.set_attribute("key", "value")
        """
        span = self.start_span(name, parent, kind, attributes)
        try:
            yield span
            self.end_span(span, SpanStatus.OK)
        except Exception as e:
            span.set_status(SpanStatus.ERROR, str(e))
            span.add_event("exception", {"message": str(e), "type": type(e).__name__})
            self.end_span(span, SpanStatus.ERROR)
            raise
    
    def get_current_span(self, trace_id: str = None) -> Optional[Span]:
        """Get the current active span."""
        if trace_id and trace_id in self._active_traces:
            spans = self._active_traces[trace_id]
            return spans[-1] if spans else None
        
        # Return most recent span from any trace
        for spans in self._active_traces.values():
            if spans:
                return spans[-1]
        return None
    
    def add_exporter(self, exporter: Callable[[Span], None]):
        """Add a span exporter."""
        self.exporters.append(exporter)
    
    def get_traces(self, limit: int = 100) -> Dict[str, List[Span]]:
        """Get completed traces grouped by trace_id."""
        traces: Dict[str, List[Span]] = {}
        
        for span in self.completed_spans[-limit:]:
            trace_id = span.context.trace_id
            if trace_id not in traces:
                traces[trace_id] = []
            traces[trace_id].append(span)
        
        return traces
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get tracer statistics."""
        if not self.completed_spans:
            return {
                "total_spans": 0,
                "active_traces": len(self._active_traces)
            }
        
        durations = [s.duration_ms for s in self.completed_spans if s.duration_ms > 0]
        error_count = sum(1 for s in self.completed_spans if s.status == SpanStatus.ERROR)
        
        return {
            "total_spans": len(self.completed_spans),
            "active_traces": len(self._active_traces),
            "error_rate": error_count / len(self.completed_spans),
            "avg_duration_ms": sum(durations) / len(durations) if durations else 0,
            "max_duration_ms": max(durations) if durations else 0,
            "spans_by_kind": {
                k.value: sum(1 for s in self.completed_spans if s.kind == k)
                for k in SpanKind
            }
        }
    
    def clear(self):
        """Clear all recorded spans."""
        self.completed_spans.clear()
        self._active_traces.clear()


# Console exporter for debugging
def console_exporter(span: Span):
    """Export span to console."""
    print(f"[TRACE] {span.name} | {span.duration_ms:.2f}ms | {span.status.value}")


# Global tracer instance
_global_tracer: Optional[AgentTracer] = None


def get_tracer(service_name: str = "aion-agent") -> AgentTracer:
    """Get the global tracer instance."""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = AgentTracer(service_name)
    return _global_tracer
