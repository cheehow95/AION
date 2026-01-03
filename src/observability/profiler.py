"""
AION Agent Profiler
===================

Performance profiling for agent operations.
Identifies bottlenecks and optimization opportunities.
"""

from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import contextmanager
import time
import statistics


@dataclass
class ProfileSample:
    """A single profiling sample."""
    name: str
    start_time: float
    end_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    children: List['ProfileSample'] = field(default_factory=list)
    
    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000
    
    @property
    def self_time_ms(self) -> float:
        child_time = sum(c.duration_ms for c in self.children)
        return self.duration_ms - child_time


@dataclass
class ProfileResult:
    """Results of a profiling session."""
    session_name: str
    start_time: datetime
    end_time: datetime
    samples: List[ProfileSample]
    
    @property
    def total_duration_ms(self) -> float:
        return (self.end_time - self.start_time).total_seconds() * 1000
    
    def get_hotspots(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Get the slowest operations."""
        all_samples = self._flatten_samples(self.samples)
        sorted_samples = sorted(all_samples, key=lambda s: s.duration_ms, reverse=True)
        
        return [
            {
                "name": s.name,
                "duration_ms": s.duration_ms,
                "self_time_ms": s.self_time_ms,
                "metadata": s.metadata
            }
            for s in sorted_samples[:top_n]
        ]
    
    def get_aggregated(self) -> Dict[str, Dict[str, float]]:
        """Get aggregated stats by operation name."""
        all_samples = self._flatten_samples(self.samples)
        
        by_name: Dict[str, List[float]] = {}
        for sample in all_samples:
            if sample.name not in by_name:
                by_name[sample.name] = []
            by_name[sample.name].append(sample.duration_ms)
        
        result = {}
        for name, durations in by_name.items():
            result[name] = {
                "count": len(durations),
                "total_ms": sum(durations),
                "avg_ms": statistics.mean(durations),
                "min_ms": min(durations),
                "max_ms": max(durations),
                "std_ms": statistics.stdev(durations) if len(durations) > 1 else 0
            }
        
        return result
    
    def _flatten_samples(self, samples: List[ProfileSample]) -> List[ProfileSample]:
        """Flatten nested samples."""
        result = []
        for sample in samples:
            result.append(sample)
            result.extend(self._flatten_samples(sample.children))
        return result
    
    def to_flamegraph(self) -> Dict[str, Any]:
        """Convert to flamegraph-compatible format."""
        def build_node(sample: ProfileSample) -> Dict[str, Any]:
            return {
                "name": sample.name,
                "value": sample.duration_ms,
                "children": [build_node(c) for c in sample.children]
            }
        
        return {
            "name": self.session_name,
            "value": self.total_duration_ms,
            "children": [build_node(s) for s in self.samples]
        }
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session": self.session_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "total_duration_ms": self.total_duration_ms,
            "hotspots": self.get_hotspots(),
            "aggregated": self.get_aggregated()
        }


class AgentProfiler:
    """
    Profiler for AION agent performance.
    
    Provides:
    - Call stack profiling
    - Hotspot detection
    - Memory tracking
    - CPU time measurement
    """
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._sessions: Dict[str, ProfileResult] = {}
        self._current_session: Optional[str] = None
        self._sample_stack: List[ProfileSample] = []
        self._session_start: Optional[datetime] = None
        self._session_samples: List[ProfileSample] = []
    
    def start_session(self, name: str = "default"):
        """Start a new profiling session."""
        self._current_session = name
        self._session_start = datetime.now()
        self._sample_stack = []
        self._session_samples = []
    
    def end_session(self) -> Optional[ProfileResult]:
        """End current session and return results."""
        if not self._current_session:
            return None
        
        result = ProfileResult(
            session_name=self._current_session,
            start_time=self._session_start or datetime.now(),
            end_time=datetime.now(),
            samples=self._session_samples
        )
        
        self._sessions[self._current_session] = result
        self._current_session = None
        
        return result
    
    @contextmanager
    def profile(self, name: str, **metadata):
        """
        Context manager for profiling a code block.
        
        Usage:
            with profiler.profile("operation"):
                # do work
        """
        if not self.enabled or not self._current_session:
            yield
            return
        
        sample = ProfileSample(
            name=name,
            start_time=time.perf_counter(),
            end_time=0,
            metadata=metadata
        )
        
        # Add to parent if exists
        if self._sample_stack:
            self._sample_stack[-1].children.append(sample)
        else:
            self._session_samples.append(sample)
        
        self._sample_stack.append(sample)
        
        try:
            yield sample
        finally:
            sample.end_time = time.perf_counter()
            self._sample_stack.pop()
    
    def record_sample(self, name: str, duration_ms: float, **metadata):
        """Manually record a profiling sample."""
        if not self.enabled or not self._current_session:
            return
        
        now = time.perf_counter()
        sample = ProfileSample(
            name=name,
            start_time=now - duration_ms / 1000,
            end_time=now,
            metadata=metadata
        )
        
        if self._sample_stack:
            self._sample_stack[-1].children.append(sample)
        else:
            self._session_samples.append(sample)
    
    def get_session(self, name: str) -> Optional[ProfileResult]:
        """Get a completed profiling session."""
        return self._sessions.get(name)
    
    def get_all_sessions(self) -> Dict[str, ProfileResult]:
        """Get all completed sessions."""
        return dict(self._sessions)
    
    def compare_sessions(
        self,
        session1: str,
        session2: str
    ) -> Dict[str, Dict[str, float]]:
        """Compare two profiling sessions."""
        s1 = self._sessions.get(session1)
        s2 = self._sessions.get(session2)
        
        if not s1 or not s2:
            return {}
        
        agg1 = s1.get_aggregated()
        agg2 = s2.get_aggregated()
        
        all_names = set(agg1.keys()) | set(agg2.keys())
        
        comparison = {}
        for name in all_names:
            v1 = agg1.get(name, {}).get("avg_ms", 0)
            v2 = agg2.get(name, {}).get("avg_ms", 0)
            
            comparison[name] = {
                "session1_avg_ms": v1,
                "session2_avg_ms": v2,
                "diff_ms": v2 - v1,
                "diff_percent": ((v2 - v1) / v1 * 100) if v1 > 0 else 0
            }
        
        return comparison
    
    def clear(self):
        """Clear all sessions."""
        self._sessions.clear()


# Global profiler
_global_profiler: Optional[AgentProfiler] = None


def get_profiler() -> AgentProfiler:
    """Get the global profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = AgentProfiler()
    return _global_profiler
