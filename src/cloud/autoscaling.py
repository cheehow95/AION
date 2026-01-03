"""
AION Cloud-Native Runtime - Horizontal Autoscaling
===================================================

Horizontal autoscaling:
- Metrics Collection: CPU, memory, queue depth
- Scaling Policies: Rule-based and predictive scaling
- Scale-to-Zero: Idle agent hibernation
- Burst Handling: Rapid scale-up for spikes

Auto-generated for Phase 5: Scale
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
import statistics


class ScalingDirection(Enum):
    UP = "up"
    DOWN = "down"
    NONE = "none"


@dataclass
class ScalingPolicy:
    """Policy for autoscaling decisions."""
    name: str = ""
    metric: str = "cpu"
    target_value: float = 70.0  # Target utilization %
    min_replicas: int = 1
    max_replicas: int = 10
    scale_up_cooldown: float = 60.0  # Seconds
    scale_down_cooldown: float = 300.0
    scale_up_step: int = 2
    scale_down_step: int = 1


@dataclass
class ScalingEvent:
    """A scaling event."""
    timestamp: datetime = field(default_factory=datetime.now)
    direction: ScalingDirection = ScalingDirection.NONE
    from_replicas: int = 0
    to_replicas: int = 0
    trigger_metric: str = ""
    trigger_value: float = 0.0
    reason: str = ""


class MetricsCollector:
    """Collects metrics for autoscaling decisions."""
    
    def __init__(self, window_size: int = 60):
        self.window_size = window_size
        self.metrics: Dict[str, List[tuple]] = {}  # metric -> [(timestamp, value)]
    
    def record(self, metric: str, value: float):
        """Record a metric value."""
        if metric not in self.metrics:
            self.metrics[metric] = []
        
        now = datetime.now()
        self.metrics[metric].append((now, value))
        
        # Trim old values
        cutoff = now - timedelta(seconds=self.window_size * 2)
        self.metrics[metric] = [(t, v) for t, v in self.metrics[metric] if t > cutoff]
    
    def get_average(self, metric: str, window: int = None) -> Optional[float]:
        """Get average value over window."""
        window = window or self.window_size
        if metric not in self.metrics:
            return None
        
        cutoff = datetime.now() - timedelta(seconds=window)
        values = [v for t, v in self.metrics[metric] if t > cutoff]
        
        return statistics.mean(values) if values else None
    
    def get_trend(self, metric: str, window: int = None) -> float:
        """Get trend (positive = increasing)."""
        window = window or self.window_size
        if metric not in self.metrics:
            return 0.0
        
        cutoff = datetime.now() - timedelta(seconds=window)
        values = [v for t, v in self.metrics[metric] if t > cutoff]
        
        if len(values) < 2:
            return 0.0
        
        return (values[-1] - values[0]) / len(values)


class HorizontalAutoscaler:
    """Horizontal pod autoscaler for AION agents."""
    
    def __init__(self):
        self.policies: Dict[str, ScalingPolicy] = {}
        self.metrics = MetricsCollector()
        self.current_replicas: Dict[str, int] = {}
        self.last_scale: Dict[str, datetime] = {}
        self.events: List[ScalingEvent] = []
    
    def add_policy(self, target: str, policy: ScalingPolicy):
        """Add scaling policy for a target."""
        self.policies[target] = policy
        self.current_replicas[target] = policy.min_replicas
    
    def record_metrics(self, target: str, cpu: float = None, 
                       memory: float = None, queue_depth: int = None):
        """Record metrics for a target."""
        if cpu is not None:
            self.metrics.record(f"{target}_cpu", cpu)
        if memory is not None:
            self.metrics.record(f"{target}_memory", memory)
        if queue_depth is not None:
            self.metrics.record(f"{target}_queue", float(queue_depth))
    
    def evaluate(self, target: str) -> ScalingEvent:
        """Evaluate if scaling is needed."""
        if target not in self.policies:
            return ScalingEvent(direction=ScalingDirection.NONE)
        
        policy = self.policies[target]
        current = self.current_replicas.get(target, policy.min_replicas)
        
        # Get metric value
        metric_key = f"{target}_{policy.metric}"
        value = self.metrics.get_average(metric_key)
        
        if value is None:
            return ScalingEvent(direction=ScalingDirection.NONE)
        
        # Check cooldown
        last = self.last_scale.get(target)
        if last:
            elapsed = (datetime.now() - last).total_seconds()
            cooldown = policy.scale_up_cooldown if value > policy.target_value else policy.scale_down_cooldown
            if elapsed < cooldown:
                return ScalingEvent(direction=ScalingDirection.NONE, reason="cooldown")
        
        # Determine direction
        if value > policy.target_value * 1.2 and current < policy.max_replicas:
            new_replicas = min(current + policy.scale_up_step, policy.max_replicas)
            return ScalingEvent(
                direction=ScalingDirection.UP,
                from_replicas=current,
                to_replicas=new_replicas,
                trigger_metric=policy.metric,
                trigger_value=value,
                reason=f"{policy.metric} at {value:.1f}% > {policy.target_value}%"
            )
        elif value < policy.target_value * 0.5 and current > policy.min_replicas:
            new_replicas = max(current - policy.scale_down_step, policy.min_replicas)
            return ScalingEvent(
                direction=ScalingDirection.DOWN,
                from_replicas=current,
                to_replicas=new_replicas,
                trigger_metric=policy.metric,
                trigger_value=value,
                reason=f"{policy.metric} at {value:.1f}% < {policy.target_value * 0.5:.1f}%"
            )
        
        return ScalingEvent(direction=ScalingDirection.NONE)
    
    async def apply_scaling(self, target: str, event: ScalingEvent) -> bool:
        """Apply a scaling decision."""
        if event.direction == ScalingDirection.NONE:
            return False
        
        self.current_replicas[target] = event.to_replicas
        self.last_scale[target] = datetime.now()
        self.events.append(event)
        
        return True
    
    def get_status(self, target: str) -> Dict[str, Any]:
        """Get autoscaler status for target."""
        policy = self.policies.get(target)
        if not policy:
            return {}
        
        return {
            'current_replicas': self.current_replicas.get(target, 0),
            'min_replicas': policy.min_replicas,
            'max_replicas': policy.max_replicas,
            'target_metric': policy.metric,
            'current_value': self.metrics.get_average(f"{target}_{policy.metric}"),
            'recent_events': len([e for e in self.events[-10:] 
                                 if e.direction != ScalingDirection.NONE])
        }


class ScaleToZero:
    """Scale-to-zero capability for idle agents."""
    
    def __init__(self, idle_threshold: float = 300.0):
        self.idle_threshold = idle_threshold  # Seconds
        self.last_activity: Dict[str, datetime] = {}
        self.hibernated: Dict[str, datetime] = {}
    
    def record_activity(self, target: str):
        """Record activity on a target."""
        self.last_activity[target] = datetime.now()
        if target in self.hibernated:
            del self.hibernated[target]
    
    def check_idle(self, target: str) -> bool:
        """Check if target should be scaled to zero."""
        if target in self.hibernated:
            return True
        
        last = self.last_activity.get(target)
        if not last:
            return False
        
        idle_time = (datetime.now() - last).total_seconds()
        return idle_time > self.idle_threshold
    
    def hibernate(self, target: str):
        """Mark target as hibernated."""
        self.hibernated[target] = datetime.now()
    
    def wake(self, target: str):
        """Wake a hibernated target."""
        if target in self.hibernated:
            del self.hibernated[target]
        self.record_activity(target)
    
    def get_hibernated(self) -> List[str]:
        """Get list of hibernated targets."""
        return list(self.hibernated.keys())


async def demo_autoscaling():
    """Demonstrate autoscaling."""
    print("📈 Horizontal Autoscaling Demo")
    print("=" * 50)
    
    autoscaler = HorizontalAutoscaler()
    scale_to_zero = ScaleToZero(idle_threshold=5.0)
    
    # Add policy
    policy = ScalingPolicy(
        name="reasoning-agent",
        metric="cpu",
        target_value=70.0,
        min_replicas=1,
        max_replicas=5,
        scale_up_cooldown=1.0,
        scale_down_cooldown=2.0
    )
    autoscaler.add_policy("reasoning-agent", policy)
    
    print("\n📊 Simulating load...")
    
    # Simulate increasing load
    for cpu in [30, 50, 75, 85, 90, 60, 40, 20]:
        autoscaler.record_metrics("reasoning-agent", cpu=float(cpu))
        await asyncio.sleep(0.1)
        
        event = autoscaler.evaluate("reasoning-agent")
        if event.direction != ScalingDirection.NONE:
            await autoscaler.apply_scaling("reasoning-agent", event)
            print(f"  CPU: {cpu}% → Scale {event.direction.value}: "
                  f"{event.from_replicas} → {event.to_replicas}")
    
    print(f"\n📊 Final status: {autoscaler.get_status('reasoning-agent')}")
    
    # Scale to zero demo
    print("\n💤 Scale-to-Zero Demo:")
    scale_to_zero.record_activity("idle-agent")
    await asyncio.sleep(0.1)
    print(f"  Is idle: {scale_to_zero.check_idle('idle-agent')}")
    
    print("\n✅ Autoscaling demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_autoscaling())
