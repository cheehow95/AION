"""
AION Pulse Task Automation - Triggers
======================================

Event-based task triggers:
- Time-based triggers
- Event triggers (webhooks)
- Condition-based triggers
- Trigger combinations

Enables reactive task automation.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, time
from enum import Enum


class TriggerType(Enum):
    """Types of triggers."""
    TIME = "time"
    EVENT = "event"
    CONDITION = "condition"
    COMPOSITE = "composite"


@dataclass
class TriggerResult:
    """Result of trigger evaluation."""
    triggered: bool = False
    trigger_id: str = ""
    reason: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class Trigger(ABC):
    """Base class for triggers."""
    
    def __init__(self, trigger_id: str = "", name: str = ""):
        self.trigger_id = trigger_id or f"trigger_{id(self)}"
        self.name = name
        self.enabled = True
        self.fire_count = 0
        self.last_fired: Optional[datetime] = None
    
    @abstractmethod
    async def evaluate(self, context: Dict[str, Any] = None) -> TriggerResult:
        """Evaluate if trigger should fire."""
        pass
    
    def _record_fire(self, result: TriggerResult):
        """Record trigger firing."""
        if result.triggered:
            self.fire_count += 1
            self.last_fired = datetime.now()


class TimeTrigger(Trigger):
    """Time-based trigger."""
    
    def __init__(self, trigger_id: str = "", name: str = "",
                 at_time: time = None,
                 days_of_week: List[int] = None,
                 interval_seconds: int = None):
        super().__init__(trigger_id, name)
        self.at_time = at_time
        self.days_of_week = days_of_week or [0, 1, 2, 3, 4, 5, 6]  # All days
        self.interval_seconds = interval_seconds
        self._last_interval_check: Optional[datetime] = None
    
    async def evaluate(self, context: Dict[str, Any] = None) -> TriggerResult:
        """Check if time condition is met."""
        if not self.enabled:
            return TriggerResult(triggered=False, trigger_id=self.trigger_id)
        
        now = datetime.now()
        triggered = False
        reason = ""
        
        # Check specific time
        if self.at_time:
            current_time = now.time()
            time_diff = abs(
                (current_time.hour * 60 + current_time.minute) -
                (self.at_time.hour * 60 + self.at_time.minute)
            )
            day_matches = now.weekday() in self.days_of_week
            
            if time_diff <= 1 and day_matches:  # Within 1 minute
                triggered = True
                reason = f"Time reached: {self.at_time}"
        
        # Check interval
        if self.interval_seconds:
            if self._last_interval_check is None:
                self._last_interval_check = now
            
            elapsed = (now - self._last_interval_check).total_seconds()
            if elapsed >= self.interval_seconds:
                triggered = True
                reason = f"Interval elapsed: {self.interval_seconds}s"
                self._last_interval_check = now
        
        result = TriggerResult(
            triggered=triggered,
            trigger_id=self.trigger_id,
            reason=reason
        )
        self._record_fire(result)
        return result


class EventTrigger(Trigger):
    """Event-based trigger (webhooks, signals)."""
    
    def __init__(self, trigger_id: str = "", name: str = "",
                 event_type: str = "",
                 filter_func: Callable = None):
        super().__init__(trigger_id, name)
        self.event_type = event_type
        self.filter_func = filter_func
        self.pending_events: List[Dict[str, Any]] = []
    
    def receive_event(self, event: Dict[str, Any]):
        """Receive an event."""
        if event.get('type') == self.event_type or not self.event_type:
            if self.filter_func:
                if self.filter_func(event):
                    self.pending_events.append(event)
            else:
                self.pending_events.append(event)
    
    async def evaluate(self, context: Dict[str, Any] = None) -> TriggerResult:
        """Check for pending events."""
        if not self.enabled or not self.pending_events:
            return TriggerResult(triggered=False, trigger_id=self.trigger_id)
        
        event = self.pending_events.pop(0)
        
        result = TriggerResult(
            triggered=True,
            trigger_id=self.trigger_id,
            reason=f"Event received: {event.get('type', 'unknown')}",
            data=event
        )
        self._record_fire(result)
        return result


class ConditionTrigger(Trigger):
    """Condition-based trigger."""
    
    def __init__(self, trigger_id: str = "", name: str = "",
                 condition_func: Callable = None,
                 check_value_key: str = None,
                 operator: str = "==",
                 threshold: Any = None):
        super().__init__(trigger_id, name)
        self.condition_func = condition_func
        self.check_value_key = check_value_key
        self.operator = operator
        self.threshold = threshold
    
    async def evaluate(self, context: Dict[str, Any] = None) -> TriggerResult:
        """Evaluate condition."""
        if not self.enabled:
            return TriggerResult(triggered=False, trigger_id=self.trigger_id)
        
        context = context or {}
        triggered = False
        reason = ""
        
        # Custom function
        if self.condition_func:
            try:
                triggered = self.condition_func(context)
                reason = "Custom condition met"
            except Exception as e:
                reason = f"Condition error: {e}"
        
        # Value comparison
        elif self.check_value_key and self.threshold is not None:
            value = context.get(self.check_value_key)
            if value is not None:
                operators = {
                    '==': lambda a, b: a == b,
                    '!=': lambda a, b: a != b,
                    '>': lambda a, b: a > b,
                    '>=': lambda a, b: a >= b,
                    '<': lambda a, b: a < b,
                    '<=': lambda a, b: a <= b,
                }
                op_func = operators.get(self.operator, operators['=='])
                triggered = op_func(value, self.threshold)
                reason = f"{self.check_value_key} {self.operator} {self.threshold}"
        
        result = TriggerResult(
            triggered=triggered,
            trigger_id=self.trigger_id,
            reason=reason,
            data=context
        )
        self._record_fire(result)
        return result


class TriggerManager:
    """Manages multiple triggers."""
    
    def __init__(self):
        self.triggers: Dict[str, Trigger] = {}
        self.trigger_actions: Dict[str, Callable] = {}
        self.history: List[TriggerResult] = []
    
    def register(self, trigger: Trigger, action: Callable = None):
        """Register a trigger with optional action."""
        self.triggers[trigger.trigger_id] = trigger
        if action:
            self.trigger_actions[trigger.trigger_id] = action
    
    def unregister(self, trigger_id: str):
        """Unregister a trigger."""
        if trigger_id in self.triggers:
            del self.triggers[trigger_id]
        if trigger_id in self.trigger_actions:
            del self.trigger_actions[trigger_id]
    
    async def check_all(self, context: Dict[str, Any] = None) -> List[TriggerResult]:
        """Check all triggers."""
        results = []
        
        for trigger in self.triggers.values():
            result = await trigger.evaluate(context)
            if result.triggered:
                results.append(result)
                self.history.append(result)
                
                # Execute action if registered
                action = self.trigger_actions.get(trigger.trigger_id)
                if action:
                    if asyncio.iscoroutinefunction(action):
                        await action(result)
                    else:
                        action(result)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get trigger statistics."""
        return {
            'total_triggers': len(self.triggers),
            'enabled': sum(1 for t in self.triggers.values() if t.enabled),
            'total_fires': len(self.history),
            'by_trigger': {
                tid: t.fire_count for tid, t in self.triggers.items()
            }
        }


async def demo_triggers():
    """Demonstrate trigger system."""
    print("🎯 Pulse Trigger System Demo")
    print("=" * 50)
    
    manager = TriggerManager()
    
    # Time trigger (interval-based for demo)
    time_trigger = TimeTrigger(
        trigger_id="interval_check",
        name="Interval Trigger",
        interval_seconds=1
    )
    
    # Event trigger
    event_trigger = EventTrigger(
        trigger_id="new_message",
        name="Message Trigger",
        event_type="message"
    )
    
    # Condition trigger
    condition_trigger = ConditionTrigger(
        trigger_id="high_cpu",
        name="CPU Alert",
        check_value_key="cpu_usage",
        operator=">",
        threshold=80
    )
    
    # Register with actions
    manager.register(time_trigger, lambda r: print(f"   ⏰ {r.reason}"))
    manager.register(event_trigger, lambda r: print(f"   📨 {r.reason}"))
    manager.register(condition_trigger, lambda r: print(f"   ⚠️ {r.reason}"))
    
    print(f"\n📊 Registered Triggers: {len(manager.triggers)}")
    
    # Simulate events
    event_trigger.receive_event({'type': 'message', 'content': 'Hello!'})
    
    # Check triggers
    print("\n🔎 Checking triggers...")
    
    # First check
    results = await manager.check_all({'cpu_usage': 50})
    print(f"   Triggered: {len(results)}")
    
    await asyncio.sleep(1.1)  # Wait for interval
    
    # Second check with high CPU
    results = await manager.check_all({'cpu_usage': 95})
    print(f"   Triggered: {len(results)}")
    
    print(f"\n📊 Stats: {manager.get_stats()}")
    print("\n✅ Trigger demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_triggers())
