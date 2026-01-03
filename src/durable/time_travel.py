"""
AION Durable Execution - Time-Travel Debugging
===============================================

Time-travel debugging capabilities:
- Event Sourcing: All state changes as events
- History Replay: Step-by-step execution replay
- Breakpoint Injection: Retroactive debugging points
- State Diffing: Visual state comparison

Auto-generated for Phase 4: Autonomy
"""

import asyncio
import uuid
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Set
from datetime import datetime
from enum import Enum
from copy import deepcopy


class EventType(Enum):
    """Types of state events."""
    STATE_CHANGE = "state_change"
    ACTION_START = "action_start"
    ACTION_END = "action_end"
    ERROR = "error"
    CHECKPOINT = "checkpoint"
    SIGNAL = "signal"


@dataclass
class StateEvent:
    """An event that modifies or observes state."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: EventType = EventType.STATE_CHANGE
    timestamp: datetime = field(default_factory=datetime.now)
    sequence: int = 0
    key: str = ""
    old_value: Any = None
    new_value: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'type': self.type.value,
            'timestamp': self.timestamp.isoformat(),
            'sequence': self.sequence,
            'key': self.key,
            'old_value': self.old_value,
            'new_value': self.new_value,
            'metadata': self.metadata
        }


class EventStore:
    """Store for event sourcing."""
    
    def __init__(self):
        self.events: List[StateEvent] = []
        self.sequence = 0
        self.snapshots: Dict[int, Dict[str, Any]] = {}
        self.snapshot_interval = 100
    
    def append(self, event: StateEvent) -> int:
        """Append an event to the store."""
        self.sequence += 1
        event.sequence = self.sequence
        self.events.append(event)
        
        # Create snapshot periodically
        if self.sequence % self.snapshot_interval == 0:
            self.snapshots[self.sequence] = self.reconstruct_state(self.sequence)
        
        return self.sequence
    
    def record_change(self, key: str, old_value: Any, new_value: Any,
                      metadata: Dict[str, Any] = None) -> int:
        """Record a state change event."""
        event = StateEvent(
            type=EventType.STATE_CHANGE,
            key=key,
            old_value=deepcopy(old_value),
            new_value=deepcopy(new_value),
            metadata=metadata or {}
        )
        return self.append(event)
    
    def record_action(self, action_name: str, started: bool,
                      result: Any = None, error: str = None) -> int:
        """Record action start/end."""
        event = StateEvent(
            type=EventType.ACTION_START if started else EventType.ACTION_END,
            key=action_name,
            new_value=result,
            metadata={'error': error} if error else {}
        )
        return self.append(event)
    
    def get_events(self, from_seq: int = 0, to_seq: int = None,
                   event_type: EventType = None) -> List[StateEvent]:
        """Get events in a range."""
        to_seq = to_seq or self.sequence
        result = []
        for e in self.events:
            if e.sequence < from_seq:
                continue
            if e.sequence > to_seq:
                break
            if event_type and e.type != event_type:
                continue
            result.append(e)
        return result
    
    def reconstruct_state(self, at_sequence: int) -> Dict[str, Any]:
        """Reconstruct state at a specific sequence number."""
        # Find nearest snapshot
        snapshot_seq = 0
        state = {}
        
        for seq in sorted(self.snapshots.keys()):
            if seq <= at_sequence:
                snapshot_seq = seq
                state = deepcopy(self.snapshots[seq])
        
        # Apply events after snapshot
        for event in self.events:
            if event.sequence <= snapshot_seq:
                continue
            if event.sequence > at_sequence:
                break
            if event.type == EventType.STATE_CHANGE:
                state[event.key] = deepcopy(event.new_value)
        
        return state
    
    def get_key_history(self, key: str) -> List[StateEvent]:
        """Get all events for a specific key."""
        return [e for e in self.events if e.key == key]


@dataclass
class Breakpoint:
    """A debugging breakpoint."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    condition: Optional[Callable[[Dict[str, Any]], bool]] = None
    key_watch: Optional[str] = None
    sequence_at: Optional[int] = None
    hit_count: int = 0
    enabled: bool = True


class TimeTravelDebugger:
    """Time-travel debugger for workflow execution."""
    
    def __init__(self, event_store: EventStore = None):
        self.store = event_store or EventStore()
        self.current_position: int = 0
        self.breakpoints: Dict[str, Breakpoint] = {}
        self.watch_keys: Set[str] = set()
        self.callbacks: List[Callable] = []
    
    def step_forward(self, steps: int = 1) -> Dict[str, Any]:
        """Step forward through history."""
        target = min(self.current_position + steps, self.store.sequence)
        self.current_position = target
        return self._get_current_state()
    
    def step_backward(self, steps: int = 1) -> Dict[str, Any]:
        """Step backward through history."""
        target = max(self.current_position - steps, 0)
        self.current_position = target
        return self._get_current_state()
    
    def goto(self, sequence: int) -> Dict[str, Any]:
        """Jump to a specific sequence number."""
        self.current_position = max(0, min(sequence, self.store.sequence))
        return self._get_current_state()
    
    def _get_current_state(self) -> Dict[str, Any]:
        """Get state at current position."""
        return self.store.reconstruct_state(self.current_position)
    
    def add_breakpoint(self, condition: Callable = None, 
                       key: str = None, at_sequence: int = None) -> str:
        """Add a breakpoint."""
        bp = Breakpoint(
            condition=condition,
            key_watch=key,
            sequence_at=at_sequence
        )
        self.breakpoints[bp.id] = bp
        return bp.id
    
    def remove_breakpoint(self, bp_id: str):
        """Remove a breakpoint."""
        self.breakpoints.pop(bp_id, None)
    
    def check_breakpoints(self, state: Dict[str, Any], event: StateEvent) -> List[str]:
        """Check if any breakpoints are hit."""
        hits = []
        for bp_id, bp in self.breakpoints.items():
            if not bp.enabled:
                continue
            
            hit = False
            
            if bp.sequence_at and event.sequence == bp.sequence_at:
                hit = True
            elif bp.key_watch and event.key == bp.key_watch:
                hit = True
            elif bp.condition:
                try:
                    if bp.condition(state):
                        hit = True
                except Exception:
                    pass
            
            if hit:
                bp.hit_count += 1
                hits.append(bp_id)
        
        return hits
    
    def watch(self, key: str):
        """Add a key to watch list."""
        self.watch_keys.add(key)
    
    def unwatch(self, key: str):
        """Remove a key from watch list."""
        self.watch_keys.discard(key)
    
    def diff_states(self, seq1: int, seq2: int) -> Dict[str, Any]:
        """Compare states at two sequence points."""
        state1 = self.store.reconstruct_state(seq1)
        state2 = self.store.reconstruct_state(seq2)
        
        all_keys = set(state1.keys()) | set(state2.keys())
        diff = {
            'added': {},
            'removed': {},
            'modified': {}
        }
        
        for key in all_keys:
            if key not in state1:
                diff['added'][key] = state2[key]
            elif key not in state2:
                diff['removed'][key] = state1[key]
            elif state1[key] != state2[key]:
                diff['modified'][key] = {'old': state1[key], 'new': state2[key]}
        
        return diff
    
    def find_event(self, key: str = None, value: Any = None,
                   from_seq: int = 0) -> Optional[StateEvent]:
        """Find first event matching criteria."""
        for event in self.store.events:
            if event.sequence < from_seq:
                continue
            if key and event.key != key:
                continue
            if value is not None and event.new_value != value:
                continue
            return event
        return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get debugger status."""
        return {
            'current_position': self.current_position,
            'total_events': self.store.sequence,
            'breakpoints': len(self.breakpoints),
            'watched_keys': list(self.watch_keys),
            'current_state': self._get_current_state()
        }


class ReplaySession:
    """Session for replaying workflow execution."""
    
    def __init__(self, event_store: EventStore):
        self.store = event_store
        self.debugger = TimeTravelDebugger(event_store)
        self.replay_speed: float = 1.0
        self.paused: bool = False
        self.listeners: List[Callable[[StateEvent, Dict], None]] = []
    
    def add_listener(self, callback: Callable[[StateEvent, Dict], None]):
        """Add replay event listener."""
        self.listeners.append(callback)
    
    def _notify(self, event: StateEvent, state: Dict[str, Any]):
        """Notify listeners of event."""
        for listener in self.listeners:
            try:
                listener(event, state)
            except Exception:
                pass
    
    async def replay(self, from_seq: int = 0, to_seq: int = None,
                     realtime: bool = False):
        """Replay events in sequence."""
        to_seq = to_seq or self.store.sequence
        events = self.store.get_events(from_seq, to_seq)
        
        prev_time = None
        
        for event in events:
            while self.paused:
                await asyncio.sleep(0.1)
            
            # Check breakpoints
            state = self.store.reconstruct_state(event.sequence)
            hits = self.debugger.check_breakpoints(state, event)
            if hits:
                self.paused = True
                self._notify(event, state)
                continue
            
            # Simulate timing
            if realtime and prev_time:
                delay = (event.timestamp - prev_time).total_seconds()
                await asyncio.sleep(delay / self.replay_speed)
            
            self.debugger.current_position = event.sequence
            self._notify(event, state)
            
            prev_time = event.timestamp
    
    def pause(self):
        """Pause replay."""
        self.paused = True
    
    def resume(self):
        """Resume replay."""
        self.paused = False
    
    def set_speed(self, speed: float):
        """Set replay speed multiplier."""
        self.replay_speed = max(0.1, min(10.0, speed))


async def demo_time_travel():
    """Demonstrate time-travel debugging."""
    print("⏰ Time-Travel Debugging Demo")
    print("=" * 50)
    
    store = EventStore()
    debugger = TimeTravelDebugger(store)
    
    # Simulate workflow execution with events
    print("\n📝 Recording execution events...")
    
    store.record_change("status", None, "started")
    store.record_action("validate", started=True)
    store.record_change("order_id", None, "ORD-123")
    store.record_action("validate", started=False, result="valid")
    store.record_change("status", "started", "validating")
    store.record_action("process", started=True)
    store.record_change("payment_id", None, "PAY-456")
    store.record_action("process", started=False, result="success")
    store.record_change("status", "validating", "processed")
    store.record_change("shipped", False, True)
    store.record_change("status", "processed", "completed")
    
    print(f"  Total events: {store.sequence}")
    
    # Time travel operations
    print("\n🔍 Time-travel debugging:")
    
    # Go to middle
    state = debugger.goto(5)
    print(f"  At sequence 5: status={state.get('status')}, order_id={state.get('order_id')}")
    
    # Step forward
    state = debugger.step_forward(3)
    print(f"  After 3 steps: status={state.get('status')}, payment_id={state.get('payment_id')}")
    
    # Compare states
    diff = debugger.diff_states(1, store.sequence)
    print(f"\n  State diff (1 → {store.sequence}):")
    print(f"    Added: {list(diff['added'].keys())}")
    print(f"    Modified: {list(diff['modified'].keys())}")
    
    # Key history
    history = store.get_key_history("status")
    print(f"\n  'status' history: {[e.new_value for e in history]}")
    
    # Breakpoint
    bp_id = debugger.add_breakpoint(key="payment_id")
    print(f"\n  Added breakpoint on 'payment_id': {bp_id}")
    
    print(f"\n📊 Debugger status: {debugger.get_status()}")
    print("\n✅ Time-travel demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_time_travel())
