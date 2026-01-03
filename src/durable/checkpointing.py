"""
AION Durable Execution - Automatic Checkpointing
=================================================

State checkpointing for fault tolerance:
- State Snapshots: Full agent state serialization
- Incremental Checkpoints: Delta-based state updates
- Checkpoint Storage: Configurable persistence backends
- Recovery Points: Automatic recovery from latest checkpoint

Auto-generated for Phase 4: Autonomy
"""

import asyncio
import uuid
import json
import hashlib
import pickle
import gzip
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from datetime import datetime
from pathlib import Path
from enum import Enum


class CheckpointType(Enum):
    """Types of checkpoints."""
    FULL = "full"
    INCREMENTAL = "incremental"
    SNAPSHOT = "snapshot"


@dataclass
class Checkpoint:
    """A state checkpoint."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: CheckpointType = CheckpointType.FULL
    workflow_id: str = ""
    sequence: int = 0
    state: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    size_bytes: int = 0
    checksum: str = ""
    
    def compute_checksum(self) -> str:
        """Compute checksum of state."""
        data = json.dumps(self.state, sort_keys=True, default=str)
        self.checksum = hashlib.sha256(data.encode()).hexdigest()[:16]
        return self.checksum
    
    def serialize(self) -> bytes:
        """Serialize checkpoint to bytes."""
        data = {
            'id': self.id,
            'type': self.type.value,
            'workflow_id': self.workflow_id,
            'sequence': self.sequence,
            'state': self.state,
            'parent_id': self.parent_id,
            'created_at': self.created_at.isoformat(),
            'checksum': self.checksum
        }
        raw = json.dumps(data).encode()
        self.size_bytes = len(raw)
        return gzip.compress(raw)
    
    @classmethod
    def deserialize(cls, data: bytes) -> 'Checkpoint':
        """Deserialize checkpoint from bytes."""
        raw = gzip.decompress(data)
        d = json.loads(raw)
        cp = cls(
            id=d['id'],
            type=CheckpointType(d['type']),
            workflow_id=d['workflow_id'],
            sequence=d['sequence'],
            state=d['state'],
            parent_id=d.get('parent_id'),
            created_at=datetime.fromisoformat(d['created_at']),
            checksum=d.get('checksum', '')
        )
        cp.size_bytes = len(raw)
        return cp


@dataclass
class IncrementalCheckpoint:
    """Incremental checkpoint storing only deltas."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: str = ""
    workflow_id: str = ""
    sequence: int = 0
    added: Dict[str, Any] = field(default_factory=dict)
    modified: Dict[str, Any] = field(default_factory=dict)
    deleted: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.now)
    
    def apply_to(self, base_state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply delta to base state."""
        result = dict(base_state)
        for key in self.deleted:
            result.pop(key, None)
        result.update(self.added)
        result.update(self.modified)
        return result
    
    @classmethod
    def compute_delta(cls, old_state: Dict[str, Any], 
                      new_state: Dict[str, Any],
                      parent_id: str = "") -> 'IncrementalCheckpoint':
        """Compute delta between states."""
        added = {}
        modified = {}
        deleted = set()
        
        old_keys = set(old_state.keys())
        new_keys = set(new_state.keys())
        
        # Deleted keys
        deleted = old_keys - new_keys
        
        # Added keys
        for key in new_keys - old_keys:
            added[key] = new_state[key]
        
        # Modified keys
        for key in old_keys & new_keys:
            if old_state[key] != new_state[key]:
                modified[key] = new_state[key]
        
        return cls(
            parent_id=parent_id,
            added=added,
            modified=modified,
            deleted=deleted
        )


class CheckpointStorage(ABC):
    """Abstract storage backend for checkpoints."""
    
    @abstractmethod
    async def save(self, checkpoint: Checkpoint) -> bool:
        pass
    
    @abstractmethod
    async def load(self, checkpoint_id: str) -> Optional[Checkpoint]:
        pass
    
    @abstractmethod
    async def list_checkpoints(self, workflow_id: str) -> List[str]:
        pass
    
    @abstractmethod
    async def delete(self, checkpoint_id: str) -> bool:
        pass


class MemoryCheckpointStorage(CheckpointStorage):
    """In-memory checkpoint storage."""
    
    def __init__(self):
        self.checkpoints: Dict[str, bytes] = {}
        self.workflow_index: Dict[str, List[str]] = {}
    
    async def save(self, checkpoint: Checkpoint) -> bool:
        data = checkpoint.serialize()
        self.checkpoints[checkpoint.id] = data
        
        if checkpoint.workflow_id not in self.workflow_index:
            self.workflow_index[checkpoint.workflow_id] = []
        self.workflow_index[checkpoint.workflow_id].append(checkpoint.id)
        
        return True
    
    async def load(self, checkpoint_id: str) -> Optional[Checkpoint]:
        data = self.checkpoints.get(checkpoint_id)
        if data:
            return Checkpoint.deserialize(data)
        return None
    
    async def list_checkpoints(self, workflow_id: str) -> List[str]:
        return self.workflow_index.get(workflow_id, [])
    
    async def delete(self, checkpoint_id: str) -> bool:
        if checkpoint_id in self.checkpoints:
            del self.checkpoints[checkpoint_id]
            return True
        return False


class FileCheckpointStorage(CheckpointStorage):
    """File-based checkpoint storage."""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def _get_path(self, checkpoint_id: str) -> Path:
        return self.base_path / f"{checkpoint_id}.ckpt"
    
    async def save(self, checkpoint: Checkpoint) -> bool:
        data = checkpoint.serialize()
        path = self._get_path(checkpoint.id)
        path.write_bytes(data)
        return True
    
    async def load(self, checkpoint_id: str) -> Optional[Checkpoint]:
        path = self._get_path(checkpoint_id)
        if path.exists():
            data = path.read_bytes()
            return Checkpoint.deserialize(data)
        return None
    
    async def list_checkpoints(self, workflow_id: str) -> List[str]:
        checkpoints = []
        for path in self.base_path.glob("*.ckpt"):
            try:
                cp = await self.load(path.stem)
                if cp and cp.workflow_id == workflow_id:
                    checkpoints.append(cp.id)
            except Exception:
                pass
        return checkpoints
    
    async def delete(self, checkpoint_id: str) -> bool:
        path = self._get_path(checkpoint_id)
        if path.exists():
            path.unlink()
            return True
        return False


class CheckpointManager:
    """Manages checkpoint creation and recovery."""
    
    def __init__(self, storage: CheckpointStorage = None,
                 checkpoint_interval: int = 10,
                 max_checkpoints: int = 100):
        self.storage = storage or MemoryCheckpointStorage()
        self.checkpoint_interval = checkpoint_interval
        self.max_checkpoints = max_checkpoints
        
        # State tracking
        self.current_state: Dict[str, Dict[str, Any]] = {}  # workflow_id -> state
        self.sequences: Dict[str, int] = {}  # workflow_id -> sequence
        self.last_checkpoint: Dict[str, str] = {}  # workflow_id -> checkpoint_id
    
    async def create_checkpoint(self, workflow_id: str, 
                                state: Dict[str, Any],
                                checkpoint_type: CheckpointType = None) -> Checkpoint:
        """Create a new checkpoint."""
        sequence = self.sequences.get(workflow_id, 0) + 1
        self.sequences[workflow_id] = sequence
        
        # Determine checkpoint type
        if checkpoint_type is None:
            if sequence % self.checkpoint_interval == 0:
                checkpoint_type = CheckpointType.FULL
            else:
                checkpoint_type = CheckpointType.INCREMENTAL
        
        parent_id = self.last_checkpoint.get(workflow_id)
        
        checkpoint = Checkpoint(
            type=checkpoint_type,
            workflow_id=workflow_id,
            sequence=sequence,
            state=state,
            parent_id=parent_id
        )
        checkpoint.compute_checksum()
        
        await self.storage.save(checkpoint)
        
        self.current_state[workflow_id] = state
        self.last_checkpoint[workflow_id] = checkpoint.id
        
        # Cleanup old checkpoints
        await self._cleanup(workflow_id)
        
        return checkpoint
    
    async def _cleanup(self, workflow_id: str):
        """Remove old checkpoints beyond limit."""
        checkpoint_ids = await self.storage.list_checkpoints(workflow_id)
        if len(checkpoint_ids) > self.max_checkpoints:
            to_delete = len(checkpoint_ids) - self.max_checkpoints
            for cp_id in checkpoint_ids[:to_delete]:
                await self.storage.delete(cp_id)
    
    async def recover(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Recover latest state for a workflow."""
        checkpoint_ids = await self.storage.list_checkpoints(workflow_id)
        if not checkpoint_ids:
            return None
        
        # Load latest checkpoint
        latest = await self.storage.load(checkpoint_ids[-1])
        if not latest:
            return None
        
        # If full checkpoint, return directly
        if latest.type == CheckpointType.FULL:
            self.current_state[workflow_id] = latest.state
            self.last_checkpoint[workflow_id] = latest.id
            return latest.state
        
        # For incremental, need to rebuild
        chain = [latest]
        current_id = latest.parent_id
        
        while current_id:
            cp = await self.storage.load(current_id)
            if not cp:
                break
            chain.append(cp)
            if cp.type == CheckpointType.FULL:
                break
            current_id = cp.parent_id
        
        # Rebuild state
        chain.reverse()
        state = chain[0].state if chain else {}
        
        for cp in chain[1:]:
            if isinstance(cp.state, dict):
                state.update(cp.state)
        
        self.current_state[workflow_id] = state
        self.last_checkpoint[workflow_id] = latest.id
        return state
    
    async def get_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Get a specific checkpoint."""
        return await self.storage.load(checkpoint_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get checkpointing statistics."""
        return {
            'tracked_workflows': len(self.current_state),
            'total_sequences': sum(self.sequences.values()),
            'last_checkpoints': len(self.last_checkpoint)
        }


async def demo_checkpointing():
    """Demonstrate checkpointing system."""
    print("💾 Checkpointing System Demo")
    print("=" * 50)
    
    manager = CheckpointManager(checkpoint_interval=5)
    
    # Simulate workflow execution with checkpoints
    workflow_id = "test-workflow-1"
    state = {"step": 0, "data": []}
    
    print("\n📝 Creating checkpoints...")
    for i in range(1, 11):
        state["step"] = i
        state["data"].append(f"item_{i}")
        
        cp = await manager.create_checkpoint(workflow_id, dict(state))
        print(f"  Checkpoint {i}: type={cp.type.value}, size={cp.size_bytes}b")
    
    print("\n🔄 Simulating recovery...")
    recovered = await manager.recover(workflow_id)
    print(f"  Recovered state: step={recovered['step']}, items={len(recovered['data'])}")
    
    print(f"\n📊 Statistics: {manager.get_statistics()}")
    print("\n✅ Checkpointing demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_checkpointing())
