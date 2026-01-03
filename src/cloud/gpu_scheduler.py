"""
AION Cloud-Native Runtime - GPU Scheduler
==========================================

GPU resource scheduling:
- GPU Detection: Available GPU inventory
- Affinity Scheduling: GPU type preferences
- Memory Management: VRAM allocation tracking
- Multi-GPU: Distributed model parallel execution

Auto-generated for Phase 5: Scale
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from datetime import datetime
from enum import Enum


class GPUType(Enum):
    """Types of GPUs."""
    NVIDIA_A100 = "nvidia-a100"
    NVIDIA_H100 = "nvidia-h100"
    NVIDIA_A10 = "nvidia-a10"
    NVIDIA_T4 = "nvidia-t4"
    NVIDIA_V100 = "nvidia-v100"
    AMD_MI300 = "amd-mi300"
    UNKNOWN = "unknown"


@dataclass
class GPUResource:
    """A GPU resource."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    gpu_type: GPUType = GPUType.UNKNOWN
    device_index: int = 0
    node: str = ""
    total_memory_mb: int = 0
    used_memory_mb: int = 0
    compute_capability: str = ""
    allocated_to: Optional[str] = None
    last_updated: datetime = field(default_factory=datetime.now)
    
    @property
    def available_memory_mb(self) -> int:
        return self.total_memory_mb - self.used_memory_mb
    
    @property
    def utilization(self) -> float:
        if self.total_memory_mb == 0:
            return 0.0
        return self.used_memory_mb / self.total_memory_mb


@dataclass
class GPURequest:
    """A request for GPU resources."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    requester: str = ""
    memory_mb: int = 0
    gpu_count: int = 1
    preferred_types: List[GPUType] = field(default_factory=list)
    exclusive: bool = False
    priority: int = 0  # Higher = more priority


class VRAMManager:
    """Manages VRAM allocation across GPUs."""
    
    def __init__(self):
        self.allocations: Dict[str, Dict[str, int]] = {}  # gpu_id -> {alloc_id: size}
    
    def allocate(self, gpu_id: str, allocation_id: str, size_mb: int) -> bool:
        """Allocate VRAM on a GPU."""
        if gpu_id not in self.allocations:
            self.allocations[gpu_id] = {}
        
        self.allocations[gpu_id][allocation_id] = size_mb
        return True
    
    def deallocate(self, gpu_id: str, allocation_id: str) -> int:
        """Deallocate VRAM, returns freed size."""
        if gpu_id not in self.allocations:
            return 0
        
        return self.allocations[gpu_id].pop(allocation_id, 0)
    
    def get_used(self, gpu_id: str) -> int:
        """Get total used VRAM on a GPU."""
        if gpu_id not in self.allocations:
            return 0
        return sum(self.allocations[gpu_id].values())
    
    def get_allocations(self, gpu_id: str) -> Dict[str, int]:
        """Get all allocations on a GPU."""
        return dict(self.allocations.get(gpu_id, {}))


class AffinityScheduler:
    """GPU affinity-based scheduling."""
    
    def __init__(self):
        self.affinities: Dict[str, List[GPUType]] = {}  # workload -> preferred types
        self.anti_affinities: Dict[str, Set[str]] = {}  # workload -> gpus to avoid
    
    def set_affinity(self, workload: str, gpu_types: List[GPUType]):
        """Set GPU type affinity for a workload."""
        self.affinities[workload] = gpu_types
    
    def set_anti_affinity(self, workload: str, gpu_ids: Set[str]):
        """Set GPUs to avoid for a workload."""
        self.anti_affinities[workload] = gpu_ids
    
    def score_gpu(self, workload: str, gpu: GPUResource) -> float:
        """Score a GPU for a workload (higher = better)."""
        score = 1.0
        
        # Check affinity
        preferred = self.affinities.get(workload, [])
        if preferred and gpu.gpu_type in preferred:
            score += 1.0
        
        # Check anti-affinity
        avoid = self.anti_affinities.get(workload, set())
        if gpu.id in avoid:
            score -= 2.0
        
        # Prefer less utilized GPUs
        score += (1 - gpu.utilization)
        
        return score


class GPUScheduler:
    """Scheduler for GPU resources."""
    
    def __init__(self):
        self.gpus: Dict[str, GPUResource] = {}
        self.vram = VRAMManager()
        self.affinity = AffinityScheduler()
        self.pending_requests: List[GPURequest] = []
        self.fulfilled: Dict[str, List[str]] = {}  # request_id -> gpu_ids
    
    def register_gpu(self, gpu: GPUResource):
        """Register a GPU."""
        self.gpus[gpu.id] = gpu
    
    def unregister_gpu(self, gpu_id: str):
        """Unregister a GPU."""
        self.gpus.pop(gpu_id, None)
    
    def discover_gpus(self) -> List[GPUResource]:
        """Discover available GPUs (simulated)."""
        # In real implementation, would use nvidia-smi, etc.
        discovered = [
            GPUResource(gpu_type=GPUType.NVIDIA_A100, device_index=0, 
                       node="node-1", total_memory_mb=40960),
            GPUResource(gpu_type=GPUType.NVIDIA_A100, device_index=1,
                       node="node-1", total_memory_mb=40960),
            GPUResource(gpu_type=GPUType.NVIDIA_T4, device_index=0,
                       node="node-2", total_memory_mb=16384),
        ]
        
        for gpu in discovered:
            self.register_gpu(gpu)
        
        return discovered
    
    def request(self, request: GPURequest) -> Optional[List[str]]:
        """Request GPU allocation."""
        # Sort pending by priority
        self.pending_requests.append(request)
        self.pending_requests.sort(key=lambda r: r.priority, reverse=True)
        
        return self._try_fulfill(request)
    
    def _try_fulfill(self, request: GPURequest) -> Optional[List[str]]:
        """Try to fulfill a GPU request."""
        candidates = []
        
        for gpu in self.gpus.values():
            if gpu.allocated_to and request.exclusive:
                continue
            if gpu.available_memory_mb < request.memory_mb:
                continue
            
            score = self.affinity.score_gpu(request.requester, gpu)
            
            # Boost score for preferred types
            if request.preferred_types and gpu.gpu_type in request.preferred_types:
                score += 2.0
            
            candidates.append((gpu, score))
        
        # Sort by score
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        if len(candidates) < request.gpu_count:
            return None
        
        # Allocate
        allocated = []
        for gpu, _ in candidates[:request.gpu_count]:
            self.vram.allocate(gpu.id, request.id, request.memory_mb)
            gpu.used_memory_mb += request.memory_mb
            if request.exclusive:
                gpu.allocated_to = request.requester
            allocated.append(gpu.id)
        
        self.fulfilled[request.id] = allocated
        self.pending_requests = [r for r in self.pending_requests if r.id != request.id]
        
        return allocated
    
    def release(self, request_id: str):
        """Release GPU allocation."""
        if request_id not in self.fulfilled:
            return
        
        for gpu_id in self.fulfilled[request_id]:
            freed = self.vram.deallocate(gpu_id, request_id)
            if gpu_id in self.gpus:
                self.gpus[gpu_id].used_memory_mb -= freed
                self.gpus[gpu_id].allocated_to = None
        
        del self.fulfilled[request_id]
    
    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status."""
        total_memory = sum(g.total_memory_mb for g in self.gpus.values())
        used_memory = sum(g.used_memory_mb for g in self.gpus.values())
        
        by_type = {}
        for g in self.gpus.values():
            by_type[g.gpu_type.value] = by_type.get(g.gpu_type.value, 0) + 1
        
        return {
            'total_gpus': len(self.gpus),
            'by_type': by_type,
            'total_memory_gb': total_memory / 1024,
            'used_memory_gb': used_memory / 1024,
            'utilization': used_memory / total_memory if total_memory > 0 else 0,
            'pending_requests': len(self.pending_requests),
            'active_allocations': len(self.fulfilled)
        }


async def demo_gpu_scheduler():
    """Demonstrate GPU scheduler."""
    print("🎮 GPU Scheduler Demo")
    print("=" * 50)
    
    scheduler = GPUScheduler()
    
    # Discover GPUs
    print("\n🔍 Discovering GPUs...")
    gpus = scheduler.discover_gpus()
    for gpu in gpus:
        print(f"  {gpu.gpu_type.value} on {gpu.node}: {gpu.total_memory_mb}MB")
    
    # Set affinity
    scheduler.affinity.set_affinity("llm-inference", [GPUType.NVIDIA_A100, GPUType.NVIDIA_H100])
    
    # Request GPU
    print("\n📋 Requesting GPU for LLM inference...")
    request = GPURequest(
        requester="llm-inference",
        memory_mb=20000,
        gpu_count=1,
        preferred_types=[GPUType.NVIDIA_A100]
    )
    
    allocated = scheduler.request(request)
    if allocated:
        print(f"  Allocated: {allocated}")
    
    # Another request
    print("\n📋 Requesting multi-GPU for training...")
    request2 = GPURequest(
        requester="training",
        memory_mb=10000,
        gpu_count=2
    )
    
    allocated2 = scheduler.request(request2)
    if allocated2:
        print(f"  Allocated: {allocated2}")
    else:
        print("  Insufficient resources")
    
    print(f"\n📊 Scheduler Status: {scheduler.get_status()}")
    
    # Release
    scheduler.release(request.id)
    print(f"\n♻️ Released allocation: {scheduler.get_status()['active_allocations']} remaining")
    
    print("\n✅ GPU scheduler demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_gpu_scheduler())
