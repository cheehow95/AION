"""
AION Cloud-Native Runtime - Multi-Region Deployment
====================================================

Multi-region deployment:
- Region Discovery: Available deployment regions
- Data Residency: Compliance-aware placement
- Failover: Automatic regional failover
- Latency Optimization: Request routing to closest region

Auto-generated for Phase 5: Scale
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from datetime import datetime
from enum import Enum
import random


class RegionStatus(Enum):
    """Status of a region."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"
    UNKNOWN = "unknown"


class ComplianceRegime(Enum):
    """Data compliance regimes."""
    GDPR = "gdpr"
    HIPAA = "hipaa"
    SOC2 = "soc2"
    PCI_DSS = "pci_dss"
    NONE = "none"


@dataclass
class Region:
    """A deployment region."""
    id: str = ""
    name: str = ""
    cloud_provider: str = ""
    location: str = ""
    coordinates: tuple = (0.0, 0.0)  # (lat, lon)
    status: RegionStatus = RegionStatus.UNKNOWN
    capacity: int = 100
    current_load: int = 0
    compliance: Set[ComplianceRegime] = field(default_factory=set)
    latency_ms: Dict[str, float] = field(default_factory=dict)  # region_id -> latency
    last_health_check: datetime = field(default_factory=datetime.now)
    
    @property
    def available_capacity(self) -> int:
        return max(0, self.capacity - self.current_load)
    
    @property
    def utilization(self) -> float:
        return self.current_load / self.capacity if self.capacity > 0 else 0


@dataclass
class RegionDeployment:
    """A deployment in a region."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    region_id: str = ""
    resource_type: str = ""
    resource_name: str = ""
    replicas: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class RegionManager:
    """Manages multi-region deployments."""
    
    def __init__(self):
        self.regions: Dict[str, Region] = {}
        self.deployments: Dict[str, RegionDeployment] = {}
        self.primary_region: Optional[str] = None
    
    def register_region(self, region: Region):
        """Register a region."""
        self.regions[region.id] = region
        if self.primary_region is None:
            self.primary_region = region.id
    
    def get_regions_by_compliance(self, required: Set[ComplianceRegime]) -> List[Region]:
        """Get regions that meet compliance requirements."""
        return [r for r in self.regions.values() if required <= r.compliance]
    
    def get_healthy_regions(self) -> List[Region]:
        """Get all healthy regions."""
        return [r for r in self.regions.values() 
                if r.status in [RegionStatus.HEALTHY, RegionStatus.DEGRADED]]
    
    def select_region(self, compliance: Set[ComplianceRegime] = None,
                      preferred: List[str] = None) -> Optional[Region]:
        """Select best region for deployment."""
        candidates = self.get_healthy_regions()
        
        if compliance:
            candidates = [r for r in candidates if compliance <= r.compliance]
        
        if not candidates:
            return None
        
        # Prefer specified regions
        if preferred:
            for pref in preferred:
                for r in candidates:
                    if r.id == pref:
                        return r
        
        # Select by available capacity
        return max(candidates, key=lambda r: r.available_capacity)
    
    def deploy(self, resource_type: str, resource_name: str,
               regions: List[str] = None, replicas_per_region: int = 1,
               compliance: Set[ComplianceRegime] = None) -> List[RegionDeployment]:
        """Deploy to multiple regions."""
        deployments = []
        
        target_regions = []
        if regions:
            target_regions = [self.regions[r] for r in regions if r in self.regions]
        else:
            # Auto-select regions
            candidates = self.get_healthy_regions()
            if compliance:
                candidates = [r for r in candidates if compliance <= r.compliance]
            target_regions = candidates[:3]  # Deploy to up to 3 regions
        
        for region in target_regions:
            deployment = RegionDeployment(
                region_id=region.id,
                resource_type=resource_type,
                resource_name=resource_name,
                replicas=replicas_per_region
            )
            self.deployments[deployment.id] = deployment
            region.current_load += replicas_per_region
            deployments.append(deployment)
        
        return deployments
    
    def get_status(self) -> Dict[str, Any]:
        """Get multi-region status."""
        healthy = sum(1 for r in self.regions.values() if r.status == RegionStatus.HEALTHY)
        return {
            'total_regions': len(self.regions),
            'healthy_regions': healthy,
            'total_deployments': len(self.deployments),
            'primary_region': self.primary_region
        }


class FailoverController:
    """Controls regional failover."""
    
    def __init__(self, region_manager: RegionManager):
        self.manager = region_manager
        self.failover_history: List[Dict[str, Any]] = []
        self.failover_targets: Dict[str, str] = {}  # region -> failover_region
    
    def set_failover_target(self, region_id: str, target_id: str):
        """Set failover target for a region."""
        self.failover_targets[region_id] = target_id
    
    def auto_configure_failover(self):
        """Auto-configure failover based on latency."""
        for region in self.manager.regions.values():
            best_target = None
            best_latency = float('inf')
            
            for other_id, latency in region.latency_ms.items():
                if other_id == region.id:
                    continue
                other = self.manager.regions.get(other_id)
                if other and other.status == RegionStatus.HEALTHY:
                    if latency < best_latency:
                        best_latency = latency
                        best_target = other_id
            
            if best_target:
                self.failover_targets[region.id] = best_target
    
    async def trigger_failover(self, failed_region_id: str) -> Optional[str]:
        """Trigger failover from a failed region."""
        if failed_region_id not in self.manager.regions:
            return None
        
        failed = self.manager.regions[failed_region_id]
        failed.status = RegionStatus.UNHEALTHY
        
        # Find target
        target_id = self.failover_targets.get(failed_region_id)
        if not target_id:
            # Auto-select
            healthy = self.manager.get_healthy_regions()
            if healthy:
                target_id = healthy[0].id
        
        if not target_id:
            return None
        
        target = self.manager.regions.get(target_id)
        if not target or target.status == RegionStatus.UNHEALTHY:
            return None
        
        # Migrate deployments
        for deployment in list(self.manager.deployments.values()):
            if deployment.region_id == failed_region_id:
                deployment.region_id = target_id
                target.current_load += deployment.replicas
        
        # Update primary if needed
        if self.manager.primary_region == failed_region_id:
            self.manager.primary_region = target_id
        
        self.failover_history.append({
            'from_region': failed_region_id,
            'to_region': target_id,
            'timestamp': datetime.now().isoformat(),
            'deployments_migrated': sum(1 for d in self.manager.deployments.values() 
                                        if d.region_id == target_id)
        })
        
        return target_id


class LatencyOptimizer:
    """Optimizes request routing based on latency."""
    
    def __init__(self, region_manager: RegionManager):
        self.manager = region_manager
        self.client_latencies: Dict[str, Dict[str, float]] = {}  # client -> {region -> latency}
    
    def record_latency(self, client_id: str, region_id: str, latency_ms: float):
        """Record latency measurement."""
        if client_id not in self.client_latencies:
            self.client_latencies[client_id] = {}
        self.client_latencies[client_id][region_id] = latency_ms
    
    def get_optimal_region(self, client_id: str, 
                           required_compliance: Set[ComplianceRegime] = None) -> Optional[str]:
        """Get optimal region for a client."""
        latencies = self.client_latencies.get(client_id, {})
        
        candidates = []
        for region_id, latency in latencies.items():
            region = self.manager.regions.get(region_id)
            if not region or region.status == RegionStatus.UNHEALTHY:
                continue
            if required_compliance and not (required_compliance <= region.compliance):
                continue
            candidates.append((region_id, latency))
        
        if not candidates:
            # Fallback: use any healthy region
            healthy = self.manager.get_healthy_regions()
            if healthy:
                return healthy[0].id
            return None
        
        # Return lowest latency
        return min(candidates, key=lambda x: x[1])[0]
    
    def estimate_latency(self, client_coords: tuple, region_id: str) -> float:
        """Estimate latency based on coordinates."""
        region = self.manager.regions.get(region_id)
        if not region:
            return float('inf')
        
        # Simple distance-based estimate
        import math
        lat1, lon1 = client_coords
        lat2, lon2 = region.coordinates
        dist = math.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)
        
        # Rough estimate: 10ms per degree
        return dist * 10


async def demo_multi_region():
    """Demonstrate multi-region deployment."""
    print("🌍 Multi-Region Deployment Demo")
    print("=" * 50)
    
    manager = RegionManager()
    
    # Register regions
    regions = [
        Region(id="us-east-1", name="US East", cloud_provider="aws",
               location="Virginia", coordinates=(37.5, -77.5),
               status=RegionStatus.HEALTHY, capacity=100,
               compliance={ComplianceRegime.SOC2, ComplianceRegime.HIPAA}),
        Region(id="eu-west-1", name="EU West", cloud_provider="aws",
               location="Ireland", coordinates=(53.3, -6.3),
               status=RegionStatus.HEALTHY, capacity=80,
               compliance={ComplianceRegime.GDPR, ComplianceRegime.SOC2}),
        Region(id="ap-east-1", name="Asia Pacific", cloud_provider="aws",
               location="Hong Kong", coordinates=(22.3, 114.2),
               status=RegionStatus.HEALTHY, capacity=60,
               compliance={ComplianceRegime.SOC2}),
    ]
    
    for region in regions:
        manager.register_region(region)
    
    # Set up inter-region latencies
    regions[0].latency_ms = {"eu-west-1": 80, "ap-east-1": 200}
    regions[1].latency_ms = {"us-east-1": 80, "ap-east-1": 150}
    regions[2].latency_ms = {"us-east-1": 200, "eu-west-1": 150}
    
    print(f"\n📍 Registered {len(regions)} regions")
    
    # Deploy with compliance
    print("\n📦 Deploying with GDPR compliance...")
    deployments = manager.deploy(
        resource_type="AIOnAgent",
        resource_name="gdpr-agent",
        compliance={ComplianceRegime.GDPR}
    )
    print(f"  Deployed to: {[d.region_id for d in deployments]}")
    
    # Setup failover
    failover = FailoverController(manager)
    failover.auto_configure_failover()
    print(f"\n🔄 Failover targets configured")
    
    # Simulate failover
    print("\n⚠️ Simulating region failure...")
    target = await failover.trigger_failover("eu-west-1")
    print(f"  Failed over to: {target}")
    
    # Latency optimization
    optimizer = LatencyOptimizer(manager)
    optimizer.record_latency("client-1", "us-east-1", 20)
    optimizer.record_latency("client-1", "ap-east-1", 180)
    
    optimal = optimizer.get_optimal_region("client-1")
    print(f"\n🚀 Optimal region for client-1: {optimal}")
    
    print(f"\n📊 Status: {manager.get_status()}")
    print("\n✅ Multi-region demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_multi_region())
