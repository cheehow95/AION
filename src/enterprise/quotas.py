"""
AION Usage Quota Management
===========================

Quota policies, usage tracking, and enforcement
for resource management and cost control.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
import uuid


# =============================================================================
# QUOTA TYPES
# =============================================================================

class QuotaType(Enum):
    """Types of quotas that can be tracked."""
    TOKENS = "tokens"
    REQUESTS = "requests"
    COST = "cost"
    STORAGE = "storage"
    API_CALLS = "api_calls"
    COMPUTE_MINUTES = "compute_minutes"
    BANDWIDTH = "bandwidth"
    AGENTS = "agents"
    EXECUTIONS = "executions"
    CUSTOM = "custom"


class ResetPeriod(Enum):
    """Period for quota reset."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"
    NEVER = "never"


class EnforcementAction(Enum):
    """Actions when quota is exceeded."""
    ALLOW = "allow"         # Log but allow
    WARN = "warn"           # Warn but allow
    THROTTLE = "throttle"   # Rate limit
    BLOCK = "block"         # Block request


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class QuotaPolicy:
    """A quota policy definition."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    quota_type: QuotaType = QuotaType.REQUESTS
    
    # Limits
    limit: float = 1000.0
    soft_limit: Optional[float] = None  # Warning threshold
    burst_limit: Optional[float] = None  # Short-term burst allowance
    
    # Reset
    reset_period: ResetPeriod = ResetPeriod.DAILY
    
    # Enforcement
    enforcement: EnforcementAction = EnforcementAction.WARN
    
    # Scope
    scope: str = "global"  # "global", "user", "agent", "project"
    
    # Metadata
    description: str = ""
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    
    @property
    def has_soft_limit(self) -> bool:
        return self.soft_limit is not None and self.soft_limit < self.limit


@dataclass
class QuotaUsage:
    """Current usage for a quota."""
    policy_id: str
    entity_id: str  # User/agent/project ID
    
    # Current usage
    current: float = 0.0
    
    # History
    period_start: datetime = field(default_factory=datetime.now)
    period_end: Optional[datetime] = None
    
    # Peak
    peak_usage: float = 0.0
    peak_time: Optional[datetime] = None
    
    # Stats
    total_consumed: float = 0.0
    request_count: int = 0
    
    @property
    def remaining(self) -> float:
        """Calculate remaining quota (requires policy context)."""
        return max(0, self.total_consumed - self.current)


@dataclass
class QuotaAlert:
    """An alert triggered by quota usage."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    policy_id: str = ""
    entity_id: str = ""
    alert_type: str = ""  # "approaching", "exceeded", "reset"
    threshold: float = 0.0
    current_usage: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    message: str = ""
    acknowledged: bool = False


@dataclass
class QuotaCheckResult:
    """Result of a quota check."""
    allowed: bool
    policy_id: str
    current_usage: float
    limit: float
    remaining: float
    enforcement: EnforcementAction
    alerts: List[QuotaAlert] = field(default_factory=list)
    message: str = ""
    
    @property
    def percentage_used(self) -> float:
        if self.limit == 0:
            return 0.0
        return (self.current_usage / self.limit) * 100


# =============================================================================
# QUOTA MANAGER
# =============================================================================

class QuotaManager:
    """
    Manage quotas, track usage, and enforce limits.
    """
    
    def __init__(self):
        self._policies: Dict[str, QuotaPolicy] = {}
        self._usage: Dict[str, Dict[str, QuotaUsage]] = {}  # policy_id -> entity_id -> usage
        self._alerts: List[QuotaAlert] = []
        self._alert_callbacks: List[Callable[[QuotaAlert], None]] = []
    
    def create_policy(
        self,
        name: str,
        quota_type: QuotaType,
        limit: float,
        soft_limit: float = None,
        reset_period: ResetPeriod = ResetPeriod.DAILY,
        enforcement: EnforcementAction = EnforcementAction.WARN,
        scope: str = "global",
        **kwargs
    ) -> QuotaPolicy:
        """
        Create a new quota policy.
        
        Args:
            name: Policy name
            quota_type: Type of quota
            limit: Maximum allowed
            soft_limit: Warning threshold
            reset_period: When to reset
            enforcement: Action when exceeded
            scope: Scope of policy
            
        Returns:
            Created QuotaPolicy
        """
        policy = QuotaPolicy(
            name=name,
            quota_type=quota_type,
            limit=limit,
            soft_limit=soft_limit or limit * 0.8,
            reset_period=reset_period,
            enforcement=enforcement,
            scope=scope,
            **kwargs
        )
        
        self._policies[policy.id] = policy
        self._usage[policy.id] = {}
        
        return policy
    
    def get_policy(self, policy_id: str) -> Optional[QuotaPolicy]:
        """Get a policy by ID."""
        return self._policies.get(policy_id)
    
    def get_policy_by_name(self, name: str) -> Optional[QuotaPolicy]:
        """Get a policy by name."""
        for policy in self._policies.values():
            if policy.name == name:
                return policy
        return None
    
    def list_policies(self) -> List[QuotaPolicy]:
        """List all policies."""
        return list(self._policies.values())
    
    def check_quota(
        self,
        policy_id: str,
        entity_id: str,
        amount: float = 1.0
    ) -> QuotaCheckResult:
        """
        Check if a quota allows an operation.
        
        Args:
            policy_id: Policy to check
            entity_id: Entity (user/agent) ID
            amount: Amount to check
            
        Returns:
            QuotaCheckResult with decision
        """
        policy = self._policies.get(policy_id)
        if not policy:
            return QuotaCheckResult(
                allowed=True,
                policy_id=policy_id,
                current_usage=0,
                limit=0,
                remaining=0,
                enforcement=EnforcementAction.ALLOW,
                message="Policy not found"
            )
        
        if not policy.enabled:
            return QuotaCheckResult(
                allowed=True,
                policy_id=policy_id,
                current_usage=0,
                limit=policy.limit,
                remaining=policy.limit,
                enforcement=EnforcementAction.ALLOW,
                message="Policy disabled"
            )
        
        # Get or create usage tracking
        usage = self._get_or_create_usage(policy_id, entity_id)
        
        # Check for period reset
        self._check_period_reset(policy, usage)
        
        projected = usage.current + amount
        alerts = []
        
        # Check soft limit
        if policy.has_soft_limit and projected >= policy.soft_limit and usage.current < policy.soft_limit:
            alert = self._create_alert(policy, entity_id, "approaching", policy.soft_limit, projected)
            alerts.append(alert)
        
        # Check limit
        if projected > policy.limit:
            alert = self._create_alert(policy, entity_id, "exceeded", policy.limit, projected)
            alerts.append(alert)
            
            allowed = policy.enforcement in [EnforcementAction.ALLOW, EnforcementAction.WARN]
            return QuotaCheckResult(
                allowed=allowed,
                policy_id=policy_id,
                current_usage=usage.current,
                limit=policy.limit,
                remaining=max(0, policy.limit - usage.current),
                enforcement=policy.enforcement,
                alerts=alerts,
                message=f"Quota exceeded: {usage.current:.0f}/{policy.limit:.0f}"
            )
        
        return QuotaCheckResult(
            allowed=True,
            policy_id=policy_id,
            current_usage=usage.current,
            limit=policy.limit,
            remaining=policy.limit - usage.current,
            enforcement=policy.enforcement,
            alerts=alerts,
            message="OK"
        )
    
    def consume(
        self,
        policy_id: str,
        entity_id: str,
        amount: float = 1.0
    ) -> QuotaCheckResult:
        """
        Consume quota (check and record usage).
        
        Args:
            policy_id: Policy to consume from
            entity_id: Entity ID
            amount: Amount to consume
            
        Returns:
            QuotaCheckResult with updated usage
        """
        result = self.check_quota(policy_id, entity_id, amount)
        
        if result.allowed:
            usage = self._get_or_create_usage(policy_id, entity_id)
            usage.current += amount
            usage.total_consumed += amount
            usage.request_count += 1
            
            # Update peak
            if usage.current > usage.peak_usage:
                usage.peak_usage = usage.current
                usage.peak_time = datetime.now()
            
            # Update result
            result.current_usage = usage.current
            result.remaining = max(0, result.limit - usage.current)
        
        return result
    
    def get_usage(self, policy_id: str, entity_id: str) -> Optional[QuotaUsage]:
        """Get current usage for an entity."""
        if policy_id not in self._usage:
            return None
        return self._usage[policy_id].get(entity_id)
    
    def reset_usage(self, policy_id: str, entity_id: str = None) -> bool:
        """
        Reset usage for a policy.
        
        Args:
            policy_id: Policy ID
            entity_id: Specific entity (None = all)
            
        Returns:
            True if reset successful
        """
        if policy_id not in self._usage:
            return False
        
        if entity_id:
            if entity_id in self._usage[policy_id]:
                usage = self._usage[policy_id][entity_id]
                usage.current = 0
                usage.period_start = datetime.now()
                return True
            return False
        else:
            for usage in self._usage[policy_id].values():
                usage.current = 0
                usage.period_start = datetime.now()
            return True
    
    def set_alert_threshold(
        self,
        policy_id: str,
        threshold_percent: float,
        callback: Callable[[QuotaAlert], None] = None
    ) -> None:
        """Set custom alert threshold."""
        policy = self._policies.get(policy_id)
        if policy:
            policy.soft_limit = policy.limit * (threshold_percent / 100)
        
        if callback:
            self._alert_callbacks.append(callback)
    
    def get_alerts(
        self,
        policy_id: str = None,
        entity_id: str = None,
        unacknowledged_only: bool = False
    ) -> List[QuotaAlert]:
        """Get alerts."""
        results = self._alerts
        
        if policy_id:
            results = [a for a in results if a.policy_id == policy_id]
        if entity_id:
            results = [a for a in results if a.entity_id == entity_id]
        if unacknowledged_only:
            results = [a for a in results if not a.acknowledged]
        
        return results
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self._alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                return True
        return False
    
    def get_summary(self, entity_id: str = None) -> Dict[str, Any]:
        """Get usage summary."""
        summary = {
            "total_policies": len(self._policies),
            "active_policies": len([p for p in self._policies.values() if p.enabled]),
            "total_alerts": len(self._alerts),
            "unacknowledged_alerts": len([a for a in self._alerts if not a.acknowledged]),
            "usage": {}
        }
        
        for policy_id, policy in self._policies.items():
            usage_data = {
                "name": policy.name,
                "type": policy.quota_type.value,
                "limit": policy.limit,
                "entities": {}
            }
            
            if policy_id in self._usage:
                for eid, usage in self._usage[policy_id].items():
                    if entity_id is None or eid == entity_id:
                        usage_data["entities"][eid] = {
                            "current": usage.current,
                            "remaining": policy.limit - usage.current,
                            "percentage": (usage.current / policy.limit * 100) if policy.limit > 0 else 0
                        }
            
            summary["usage"][policy_id] = usage_data
        
        return summary
    
    # -------------------------------------------------------------------------
    # Internal methods
    # -------------------------------------------------------------------------
    
    def _get_or_create_usage(self, policy_id: str, entity_id: str) -> QuotaUsage:
        """Get or create usage tracking."""
        if policy_id not in self._usage:
            self._usage[policy_id] = {}
        
        if entity_id not in self._usage[policy_id]:
            self._usage[policy_id][entity_id] = QuotaUsage(
                policy_id=policy_id,
                entity_id=entity_id
            )
        
        return self._usage[policy_id][entity_id]
    
    def _check_period_reset(self, policy: QuotaPolicy, usage: QuotaUsage) -> None:
        """Check if usage period should reset."""
        now = datetime.now()
        should_reset = False
        
        if policy.reset_period == ResetPeriod.HOURLY:
            should_reset = (now - usage.period_start) >= timedelta(hours=1)
        elif policy.reset_period == ResetPeriod.DAILY:
            should_reset = (now - usage.period_start) >= timedelta(days=1)
        elif policy.reset_period == ResetPeriod.WEEKLY:
            should_reset = (now - usage.period_start) >= timedelta(weeks=1)
        elif policy.reset_period == ResetPeriod.MONTHLY:
            should_reset = (now - usage.period_start) >= timedelta(days=30)
        
        if should_reset:
            usage.current = 0
            usage.period_start = now
            
            # Create reset alert
            self._create_alert(policy, usage.entity_id, "reset", 0, 0)
    
    def _create_alert(
        self,
        policy: QuotaPolicy,
        entity_id: str,
        alert_type: str,
        threshold: float,
        current: float
    ) -> QuotaAlert:
        """Create and store an alert."""
        messages = {
            "approaching": f"Approaching quota limit ({current:.0f}/{policy.limit:.0f})",
            "exceeded": f"Quota exceeded ({current:.0f}/{policy.limit:.0f})",
            "reset": "Quota period reset"
        }
        
        alert = QuotaAlert(
            policy_id=policy.id,
            entity_id=entity_id,
            alert_type=alert_type,
            threshold=threshold,
            current_usage=current,
            message=messages.get(alert_type, alert_type)
        )
        
        self._alerts.append(alert)
        
        # Notify callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception:
                pass
        
        return alert


# =============================================================================
# DEMO
# =============================================================================

def demo_quotas():
    """Demonstrate quota management."""
    print("📊 Quota Management Demo")
    print("-" * 40)
    
    manager = QuotaManager()
    
    # Create policies
    tokens_policy = manager.create_policy(
        name="daily_tokens",
        quota_type=QuotaType.TOKENS,
        limit=10000,
        soft_limit=8000,
        reset_period=ResetPeriod.DAILY,
        enforcement=EnforcementAction.WARN
    )
    print(f"Created policy: {tokens_policy.name} ({tokens_policy.limit} limit)")
    
    requests_policy = manager.create_policy(
        name="hourly_requests",
        quota_type=QuotaType.REQUESTS,
        limit=100,
        reset_period=ResetPeriod.HOURLY,
        enforcement=EnforcementAction.BLOCK
    )
    print(f"Created policy: {requests_policy.name} ({requests_policy.limit} limit)")
    
    # Simulate usage
    user_id = "user_123"
    
    # Consume some quota
    for i in range(5):
        result = manager.consume(tokens_policy.id, user_id, 1500)
        print(f"  Consumed 1500 tokens: {result.current_usage}/{result.limit}")
    
    # Check quota
    result = manager.check_quota(tokens_policy.id, user_id, 2000)
    print(f"\nCheck 2000 more: allowed={result.allowed}, remaining={result.remaining}")
    
    # Try to exceed
    result = manager.consume(tokens_policy.id, user_id, 3000)
    print(f"Consume 3000: allowed={result.allowed} ({result.message})")
    
    # Check alerts
    alerts = manager.get_alerts(unacknowledged_only=True)
    print(f"\nAlerts: {len(alerts)}")
    for alert in alerts[:3]:
        print(f"  - {alert.alert_type}: {alert.message}")
    
    # Get summary
    summary = manager.get_summary(user_id)
    print(f"\nSummary: {summary['total_policies']} policies, {summary['unacknowledged_alerts']} alerts")
    
    print("-" * 40)
    print("✅ Quota demo complete!")


if __name__ == "__main__":
    demo_quotas()
