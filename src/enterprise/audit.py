"""
AION Compliance Audit Logging
=============================

Enterprise-grade audit logging with tamper-evident records,
compliance levels, and secure export functionality.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from enum import Enum
import uuid
import hashlib
import json


# =============================================================================
# AUDIT LEVELS
# =============================================================================

class AuditLevel(Enum):
    """Audit event severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    SECURITY = "security"
    COMPLIANCE = "compliance"
    CRITICAL = "critical"


class ActionType(Enum):
    """Types of auditable actions."""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"
    AUTHENTICATE = "authenticate"
    AUTHORIZE = "authorize"
    EXPORT = "export"
    IMPORT = "import"
    CONFIG_CHANGE = "config_change"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class AuditEvent:
    """An audit log event."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    level: AuditLevel = AuditLevel.INFO
    
    # Who
    actor: str = ""  # User/agent ID
    actor_type: str = "user"  # user, agent, system
    ip_address: str = ""
    session_id: str = ""
    
    # What
    action: ActionType = ActionType.READ
    resource_type: str = ""  # What type of resource
    resource_id: str = ""    # Specific resource
    
    # Result
    success: bool = True
    error_message: str = ""
    
    # Context
    details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Integrity
    previous_hash: str = ""
    event_hash: str = ""
    
    def __post_init__(self):
        if not self.event_hash:
            self.event_hash = self._compute_hash()
    
    def _compute_hash(self) -> str:
        """Compute event hash for integrity."""
        data = {
            "timestamp": self.timestamp.isoformat(),
            "actor": self.actor,
            "action": self.action.value,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "success": self.success,
            "previous_hash": self.previous_hash
        }
        content = json.dumps(data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:32]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "actor": self.actor,
            "actor_type": self.actor_type,
            "ip_address": self.ip_address,
            "action": self.action.value,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "success": self.success,
            "error_message": self.error_message,
            "details": self.details,
            "event_hash": self.event_hash,
            "previous_hash": self.previous_hash,
        }


@dataclass
class AuditQuery:
    """Query parameters for audit log search."""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    levels: List[AuditLevel] = field(default_factory=list)
    actors: List[str] = field(default_factory=list)
    actions: List[ActionType] = field(default_factory=list)
    resource_types: List[str] = field(default_factory=list)
    success_only: Optional[bool] = None
    limit: int = 100
    offset: int = 0


@dataclass
class AuditStats:
    """Statistics for audit logs."""
    total_events: int = 0
    by_level: Dict[str, int] = field(default_factory=dict)
    by_action: Dict[str, int] = field(default_factory=dict)
    by_actor: Dict[str, int] = field(default_factory=dict)
    success_count: int = 0
    failure_count: int = 0
    time_range_start: Optional[datetime] = None
    time_range_end: Optional[datetime] = None


# =============================================================================
# AUDIT LOGGER
# =============================================================================

class AuditLogger:
    """
    Enterprise audit logging system.
    
    Provides tamper-evident logging, compliance filtering,
    and secure export capabilities.
    """
    
    def __init__(self, storage_path: str = None):
        self.storage_path = storage_path
        self._events: List[AuditEvent] = []
        self._last_hash = ""
        self._subscribers: List[Callable[[AuditEvent], None]] = []
        self._retention_days = 90
    
    def log(
        self,
        action: ActionType,
        resource_type: str,
        resource_id: str = "",
        actor: str = "",
        level: AuditLevel = AuditLevel.INFO,
        success: bool = True,
        error_message: str = "",
        details: Dict[str, Any] = None,
        **kwargs
    ) -> AuditEvent:
        """
        Log an audit event.
        
        Args:
            action: Action type
            resource_type: Type of resource accessed
            resource_id: Specific resource ID
            actor: User/agent performing action
            level: Audit level
            success: Whether action succeeded
            error_message: Error details if failed
            details: Additional context
            **kwargs: Extra metadata
            
        Returns:
            Created AuditEvent
        """
        event = AuditEvent(
            level=level,
            actor=actor,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            success=success,
            error_message=error_message,
            details=details or {},
            metadata=kwargs,
            previous_hash=self._last_hash
        )
        
        self._events.append(event)
        self._last_hash = event.event_hash
        
        # Notify subscribers
        for subscriber in self._subscribers:
            try:
                subscriber(event)
            except Exception:
                pass
        
        return event
    
    def log_security(
        self,
        action: ActionType,
        resource_type: str,
        actor: str = "",
        details: Dict[str, Any] = None,
        **kwargs
    ) -> AuditEvent:
        """Log a security event."""
        return self.log(
            action=action,
            resource_type=resource_type,
            actor=actor,
            level=AuditLevel.SECURITY,
            details=details,
            **kwargs
        )
    
    def log_compliance(
        self,
        action: ActionType,
        resource_type: str,
        resource_id: str = "",
        actor: str = "",
        details: Dict[str, Any] = None,
        **kwargs
    ) -> AuditEvent:
        """Log a compliance event."""
        return self.log(
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            actor=actor,
            level=AuditLevel.COMPLIANCE,
            details=details,
            **kwargs
        )
    
    def log_access(
        self,
        resource_type: str,
        resource_id: str = "",
        actor: str = "",
        access_granted: bool = True,
        reason: str = "",
        **kwargs
    ) -> AuditEvent:
        """Log an access control event."""
        return self.log(
            action=ActionType.AUTHORIZE,
            resource_type=resource_type,
            resource_id=resource_id,
            actor=actor,
            level=AuditLevel.SECURITY,
            success=access_granted,
            error_message="" if access_granted else reason,
            details={"access_granted": access_granted, "reason": reason},
            **kwargs
        )
    
    def query(self, query: AuditQuery) -> List[AuditEvent]:
        """
        Query audit logs.
        
        Args:
            query: Query parameters
            
        Returns:
            Matching events
        """
        results = []
        
        for event in self._events:
            # Time filters
            if query.start_time and event.timestamp < query.start_time:
                continue
            if query.end_time and event.timestamp > query.end_time:
                continue
            
            # Level filter
            if query.levels and event.level not in query.levels:
                continue
            
            # Actor filter
            if query.actors and event.actor not in query.actors:
                continue
            
            # Action filter
            if query.actions and event.action not in query.actions:
                continue
            
            # Resource type filter
            if query.resource_types and event.resource_type not in query.resource_types:
                continue
            
            # Success filter
            if query.success_only is not None and event.success != query.success_only:
                continue
            
            results.append(event)
        
        # Apply pagination
        return results[query.offset:query.offset + query.limit]
    
    def get_stats(
        self,
        start_time: datetime = None,
        end_time: datetime = None
    ) -> AuditStats:
        """Get audit statistics."""
        events = self._events
        
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]
        
        stats = AuditStats(total_events=len(events))
        
        for event in events:
            # By level
            level_key = event.level.value
            stats.by_level[level_key] = stats.by_level.get(level_key, 0) + 1
            
            # By action
            action_key = event.action.value
            stats.by_action[action_key] = stats.by_action.get(action_key, 0) + 1
            
            # By actor
            if event.actor:
                stats.by_actor[event.actor] = stats.by_actor.get(event.actor, 0) + 1
            
            # Success/failure
            if event.success:
                stats.success_count += 1
            else:
                stats.failure_count += 1
        
        if events:
            stats.time_range_start = min(e.timestamp for e in events)
            stats.time_range_end = max(e.timestamp for e in events)
        
        return stats
    
    def verify_integrity(self) -> Dict[str, Any]:
        """
        Verify integrity of the audit log.
        
        Returns:
            Verification result
        """
        if not self._events:
            return {"valid": True, "events_checked": 0}
        
        valid = True
        invalid_events = []
        
        expected_prev = ""
        for event in self._events:
            # Check chain
            if event.previous_hash != expected_prev:
                valid = False
                invalid_events.append(event.id)
            
            # Verify hash
            recomputed = event._compute_hash()
            if recomputed != event.event_hash:
                valid = False
                invalid_events.append(event.id)
            
            expected_prev = event.event_hash
        
        return {
            "valid": valid,
            "events_checked": len(self._events),
            "invalid_events": invalid_events
        }
    
    def export(
        self,
        format: str = "json",
        query: AuditQuery = None
    ) -> str:
        """
        Export audit logs.
        
        Args:
            format: Export format (json, csv)
            query: Optional query filter
            
        Returns:
            Exported data as string
        """
        events = self.query(query) if query else self._events
        
        if format == "json":
            return json.dumps([e.to_dict() for e in events], indent=2)
        
        elif format == "csv":
            lines = ["timestamp,level,actor,action,resource_type,resource_id,success"]
            for e in events:
                lines.append(
                    f"{e.timestamp.isoformat()},{e.level.value},{e.actor},"
                    f"{e.action.value},{e.resource_type},{e.resource_id},{e.success}"
                )
            return "\n".join(lines)
        
        return ""
    
    def subscribe(self, callback: Callable[[AuditEvent], None]) -> None:
        """Subscribe to new audit events."""
        self._subscribers.append(callback)
    
    def unsubscribe(self, callback: Callable[[AuditEvent], None]) -> None:
        """Unsubscribe from audit events."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)
    
    def cleanup(self, before: datetime) -> int:
        """
        Remove events before a date.
        
        Args:
            before: Remove events before this time
            
        Returns:
            Number of events removed
        """
        original_count = len(self._events)
        self._events = [e for e in self._events if e.timestamp >= before]
        return original_count - len(self._events)


# =============================================================================
# DEMO
# =============================================================================

def demo_audit():
    """Demonstrate audit logging."""
    print("📋 Audit Logging Demo")
    print("-" * 40)
    
    logger = AuditLogger()
    
    # Log various events
    logger.log(
        action=ActionType.AUTHENTICATE,
        resource_type="session",
        actor="user@example.com",
        level=AuditLevel.INFO,
        details={"method": "password"}
    )
    
    logger.log(
        action=ActionType.READ,
        resource_type="document",
        resource_id="doc-123",
        actor="user@example.com"
    )
    
    logger.log_security(
        action=ActionType.AUTHENTICATE,
        resource_type="session",
        actor="unknown",
        success=False,
        error_message="Invalid credentials"
    )
    
    logger.log_compliance(
        action=ActionType.EXPORT,
        resource_type="user_data",
        resource_id="export-456",
        actor="admin@example.com",
        details={"records": 1000}
    )
    
    print(f"Logged {len(logger._events)} events")
    
    # Query
    query = AuditQuery(levels=[AuditLevel.SECURITY])
    security_events = logger.query(query)
    print(f"Security events: {len(security_events)}")
    
    # Stats
    stats = logger.get_stats()
    print(f"Stats: {stats.success_count} success, {stats.failure_count} failures")
    print(f"By level: {stats.by_level}")
    
    # Verify integrity
    integrity = logger.verify_integrity()
    print(f"Integrity check: {'VALID' if integrity['valid'] else 'INVALID'}")
    
    # Export
    exported = logger.export("json")
    print(f"Exported {len(exported)} bytes")
    
    print("-" * 40)
    print("✅ Audit demo complete!")


if __name__ == "__main__":
    demo_audit()
