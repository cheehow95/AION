"""
AION MCP Security Manager
=========================

Provides security features for MCP communications:
- Authentication and authorization
- Rate limiting
- Access control
- Credential management
"""

import hashlib
import secrets
import time
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class Permission(Enum):
    """MCP operation permissions."""
    TOOLS_LIST = "tools:list"
    TOOLS_CALL = "tools:call"
    RESOURCES_LIST = "resources:list"
    RESOURCES_READ = "resources:read"
    RESOURCES_SUBSCRIBE = "resources:subscribe"
    PROMPTS_LIST = "prompts:list"
    PROMPTS_GET = "prompts:get"
    ADMIN = "admin:*"


@dataclass
class Credential:
    """Stored credential for an MCP server."""
    server_uri: str
    credential_type: str  # "api_key", "oauth", "certificate"
    value: str
    expires_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at


@dataclass
class AccessPolicy:
    """Access control policy."""
    name: str
    allowed_tools: Set[str] = field(default_factory=set)  # Empty = all allowed
    denied_tools: Set[str] = field(default_factory=set)
    allowed_resources: Set[str] = field(default_factory=set)
    denied_resources: Set[str] = field(default_factory=set)
    permissions: Set[Permission] = field(default_factory=set)
    rate_limit: Optional[int] = None  # requests per minute
    
    def can_access_tool(self, tool_name: str) -> bool:
        if tool_name in self.denied_tools:
            return False
        if not self.allowed_tools:  # Empty = all allowed
            return True
        return tool_name in self.allowed_tools
    
    def can_access_resource(self, uri: str) -> bool:
        if uri in self.denied_resources:
            return False
        if not self.allowed_resources:
            return True
        return uri in self.allowed_resources


@dataclass
class RateLimitEntry:
    """Tracks rate limit state for a client."""
    client_id: str
    requests: List[datetime] = field(default_factory=list)
    blocked_until: Optional[datetime] = None


class MCPSecurityManager:
    """
    Manages security for MCP communications.
    
    Features:
    - Credential management
    - Access control policies
    - Rate limiting
    - Request validation
    """
    
    def __init__(self):
        # Stored credentials
        self.credentials: Dict[str, Credential] = {}
        
        # Access policies
        self.policies: Dict[str, AccessPolicy] = {}
        self.client_policies: Dict[str, str] = {}  # client_id -> policy_name
        
        # Rate limiting
        self.rate_limits: Dict[str, RateLimitEntry] = {}
        self.default_rate_limit = 100  # requests per minute
        
        # Audit log
        self.audit_log: List[Dict[str, Any]] = []
        self.max_audit_entries = 10000
        
        # Create default policies
        self._create_default_policies()
    
    def _create_default_policies(self):
        """Create default access policies."""
        # Full access policy
        self.policies["full_access"] = AccessPolicy(
            name="full_access",
            permissions={p for p in Permission}
        )
        
        # Read-only policy
        self.policies["read_only"] = AccessPolicy(
            name="read_only",
            permissions={
                Permission.TOOLS_LIST,
                Permission.RESOURCES_LIST,
                Permission.RESOURCES_READ,
                Permission.PROMPTS_LIST,
                Permission.PROMPTS_GET
            }
        )
        
        # Tools only policy
        self.policies["tools_only"] = AccessPolicy(
            name="tools_only",
            permissions={
                Permission.TOOLS_LIST,
                Permission.TOOLS_CALL
            }
        )
    
    # ============ Credential Management ============
    
    def store_credential(
        self,
        server_uri: str,
        credential_type: str,
        value: str,
        expires_in: Optional[timedelta] = None
    ) -> str:
        """
        Store a credential for an MCP server.
        
        Args:
            server_uri: URI of the MCP server
            credential_type: Type of credential
            value: The credential value (will be hashed for storage)
            expires_in: Optional expiration time
        
        Returns:
            Credential ID
        """
        expires_at = None
        if expires_in:
            expires_at = datetime.now() + expires_in
        
        # Hash sensitive credentials
        hashed_value = hashlib.sha256(value.encode()).hexdigest()
        
        credential_id = f"{server_uri}:{secrets.token_hex(8)}"
        self.credentials[credential_id] = Credential(
            server_uri=server_uri,
            credential_type=credential_type,
            value=hashed_value,
            expires_at=expires_at
        )
        
        self._audit("credential_stored", {
            "credential_id": credential_id,
            "server_uri": server_uri,
            "type": credential_type
        })
        
        return credential_id
    
    def get_credential(self, server_uri: str) -> Optional[Credential]:
        """Get the credential for a server."""
        for cred in self.credentials.values():
            if cred.server_uri == server_uri and not cred.is_expired():
                return cred
        return None
    
    def revoke_credential(self, credential_id: str):
        """Revoke a credential."""
        if credential_id in self.credentials:
            del self.credentials[credential_id]
            self._audit("credential_revoked", {"credential_id": credential_id})
    
    def rotate_credentials(self):
        """Remove expired credentials."""
        expired = [
            cid for cid, cred in self.credentials.items()
            if cred.is_expired()
        ]
        for cid in expired:
            del self.credentials[cid]
        
        if expired:
            self._audit("credentials_rotated", {"count": len(expired)})
    
    # ============ Access Control ============
    
    def create_policy(
        self,
        name: str,
        permissions: Set[Permission] = None,
        allowed_tools: Set[str] = None,
        denied_tools: Set[str] = None,
        rate_limit: int = None
    ) -> AccessPolicy:
        """Create a new access policy."""
        policy = AccessPolicy(
            name=name,
            permissions=permissions or set(),
            allowed_tools=allowed_tools or set(),
            denied_tools=denied_tools or set(),
            rate_limit=rate_limit
        )
        self.policies[name] = policy
        
        self._audit("policy_created", {"name": name})
        return policy
    
    def assign_policy(self, client_id: str, policy_name: str):
        """Assign a policy to a client."""
        if policy_name not in self.policies:
            raise ValueError(f"Policy not found: {policy_name}")
        
        self.client_policies[client_id] = policy_name
        self._audit("policy_assigned", {
            "client_id": client_id,
            "policy": policy_name
        })
    
    def get_policy(self, client_id: str) -> AccessPolicy:
        """Get the policy for a client."""
        policy_name = self.client_policies.get(client_id, "read_only")
        return self.policies.get(policy_name, self.policies["read_only"])
    
    def check_permission(
        self,
        client_id: str,
        permission: Permission
    ) -> bool:
        """Check if a client has a permission."""
        policy = self.get_policy(client_id)
        
        # Admin has all permissions
        if Permission.ADMIN in policy.permissions:
            return True
        
        return permission in policy.permissions
    
    def check_tool_access(
        self,
        client_id: str,
        tool_name: str
    ) -> bool:
        """Check if a client can access a tool."""
        policy = self.get_policy(client_id)
        
        # Must have TOOLS_CALL permission
        if not self.check_permission(client_id, Permission.TOOLS_CALL):
            return False
        
        return policy.can_access_tool(tool_name)
    
    def check_resource_access(
        self,
        client_id: str,
        resource_uri: str
    ) -> bool:
        """Check if a client can access a resource."""
        policy = self.get_policy(client_id)
        
        # Must have RESOURCES_READ permission
        if not self.check_permission(client_id, Permission.RESOURCES_READ):
            return False
        
        return policy.can_access_resource(resource_uri)
    
    # ============ Rate Limiting ============
    
    def check_rate_limit(self, client_id: str) -> tuple[bool, Optional[int]]:
        """
        Check if a client is within rate limits.
        
        Returns:
            (allowed, retry_after_seconds)
        """
        now = datetime.now()
        
        # Get or create rate limit entry
        if client_id not in self.rate_limits:
            self.rate_limits[client_id] = RateLimitEntry(client_id=client_id)
        
        entry = self.rate_limits[client_id]
        
        # Check if blocked
        if entry.blocked_until and now < entry.blocked_until:
            retry_after = (entry.blocked_until - now).seconds
            return False, retry_after
        
        # Clean old requests
        one_minute_ago = now - timedelta(minutes=1)
        entry.requests = [r for r in entry.requests if r > one_minute_ago]
        
        # Get rate limit for this client
        policy = self.get_policy(client_id)
        limit = policy.rate_limit or self.default_rate_limit
        
        # Check limit
        if len(entry.requests) >= limit:
            # Block for increasing duration based on violations
            block_duration = min(60, len(entry.requests) - limit + 1) * 10
            entry.blocked_until = now + timedelta(seconds=block_duration)
            
            self._audit("rate_limit_exceeded", {
                "client_id": client_id,
                "requests": len(entry.requests),
                "limit": limit
            })
            
            return False, block_duration
        
        # Record request
        entry.requests.append(now)
        return True, None
    
    # ============ Request Validation ============
    
    def validate_request(
        self,
        client_id: str,
        method: str,
        params: Dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """
        Validate an MCP request.
        
        Returns:
            (valid, error_message)
        """
        # Check rate limit
        allowed, retry_after = self.check_rate_limit(client_id)
        if not allowed:
            return False, f"Rate limit exceeded. Retry after {retry_after} seconds."
        
        # Check permission based on method
        permission_map = {
            "tools/list": Permission.TOOLS_LIST,
            "tools/call": Permission.TOOLS_CALL,
            "resources/list": Permission.RESOURCES_LIST,
            "resources/read": Permission.RESOURCES_READ,
            "resources/subscribe": Permission.RESOURCES_SUBSCRIBE,
            "prompts/list": Permission.PROMPTS_LIST,
            "prompts/get": Permission.PROMPTS_GET
        }
        
        required_permission = permission_map.get(method)
        if required_permission and not self.check_permission(client_id, required_permission):
            self._audit("permission_denied", {
                "client_id": client_id,
                "method": method
            })
            return False, f"Permission denied: {required_permission.value}"
        
        # Additional checks for specific methods
        if method == "tools/call":
            tool_name = params.get("name", "")
            if not self.check_tool_access(client_id, tool_name):
                return False, f"Access denied to tool: {tool_name}"
        
        if method == "resources/read":
            uri = params.get("uri", "")
            if not self.check_resource_access(client_id, uri):
                return False, f"Access denied to resource: {uri}"
        
        return True, None
    
    # ============ Audit Logging ============
    
    def _audit(self, event_type: str, details: Dict[str, Any]):
        """Record an audit event."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            **details
        }
        
        self.audit_log.append(entry)
        
        # Trim log if too large
        if len(self.audit_log) > self.max_audit_entries:
            self.audit_log = self.audit_log[-self.max_audit_entries:]
    
    def get_audit_log(
        self,
        event_type: str = None,
        client_id: str = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get audit log entries."""
        entries = self.audit_log
        
        if event_type:
            entries = [e for e in entries if e.get("event_type") == event_type]
        
        if client_id:
            entries = [e for e in entries if e.get("client_id") == client_id]
        
        return entries[-limit:]
    
    def clear_audit_log(self):
        """Clear the audit log."""
        self.audit_log.clear()
