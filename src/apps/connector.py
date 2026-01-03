"""
AION App Integration Directory - App Connector
================================================

App connection framework:
- OAuth integration
- Session management
- Data exchange protocols
- Secure execution sandbox

Enables secure third-party app connections.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
import uuid
import hashlib


class ConnectionStatus(Enum):
    """App connection status."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    EXPIRED = "expired"


@dataclass
class OAuthConfig:
    """OAuth configuration for an app."""
    client_id: str = ""
    client_secret: str = ""  # In production, use secure storage
    auth_url: str = ""
    token_url: str = ""
    scopes: List[str] = field(default_factory=list)
    redirect_uri: str = ""


@dataclass
class OAuthToken:
    """OAuth token."""
    access_token: str = ""
    refresh_token: str = ""
    token_type: str = "Bearer"
    expires_at: datetime = field(default_factory=lambda: datetime.now() + timedelta(hours=1))
    scopes: List[str] = field(default_factory=list)
    
    @property
    def is_expired(self) -> bool:
        return datetime.now() > self.expires_at


@dataclass
class AppSession:
    """A session with a connected app."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    app_id: str = ""
    user_id: str = ""
    status: ConnectionStatus = ConnectionStatus.DISCONNECTED
    token: Optional[OAuthToken] = None
    connected_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AppConnector:
    """Manages connections to third-party apps."""
    
    def __init__(self):
        self.configs: Dict[str, OAuthConfig] = {}
        self.sessions: Dict[str, AppSession] = {}  # session_id -> session
        self.user_sessions: Dict[str, Dict[str, str]] = {}  # user_id -> {app_id: session_id}
        self.action_handlers: Dict[str, Dict[str, Callable]] = {}  # app_id -> {action: handler}
    
    def register_config(self, app_id: str, config: OAuthConfig):
        """Register OAuth config for an app."""
        self.configs[app_id] = config
    
    def register_action(self, app_id: str, action: str, handler: Callable):
        """Register an action handler for an app."""
        if app_id not in self.action_handlers:
            self.action_handlers[app_id] = {}
        self.action_handlers[app_id][action] = handler
    
    async def connect(self, app_id: str, user_id: str,
                      auth_code: str = None) -> AppSession:
        """Connect to an app."""
        session = AppSession(
            app_id=app_id,
            user_id=user_id,
            status=ConnectionStatus.CONNECTING
        )
        
        config = self.configs.get(app_id)
        
        if config and auth_code:
            # Exchange code for token (simulated)
            token = await self._exchange_token(config, auth_code)
            session.token = token
            session.status = ConnectionStatus.CONNECTED
            session.connected_at = datetime.now()
        elif not config:
            # No OAuth required
            session.status = ConnectionStatus.CONNECTED
            session.connected_at = datetime.now()
        else:
            session.status = ConnectionStatus.ERROR
        
        self.sessions[session.id] = session
        
        # Track user sessions
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {}
        self.user_sessions[user_id][app_id] = session.id
        
        return session
    
    async def _exchange_token(self, config: OAuthConfig, code: str) -> OAuthToken:
        """Exchange auth code for token (simulated)."""
        # In production, make actual OAuth request
        await asyncio.sleep(0.1)  # Simulate network delay
        
        return OAuthToken(
            access_token=f"access_{uuid.uuid4().hex[:16]}",
            refresh_token=f"refresh_{uuid.uuid4().hex[:16]}",
            scopes=config.scopes,
            expires_at=datetime.now() + timedelta(hours=1)
        )
    
    async def refresh_token(self, session_id: str) -> bool:
        """Refresh an expired token."""
        session = self.sessions.get(session_id)
        if not session or not session.token:
            return False
        
        config = self.configs.get(session.app_id)
        if not config:
            return False
        
        # Simulate token refresh
        session.token = OAuthToken(
            access_token=f"access_{uuid.uuid4().hex[:16]}",
            refresh_token=session.token.refresh_token,
            scopes=session.token.scopes,
            expires_at=datetime.now() + timedelta(hours=1)
        )
        
        return True
    
    async def disconnect(self, session_id: str) -> bool:
        """Disconnect from an app."""
        session = self.sessions.get(session_id)
        if not session:
            return False
        
        session.status = ConnectionStatus.DISCONNECTED
        session.token = None
        
        # Remove from user sessions
        if session.user_id in self.user_sessions:
            if session.app_id in self.user_sessions[session.user_id]:
                del self.user_sessions[session.user_id][session.app_id]
        
        return True
    
    def get_session(self, user_id: str, app_id: str) -> Optional[AppSession]:
        """Get user's session for an app."""
        session_id = self.user_sessions.get(user_id, {}).get(app_id)
        if session_id:
            return self.sessions.get(session_id)
        return None
    
    def get_user_connections(self, user_id: str) -> List[AppSession]:
        """Get all connected apps for a user."""
        session_ids = self.user_sessions.get(user_id, {}).values()
        sessions = [self.sessions.get(sid) for sid in session_ids]
        return [s for s in sessions if s and s.status == ConnectionStatus.CONNECTED]
    
    async def execute_action(self, session_id: str, action: str,
                            params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute an action on a connected app."""
        session = self.sessions.get(session_id)
        if not session or session.status != ConnectionStatus.CONNECTED:
            return {'error': 'Not connected'}
        
        # Check token expiry
        if session.token and session.token.is_expired:
            refreshed = await self.refresh_token(session_id)
            if not refreshed:
                session.status = ConnectionStatus.EXPIRED
                return {'error': 'Token expired'}
        
        # Get action handler
        handlers = self.action_handlers.get(session.app_id, {})
        handler = handlers.get(action)
        
        if not handler:
            return {'error': f'Unknown action: {action}'}
        
        session.last_used = datetime.now()
        
        # Execute handler
        try:
            if asyncio.iscoroutinefunction(handler):
                result = await handler(session, params or {})
            else:
                result = handler(session, params or {})
            return {'success': True, 'result': result}
        except Exception as e:
            return {'error': str(e)}
    
    def get_auth_url(self, app_id: str, state: str = "") -> Optional[str]:
        """Get OAuth authorization URL."""
        config = self.configs.get(app_id)
        if not config:
            return None
        
        state = state or str(uuid.uuid4())
        scopes = '+'.join(config.scopes)
        
        return (
            f"{config.auth_url}?"
            f"client_id={config.client_id}&"
            f"redirect_uri={config.redirect_uri}&"
            f"scope={scopes}&"
            f"state={state}&"
            f"response_type=code"
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connector statistics."""
        connected = sum(1 for s in self.sessions.values() 
                       if s.status == ConnectionStatus.CONNECTED)
        
        return {
            'total_sessions': len(self.sessions),
            'connected': connected,
            'registered_apps': len(self.configs),
            'action_handlers': sum(len(h) for h in self.action_handlers.values())
        }


async def demo_connector():
    """Demonstrate app connector."""
    print("🔌 App Connector Demo")
    print("=" * 50)
    
    connector = AppConnector()
    
    # Register OAuth config
    connector.register_config("github", OAuthConfig(
        client_id="github_client_123",
        client_secret="secret",
        auth_url="https://github.com/login/oauth/authorize",
        token_url="https://github.com/login/oauth/access_token",
        scopes=["repo", "read:user"],
        redirect_uri="https://aion.io/callback"
    ))
    
    # Register action handlers
    async def list_repos(session: AppSession, params: Dict) -> List[str]:
        return ["repo1", "repo2", "repo3"]
    
    async def create_issue(session: AppSession, params: Dict) -> Dict:
        return {"issue_id": "123", "title": params.get("title", "New Issue")}
    
    connector.register_action("github", "list_repos", list_repos)
    connector.register_action("github", "create_issue", create_issue)
    
    # Get auth URL
    auth_url = connector.get_auth_url("github")
    print(f"\n🔐 Auth URL: {auth_url[:60]}...")
    
    # Connect (simulating callback with auth code)
    session = await connector.connect("github", "user1", auth_code="code_abc123")
    print(f"\n📱 Connected: {session.status.value}")
    print(f"   Token expires: {session.token.expires_at if session.token else 'N/A'}")
    
    # Execute actions
    print("\n🎬 Executing actions:")
    
    result = await connector.execute_action(session.id, "list_repos")
    print(f"   list_repos: {result}")
    
    result = await connector.execute_action(session.id, "create_issue", 
                                           {"title": "Bug fix needed"})
    print(f"   create_issue: {result}")
    
    # User connections
    connections = connector.get_user_connections("user1")
    print(f"\n👤 User1 connections: {len(connections)}")
    for conn in connections:
        print(f"   • {conn.app_id} (session: {conn.id})")
    
    print(f"\n📊 Stats: {connector.get_stats()}")
    
    # Disconnect
    await connector.disconnect(session.id)
    print(f"\n🔌 Disconnected: {session.status.value}")
    
    print("\n✅ Connector demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_connector())
