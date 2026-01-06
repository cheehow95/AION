"""
Supabase Authentication Service
===============================
Handles user authentication via Supabase Auth.
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass
from .client import get_supabase_client


@dataclass
class User:
    """Authenticated user data."""
    id: str
    email: str
    created_at: str
    metadata: Dict[str, Any] = None


@dataclass  
class Session:
    """Authentication session."""
    access_token: str
    refresh_token: str
    expires_at: int
    user: User


class SupabaseAuth:
    """
    Authentication service using Supabase Auth.
    
    Provides email/password auth, OAuth, and session management.
    """
    
    def __init__(self):
        self.client = get_supabase_client()
    
    def sign_up(self, email: str, password: str) -> Session:
        """
        Register a new user.
        
        Args:
            email: User's email address
            password: User's password (min 6 characters)
            
        Returns:
            Session with access token and user info
        """
        response = self.client.auth.sign_up({
            "email": email,
            "password": password
        })
        return self._parse_session(response)
    
    def sign_in(self, email: str, password: str) -> Session:
        """
        Sign in with email and password.
        
        Args:
            email: User's email
            password: User's password
            
        Returns:
            Session with access token
        """
        response = self.client.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        return self._parse_session(response)
    
    def sign_out(self) -> bool:
        """
        Sign out the current user.
        
        Returns:
            True if successful
        """
        self.client.auth.sign_out()
        return True
    
    def get_user(self, access_token: str = None) -> Optional[User]:
        """
        Get current authenticated user.
        
        Args:
            access_token: Optional JWT token (uses session if not provided)
            
        Returns:
            User object or None if not authenticated
        """
        try:
            response = self.client.auth.get_user(access_token)
            if response and response.user:
                return User(
                    id=response.user.id,
                    email=response.user.email,
                    created_at=str(response.user.created_at),
                    metadata=response.user.user_metadata
                )
        except Exception:
            return None
        return None
    
    def refresh_session(self, refresh_token: str) -> Session:
        """
        Refresh an expired session.
        
        Args:
            refresh_token: The refresh token from previous session
            
        Returns:
            New session with fresh tokens
        """
        response = self.client.auth.refresh_session(refresh_token)
        return self._parse_session(response)
    
    def get_oauth_url(self, provider: str, redirect_to: str = None) -> str:
        """
        Get OAuth authorization URL.
        
        Args:
            provider: OAuth provider ('google', 'github', 'discord', etc.)
            redirect_to: URL to redirect after auth
            
        Returns:
            Authorization URL to redirect user to
        """
        options = {}
        if redirect_to:
            options["redirect_to"] = redirect_to
            
        response = self.client.auth.sign_in_with_oauth({
            "provider": provider,
            "options": options
        })
        return response.url
    
    def _parse_session(self, response) -> Session:
        """Parse auth response into Session object."""
        session = response.session
        user = response.user
        
        return Session(
            access_token=session.access_token,
            refresh_token=session.refresh_token,
            expires_at=session.expires_at,
            user=User(
                id=user.id,
                email=user.email,
                created_at=str(user.created_at),
                metadata=user.user_metadata
            )
        )
