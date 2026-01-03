"""
AION App Integration Directory - Package Initialization
========================================================

Third-party app integration like GPT-5.2 Apps:
- App catalog and discovery
- App connector framework
- OAuth integration
"""

from .directory import (
    App,
    AppCategory,
    AppDirectory,
    AppRating
)

from .connector import (
    AppConnector,
    ConnectionStatus,
    AppSession,
    OAuthConfig
)

__all__ = [
    # Directory
    'App',
    'AppCategory',
    'AppDirectory',
    'AppRating',
    # Connector
    'AppConnector',
    'ConnectionStatus',
    'AppSession',
    'OAuthConfig',
]
