"""
AION Supabase Integration Module
================================
Provides authentication, database, and storage services via Supabase.
"""

from .client import get_supabase_client
from .auth import SupabaseAuth
from .database import SupabaseDB
from .storage import SupabaseStorage

__all__ = [
    'get_supabase_client',
    'SupabaseAuth',
    'SupabaseDB',
    'SupabaseStorage'
]
