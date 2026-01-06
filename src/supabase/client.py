"""
Supabase Client Factory
=======================
Singleton client for Supabase connection.
"""

import os
from typing import Optional
from functools import lru_cache
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()


@lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    """
    Get the Supabase client singleton.
    
    Returns:
        Client: Supabase client instance
        
    Raises:
        ValueError: If SUPABASE_URL or SUPABASE_ANON_KEY not set
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY")
    
    if not url or not key:
        raise ValueError(
            "Supabase credentials not configured. "
            "Set SUPABASE_URL and SUPABASE_ANON_KEY in .env file."
        )
    
    return create_client(url, key)


def get_service_client() -> Client:
    """
    Get Supabase client with service role key (admin access).
    Use only for server-side operations that bypass RLS.
    
    Returns:
        Client: Supabase client with service role
    """
    url = os.getenv("SUPABASE_URL")
    service_key = os.getenv("SUPABASE_SERVICE_KEY")
    
    if not url or not service_key:
        raise ValueError(
            "Supabase service credentials not configured. "
            "Set SUPABASE_URL and SUPABASE_SERVICE_KEY in .env file."
        )
    
    return create_client(url, service_key)
