"""
PostgreSQL Connection Pool
==========================
Manages database connections with pooling for performance.
"""

import os
from typing import Optional
from contextlib import contextmanager
from dotenv import load_dotenv

import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor

# Load environment variables
load_dotenv()


class PostgresPool:
    """
    PostgreSQL connection pool manager.
    
    Uses environment variables for configuration:
    - POSTGRES_HOST (default: localhost)
    - POSTGRES_PORT (default: 5432)
    - POSTGRES_DB (default: aion)
    - POSTGRES_USER (default: postgres)
    - POSTGRES_PASSWORD
    """
    
    _pool: Optional[pool.ThreadedConnectionPool] = None
    
    @classmethod
    def get_config(cls) -> dict:
        """Get database configuration from environment."""
        return {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': int(os.getenv('POSTGRES_PORT', '5432')),
            'database': os.getenv('POSTGRES_DB', 'aion'),
            'user': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', ''),
        }
    
    @classmethod
    def initialize(cls, min_connections: int = 2, max_connections: int = 10):
        """
        Initialize the connection pool.
        
        Args:
            min_connections: Minimum pool size
            max_connections: Maximum pool size
        """
        if cls._pool is not None:
            return
        
        config = cls.get_config()
        
        try:
            cls._pool = pool.ThreadedConnectionPool(
                min_connections,
                max_connections,
                **config
            )
            print(f"✓ PostgreSQL pool initialized ({config['host']}:{config['port']}/{config['database']})")
        except Exception as e:
            print(f"⚠ PostgreSQL connection failed: {e}")
            cls._pool = None
    
    @classmethod
    def get_connection(cls):
        """Get a connection from the pool."""
        if cls._pool is None:
            cls.initialize()
        
        if cls._pool is None:
            raise ConnectionError("PostgreSQL pool not available")
        
        return cls._pool.getconn()
    
    @classmethod
    def return_connection(cls, conn):
        """Return a connection to the pool."""
        if cls._pool and conn:
            cls._pool.putconn(conn)
    
    @classmethod
    def close_all(cls):
        """Close all connections in the pool."""
        if cls._pool:
            cls._pool.closeall()
            cls._pool = None


@contextmanager
def get_connection():
    """
    Context manager for database connections.
    
    Usage:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM users")
    """
    conn = PostgresPool.get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        PostgresPool.return_connection(conn)
