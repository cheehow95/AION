"""
PostgreSQL Migration System
===========================
Simple migration runner for database schema management.
"""

import os
from typing import List
from datetime import datetime
from .connection import get_connection


class Migrations:
    """
    Database migration manager.
    
    Runs SQL migration files in order and tracks applied migrations.
    """
    
    def __init__(self, migrations_dir: str = None):
        """
        Initialize migration manager.
        
        Args:
            migrations_dir: Directory containing .sql migration files
        """
        if migrations_dir is None:
            # Default to src/postgres/migrations
            base_dir = os.path.dirname(os.path.abspath(__file__))
            migrations_dir = os.path.join(base_dir, 'migrations')
        
        self.migrations_dir = migrations_dir
        self._ensure_migrations_table()
    
    def _ensure_migrations_table(self):
        """Create migrations tracking table if not exists."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS _migrations (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(255) NOT NULL UNIQUE,
                        applied_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
    
    def _get_applied(self) -> List[str]:
        """Get list of already applied migrations."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT name FROM _migrations ORDER BY id")
                return [row[0] for row in cur.fetchall()]
    
    def _get_pending(self) -> List[str]:
        """Get list of pending migration files."""
        applied = set(self._get_applied())
        
        if not os.path.exists(self.migrations_dir):
            return []
        
        files = sorted([
            f for f in os.listdir(self.migrations_dir)
            if f.endswith('.sql') and f not in applied
        ])
        
        return files
    
    def apply(self, name: str = None) -> int:
        """
        Apply pending migrations.
        
        Args:
            name: Specific migration to apply, or None for all pending
            
        Returns:
            Number of applied migrations
        """
        if name:
            pending = [name] if name in self._get_pending() else []
        else:
            pending = self._get_pending()
        
        count = 0
        for migration in pending:
            filepath = os.path.join(self.migrations_dir, migration)
            
            with open(filepath, 'r') as f:
                sql = f.read()
            
            with get_connection() as conn:
                with conn.cursor() as cur:
                    # Execute migration
                    cur.execute(sql)
                    
                    # Record migration
                    cur.execute(
                        "INSERT INTO _migrations (name) VALUES (%s)",
                        (migration,)
                    )
            
            print(f"✓ Applied: {migration}")
            count += 1
        
        return count
    
    def status(self) -> dict:
        """
        Get migration status.
        
        Returns:
            Dict with applied and pending migrations
        """
        return {
            'applied': self._get_applied(),
            'pending': self._get_pending()
        }
    
    def create(self, name: str) -> str:
        """
        Create a new migration file.
        
        Args:
            name: Migration name (will be prefixed with timestamp)
            
        Returns:
            Path to created file
        """
        os.makedirs(self.migrations_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        filename = f"{timestamp}_{name}.sql"
        filepath = os.path.join(self.migrations_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write(f"-- Migration: {name}\n")
            f.write(f"-- Created: {datetime.now().isoformat()}\n\n")
            f.write("-- Add your SQL here\n")
        
        return filepath
