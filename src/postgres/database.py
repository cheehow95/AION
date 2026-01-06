"""
PostgreSQL Database Service
===========================
High-level database operations with connection pooling.
"""

from typing import Dict, List, Any, Optional, Tuple
from psycopg2.extras import RealDictCursor
from .connection import get_connection


class PostgresDB:
    """
    Database service for PostgreSQL.
    
    Provides typed CRUD operations with automatic connection management.
    """
    
    def execute(self, query: str, params: tuple = None) -> int:
        """
        Execute a query (INSERT, UPDATE, DELETE).
        
        Args:
            query: SQL query with %s placeholders
            params: Query parameters
            
        Returns:
            Number of affected rows
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                return cur.rowcount
    
    def execute_many(self, query: str, params_list: List[tuple]) -> int:
        """
        Execute a query for multiple parameter sets.
        
        Args:
            query: SQL query with %s placeholders
            params_list: List of parameter tuples
            
        Returns:
            Total affected rows
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(query, params_list)
                return cur.rowcount
    
    def fetch_one(self, query: str, params: tuple = None) -> Optional[Dict]:
        """
        Fetch a single row.
        
        Args:
            query: SELECT query
            params: Query parameters
            
        Returns:
            Row as dict or None
        """
        with get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                row = cur.fetchone()
                return dict(row) if row else None
    
    def fetch_all(self, query: str, params: tuple = None) -> List[Dict]:
        """
        Fetch all matching rows.
        
        Args:
            query: SELECT query
            params: Query parameters
            
        Returns:
            List of rows as dicts
        """
        with get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                return [dict(row) for row in cur.fetchall()]
    
    def insert(self, table: str, data: Dict[str, Any]) -> Dict:
        """
        Insert a row and return it.
        
        Args:
            table: Table name
            data: Column values
            
        Returns:
            Inserted row with generated fields
        """
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['%s'] * len(data))
        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders}) RETURNING *"
        
        return self.fetch_one(query, tuple(data.values()))
    
    def insert_many(self, table: str, rows: List[Dict[str, Any]]) -> int:
        """
        Insert multiple rows.
        
        Args:
            table: Table name  
            rows: List of row data
            
        Returns:
            Number of inserted rows
        """
        if not rows:
            return 0
        
        columns = ', '.join(rows[0].keys())
        placeholders = ', '.join(['%s'] * len(rows[0]))
        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        
        params_list = [tuple(row.values()) for row in rows]
        return self.execute_many(query, params_list)
    
    def select(
        self,
        table: str,
        columns: str = "*",
        where: Dict[str, Any] = None,
        order_by: str = None,
        limit: int = None,
        offset: int = None
    ) -> List[Dict]:
        """
        Query rows from a table.
        
        Args:
            table: Table name
            columns: Columns to select
            where: Filter conditions (column=value)
            order_by: Order clause (prefix with - for DESC)
            limit: Max rows
            offset: Skip rows
            
        Returns:
            List of matching rows
        """
        query = f"SELECT {columns} FROM {table}"
        params = []
        
        if where:
            conditions = []
            for col, val in where.items():
                conditions.append(f"{col} = %s")
                params.append(val)
            query += " WHERE " + " AND ".join(conditions)
        
        if order_by:
            if order_by.startswith('-'):
                query += f" ORDER BY {order_by[1:]} DESC"
            else:
                query += f" ORDER BY {order_by}"
        
        if limit:
            query += f" LIMIT {limit}"
        
        if offset:
            query += f" OFFSET {offset}"
        
        return self.fetch_all(query, tuple(params) if params else None)
    
    def select_one(self, table: str, where: Dict[str, Any]) -> Optional[Dict]:
        """
        Get a single row by filter.
        
        Args:
            table: Table name
            where: Filter conditions
            
        Returns:
            Matching row or None
        """
        rows = self.select(table, where=where, limit=1)
        return rows[0] if rows else None
    
    def update(
        self,
        table: str,
        where: Dict[str, Any],
        data: Dict[str, Any]
    ) -> int:
        """
        Update rows matching filter.
        
        Args:
            table: Table name
            where: Filter conditions
            data: New values
            
        Returns:
            Number of updated rows
        """
        set_clause = ', '.join([f"{col} = %s" for col in data.keys()])
        where_clause = ' AND '.join([f"{col} = %s" for col in where.keys()])
        
        query = f"UPDATE {table} SET {set_clause} WHERE {where_clause}"
        params = tuple(data.values()) + tuple(where.values())
        
        return self.execute(query, params)
    
    def delete(self, table: str, where: Dict[str, Any]) -> int:
        """
        Delete rows matching filter.
        
        Args:
            table: Table name
            where: Filter conditions
            
        Returns:
            Number of deleted rows
        """
        where_clause = ' AND '.join([f"{col} = %s" for col in where.keys()])
        query = f"DELETE FROM {table} WHERE {where_clause}"
        
        return self.execute(query, tuple(where.values()))
    
    def upsert(
        self,
        table: str,
        data: Dict[str, Any],
        conflict_columns: List[str]
    ) -> Dict:
        """
        Insert or update on conflict.
        
        Args:
            table: Table name
            data: Row data
            conflict_columns: Columns for conflict detection
            
        Returns:
            Upserted row
        """
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['%s'] * len(data))
        conflict = ', '.join(conflict_columns)
        
        update_cols = [k for k in data.keys() if k not in conflict_columns]
        update_clause = ', '.join([f"{col} = EXCLUDED.{col}" for col in update_cols])
        
        query = f"""
            INSERT INTO {table} ({columns}) VALUES ({placeholders})
            ON CONFLICT ({conflict}) DO UPDATE SET {update_clause}
            RETURNING *
        """
        
        return self.fetch_one(query, tuple(data.values()))
    
    def count(self, table: str, where: Dict[str, Any] = None) -> int:
        """
        Count rows in a table.
        
        Args:
            table: Table name
            where: Optional filter
            
        Returns:
            Row count
        """
        query = f"SELECT COUNT(*) as count FROM {table}"
        params = []
        
        if where:
            conditions = [f"{col} = %s" for col in where.keys()]
            query += " WHERE " + " AND ".join(conditions)
            params = list(where.values())
        
        result = self.fetch_one(query, tuple(params) if params else None)
        return result['count'] if result else 0
    
    def raw(self, query: str, params: tuple = None) -> List[Dict]:
        """
        Execute raw SQL query.
        
        Args:
            query: Raw SQL
            params: Query parameters
            
        Returns:
            Query results
        """
        return self.fetch_all(query, params)
