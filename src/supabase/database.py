"""
Supabase Database Service
=========================
Provides CRUD operations for Supabase PostgreSQL database.
"""

from typing import Dict, List, Any, Optional
from .client import get_supabase_client


class SupabaseDB:
    """
    Database service for Supabase PostgreSQL.
    
    Provides typed CRUD operations with support for filters and RPC.
    """
    
    def __init__(self):
        self.client = get_supabase_client()
    
    def insert(self, table: str, data: Dict[str, Any]) -> Dict:
        """
        Insert a row into a table.
        
        Args:
            table: Table name
            data: Row data as dict
            
        Returns:
            Inserted row with generated fields
        """
        response = self.client.table(table).insert(data).execute()
        return response.data[0] if response.data else {}
    
    def insert_many(self, table: str, rows: List[Dict[str, Any]]) -> List[Dict]:
        """
        Insert multiple rows.
        
        Args:
            table: Table name
            rows: List of row data dicts
            
        Returns:
            List of inserted rows
        """
        response = self.client.table(table).insert(rows).execute()
        return response.data
    
    def select(
        self, 
        table: str, 
        columns: str = "*",
        filters: Dict[str, Any] = None,
        order_by: str = None,
        limit: int = None,
        offset: int = None
    ) -> List[Dict]:
        """
        Query rows from a table.
        
        Args:
            table: Table name
            columns: Columns to select (default "*")
            filters: Dict of column=value filters
            order_by: Column to order by (prefix with - for DESC)
            limit: Max rows to return
            offset: Number of rows to skip
            
        Returns:
            List of matching rows
        """
        query = self.client.table(table).select(columns)
        
        # Apply filters
        if filters:
            for key, value in filters.items():
                query = query.eq(key, value)
        
        # Apply ordering
        if order_by:
            desc = order_by.startswith("-")
            col = order_by.lstrip("-")
            query = query.order(col, desc=desc)
        
        # Apply pagination
        if limit:
            query = query.limit(limit)
        if offset:
            query = query.range(offset, offset + (limit or 100) - 1)
        
        response = query.execute()
        return response.data
    
    def select_one(self, table: str, filters: Dict[str, Any]) -> Optional[Dict]:
        """
        Get a single row by filters.
        
        Args:
            table: Table name
            filters: Dict of column=value filters
            
        Returns:
            Matching row or None
        """
        results = self.select(table, filters=filters, limit=1)
        return results[0] if results else None
    
    def update(
        self, 
        table: str, 
        filters: Dict[str, Any], 
        data: Dict[str, Any]
    ) -> List[Dict]:
        """
        Update rows matching filters.
        
        Args:
            table: Table name
            filters: Dict of column=value to match
            data: New values to set
            
        Returns:
            Updated rows
        """
        query = self.client.table(table).update(data)
        
        for key, value in filters.items():
            query = query.eq(key, value)
        
        response = query.execute()
        return response.data
    
    def delete(self, table: str, filters: Dict[str, Any]) -> bool:
        """
        Delete rows matching filters.
        
        Args:
            table: Table name
            filters: Dict of column=value to match
            
        Returns:
            True if successful
        """
        query = self.client.table(table).delete()
        
        for key, value in filters.items():
            query = query.eq(key, value)
        
        query.execute()
        return True
    
    def upsert(self, table: str, data: Dict[str, Any]) -> Dict:
        """
        Insert or update a row (based on primary key).
        
        Args:
            table: Table name
            data: Row data including primary key
            
        Returns:
            Upserted row
        """
        response = self.client.table(table).upsert(data).execute()
        return response.data[0] if response.data else {}
    
    def rpc(self, function_name: str, params: Dict[str, Any] = None) -> Any:
        """
        Call a PostgreSQL stored procedure/function.
        
        Args:
            function_name: Name of the function
            params: Function parameters
            
        Returns:
            Function result
        """
        response = self.client.rpc(function_name, params or {}).execute()
        return response.data
    
    def count(self, table: str, filters: Dict[str, Any] = None) -> int:
        """
        Count rows in a table.
        
        Args:
            table: Table name
            filters: Optional filters
            
        Returns:
            Row count
        """
        query = self.client.table(table).select("*", count="exact")
        
        if filters:
            for key, value in filters.items():
                query = query.eq(key, value)
        
        response = query.execute()
        return response.count or 0
