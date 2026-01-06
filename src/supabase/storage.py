"""
Supabase Storage Service
========================
Provides file storage operations via Supabase Storage.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from .client import get_supabase_client


@dataclass
class FileInfo:
    """File metadata from storage."""
    name: str
    id: str
    size: int
    created_at: str
    metadata: Dict[str, Any] = None


class SupabaseStorage:
    """
    Storage service for Supabase Storage.
    
    Provides file upload, download, and management for buckets.
    """
    
    # Default bucket names
    AVATARS_BUCKET = "avatars"
    UPLOADS_BUCKET = "uploads"
    EXPORTS_BUCKET = "exports"
    
    def __init__(self):
        self.client = get_supabase_client()
    
    def upload(
        self, 
        bucket: str, 
        path: str, 
        file_data: bytes,
        content_type: str = "application/octet-stream"
    ) -> str:
        """
        Upload a file to storage.
        
        Args:
            bucket: Bucket name
            path: File path within bucket (e.g., "user123/image.png")
            file_data: File contents as bytes
            content_type: MIME type
            
        Returns:
            Public URL of uploaded file
        """
        response = self.client.storage.from_(bucket).upload(
            path,
            file_data,
            {"content-type": content_type}
        )
        return self.get_public_url(bucket, path)
    
    def download(self, bucket: str, path: str) -> bytes:
        """
        Download a file from storage.
        
        Args:
            bucket: Bucket name
            path: File path within bucket
            
        Returns:
            File contents as bytes
        """
        response = self.client.storage.from_(bucket).download(path)
        return response
    
    def delete(self, bucket: str, paths: List[str]) -> bool:
        """
        Delete files from storage.
        
        Args:
            bucket: Bucket name
            paths: List of file paths to delete
            
        Returns:
            True if successful
        """
        self.client.storage.from_(bucket).remove(paths)
        return True
    
    def get_public_url(self, bucket: str, path: str) -> str:
        """
        Get public URL for a file.
        
        Args:
            bucket: Bucket name
            path: File path
            
        Returns:
            Public URL (only works for public buckets)
        """
        return self.client.storage.from_(bucket).get_public_url(path)
    
    def get_signed_url(
        self, 
        bucket: str, 
        path: str, 
        expires_in: int = 3600
    ) -> str:
        """
        Get signed URL for private file access.
        
        Args:
            bucket: Bucket name
            path: File path
            expires_in: Seconds until URL expires (default 1 hour)
            
        Returns:
            Signed URL with temporary access
        """
        response = self.client.storage.from_(bucket).create_signed_url(
            path, 
            expires_in
        )
        return response["signedURL"]
    
    def list_files(
        self, 
        bucket: str, 
        prefix: str = "",
        limit: int = 100,
        offset: int = 0
    ) -> List[FileInfo]:
        """
        List files in a bucket/folder.
        
        Args:
            bucket: Bucket name
            prefix: Folder prefix to filter by
            limit: Max files to return
            offset: Pagination offset
            
        Returns:
            List of FileInfo objects
        """
        response = self.client.storage.from_(bucket).list(
            prefix,
            {"limit": limit, "offset": offset}
        )
        
        return [
            FileInfo(
                name=f.get("name"),
                id=f.get("id"),
                size=f.get("metadata", {}).get("size", 0),
                created_at=f.get("created_at"),
                metadata=f.get("metadata")
            )
            for f in response
        ]
    
    def move(self, bucket: str, from_path: str, to_path: str) -> bool:
        """
        Move/rename a file.
        
        Args:
            bucket: Bucket name
            from_path: Current path
            to_path: New path
            
        Returns:
            True if successful
        """
        self.client.storage.from_(bucket).move(from_path, to_path)
        return True
    
    def copy(self, bucket: str, from_path: str, to_path: str) -> bool:
        """
        Copy a file.
        
        Args:
            bucket: Bucket name
            from_path: Source path
            to_path: Destination path
            
        Returns:
            True if successful
        """
        self.client.storage.from_(bucket).copy(from_path, to_path)
        return True
    
    def create_bucket(
        self, 
        name: str, 
        public: bool = False,
        file_size_limit: int = None,
        allowed_mime_types: List[str] = None
    ) -> bool:
        """
        Create a new storage bucket.
        
        Args:
            name: Bucket name
            public: Whether bucket is publicly accessible
            file_size_limit: Max file size in bytes
            allowed_mime_types: List of allowed MIME types
            
        Returns:
            True if successful
        """
        options = {"public": public}
        if file_size_limit:
            options["file_size_limit"] = file_size_limit
        if allowed_mime_types:
            options["allowed_mime_types"] = allowed_mime_types
            
        self.client.storage.create_bucket(name, options)
        return True
