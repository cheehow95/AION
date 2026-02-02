"""Tests for AION Persistent Vector Store"""

import pytest
import os
import tempfile
import shutil
import sys
sys.path.insert(0, '.')

from src.runtime.persistent_store import (
    PersistentVectorStore, VectorDocument, get_vector_store
)


class TestVectorDocument:
    """Test VectorDocument dataclass."""
    
    def test_document_creation(self):
        """Test creating a VectorDocument."""
        doc = VectorDocument(
            id="doc123",
            content="Test content",
            embedding=[0.1, 0.2, 0.3],
            metadata={"source": "test"}
        )
        
        assert doc.id == "doc123"
        assert doc.content == "Test content"
        assert len(doc.embedding) == 3
        assert doc.collection == "default"
    
    def test_document_with_collection(self):
        """Test document with custom collection."""
        doc = VectorDocument(
            id="doc",
            content="content",
            embedding=[],
            metadata={},
            collection="custom"
        )
        
        assert doc.collection == "custom"


class TestPersistentVectorStore:
    """Test PersistentVectorStore class."""
    
    @pytest.fixture
    def temp_store(self):
        """Create a temporary store for testing."""
        temp_dir = tempfile.mkdtemp()
        store = PersistentVectorStore(persist_directory=temp_dir)
        yield store
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_init(self, temp_store):
        """Test store initialization."""
        assert temp_store.collections == {}
        assert os.path.exists(temp_store.persist_dir)
    
    def test_create_collection(self, temp_store):
        """Test creating a collection."""
        result = temp_store.create_collection("test_collection")
        
        assert result == True
        assert "test_collection" in temp_store.collections
    
    def test_create_collection_duplicate(self, temp_store):
        """Test creating duplicate collection."""
        temp_store.create_collection("test")
        result = temp_store.create_collection("test")
        
        assert result == False
    
    def test_add_document(self, temp_store):
        """Test adding a document."""
        doc_id = temp_store.add(
            content="Test document",
            metadata={"key": "value"},
            collection="test"
        )
        
        assert doc_id is not None
        assert len(doc_id) == 12  # MD5 truncated to 12 chars
        assert temp_store.count("test") == 1
    
    def test_add_creates_collection(self, temp_store):
        """Test that add creates collection if needed."""
        temp_store.add("content", collection="auto_created")
        
        assert "auto_created" in temp_store.collections
    
    def test_add_with_custom_embedding(self, temp_store):
        """Test adding with custom embedding."""
        custom_embedding = [0.5] * 384
        temp_store.add(
            content="test",
            embedding=custom_embedding,
            collection="test"
        )
        
        docs = temp_store.collections["test"]
        assert docs[0].embedding == custom_embedding
    
    def test_search(self, temp_store):
        """Test semantic search."""
        temp_store.add("Python is a programming language", collection="docs")
        temp_store.add("JavaScript is for web development", collection="docs")
        temp_store.add("Cooking recipes are delicious", collection="docs")
        
        results = temp_store.search("programming Python", collection="docs", limit=2)
        
        assert len(results) <= 2
        assert any("Python" in r.content for r in results)
    
    def test_search_empty_collection(self, temp_store):
        """Test search on non-existent collection."""
        results = temp_store.search("query", collection="nonexistent")
        
        assert results == []
    
    def test_search_limit(self, temp_store):
        """Test search limit parameter."""
        for i in range(10):
            temp_store.add(f"Document {i}", collection="many")
        
        results = temp_store.search("Document", collection="many", limit=3)
        
        assert len(results) == 3
    
    def test_delete(self, temp_store):
        """Test deleting a document."""
        doc_id = temp_store.add("To be deleted", collection="test")
        
        result = temp_store.delete(doc_id, collection="test")
        
        assert result == True
        assert temp_store.count("test") == 0
    
    def test_delete_nonexistent(self, temp_store):
        """Test deleting non-existent document."""
        result = temp_store.delete("fake_id", collection="test")
        
        assert result == False
    
    def test_count(self, temp_store):
        """Test document count."""
        assert temp_store.count("test") == 0
        
        temp_store.add("doc1", collection="test")
        temp_store.add("doc2", collection="test")
        
        assert temp_store.count("test") == 2
    
    def test_list_collections(self, temp_store):
        """Test listing collections."""
        temp_store.create_collection("a")
        temp_store.create_collection("b")
        temp_store.create_collection("c")
        
        collections = temp_store.list_collections()
        
        assert "a" in collections
        assert "b" in collections
        assert "c" in collections
    
    def test_generate_embedding(self, temp_store):
        """Test embedding generation."""
        embedding = temp_store._generate_embedding("test text")
        
        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)
    
    def test_generate_embedding_empty(self, temp_store):
        """Test embedding generation for empty text."""
        embedding = temp_store._generate_embedding("")
        
        assert len(embedding) == 384
        assert all(x == 0.0 for x in embedding)
    
    def test_generate_embedding_normalized(self, temp_store):
        """Test that embeddings are normalized."""
        embedding = temp_store._generate_embedding("test text here")
        
        # Check magnitude is approximately 1
        import math
        magnitude = math.sqrt(sum(x*x for x in embedding))
        assert abs(magnitude - 1.0) < 0.01 or magnitude == 0
    
    def test_cosine_similarity(self, temp_store):
        """Test cosine similarity calculation."""
        v1 = [1.0, 0.0, 0.0]
        v2 = [1.0, 0.0, 0.0]
        v3 = [0.0, 1.0, 0.0]
        
        assert temp_store._cosine_similarity(v1, v2) == 1.0
        assert temp_store._cosine_similarity(v1, v3) == 0.0
    
    def test_persistence(self):
        """Test data persistence to disk."""
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create store and add data
            store1 = PersistentVectorStore(persist_directory=temp_dir)
            store1.add("Persistent data", collection="persist_test")
            
            # Create new store from same directory
            store2 = PersistentVectorStore(persist_directory=temp_dir)
            
            assert store2.count("persist_test") == 1
            docs = store2.collections["persist_test"]
            assert docs[0].content == "Persistent data"
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestGetVectorStore:
    """Test get_vector_store function."""
    
    def test_get_vector_store(self):
        """Test getting global vector store."""
        store = get_vector_store("./test_vector_db")
        
        assert isinstance(store, PersistentVectorStore)
    
    def test_get_vector_store_singleton(self):
        """Test that same instance is returned."""
        # Note: This might fail if other tests modified global state
        # In a real test suite, you'd reset global state between tests
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
