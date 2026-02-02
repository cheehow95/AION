"""Tests for AION AST Cache"""

import pytest
import sys
sys.path.insert(0, '.')

from src.ast_cache import ASTCache, cached_parse, get_cache


class TestASTCache:
    """Test cases for the AST caching system."""
    
    def test_init(self):
        """Test cache initialization."""
        cache = ASTCache(max_size=50)
        assert cache.max_size == 50
        assert len(cache.cache) == 0
        assert cache.stats['hits'] == 0
        assert cache.stats['misses'] == 0
    
    def test_put_and_get(self):
        """Test storing and retrieving ASTs."""
        cache = ASTCache()
        
        # Create a mock AST
        mock_ast = {'type': 'Program', 'declarations': []}
        source = "agent Test {}"
        
        # Put in cache
        cache.put(source, mock_ast)
        
        # Get from cache
        result = cache.get(source)
        assert result is not None
        assert result == mock_ast
        assert cache.stats['hits'] == 1
    
    def test_cache_miss(self):
        """Test cache miss behavior."""
        cache = ASTCache()
        
        result = cache.get("nonexistent source")
        assert result is None
        assert cache.stats['misses'] == 1
    
    def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        import time
        cache = ASTCache(max_size=3)
        
        # Fill the cache with delays to ensure distinct timestamps
        cache.put("source1", {'ast': 1})
        time.sleep(0.01)
        cache.put("source2", {'ast': 2})
        time.sleep(0.01)
        cache.put("source3", {'ast': 3})
        
        # Access source1 to make it recently used
        cache.get("source1")
        time.sleep(0.01)
        
        # Add new item - should evict source2 (least recently used)
        cache.put("source4", {'ast': 4})
        
        assert cache.get("source1") is not None  # Still present
        assert cache.get("source3") is not None  # Still present
        assert cache.get("source4") is not None  # Newly added
        assert cache.stats['evictions'] >= 1
    
    def test_clear(self):
        """Test cache clearing."""
        cache = ASTCache()
        cache.put("source", {'ast': 1})
        
        cache.clear()
        
        assert len(cache.cache) == 0
        assert cache.get("source") is None
    
    def test_hit_rate(self):
        """Test hit rate calculation."""
        cache = ASTCache()
        
        # Initially zero
        assert cache.hit_rate == 0.0
        
        # Add and hit
        cache.put("source", {'ast': 1})
        cache.get("source")  # hit
        cache.get("source")  # hit
        cache.get("other")   # miss
        
        # 2 hits, 1 miss = 2/3 hit rate
        assert cache.hit_rate == pytest.approx(2/3)
    
    def test_hash_consistency(self):
        """Test that same source produces same hash."""
        cache = ASTCache()
        
        source = "agent Test { goal \"Help\" }"
        hash1 = cache._hash(source)
        hash2 = cache._hash(source)
        
        assert hash1 == hash2
    
    def test_different_source_different_hash(self):
        """Test that different sources produce different hashes."""
        cache = ASTCache()
        
        hash1 = cache._hash("agent A {}")
        hash2 = cache._hash("agent B {}")
        
        assert hash1 != hash2


class TestCachedParse:
    """Test the cached_parse function."""
    
    def test_cached_parse_returns_ast(self):
        """Test that cached_parse returns valid AST."""
        # Clear global cache first
        get_cache().clear()
        
        source = "agent Test { goal \"Testing\" }"
        result = cached_parse(source)
        
        assert result is not None
        assert hasattr(result, 'declarations')
    
    def test_cached_parse_caches_result(self):
        """Test that second parse uses cache."""
        get_cache().clear()
        
        source = "agent CacheTest { goal \"Test caching\" }"
        
        # First parse - miss
        result1 = cached_parse(source)
        initial_misses = get_cache().stats['misses']
        
        # Second parse - hit
        result2 = cached_parse(source)
        
        assert get_cache().stats['hits'] > 0
        assert result1.declarations[0].name == result2.declarations[0].name
    
    def test_get_cache(self):
        """Test global cache access."""
        cache = get_cache()
        assert isinstance(cache, ASTCache)
        
        # Should return same instance
        cache2 = get_cache()
        assert cache is cache2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
