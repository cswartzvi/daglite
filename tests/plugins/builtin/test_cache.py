"""Unit tests for built-in cache store integration.

Tests in this file should NOT use evaluate(). Evaluation behavior tests are in tests/behavior/ and
cross-subsystem scenarios are in tests/integration/.
"""

import tempfile

from daglite.cache.core import CACHE_MISS
from daglite.cache.core import CacheMiss
from daglite.cache.core import default_cache_hash
from daglite.cache.store import CacheStore


class TestCacheStoreGetPut:
    """Tests for CacheStore get/put operations."""

    def test_cache_miss_returns_cache_miss(self):
        """Test that get returns CACHE_MISS on a cache miss."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = CacheStore(tmpdir)

            def test_func(x):
                return x * 2

            cache_key = default_cache_hash(test_func, {"x": 1})
            result = store.get(cache_key)
            assert isinstance(result, CacheMiss)
            assert result is CACHE_MISS

    def test_cache_hit_returns_value(self):
        """Test that get returns value on cache hit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = CacheStore(tmpdir)

            def test_func(x):
                return x * 2

            cache_key = default_cache_hash(test_func, {"x": 1})

            store.put(cache_key, 42)

            result = store.get(cache_key)
            assert result == 42

    def test_cache_hit_with_none_value(self):
        """Test that cached None values are properly returned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = CacheStore(tmpdir)

            def test_func(x):
                return None

            cache_key = default_cache_hash(test_func, {"x": 1})

            store.put(cache_key, None)

            result = store.get(cache_key)
            assert result is None

    def test_different_inputs_produce_different_cache_keys(self):
        """Test that different inputs produce different cache entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = CacheStore(tmpdir)

            def test_func(x):
                return x * 2

            # Store two different results
            key1 = default_cache_hash(test_func, {"x": 1})
            key2 = default_cache_hash(test_func, {"x": 2})

            store.put(key1, 2)
            store.put(key2, 4)

            # Verify both are cached separately
            assert store.get(key1) == 2
            assert store.get(key2) == 4


class TestDefaultCacheHash:
    """Tests for the default_cache_hash function."""

    def test_same_func_same_inputs_same_hash(self):
        """Test that identical function and inputs produce the same hash."""

        def test_func(x):
            return x * 2

        hash1 = default_cache_hash(test_func, {"x": 1})
        hash2 = default_cache_hash(test_func, {"x": 1})
        assert hash1 == hash2

    def test_different_inputs_different_hash(self):
        """Test that different inputs produce different hashes."""

        def test_func(x):
            return x * 2

        hash1 = default_cache_hash(test_func, {"x": 1})
        hash2 = default_cache_hash(test_func, {"x": 2})
        assert hash1 != hash2

    def test_different_functions_different_hash(self):
        """Test that different functions produce different hashes."""

        def func_a(x):
            return x * 2

        def func_b(x):
            return x * 3

        hash1 = default_cache_hash(func_a, {"x": 1})
        hash2 = default_cache_hash(func_b, {"x": 1})
        assert hash1 != hash2
