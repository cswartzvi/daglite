"""Unit tests for CachePlugin.

Tests in this file should NOT use evaluate(). Evaluation tests are in tests/evaluation/.
"""

import tempfile
from unittest.mock import Mock

from daglite.cache.store import FileCacheStore
from daglite.plugins.builtin.cache import CachePlugin


class TestCachePluginInit:
    """Tests for CachePlugin initialization."""

    def test_init_with_cache_store(self):
        """Test initialization with CacheStore instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileCacheStore(tmpdir)
            plugin = CachePlugin(store=store)
            assert plugin.store is store


class TestCachePluginCheckCache:
    """Tests for CachePlugin.check_cache hook."""

    def test_cache_disabled_returns_none(self):
        """Test that check_cache returns None when caching is disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileCacheStore(tmpdir)
            plugin = CachePlugin(store=store)

            func = Mock(__name__="test_func")
            metadata = Mock(name="test_task")

            result = plugin.check_cache(
                func=func,
                metadata=metadata,
                inputs={"x": 1},
                cache_enabled=False,
                cache_ttl=None,
            )

            assert result is None

    def test_cache_miss_returns_none(self):
        """Test that check_cache returns None on cache miss."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileCacheStore(tmpdir)
            plugin = CachePlugin(store=store)

            def test_func(x):
                return x * 2

            metadata = Mock(name="test_task")

            result = plugin.check_cache(
                func=test_func,
                metadata=metadata,
                inputs={"x": 1},
                cache_enabled=True,
                cache_ttl=None,
            )

            assert result is None

    def test_cache_hit_returns_wrapped_value(self):
        """Test that check_cache returns wrapped value on cache hit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileCacheStore(tmpdir)
            plugin = CachePlugin(store=store)

            def test_func(x):
                return x * 2

            metadata = Mock(name="test_task")
            inputs = {"x": 1}

            # First store a value
            plugin.update_cache(
                func=test_func,
                metadata=metadata,
                inputs=inputs,
                result=42,
                cache_enabled=True,
                cache_ttl=None,
            )

            # Then check cache
            result = plugin.check_cache(
                func=test_func,
                metadata=metadata,
                inputs=inputs,
                cache_enabled=True,
                cache_ttl=None,
            )

            assert result == {"value": 42}

    def test_cache_hit_with_none_value(self):
        """Test that cached None values are properly returned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileCacheStore(tmpdir)
            plugin = CachePlugin(store=store)

            def test_func(x):
                return None

            metadata = Mock(name="test_task")
            inputs = {"x": 1}

            # Store None
            plugin.update_cache(
                func=test_func,
                metadata=metadata,
                inputs=inputs,
                result=None,
                cache_enabled=True,
                cache_ttl=None,
            )

            # Check cache
            result = plugin.check_cache(
                func=test_func,
                metadata=metadata,
                inputs=inputs,
                cache_enabled=True,
                cache_ttl=None,
            )

            # Should return wrapper with None value
            assert result == {"value": None}


class TestCachePluginUpdateCache:
    """Tests for CachePlugin.update_cache hook."""

    def test_cache_disabled_does_nothing(self):
        """Test that update_cache does nothing when caching is disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileCacheStore(tmpdir)
            plugin = CachePlugin(store=store)

            def test_func(x):
                return x * 2

            metadata = Mock(name="test_task")

            # Should not raise
            plugin.update_cache(
                func=test_func,
                metadata=metadata,
                inputs={"x": 1},
                result=42,
                cache_enabled=False,
                cache_ttl=None,
            )

            # Verify nothing was cached (check_cache should return None)
            result = plugin.check_cache(
                func=test_func,
                metadata=metadata,
                inputs={"x": 1},
                cache_enabled=True,
                cache_ttl=None,
            )
            assert result is None

    def test_stores_wrapped_value(self):
        """Test that update_cache stores wrapped result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileCacheStore(tmpdir)
            plugin = CachePlugin(store=store)

            def test_func(x):
                return x * 2

            metadata = Mock(name="test_task")
            inputs = {"x": 5}

            plugin.update_cache(
                func=test_func,
                metadata=metadata,
                inputs=inputs,
                result=10,
                cache_enabled=True,
                cache_ttl=60,
            )

            # Verify it was stored (by checking cache)
            result = plugin.check_cache(
                func=test_func,
                metadata=metadata,
                inputs=inputs,
                cache_enabled=True,
                cache_ttl=None,
            )

            assert result == {"value": 10}

    def test_different_inputs_produce_different_cache_keys(self):
        """Test that different inputs produce different cache entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileCacheStore(tmpdir)
            plugin = CachePlugin(store=store)

            def test_func(x):
                return x * 2

            metadata = Mock(name="test_task")

            # Store two different results
            plugin.update_cache(
                func=test_func,
                metadata=metadata,
                inputs={"x": 1},
                result=2,
                cache_enabled=True,
                cache_ttl=None,
            )

            plugin.update_cache(
                func=test_func,
                metadata=metadata,
                inputs={"x": 2},
                result=4,
                cache_enabled=True,
                cache_ttl=None,
            )

            # Verify both are cached separately
            result1 = plugin.check_cache(
                func=test_func,
                metadata=metadata,
                inputs={"x": 1},
                cache_enabled=True,
                cache_ttl=None,
            )
            result2 = plugin.check_cache(
                func=test_func,
                metadata=metadata,
                inputs={"x": 2},
                cache_enabled=True,
                cache_ttl=None,
            )

            assert result1 == {"value": 2}
            assert result2 == {"value": 4}


class TestCachePluginHooks:
    """Tests for CachePlugin hook implementations."""

    def test_has_check_cache_hook(self):
        """Test that check_cache hook is implemented."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileCacheStore(tmpdir)
            plugin = CachePlugin(store=store)
            assert hasattr(plugin, "check_cache")

    def test_has_update_cache_hook(self):
        """Test that update_cache hook is implemented."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileCacheStore(tmpdir)
            plugin = CachePlugin(store=store)
            assert hasattr(plugin, "update_cache")
