"""Unit tests for CacheStore.

Tests in this file should NOT use evaluate(). Evaluation behavior tests are in tests/behavior/ and
cross-subsystem scenarios are in tests/integration/.
"""

import pickle
import tempfile
import time
from pathlib import Path

from daglite.cache.store import CacheStore


class TestCacheStore:
    """Tests for CacheStore."""

    def test_init_with_string_path(self):
        """Test initialization with string path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = CacheStore(tmpdir)
            assert store._driver is not None

    def test_init_creates_directory(self):
        """Test that initialization creates the base directory via FileDriver."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = f"{tmpdir}/nested/path"
            _ = CacheStore(nested_path)
            assert Path(nested_path).exists()

    def test_put_and_get_basic_types(self):
        """Test putting and getting basic Python types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = CacheStore(tmpdir)

            # Put various types
            store.put("key1", 42)
            store.put("key2", "hello")
            store.put("key3", 3.14)
            store.put("key4", [1, 2, 3])

            # Get and verify
            assert store.get("key1") == 42
            assert store.get("key2") == "hello"
            assert store.get("key3") == 3.14
            assert store.get("key4") == [1, 2, 3]

    def test_put_with_none_value(self):
        """Test that None values can be cached."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = CacheStore(tmpdir)

            store.put("none_key", None)
            assert store.get("none_key") is None

    def test_get_nonexistent_returns_none(self):
        """Test that getting nonexistent key returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = CacheStore(tmpdir)
            assert store.get("nonexistent") is None

    def test_put_with_ttl_not_expired(self):
        """Test that cached value is returned when TTL has not expired."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = CacheStore(tmpdir)

            store.put("key", "value", ttl=10)  # 10 second TTL
            time.sleep(0.1)  # Small delay
            assert store.get("key") == "value"

    def test_put_with_ttl_expired(self):
        """Test that None is returned when TTL has expired."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = CacheStore(tmpdir)

            store.put("key", "value", ttl=1)  # 1 second TTL
            time.sleep(1.1)  # Wait for expiration
            assert store.get("key") is None

    def test_put_without_ttl_never_expires(self):
        """Test that cached values without TTL don't expire."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = CacheStore(tmpdir)

            store.put("key", "value")  # No TTL
            time.sleep(0.1)
            assert store.get("key") == "value"

    def test_git_style_sharding(self):
        """Test that cache files are sharded git-style (XX/YYYYYY...)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = CacheStore(tmpdir)

            hash_key = "abcdef1234567890"
            store.put(hash_key, "test_value")

            # Check that file exists in ab/cdef1234567890 format
            expected_dir = Path(tmpdir) / "ab"
            assert expected_dir.exists()

            data_file = Path(tmpdir) / "ab" / "cdef1234567890"
            assert data_file.exists()

    def test_metadata_file_created_with_ttl(self):
        """Test that metadata JSON file is created when TTL is specified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = CacheStore(tmpdir)

            hash_key = "abcdef1234567890"
            store.put(hash_key, "value", ttl=60)

            # Check that metadata file exists (.meta.json extension)
            metadata_path = Path(tmpdir) / "ab" / "cdef1234567890.meta.json"
            assert metadata_path.exists()

            # Verify metadata content
            import json

            metadata = json.loads(metadata_path.read_text())
            assert "timestamp" in metadata
            assert "ttl" in metadata
            assert metadata["ttl"] == 60
            assert isinstance(metadata["timestamp"], (int, float))

    def test_metadata_file_created_without_ttl(self):
        """Test that metadata file is still created even without TTL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = CacheStore(tmpdir)

            hash_key = "abcdef1234567890"
            store.put(hash_key, "value")  # No TTL

            # Metadata file still exists (with ttl=None)
            metadata_path = Path(tmpdir) / "ab" / "cdef1234567890.meta.json"
            assert metadata_path.exists()

    def test_invalidate_removes_cache_entry(self):
        """Test that invalidate removes both data and metadata files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = CacheStore(tmpdir)

            hash_key = "abcdef1234567890"
            store.put(hash_key, "value", ttl=60)

            # Verify files exist
            data_path = Path(tmpdir) / "ab" / "cdef1234567890"
            meta_path = Path(tmpdir) / "ab" / "cdef1234567890.meta.json"
            assert data_path.exists()
            assert meta_path.exists()

            # Invalidate
            store.invalidate(hash_key)

            # Verify files are removed
            assert not data_path.exists()
            assert not meta_path.exists()

    def test_invalidate_nonexistent_key_does_nothing(self):
        """Test that invalidating nonexistent key doesn't raise error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = CacheStore(tmpdir)

            # Should not raise
            store.invalidate("nonexistent")

    def test_clear_removes_all_cache_entries(self):
        """Test that clear removes all cached entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = CacheStore(tmpdir)

            # Add multiple entries
            store.put("key1", "value1")
            store.put("key2", "value2", ttl=60)
            store.put("key3", "value3")

            # Verify they exist
            assert store.get("key1") == "value1"
            assert store.get("key2") == "value2"
            assert store.get("key3") == "value3"

            # Clear all
            store.clear()

            # Verify all are gone
            assert store.get("key1") is None
            assert store.get("key2") is None
            assert store.get("key3") is None

    def test_serialization_for_distributed_backends(self):
        """Test that CacheStore can be pickled (for Ray, Dask, etc.)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = CacheStore(tmpdir)

            # Pickle and unpickle
            pickled = pickle.dumps(store)
            restored = pickle.loads(pickled)

            # Verify it still works
            restored.put("test_key", "test_value")
            assert restored.get("test_key") == "test_value"

    def test_complex_objects(self):
        """Test caching complex Python objects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = CacheStore(tmpdir)

            # Complex nested structure
            data = {
                "nested": {"list": [1, 2, 3], "tuple": (4, 5, 6)},
                "set": {7, 8, 9},
            }

            store.put("complex", data)
            retrieved = store.get("complex")

            assert retrieved == data
            assert isinstance(retrieved["nested"]["tuple"], tuple)  # type: ignore
            assert isinstance(retrieved["set"], set)  # type: ignore

    def test_realistic_hash_key_length(self):
        """Test with realistic hash key length (like SHA256)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = CacheStore(tmpdir)

            # Simulate realistic hash (64 chars for SHA256)
            hash_key = "a1b2c3d4e5f6" + "0" * 52
            store.put(hash_key, "test_value")
            assert store.get(hash_key) == "test_value"

            # Verify sharding structure
            shard_dir = Path(tmpdir) / "a1"
            assert shard_dir.exists()
            data_files = [
                f for f in shard_dir.iterdir() if f.is_file() and ".meta.json" not in f.name
            ]
            assert len(data_files) == 1
