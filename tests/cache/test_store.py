"""Unit tests for FileCacheStore.

Tests in this file should NOT use evaluate(). Evaluation tests are in tests/evaluation/.
"""

import pickle
import tempfile
import time
from pathlib import Path

from daglite.cache.store import FileCacheStore


class TestFileCacheStore:
    """Tests for FileCacheStore."""

    def test_init_with_string_path(self):
        """Test initialization with string path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileCacheStore(tmpdir)
            assert store.base_path == tmpdir
            assert store.fs is not None

    def test_init_creates_directory(self):
        """Test that initialization creates the base directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = f"{tmpdir}/nested/path"
            _ = FileCacheStore(nested_path)
            assert Path(nested_path).exists()

    def test_put_and_get_basic_types(self):
        """Test putting and getting basic Python types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileCacheStore(tmpdir)

            # Put various types
            store.put("key1", 42)
            store.put("key2", "hello")
            store.put("key3", 3.14)
            store.put("key4", [1, 2, 3])

            # Get and verify
            assert store.get("key1", int) == 42
            assert store.get("key2", str) == "hello"
            assert store.get("key3", float) == 3.14
            assert store.get("key4", list) == [1, 2, 3]

    def test_put_with_none_value(self):
        """Test that None values can be cached."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileCacheStore(tmpdir)

            store.put("none_key", None)
            assert store.get("none_key", type(None)) is None

    def test_get_nonexistent_returns_none(self):
        """Test that getting nonexistent key returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileCacheStore(tmpdir)
            assert store.get("nonexistent", str) is None

    def test_put_with_ttl_not_expired(self):
        """Test that cached value is returned when TTL has not expired."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileCacheStore(tmpdir)

            store.put("key", "value", ttl=10)  # 10 second TTL
            time.sleep(0.1)  # Small delay
            assert store.get("key", str) == "value"

    def test_put_with_ttl_expired(self):
        """Test that None is returned when TTL has expired."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileCacheStore(tmpdir)

            store.put("key", "value", ttl=1)  # 1 second TTL
            time.sleep(1.1)  # Wait for expiration
            assert store.get("key", str) is None

    def test_put_without_ttl_never_expires(self):
        """Test that cached values without TTL don't expire."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileCacheStore(tmpdir)

            store.put("key", "value")  # No TTL
            time.sleep(0.1)
            assert store.get("key", str) == "value"

    def test_git_style_sharding(self):
        """Test that cache files are sharded git-style (XX/YYYYYY...)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileCacheStore(tmpdir)

            hash_key = "abcdef1234567890"
            store.put(hash_key, "test_value")

            # Check that file exists in ab/cdef1234567890 format (no extension)
            expected_dir = Path(tmpdir) / "ab"
            assert expected_dir.exists()

            data_file = Path(tmpdir) / "ab" / "cdef1234567890"
            assert data_file.exists()

    def test_metadata_file_created_with_ttl(self):
        """Test that metadata JSON file is created when TTL is specified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileCacheStore(tmpdir)

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

    def test_no_metadata_file_without_ttl(self):
        """Test that metadata file is still created even without TTL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileCacheStore(tmpdir)

            hash_key = "abcdef1234567890"
            store.put(hash_key, "value")  # No TTL

            # Metadata file still exists (with ttl=None)
            metadata_path = Path(tmpdir) / "ab" / "cdef1234567890.meta.json"
            assert metadata_path.exists()

    def test_invalidate_removes_cache_entry(self):
        """Test that invalidate removes both data and metadata files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileCacheStore(tmpdir)

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
            store = FileCacheStore(tmpdir)

            # Should not raise
            store.invalidate("nonexistent")

    def test_clear_removes_all_cache_entries(self):
        """Test that clear removes all cached entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileCacheStore(tmpdir)

            # Add multiple entries
            store.put("key1", "value1")
            store.put("key2", "value2", ttl=60)
            store.put("key3", "value3")

            # Verify they exist
            assert store.get("key1", str) == "value1"
            assert store.get("key2", str) == "value2"
            assert store.get("key3", str) == "value3"

            # Clear all
            store.clear()

            # Verify all are gone
            assert store.get("key1", str) is None
            assert store.get("key2", str) is None
            assert store.get("key3", str) is None

            # Verify directory structure is cleaned
            cache_files = [
                f for f in Path(tmpdir).rglob("*") if f.is_file() and ".meta.json" not in f.name
            ]
            assert len(cache_files) == 0

    def test_serialization_for_distributed_backends(self):
        """Test that FileCacheStore can be pickled (for Ray, Dask, etc.)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileCacheStore(tmpdir)

            # Pickle and unpickle
            pickled = pickle.dumps(store)
            restored = pickle.loads(pickled)

            # Verify it still works
            assert restored.base_path == tmpdir
            restored.put("test_key", "test_value")
            assert restored.get("test_key", str) == "test_value"

    def test_complex_objects(self):
        """Test caching complex Python objects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileCacheStore(tmpdir)

            # Complex nested structure
            data = {
                "nested": {"list": [1, 2, 3], "tuple": (4, 5, 6)},
                "set": {7, 8, 9},
            }

            store.put("complex", data)
            retrieved = store.get("complex", dict)

            assert retrieved == data
            assert isinstance(retrieved["nested"]["tuple"], tuple)  # type: ignore
            assert isinstance(retrieved["set"], set)  # type: ignore

    def test_realistic_hash_key_length(self):
        """Test with realistic hash key length (like SHA256)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileCacheStore(tmpdir)

            # Simulate realistic hash (64 chars for SHA256)
            hash_key = "a1b2c3d4e5f6" + "0" * 52
            store.put(hash_key, "test_value")
            assert store.get(hash_key, str) == "test_value"

            # Verify sharding structure
            shard_dir = Path(tmpdir) / "a1"
            assert shard_dir.exists()
            data_files = [
                f for f in shard_dir.iterdir() if f.is_file() and ".meta.json" not in f.name
            ]
            assert len(data_files) == 1
