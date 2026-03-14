"""Unit tests for the cache subsystem."""

from __future__ import annotations

import pickle
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from daglite.cache.core import CACHE_MISS
from daglite.cache.core import _canonical
from daglite.cache.core import default_cache_hash
from daglite.cache.store import CacheStore


class TestCacheStorePutGet:
    """Basic put and get operations."""

    def test_basic_types(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = CacheStore(td)
            store.put("aabb001122", 42)
            store.put("aabb334455", "hello")
            store.put("aabb667788", [1, 2, 3])

            assert store.get("aabb001122") == 42
            assert store.get("aabb334455") == "hello"
            assert store.get("aabb667788") == [1, 2, 3]

    def test_none_value(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = CacheStore(td)
            store.put("ffee000000", None)
            assert store.get("ffee000000") is None

    def test_complex_objects(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = CacheStore(td)
            data = {"nested": {"list": [1, 2, 3], "tuple": (4, 5, 6)}, "set": {7, 8, 9}}
            store.put("c0ffee1234", data)
            retrieved = store.get("c0ffee1234")
            assert retrieved == data

    def test_nonexistent_returns_cache_miss(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = CacheStore(td)
            assert store.get("deadbeef01") is CACHE_MISS


class TestCacheStoreTTL:
    """Time-to-live expiration behaviour."""

    def test_not_expired(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = CacheStore(td)
            store.put("cafebabe01", "value", ttl=10)
            assert store.get("cafebabe01") == "value"

    def test_expired(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = CacheStore(td)
            store.put("cafebabe01", "value", ttl=1)
            time.sleep(1.1)
            assert store.get("cafebabe01") is CACHE_MISS


class TestCacheStoreSharding:
    """Git-style two-character prefix sharding and metadata files."""

    def test_directory_layout(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = CacheStore(td)
            store.put("abcdef1234567890", "test_value")

            assert (Path(td) / "ab").exists()
            assert (Path(td) / "ab" / "cdef1234567890").exists()

    def test_metadata_json_with_ttl(self) -> None:
        import json

        with tempfile.TemporaryDirectory() as td:
            store = CacheStore(td)
            store.put("abcdef1234567890", "value", ttl=60)

            meta = json.loads((Path(td) / "ab" / "cdef1234567890.meta.json").read_text())
            assert meta["ttl"] == 60
            assert isinstance(meta["timestamp"], (int, float))


class TestCacheStoreInvalidate:
    """Invalidate removes entries; nonexistent keys are no-ops."""

    def test_removes_entry(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = CacheStore(td)
            store.put("abcdef1234567890", "value", ttl=60)
            store.invalidate("abcdef1234567890")

            assert not (Path(td) / "ab" / "cdef1234567890").exists()
            assert not (Path(td) / "ab" / "cdef1234567890.meta.json").exists()

    def test_nonexistent_is_noop(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = CacheStore(td)
            store.invalidate("deadbeef01")  # should not raise


class TestCacheStoreClear:
    def test_removes_all_entries(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = CacheStore(td)
            store.put("aabb001122", "v1")
            store.put("aabb334455", "v2")
            store.clear()

            assert store.get("aabb001122") is CACHE_MISS
            assert store.get("aabb334455") is CACHE_MISS


class TestCacheStorePickle:
    def test_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = CacheStore(td)
            restored = pickle.loads(pickle.dumps(store))
            restored.put("abcdef0011", "test_value")
            assert restored.get("abcdef0011") == "test_value"


class TestCacheStoreIsLocal:
    def test_file_driver_is_local(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = CacheStore(td)
            assert store.is_local is True

    def test_custom_driver_not_local(self) -> None:
        driver = MagicMock()
        driver.is_local = False
        store = CacheStore.__new__(CacheStore)
        store._driver = driver
        assert store.is_local is False


class TestCacheStoreCorruption:
    def test_corrupted_metadata_returns_miss(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = CacheStore(td)
            hash_key = "aabbccdd11223344"
            store.put(hash_key, 42)

            data_key = store._hash_to_key(hash_key)
            (Path(td) / f"{data_key}.meta.json").write_text("not valid json !!!")
            assert store.get(hash_key) is CACHE_MISS

    def test_corrupted_data_returns_miss(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = CacheStore(td)
            hash_key = "aabbccdd55667788"
            store.put(hash_key, 42)

            data_key = store._hash_to_key(hash_key)
            (Path(td) / data_key).write_bytes(b"not a pickle")
            assert store.get(hash_key) is CACHE_MISS


class TestCacheStoreKeyValidation:
    def test_short_key_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = CacheStore(td)
            with pytest.raises(ValueError, match="at least 3 characters"):
                store._hash_to_key("ab")

    def test_non_hex_key_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = CacheStore(td)
            with pytest.raises(ValueError, match="hexadecimal digest"):
                store._hash_to_key("gghhii")


class TestCanonical:
    """``_canonical`` normalizes containers for stable hashing."""

    def test_dict_sorted(self) -> None:
        assert _canonical({"b": 2, "a": 1}) == [("a", 1), ("b", 2)]

    def test_set_sorted(self) -> None:
        assert _canonical({3, 1, 2}) == [1, 2, 3]

    def test_list_preserved(self) -> None:
        assert _canonical([3, 1, 2]) == [3, 1, 2]

    def test_tuple_preserved(self) -> None:
        assert _canonical((3, 1, 2)) == (3, 1, 2)

    def test_nested(self) -> None:
        result = _canonical({"x": [1, {2, 3}]})
        assert isinstance(result, list)


class TestDefaultCacheHash:
    """``default_cache_hash`` produces stable, input-sensitive digests."""

    def test_same_inputs_same_hash(self) -> None:
        def fn(x):
            return x * 2

        assert default_cache_hash(fn, {"x": 1}) == default_cache_hash(fn, {"x": 1})

    def test_different_inputs_different_hash(self) -> None:
        def fn(x):
            return x * 2

        assert default_cache_hash(fn, {"x": 1}) != default_cache_hash(fn, {"x": 2})

    def test_different_functions_different_hash(self) -> None:
        def fa(x):
            return x * 2

        def fb(x):
            return x * 3

        assert default_cache_hash(fa, {"x": 1}) != default_cache_hash(fb, {"x": 1})
