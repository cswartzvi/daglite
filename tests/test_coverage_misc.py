"""Tests targeting uncovered lines in cache, workflows, mapping, and datasets.

Covers:
- cache/core.py: _canonical branches (dict, set, list, tuple) — lines 91, 93, 95, 97
- cache/store.py: is_local, corrupted metadata, corrupted data, _hash_to_key validation
- workflows.py: get_typed_params exception, SyncWorkflow.__call__, AsyncWorkflow non-awaitable
- mapping.py: _resolve_backend fallback — line 161
- datasets/store.py: _resolve_key when no task args — line 25
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from daglite.cache.core import _canonical
from daglite.cache.core import default_cache_hash
from daglite.cache.store import CACHE_MISS
from daglite.cache.store import CacheStore

# region cache/core.py — _canonical branches


class TestCanonical:
    """_canonical normalizes dicts, sets, lists, and tuples for stable hashing."""

    def test_dict(self) -> None:
        result = _canonical({"b": 2, "a": 1})
        assert result == [("a", 1), ("b", 2)]

    def test_set(self) -> None:
        result = _canonical({3, 1, 2})
        assert result == [1, 2, 3]

    def test_frozenset(self) -> None:
        result = _canonical(frozenset([3, 1, 2]))
        assert result == [1, 2, 3]

    def test_list(self) -> None:
        result = _canonical([3, 1, 2])
        assert result == [3, 1, 2]

    def test_tuple(self) -> None:
        result = _canonical((3, 1, 2))
        assert result == (3, 1, 2)

    def test_nested(self) -> None:
        result = _canonical({"x": [1, {2, 3}]})
        assert isinstance(result, list)

    def test_passthrough(self) -> None:
        assert _canonical(42) == 42
        assert _canonical("hello") == "hello"


class TestDefaultCacheHashWithContainers:
    """default_cache_hash exercises _canonical through varied arg types."""

    def test_dict_args(self) -> None:
        def fn(d: dict) -> None:
            pass

        h1 = default_cache_hash(fn, {"d": {"b": 2, "a": 1}})
        h2 = default_cache_hash(fn, {"d": {"a": 1, "b": 2}})
        assert h1 == h2

    def test_set_args(self) -> None:
        def fn(s: set) -> None:
            pass

        h = default_cache_hash(fn, {"s": {3, 1, 2}})
        assert isinstance(h, str)

    def test_list_args(self) -> None:
        def fn(lst: list) -> None:
            pass

        h = default_cache_hash(fn, {"lst": [1, 2, 3]})
        assert isinstance(h, str)

    def test_tuple_args(self) -> None:
        def fn(t: tuple) -> None:
            pass

        h = default_cache_hash(fn, {"t": (1, 2, 3)})
        assert isinstance(h, str)


# region cache/store.py — edge cases


class TestCacheStoreIsLocal:
    """CacheStore.is_local property."""

    def test_is_local_with_file_driver(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = CacheStore(td)
            assert store.is_local is True

    def test_is_local_with_custom_driver(self) -> None:
        driver = MagicMock()
        driver.is_local = False
        store = CacheStore.__new__(CacheStore)
        store._driver = driver
        assert store.is_local is False


class TestCacheStoreCorruptedData:
    """CacheStore.get handles corrupted metadata and data."""

    def test_corrupted_metadata_returns_miss(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = CacheStore(td)
            hash_key = "aabbccdd11223344"
            store.put(hash_key, 42)

            # Corrupt the metadata file
            data_key = store._hash_to_key(hash_key)
            meta_path = Path(td) / f"{data_key}.meta.json"
            meta_path.write_text("not valid json !!!")

            assert store.get(hash_key) is CACHE_MISS

    def test_corrupted_data_returns_miss(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = CacheStore(td)
            hash_key = "aabbccdd55667788"
            store.put(hash_key, 42)

            # Corrupt the data file
            data_key = store._hash_to_key(hash_key)
            data_path = Path(td) / data_key
            data_path.write_bytes(b"not a pickle")

            assert store.get(hash_key) is CACHE_MISS


class TestCacheStoreHashValidation:
    """_hash_to_key validates short and non-hex keys."""

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


# region workflows.py


class TestWorkflowCoverage:
    """Workflow gaps: get_typed_params exception, __call__, async non-awaitable."""

    def test_sync_workflow_call(self) -> None:
        from daglite.tasks import task as eager_task
        from daglite.workflows import workflow

        @eager_task
        def add(x: int, y: int) -> int:
            return x + y

        @workflow
        def my_wf(x: int, y: int) -> int:
            return add(x=x, y=y)

        result = my_wf(2, 3)
        assert result == 5

    def test_async_workflow_non_awaitable(self) -> None:
        """AsyncWorkflow._run with a sync function returns result directly."""
        from daglite.workflows import AsyncWorkflow

        def sync_fn(x: int) -> int:
            return x * 2

        wf = AsyncWorkflow(func=sync_fn, name="test", description="")
        result = asyncio.run(wf(3))
        assert result == 6

    def test_get_typed_params_exception(self) -> None:
        """get_typed_params falls back to empty hints on exception."""

        from daglite.workflows import SyncWorkflow

        def normal_fn(x: int) -> None:
            pass

        wf = SyncWorkflow(func=normal_fn, name="test", description="")

        # Force get_type_hints to raise
        with patch("daglite.workflows.typing.get_type_hints", side_effect=Exception("fail")):
            params = wf.get_typed_params()
        assert params == {"x": None}


# region mapping.py


class TestResolveBackendFallback:
    """_resolve_backend returns 'inline' when both backend and ctx are None."""

    def test_fallback_to_inline(self) -> None:
        from daglite.mapping import _resolve_backend

        assert _resolve_backend(None, None) == "inline"


# region datasets/store.py


class TestDatasetStoreResolveKey:
    """_resolve_key returns key unchanged when no task args are available."""

    def test_no_task_args(self) -> None:
        from daglite.datasets.store import _resolve_key

        assert _resolve_key("my_dataset") == "my_dataset"

    def test_with_placeholder_no_args(self) -> None:
        from daglite.datasets.store import _resolve_key

        # No active task context, so placeholders stay unresolved (returns key as-is)
        result = _resolve_key("{split}_data")
        assert result == "{split}_data"
