"""Unit tests for BackendContext serialization round-trip and JSON safety."""

from __future__ import annotations

import json

import pytest

from daglite._context import BackendContext
from daglite.cache.store import CacheStore
from daglite.datasets.store import DatasetStore


@pytest.fixture()
def tmp_cache(tmp_path):
    return CacheStore(str(tmp_path / "cache"))


@pytest.fixture()
def tmp_dataset(tmp_path):
    return DatasetStore(str(tmp_path / "datasets"))


class TestBackendContextSerialization:
    """BackendContext.to_dict() / from_dict() round-trip."""

    def test_to_dict_is_json_serializable(self, tmp_cache, tmp_dataset):
        ctx = BackendContext(
            backend="process",
            cache_store=tmp_cache,
            dataset_store=tmp_dataset,
        )
        data = ctx.to_dict()
        # Must not raise
        json.dumps(data)

    def test_to_dict_strips_reporters(self, tmp_cache):
        from unittest.mock import MagicMock

        ctx = BackendContext(
            backend="process",
            event_reporter=MagicMock(),
            dataset_reporter=MagicMock(),
            cache_store=tmp_cache,
        )
        data = ctx.to_dict()
        assert "event_reporter" not in data
        assert "dataset_reporter" not in data

    def test_to_dict_serializes_stores_as_paths(self, tmp_path):
        cache_path = str(tmp_path / "cache")
        ds_path = str(tmp_path / "datasets")
        ctx = BackendContext(
            backend="thread",
            cache_store=CacheStore(cache_path),
            dataset_store=DatasetStore(ds_path),
        )
        data = ctx.to_dict()
        assert data["cache_store"] == cache_path
        assert data["dataset_store"] == ds_path

    def test_to_dict_none_stores_stay_none(self):
        ctx = BackendContext(backend="inline")
        data = ctx.to_dict()
        assert data["cache_store"] is None
        assert data["dataset_store"] is None

    def test_round_trip(self, tmp_path):
        cache_path = str(tmp_path / "cache")
        ds_path = str(tmp_path / "datasets")
        original = BackendContext(
            backend="process",
            cache_store=CacheStore(cache_path),
            dataset_store=DatasetStore(ds_path),
            map_index=3,
        )
        data = original.to_dict()

        restored = BackendContext.from_dict(data)

        assert restored.backend == "process"
        assert restored.map_index == 3
        assert restored.event_reporter is None
        assert restored.dataset_reporter is None
        assert isinstance(restored.cache_store, CacheStore)
        assert isinstance(restored.dataset_store, DatasetStore)
        assert restored.cache_store.base_path == cache_path
        assert restored.dataset_store.base_path == ds_path

    def test_round_trip_with_reporters(self, tmp_path):
        from unittest.mock import MagicMock

        cache_path = str(tmp_path / "cache")
        original = BackendContext(
            backend="process",
            cache_store=CacheStore(cache_path),
        )
        data = original.to_dict()

        mock_event = MagicMock()
        mock_dataset = MagicMock()
        restored = BackendContext.from_dict(
            data, event_reporter=mock_event, dataset_reporter=mock_dataset
        )

        assert restored.event_reporter is mock_event
        assert restored.dataset_reporter is mock_dataset

    def test_round_trip_with_plugin_manager(self, tmp_path):
        from daglite.plugins.events import EventRegistry
        from daglite.plugins.manager import build_plugin_manager

        pm = build_plugin_manager(plugins=[], registry=EventRegistry())
        ctx = BackendContext(
            backend="process",
            plugin_manager=pm,
        )
        data = ctx.to_dict()

        # Plugin manager is serialized as a dict
        assert isinstance(data["plugin_manager"], dict)

        # JSON-safe
        json.dumps(data)

        restored = BackendContext.from_dict(data)
        assert restored.plugin_manager is not None

    def test_round_trip_none_stores(self):
        ctx = BackendContext(backend="inline")
        data = ctx.to_dict()
        restored = BackendContext.from_dict(data)
        assert restored.cache_store is None
        assert restored.dataset_store is None

    def test_json_round_trip(self, tmp_path):
        """Verify the dict survives JSON serialization (simulating machine boundaries)."""
        cache_path = str(tmp_path / "cache")
        ds_path = str(tmp_path / "datasets")
        ctx = BackendContext(
            backend="process",
            cache_store=CacheStore(cache_path),
            dataset_store=DatasetStore(ds_path),
            map_index=5,
        )
        data = ctx.to_dict()

        # Simulate JSON wire transfer
        wire = json.loads(json.dumps(data))

        restored = BackendContext.from_dict(wire)
        assert restored.backend == "process"
        assert restored.map_index == 5
        assert isinstance(restored.cache_store, CacheStore)
        assert isinstance(restored.dataset_store, DatasetStore)


class TestContextGuards:
    """Double-enter and exit-without-enter raise RuntimeError."""

    def test_double_enter_raises(self) -> None:
        ctx = BackendContext(backend="inline")
        ctx.__enter__()
        try:
            with pytest.raises(RuntimeError, match="already entered"):
                ctx.__enter__()
        finally:
            ctx.__exit__(None, None, None)

    def test_exit_without_enter_raises(self) -> None:
        ctx = BackendContext(backend="inline")
        with pytest.raises(RuntimeError, match="exit called without"):
            ctx.__exit__(None, None, None)
