"""Unit tests for context-chain resolution helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from daglite._resolvers import resolve_cache_store
from daglite._resolvers import resolve_dataset_store
from daglite.cache.store import CacheStore
from daglite.datasets.store import DatasetStore


class TestResolveCacheStore:
    """``resolve_cache_store`` resolves various input types."""

    def test_cache_store_instance_returned(self, tmp_path: Path) -> None:
        store = CacheStore(str(tmp_path))
        assert resolve_cache_store(cache=store) is store

    def test_cache_string_creates_store(self, tmp_path: Path) -> None:
        result = resolve_cache_store(cache=str(tmp_path))
        assert isinstance(result, CacheStore)

    def test_cache_store_kwarg_instance_returned(self, tmp_path: Path) -> None:
        store = CacheStore(str(tmp_path))
        result = resolve_cache_store(cache=True, cache_store=store)
        assert result is store

    def test_cache_store_kwarg_string_creates_store(self, tmp_path: Path) -> None:
        result = resolve_cache_store(cache=True, cache_store=str(tmp_path))
        assert isinstance(result, CacheStore)


class TestResolveDatasetStore:
    """``resolve_dataset_store`` resolves store from arguments or context."""

    def test_no_store_anywhere_raises(self) -> None:
        mock_settings = type("S", (), {"dataset_store": None})()
        with patch("daglite.settings.get_global_settings", return_value=mock_settings):
            with pytest.raises(RuntimeError, match="No dataset store available"):
                resolve_dataset_store(store=None)

    def test_fallback_to_settings_string(self, tmp_path: Path) -> None:
        mock_settings = type("S", (), {"dataset_store": str(tmp_path)})()
        with patch("daglite.settings.get_global_settings", return_value=mock_settings):
            result = resolve_dataset_store(store=None)
        assert isinstance(result, DatasetStore)
