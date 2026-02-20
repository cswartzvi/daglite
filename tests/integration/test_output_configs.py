"""
Integration tests for output config saving behaviour.

Replaces the white-box ``test_save_outputs.py`` with end-to-end tests that
exercise the same paths through the public API.
"""

from __future__ import annotations

import tempfile

import pytest

from daglite import task
from daglite.datasets.store import DatasetStore
from daglite.settings import DagliteSettings
from daglite.settings import set_global_settings


class TestSettingsFallbackStore:
    """When no explicit store is passed, the global settings store is used."""

    def test_settings_string_store_saves_result(self) -> None:
        """A string datastore_store path in global settings is used as fallback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            set_global_settings(DagliteSettings(datastore_store=tmpdir))
            try:

                @task
                def compute(x: int) -> int:
                    return x * 2

                compute(x=21).save("fallback_result.pkl").run()
                store = DatasetStore(tmpdir)
                assert store.exists("fallback_result.pkl")
                assert store.load("fallback_result.pkl", return_type=int) == 42
            finally:
                set_global_settings(DagliteSettings())

    def test_settings_store_instance_saves_result(self) -> None:
        """A DatasetStore instance in global settings is used as fallback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)
            set_global_settings(DagliteSettings(datastore_store=store))
            try:

                @task
                def add(a: int, b: int) -> int:
                    return a + b

                add(a=3, b=4).save("store_instance_{a}.pkl").run()
                assert store.exists("store_instance_3.pkl")
                assert store.load("store_instance_3.pkl", return_type=int) == 7
            finally:
                set_global_settings(DagliteSettings())


class TestMissingPlaceholderError:
    """A key template referencing an unknown variable raises ValueError."""

    def test_missing_placeholder_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)

            @task
            def process(data_id: str) -> str:
                return f"processed_{data_id}"

            with pytest.raises(ValueError, match="won't be available at runtime"):
                process(data_id="abc").save("output_{missing}.txt", save_store=store).run()
