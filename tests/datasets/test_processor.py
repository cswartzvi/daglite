"""Unit tests for DatasetProcessor (subclass-specific behaviour only).

Base class lifecycle tests live in tests/test_processors.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from daglite.datasets.events import DatasetSaveRequest
from daglite.datasets.processor import DatasetProcessor


class TestDatasetProcessorRequestHandling:
    """_handle_request delegates to store.save with correct arguments."""

    def test_handle_item_delegates_to_handle_request(self) -> None:
        proc = DatasetProcessor()
        store = MagicMock()

        proc._handle_item(DatasetSaveRequest(key="via_item.pkl", value="item_data", store=store))

        store.save.assert_called_once_with("via_item.pkl", "item_data", format=None, options=None)

    def test_saves_to_store(self) -> None:
        proc = DatasetProcessor()
        store = MagicMock()

        proc._handle_request(
            DatasetSaveRequest(
                key="output.txt",
                value="text content",
                store=store,
                format="text",
                options={"encoding": "utf-8"},
            )
        )

        store.save.assert_called_once_with(
            "output.txt", "text content", format="text", options={"encoding": "utf-8"}
        )

    def test_error_logged_not_raised(self, caplog) -> None:
        import logging

        proc = DatasetProcessor()
        store = MagicMock()
        store.save.side_effect = RuntimeError("Save failed")

        with caplog.at_level(logging.ERROR):
            proc._handle_request(DatasetSaveRequest(key="bad.pkl", value="data", store=store))

        assert "Save failed" in caplog.text


class TestDatasetProcessorHooks:
    """before/after dataset save hooks around store.save."""

    def test_hooks_bracket_store_save(self) -> None:
        mock_hook = MagicMock()
        proc = DatasetProcessor(hook=mock_hook)
        store = MagicMock()
        order: list[str] = []

        mock_hook.before_dataset_save.side_effect = lambda **kw: order.append("before")
        store.save.side_effect = lambda *a, **kw: order.append("save")
        mock_hook.after_dataset_save.side_effect = lambda **kw: order.append("after")

        proc._handle_request(DatasetSaveRequest(key="k", value="v", store=store))

        assert order == ["before", "save", "after"]

    def test_after_hook_not_called_on_save_error(self) -> None:
        mock_hook = MagicMock()
        proc = DatasetProcessor(hook=mock_hook)
        store = MagicMock()
        store.save.side_effect = RuntimeError("write failed")

        proc._handle_request(DatasetSaveRequest(key="k", value="v", store=store))

        mock_hook.before_dataset_save.assert_called_once()
        mock_hook.after_dataset_save.assert_not_called()
