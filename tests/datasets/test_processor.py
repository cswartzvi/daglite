"""Unit tests for DatasetProcessor."""

import queue
import time
from multiprocessing import Queue as MpQueue
from unittest.mock import MagicMock

from daglite.datasets.events import DatasetSaveRequest
from daglite.datasets.processor import DatasetProcessor


class TestDatasetProcessorSources:
    """Tests for source management."""

    def test_add_source_returns_uuid(self):
        proc = DatasetProcessor()
        queue = MpQueue()
        try:
            source_id = proc.add_source(queue)
            assert source_id is not None
            assert source_id in proc._sources
        finally:
            queue.close()
            queue.join_thread()

    def test_remove_source(self):
        proc = DatasetProcessor()
        queue = MpQueue()
        try:
            source_id = proc.add_source(queue)
            proc.remove_source(source_id)
            assert source_id not in proc._sources
        finally:
            queue.close()
            queue.join_thread()

    def test_remove_unknown_source_logs_warning(self, caplog):
        """Removing a non-existent source logs a warning."""
        from uuid import uuid4

        proc = DatasetProcessor()
        fake_id = uuid4()

        import logging

        with caplog.at_level(logging.WARNING):
            proc.remove_source(fake_id)

        assert "unknown" in caplog.text.lower() or "Tried to remove" in caplog.text


class TestDatasetProcessorLifecycle:
    """Tests for start/stop/flush lifecycle."""

    def test_start_and_stop(self):
        """Processor starts and stops cleanly."""
        proc = DatasetProcessor()
        proc.start()
        assert proc._thread is not None
        assert proc._running is True
        proc.stop()
        assert proc._thread is None
        assert proc._running is False

    def test_double_start_logs_warning(self, caplog):
        """Starting an already-started processor logs a warning."""
        import logging

        proc = DatasetProcessor()
        proc.start()
        try:
            with caplog.at_level(logging.WARNING):
                proc.start()
            assert "already started" in caplog.text.lower()
        finally:
            proc.stop()

    def test_stop_without_start_is_noop(self):
        """Stopping a never-started processor is safe."""
        proc = DatasetProcessor()
        proc.stop()  # Should not raise

    def test_flush_drains_queue(self):
        """flush() processes pending save requests."""
        proc = DatasetProcessor()
        q = queue.Queue()
        store = MagicMock()

        proc.add_source(q)

        # Put a save request on the queue
        q.put(
            DatasetSaveRequest(
                key="test.pkl",
                value={"data": 1},
                store=store,
                format="pickle",
                options=None,
            )
        )

        # Flush without starting background thread
        proc.flush(timeout=2.0)

        store.save.assert_called_once_with("test.pkl", {"data": 1}, format="pickle", options=None)

    def test_flush_timeout(self):
        """flush() respects the timeout."""
        proc = DatasetProcessor()
        # No sources, so flush should return quickly
        start = time.time()
        proc.flush(timeout=0.1)
        elapsed = time.time() - start
        assert elapsed < 1.0


class TestDatasetProcessorRequestHandling:
    """Tests for request handling."""

    def test_handle_request_saves_to_store(self):
        """_handle_request calls store.save with correct arguments."""
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

    def test_handle_request_error_logged(self, caplog):
        """Errors in _handle_request are logged, not raised."""
        import logging

        proc = DatasetProcessor()
        store = MagicMock()
        store.save.side_effect = RuntimeError("Save failed")

        with caplog.at_level(logging.ERROR):
            proc._handle_request(
                DatasetSaveRequest(
                    key="bad.pkl",
                    value="data",
                    store=store,
                )
            )

        assert "error" in caplog.text.lower() or "Save failed" in caplog.text

    def test_handle_request_without_format_and_options(self):
        """_handle_request handles missing format/options gracefully."""
        proc = DatasetProcessor()
        store = MagicMock()

        proc._handle_request(
            DatasetSaveRequest(
                key="data.pkl",
                value=42,
                store=store,
            )
        )

        store.save.assert_called_once_with("data.pkl", 42, format=None, options=None)


class TestDatasetProcessorGetItem:
    """Tests for _get_item static method."""

    def test_get_item_from_queue(self):
        """Gets request from a queue-like object."""
        q = queue.Queue()
        request = {"key": "test.pkl", "value": 1, "store": None}
        q.put(request)

        proc = DatasetProcessor()
        result = proc._get_item(q)
        assert result == request

    def test_get_item_empty_queue(self):
        """Returns None for empty queue."""
        q = queue.Queue()
        proc = DatasetProcessor()
        result = proc._get_item(q)
        assert result is None

    def test_get_item_unknown_source_type(self, caplog):
        """Logs warning for unknown source types."""
        import logging

        proc = DatasetProcessor()
        with caplog.at_level(logging.WARNING):
            result = proc._get_item("not_a_queue")
        assert result is None
        assert "unknown" in caplog.text.lower() or "Unknown" in caplog.text


class TestDatasetProcessorBackgroundProcessing:
    """Tests for background processing loop."""

    def test_background_processes_requests(self):
        """Background loop processes queued requests."""
        proc = DatasetProcessor()
        q = queue.Queue()
        store = MagicMock()

        proc.add_source(q)
        proc.start()

        q.put(
            DatasetSaveRequest(
                key="bg.pkl",
                value="background",
                store=store,
                format=None,
                options=None,
            )
        )

        # Poll instead of sleeping a fixed duration
        deadline = time.monotonic() + 5.0
        while store.save.call_count < 1 and time.monotonic() < deadline:
            time.sleep(0.05)

        proc.stop()
        store.save.assert_called_once_with("bg.pkl", "background", format=None, options=None)

    def test_multiple_requests_processed(self):
        """Background loop handles multiple requests."""
        proc = DatasetProcessor()
        q = queue.Queue()
        store = MagicMock()

        proc.add_source(q)
        proc.start()

        for i in range(5):
            q.put(
                DatasetSaveRequest(
                    key=f"item_{i}.pkl",
                    value=i,
                    store=store,
                    format=None,
                    options=None,
                )
            )

        deadline = time.monotonic() + 5.0
        while store.save.call_count < 5 and time.monotonic() < deadline:
            time.sleep(0.05)

        proc.stop()

        assert store.save.call_count == 5


class TestDatasetProcessorHooks:
    """Tests for before/after dataset save hooks in DatasetProcessor."""

    def test_fires_hooks_around_save(self):
        """_handle_request fires before and after hooks when hook is set."""
        mock_hook = MagicMock()
        proc = DatasetProcessor(hook=mock_hook)
        store = MagicMock()

        proc._handle_request(
            DatasetSaveRequest(
                key="out.pkl", value=42, store=store, format="pickle", options={"x": 1}
            )
        )

        mock_hook.before_dataset_save.assert_called_once_with(
            key="out.pkl", value=42, format="pickle", options={"x": 1}
        )
        mock_hook.after_dataset_save.assert_called_once_with(
            key="out.pkl", value=42, format="pickle", options={"x": 1}
        )

    def test_hooks_bracket_store_save(self):
        """before fires before store.save, after fires after."""
        mock_hook = MagicMock()
        proc = DatasetProcessor(hook=mock_hook)
        store = MagicMock()
        order: list[str] = []

        mock_hook.before_dataset_save.side_effect = lambda **kw: order.append("before")
        store.save.side_effect = lambda *a, **kw: order.append("save")
        mock_hook.after_dataset_save.side_effect = lambda **kw: order.append("after")

        proc._handle_request(DatasetSaveRequest(key="k", value="v", store=store))

        assert order == ["before", "save", "after"]

    def test_no_hooks_when_hook_is_none(self):
        """When no hook is passed, _handle_request just saves."""
        proc = DatasetProcessor()
        store = MagicMock()

        proc._handle_request(DatasetSaveRequest(key="k", value="v", store=store))

        store.save.assert_called_once()

    def test_after_hook_not_called_on_save_error(self):
        """If store.save raises, after hook should not fire."""
        mock_hook = MagicMock()
        proc = DatasetProcessor(hook=mock_hook)
        store = MagicMock()
        store.save.side_effect = RuntimeError("write failed")

        proc._handle_request(DatasetSaveRequest(key="k", value="v", store=store))

        mock_hook.before_dataset_save.assert_called_once()
        mock_hook.after_dataset_save.assert_not_called()
