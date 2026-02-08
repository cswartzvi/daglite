"""Unit tests for DatasetProcessor."""

import queue
import time
from multiprocessing import Queue as MpQueue
from unittest.mock import MagicMock

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
            {
                "key": "test.pkl",
                "value": {"data": 1},
                "store": store,
                "format": "pickle",
                "options": None,
            }
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
            {
                "key": "output.txt",
                "value": "text content",
                "store": store,
                "format": "text",
                "options": {"encoding": "utf-8"},
            }
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
                {
                    "key": "bad.pkl",
                    "value": "data",
                    "store": store,
                }
            )

        assert "error" in caplog.text.lower() or "Save failed" in caplog.text

    def test_handle_request_without_format_and_options(self):
        """_handle_request handles missing format/options gracefully."""
        proc = DatasetProcessor()
        store = MagicMock()

        proc._handle_request(
            {
                "key": "data.pkl",
                "value": 42,
                "store": store,
            }
        )

        store.save.assert_called_once_with("data.pkl", 42, format=None, options=None)


class TestDatasetProcessorGetRequest:
    """Tests for _get_request static method."""

    def test_get_request_from_queue(self):
        """Gets request from a queue-like object."""
        q = queue.Queue()
        request = {"key": "test.pkl", "value": 1, "store": None}
        q.put(request)

        result = DatasetProcessor._get_request(q)
        assert result == request

    def test_get_request_empty_queue(self):
        """Returns None for empty queue."""
        q = queue.Queue()
        result = DatasetProcessor._get_request(q)
        assert result is None

    def test_get_request_unknown_source_type(self, caplog):
        """Logs warning for unknown source types."""
        import logging

        with caplog.at_level(logging.WARNING):
            result = DatasetProcessor._get_request("not_a_queue")
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
            {
                "key": "bg.pkl",
                "value": "background",
                "store": store,
                "format": None,
                "options": None,
            }
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
                {
                    "key": f"item_{i}.pkl",
                    "value": i,
                    "store": store,
                    "format": None,
                    "options": None,
                }
            )

        deadline = time.monotonic() + 5.0
        while store.save.call_count < 5 and time.monotonic() < deadline:
            time.sleep(0.05)

        proc.stop()

        assert store.save.call_count == 5
