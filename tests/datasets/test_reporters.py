"""Unit tests for DatasetReporter implementations."""

import tempfile
import threading
from multiprocessing import Queue as MpQueue
from unittest.mock import MagicMock
from unittest.mock import patch

from daglite.datasets.reporters import DirectDatasetReporter
from daglite.datasets.reporters import ProcessDatasetReporter
from daglite.datasets.store import DatasetStore


class TestDirectDatasetReporter:
    """Tests for DirectDatasetReporter (inline/threaded backends)."""

    def test_is_direct(self):
        reporter = DirectDatasetReporter()
        assert reporter.is_direct is True

    def test_save_calls_store(self):
        """save() delegates to the store."""
        reporter = DirectDatasetReporter()
        store = MagicMock()

        reporter.save("key.pkl", {"a": 1}, store, format="pickle", options={"opt": True})

        store.save.assert_called_once_with(
            "key.pkl", {"a": 1}, format="pickle", options={"opt": True}
        )

    def test_save_default_format_and_options(self):
        """save() passes None format and options when not provided."""
        reporter = DirectDatasetReporter()
        store = MagicMock()

        reporter.save("key.pkl", "value", store)

        store.save.assert_called_once_with("key.pkl", "value", format=None, options=None)

    def test_thread_safety(self):
        """Multiple threads can save concurrently without data corruption."""
        reporter = DirectDatasetReporter()
        store = MagicMock()
        results = []

        def save_worker(idx):
            reporter.save(f"key_{idx}.pkl", idx, store)
            results.append(idx)

        threads = [threading.Thread(target=save_worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10
        assert store.save.call_count == 10


class TestDirectDatasetReporterHooks:
    """Tests for hook firing in DirectDatasetReporter."""

    def test_fires_before_and_after_hooks(self):
        """Hooks are called around store.save() when plugin manager is available."""
        reporter = DirectDatasetReporter()
        store = MagicMock()
        mock_hook = MagicMock()

        with patch.object(reporter, "_get_hook", return_value=mock_hook):
            reporter.save("k.pkl", {"v": 1}, store, format="pickle", options={"x": 1})

        mock_hook.before_dataset_save.assert_called_once_with(
            key="k.pkl", value={"v": 1}, format="pickle", options={"x": 1}
        )
        mock_hook.after_dataset_save.assert_called_once_with(
            key="k.pkl", value={"v": 1}, format="pickle", options={"x": 1}
        )

    def test_hooks_bracket_store_save(self):
        """before fires before store.save, after fires after."""
        reporter = DirectDatasetReporter()
        store = MagicMock()
        mock_hook = MagicMock()
        order: list[str] = []

        mock_hook.before_dataset_save.side_effect = lambda **kw: order.append("before")
        store.save.side_effect = lambda *a, **kw: order.append("save")
        mock_hook.after_dataset_save.side_effect = lambda **kw: order.append("after")

        with patch.object(reporter, "_get_hook", return_value=mock_hook):
            reporter.save("k", "v", store)

        assert order == ["before", "save", "after"]

    def test_no_hooks_when_no_plugin_manager(self):
        """When _get_hook returns None, save still works without hooks."""
        reporter = DirectDatasetReporter()
        store = MagicMock()

        with patch.object(reporter, "_get_hook", return_value=None):
            reporter.save("k", "v", store, format="text")

        store.save.assert_called_once()

    def test_get_hook_returns_none_outside_context(self):
        """_get_hook returns None when no execution context is set."""
        reporter = DirectDatasetReporter()
        assert reporter._get_hook() is None


class TestProcessDatasetReporter:
    """Tests for ProcessDatasetReporter (multiprocessing backends).

    Uses real DatasetStore instances instead of MagicMock because
    multiprocessing.Queue pickles items — MagicMock is not picklable.
    """

    def test_is_not_direct(self):
        q = MpQueue()
        try:
            reporter = ProcessDatasetReporter(q)
            assert reporter.is_direct is False
        finally:
            q.close()
            q.join_thread()

    def test_save_puts_request_on_queue(self):
        """save() serializes a request dict onto the queue."""
        q = MpQueue()
        try:
            reporter = ProcessDatasetReporter(q)
            with tempfile.TemporaryDirectory() as tmpdir:
                store = DatasetStore(tmpdir)
                reporter.save("key.pkl", {"a": 1}, store, format="pickle", options={"x": 1})

                request = q.get(timeout=2)
                assert request["key"] == "key.pkl"
                assert request["value"] == {"a": 1}
                assert request["store"].base_path == tmpdir
                assert request["format"] == "pickle"
                assert request["options"] == {"x": 1}
        finally:
            q.close()
            q.join_thread()

    def test_save_without_format_and_options(self):
        """save() defaults format and options to None."""
        q = MpQueue()
        try:
            reporter = ProcessDatasetReporter(q)
            with tempfile.TemporaryDirectory() as tmpdir:
                store = DatasetStore(tmpdir)
                reporter.save("key.pkl", "value", store)

                request = q.get(timeout=2)
                assert request["format"] is None
                assert request["options"] is None
        finally:
            q.close()
            q.join_thread()

    def test_queue_property(self):
        """queue property returns the underlying queue."""
        q = MpQueue()
        try:
            reporter = ProcessDatasetReporter(q)
            assert reporter.queue is q
        finally:
            q.close()
            q.join_thread()

    def test_close(self):
        """close() closes the underlying queue."""
        q = MpQueue()
        reporter = ProcessDatasetReporter(q)
        reporter.close()
        q.join_thread()
