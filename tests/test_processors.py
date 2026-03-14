"""Unit tests for BackgroundQueueProcessor base class."""

from __future__ import annotations

from queue import Queue
from typing import Any
from uuid import uuid4

from daglite._processor import BackgroundQueueProcessor


class _StubProcessor(BackgroundQueueProcessor):
    """Minimal concrete subclass for testing the ABC."""

    def __init__(self) -> None:
        super().__init__(name="StubProcessor")
        self.handled: list[Any] = []

    def _handle_item(self, item: Any) -> None:
        self.handled.append(item)


class TestBackgroundQueueProcessor:
    def test_add_source(self) -> None:
        proc = _StubProcessor()
        q: Queue[Any] = Queue()
        source_id = proc.add_source(q)

        assert source_id in proc._sources
        assert proc._sources[source_id] is q

    def test_remove_source(self) -> None:
        proc = _StubProcessor()
        q: Queue[Any] = Queue()
        source_id = proc.add_source(q)
        proc.remove_source(source_id)

        assert source_id not in proc._sources

    def test_remove_unknown_source_warns(self, caplog) -> None:
        proc = _StubProcessor()
        proc.remove_source(uuid4())

        assert "Tried to remove unknown StubProcessor source" in caplog.text

    def test_start_and_stop(self) -> None:
        proc = _StubProcessor()
        proc.start()

        assert proc._running is True
        assert proc._thread is not None
        assert proc._thread.is_alive()

        proc.stop()

        assert proc._running is False
        assert proc._thread is None

    def test_double_start_warns(self, caplog) -> None:
        proc = _StubProcessor()
        proc.start()
        proc.start()

        assert "StubProcessor already started" in caplog.text

        proc.stop()

    def test_stop_without_start_noop(self) -> None:
        proc = _StubProcessor()
        proc.stop()  # should not raise

    def test_flush_drains_items(self) -> None:
        proc = _StubProcessor()
        q: Queue[Any] = Queue()
        proc.add_source(q)

        q.put("a")
        q.put("b")
        proc.flush(timeout=2.0)

        assert proc.handled == ["a", "b"]
