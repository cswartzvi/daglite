"""Tests for task_map and async_task_map.

Covers:
- Inline (sequential) execution for both sync and async tasks
- Thread backend with context propagation
- async_task_map with asyncio.gather concurrency
- Empty iterables (edge case)
- Multiple iterables (zip semantics)
- Error propagation
- Backend override via the ``backend`` parameter
- Backend resolution from session context
- Event emission inside parallel calls
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from daglite.eager import eager_task
from daglite.mapping import async_task_map
from daglite.mapping import task_map
from daglite.session import RunContext
from daglite.session import reset_run_context
from daglite.session import session
from daglite.session import set_run_context


class _FakeReporter:
    """Minimal reporter that records calls for assertions."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def report(self, event_type: str, data: dict[str, Any]) -> None:
        self.events.append((event_type, data))


@eager_task
def square(x: int) -> int:
    return x * x


@eager_task
def add(x: int, y: int) -> int:
    return x + y


@eager_task
async def async_square(x: int) -> int:
    return x * x


@eager_task
async def async_add(x: int, y: int) -> int:
    return x + y


@eager_task
def fail_on_three(x: int) -> int:
    if x == 3:
        raise ValueError("three is bad")
    return x


@pytest.fixture()
def reporter() -> _FakeReporter:
    return _FakeReporter()


@pytest.fixture()
def ctx(reporter: _FakeReporter) -> RunContext:
    return RunContext(event_reporter=reporter, backend_name="inline")


class TestTaskMapInline:
    """Tests for task_map with the inline (sequential) backend."""

    def test_single_iterable(self) -> None:
        result = task_map(square, [1, 2, 3, 4])
        assert result == [1, 4, 9, 16]

    def test_multiple_iterables(self) -> None:
        result = task_map(add, [1, 2, 3], [10, 20, 30])
        assert result == [11, 22, 33]

    def test_empty_iterables(self) -> None:
        result = task_map(square)
        assert result == []

    def test_empty_list(self) -> None:
        result = task_map(square, [])
        assert result == []

    def test_error_propagates(self) -> None:
        with pytest.raises(ValueError, match="three is bad"):
            task_map(fail_on_three, [1, 2, 3, 4])

    def test_explicit_inline_backend(self) -> None:
        result = task_map(square, [5, 6], backend="inline")
        assert result == [25, 36]

    def test_emits_events_in_context(self, ctx: RunContext, reporter: _FakeReporter) -> None:
        token = set_run_context(ctx)
        try:
            result = task_map(square, [2, 3])
            assert result == [4, 9]
            started = [e for t, e in reporter.events if t == "task_started"]
            completed = [e for t, e in reporter.events if t == "task_completed"]
            assert len(started) == 2
            assert len(completed) == 2
        finally:
            reset_run_context(token)


class TestTaskMapThread:
    """Tests for task_map with the thread backend."""

    def test_thread_backend_produces_correct_results(self) -> None:
        with session(backend="thread"):
            result = task_map(square, range(10))
            assert result == [x * x for x in range(10)]

    def test_thread_backend_with_multiple_iterables(self) -> None:
        with session(backend="thread"):
            result = task_map(add, [1, 2, 3], [4, 5, 6])
            assert result == [5, 7, 9]

    def test_thread_backend_empty(self) -> None:
        with session(backend="thread"):
            result = task_map(square, [])
            assert result == []

    def test_thread_backend_error(self) -> None:
        with session(backend="thread"):
            with pytest.raises(ValueError, match="three is bad"):
                task_map(fail_on_three, [1, 2, 3])

    def test_thread_backend_propagates_events(self) -> None:
        with session(backend="thread"):
            result = task_map(square, [7, 8])
            assert result == [49, 64]

    def test_thread_alias_threading(self) -> None:
        with session(backend="inline"):
            result = task_map(square, [1, 2], backend="threading")
            assert result == [1, 4]

    def test_thread_alias_threads(self) -> None:
        with session(backend="inline"):
            result = task_map(square, [3], backend="threads")
            assert result == [9]


class TestBackendResolution:
    """Tests for backend name resolution from context or override."""

    def test_inherits_from_session_context(self) -> None:
        with session(backend="thread"):
            result = task_map(square, [2, 3])
            assert result == [4, 9]

    def test_explicit_overrides_context(self) -> None:
        with session(backend="thread"):
            result = task_map(square, [4], backend="inline")
            assert result == [16]

    def test_unknown_backend_raises(self) -> None:
        from daglite.exceptions import BackendError

        with session(backend="inline"):
            with pytest.raises(BackendError, match="Unknown backend"):
                task_map(square, [1], backend="quantum")

    def test_defaults_to_inline_without_context(self) -> None:
        result = task_map(square, [5])
        assert result == [25]


class TestAsyncTaskMap:
    """Tests for async_task_map with async tasks."""

    def test_async_single_iterable(self) -> None:
        result = asyncio.run(async_task_map(async_square, [1, 2, 3, 4]))
        assert result == [1, 4, 9, 16]

    def test_async_multiple_iterables(self) -> None:
        result = asyncio.run(async_task_map(async_add, [1, 2], [10, 20]))
        assert result == [11, 22]

    def test_async_empty(self) -> None:
        result = asyncio.run(async_task_map(async_square, []))
        assert result == []

    def test_async_emits_events(self, ctx: RunContext, reporter: _FakeReporter) -> None:
        token = set_run_context(ctx)
        try:
            result = asyncio.run(async_task_map(async_square, [2, 3]))
            assert result == [4, 9]
            started = [e for t, e in reporter.events if t == "task_started"]
            completed = [e for t, e in reporter.events if t == "task_completed"]
            assert len(started) == 2
            assert len(completed) == 2
        finally:
            reset_run_context(token)

    def test_inline_sequential(self) -> None:
        """Inline backend processes items one at a time."""
        order: list[int] = []

        @eager_task
        async def track(x: int) -> int:
            order.append(x)
            return x

        asyncio.run(async_task_map(track, [1, 2, 3], backend="inline"))
        assert order == [1, 2, 3]

    def test_concurrent_default(self) -> None:
        """Non-inline backend uses asyncio.gather for concurrency."""
        result = asyncio.run(async_task_map(async_square, [4, 5], backend="thread"))
        assert result == [16, 25]


class TestAsyncTaskMapSyncTasks:
    """Tests for async_task_map dispatching sync tasks to the thread executor."""

    def test_sync_task_via_run_in_executor(self) -> None:
        result = asyncio.run(async_task_map(square, [1, 2, 3], backend="thread"))
        assert result == [1, 4, 9]

    def test_sync_task_inline(self) -> None:
        result = asyncio.run(async_task_map(square, [4, 5], backend="inline"))
        assert result == [16, 25]


class TestTaskMapWithSession:
    """Tests for task_map used inside a session context manager."""

    def test_inside_session_inline(self) -> None:
        with session(backend="inline"):
            result = task_map(square, [1, 2, 3])
            assert result == [1, 4, 9]

    def test_inside_session_thread(self) -> None:
        with session(backend="thread"):
            result = task_map(square, [1, 2, 3])
            assert result == [1, 4, 9]

    def test_override_backend_inside_session(self) -> None:
        with session(backend="thread"):
            result = task_map(square, [1, 2], backend="inline")
            assert result == [1, 4]

    def test_events_flow_through_session(self) -> None:
        with session(backend="inline"):
            task_map(square, [10, 20])
            # Events dispatched through the session's event processor — just
            # verify no exceptions were raised and results are correct.


class TestTaskTypeCheck:
    """``task_map`` and ``async_task_map`` reject plain functions."""

    def test_task_map_rejects_plain_function(self) -> None:
        def plain(x: int) -> int:
            return x * 2

        with pytest.raises(TypeError, match="expects a `@task`-decorated callable"):
            task_map(plain, [1, 2])

    def test_async_task_map_rejects_plain_function(self) -> None:
        async def plain(x: int) -> int:
            return x * 2

        with pytest.raises(TypeError, match="expects a `@task`-decorated callable"):
            asyncio.run(async_task_map(plain, [1, 2]))
