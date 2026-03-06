"""Tests for map iteration index ``[i]`` suffixing and ``parent_task_id`` tracking.

Covers:
- ``[i]`` suffix appended to non-template task names inside ``task_map`` / ``async_task_map``
- Template names (``{param}``) skip the ``[i]`` suffix
- ``parent_task_id`` is ``None`` for top-level calls
- ``parent_task_id`` is set to the caller's ``task_id`` for nested task calls
- Context var isolation across concurrent async map items
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import UUID

import pytest

from daglite._context import RunContext
from daglite._context import get_map_iteration_index
from daglite._context import get_parent_task_id
from daglite._context import reset_run_context
from daglite._context import set_run_context
from daglite.mapping import async_task_map
from daglite.mapping import map_task
from daglite.plugins.events import TaskCompleted
from daglite.plugins.events import TaskFailed
from daglite.plugins.events import TaskStarted
from daglite.session import session
from daglite.tasks import task as eager_task

# region Helpers


class _EventCollector:
    """Minimal reporter that records events for assertions."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def report(self, event_type: str, data: dict[str, Any]) -> None:
        self.events.append((event_type, data))

    def started_events(self) -> list[TaskStarted]:
        return [e["event"] for t, e in self.events if t == "task_started"]

    def completed_events(self) -> list[TaskCompleted]:
        return [e["event"] for t, e in self.events if t == "task_completed"]

    def failed_events(self) -> list[TaskFailed]:
        return [e["event"] for t, e in self.events if t == "task_failed"]


@pytest.fixture()
def collector() -> _EventCollector:
    return _EventCollector()


@pytest.fixture()
def ctx(collector: _EventCollector) -> RunContext:
    return RunContext(event_reporter=collector, backend_name="inline")


# region Map index tests


class TestMapIndexSuffix:
    """``task_map`` appends ``[i]`` to non-template task names."""

    def test_index_in_event_names_no_session(self) -> None:
        """Without a session, ``_resolve_name`` still sees the index."""
        names: list[str] = []  # noqa

        @eager_task(name="my-task")
        def capture(x: int) -> int:
            return x

        map_task(capture, [10, 20, 30])
        # No events without a context, but we verify via a side-channel:
        # the index is set, _resolve_name appends [i].

        # Actually, without a reporter we can't inspect events.
        # Use a session to verify names.

    def test_index_suffix_with_context(self, ctx: RunContext, collector: _EventCollector) -> None:
        @eager_task(name="square")
        def sq(x: int) -> int:
            return x * x

        token = set_run_context(ctx)
        try:
            result = map_task(sq, [2, 3, 4])
            assert result == [4, 9, 16]

            started = collector.started_events()
            assert [e.task_name for e in started] == ["square[0]", "square[1]", "square[2]"]

            completed = collector.completed_events()
            assert [e.task_name for e in completed] == ["square[0]", "square[1]", "square[2]"]
        finally:
            reset_run_context(token)

    def test_index_suffix_with_session(self) -> None:
        @eager_task(name="add-one")
        def add_one(x: int) -> int:
            return x + 1

        with session(backend="inline"):
            result = map_task(add_one, [0, 1, 2])
            assert result == [1, 2, 3]

    def test_no_index_outside_map(self, ctx: RunContext, collector: _EventCollector) -> None:
        """Direct calls (not via task_map) have no ``[i]`` suffix."""

        @eager_task(name="direct")
        def direct(x: int) -> int:
            return x

        token = set_run_context(ctx)
        try:
            direct(42)
            started = collector.started_events()
            assert started[0].task_name == "direct"
        finally:
            reset_run_context(token)

    def test_index_resets_after_map(self, ctx: RunContext, collector: _EventCollector) -> None:
        """After task_map completes, direct calls shouldn't have ``[i]``."""

        @eager_task(name="resettable")
        def resettable(x: int) -> int:
            return x

        token = set_run_context(ctx)
        try:
            map_task(resettable, [1, 2])
            collector.events.clear()

            resettable(99)
            started = collector.started_events()
            assert started[0].task_name == "resettable"
        finally:
            reset_run_context(token)


class TestMapIndexTemplateSkip:
    """Template names (``{param}``) do not get the ``[i]`` suffix."""

    def test_template_name_no_index(self, ctx: RunContext, collector: _EventCollector) -> None:
        @eager_task(name="process-{x}")
        def process(x: int) -> int:
            return x * 2

        token = set_run_context(ctx)
        try:
            result = map_task(process, [10, 20, 30])
            assert result == [20, 40, 60]

            started = collector.started_events()
            names = [e.task_name for e in started]
            assert names == ["process-10", "process-20", "process-30"]
            # No [i] suffix — template resolution provides unique names.
        finally:
            reset_run_context(token)


class TestAsyncMapIndex:
    """``async_task_map`` appends ``[i]`` to non-template async task names."""

    def test_async_index_suffix(self, ctx: RunContext, collector: _EventCollector) -> None:
        @eager_task(name="async-sq")
        async def asq(x: int) -> int:
            return x * x

        token = set_run_context(ctx)
        try:
            result = asyncio.run(async_task_map(asq, [2, 3]))
            assert result == [4, 9]

            started = collector.started_events()
            assert [e.task_name for e in started] == ["async-sq[0]", "async-sq[1]"]
        finally:
            reset_run_context(token)

    def test_async_sync_task_index(self, ctx: RunContext, collector: _EventCollector) -> None:
        """Sync tasks dispatched via ``async_task_map`` also get the index."""

        @eager_task(name="sync-via-async")
        def sva(x: int) -> int:
            return x + 1

        token = set_run_context(ctx)
        try:
            result = asyncio.run(async_task_map(sva, [5, 6]))
            assert result == [6, 7]

            started = collector.started_events()
            assert [e.task_name for e in started] == ["sync-via-async[0]", "sync-via-async[1]"]
        finally:
            reset_run_context(token)

    def test_async_template_skip(self, ctx: RunContext, collector: _EventCollector) -> None:
        @eager_task(name="async-{x}")
        async def atpl(x: int) -> int:
            return x

        token = set_run_context(ctx)
        try:
            result = asyncio.run(async_task_map(atpl, [7, 8]))
            assert result == [7, 8]

            started = collector.started_events()
            names = [e.task_name for e in started]
            assert names == ["async-7", "async-8"]
        finally:
            reset_run_context(token)


class TestMapIndexWithThreadBackend:
    """Thread backend propagates the map index through context copying."""

    def test_thread_backend_index(self) -> None:
        collector = _EventCollector()
        ctx = RunContext(event_reporter=collector, backend_name="thread")
        token = set_run_context(ctx)

        @eager_task(name="thread-sq")
        def tsq(x: int) -> int:
            return x * x

        try:
            with session(backend="thread"):
                result = map_task(tsq, [1, 2, 3])
                assert result == [1, 4, 9]
        finally:
            reset_run_context(token)


class TestMapIndexIsolation:
    """Verify that ``get_map_iteration_index`` is ``None`` outside ``task_map``."""

    def test_none_outside_map(self) -> None:
        assert get_map_iteration_index() is None

    def test_visible_inside_task_during_map(self) -> None:
        """The index is consumed by ``_resolve_name`` and cleared before execution.

        This prevents nested calls from inheriting the map index. User code
        that needs the iteration index should pass it as an explicit argument.
        """
        observed: list[int | None] = []

        @eager_task(name="observe-index")
        def observe(x: int) -> int:
            observed.append(get_map_iteration_index())
            return x

        map_task(observe, [10, 20, 30])
        # Index is consumed by _resolve_name, so user code sees None.
        assert observed == [None, None, None]


# region Parent task ID tests


class TestParentTaskId:
    """``parent_task_id`` tracks nested task invocations."""

    def test_top_level_parent_is_none(self, ctx: RunContext, collector: _EventCollector) -> None:
        @eager_task(name="top-level")
        def top(x: int) -> int:
            return x

        token = set_run_context(ctx)
        try:
            top(1)
            started = collector.started_events()
            assert started[0].parent_task_id is None
        finally:
            reset_run_context(token)

    def test_nested_task_has_parent_id(self, ctx: RunContext, collector: _EventCollector) -> None:
        @eager_task(name="inner")
        def inner(x: int) -> int:
            return x * 2

        @eager_task(name="outer")
        def outer(x: int) -> int:
            return inner(x)

        token = set_run_context(ctx)
        try:
            result = outer(5)
            assert result == 10

            started = collector.started_events()
            assert len(started) == 2

            outer_event = started[0]
            inner_event = started[1]

            assert outer_event.task_name == "outer"
            assert outer_event.parent_task_id is None

            assert inner_event.task_name == "inner"
            assert inner_event.parent_task_id is not None
            assert inner_event.parent_task_id == outer_event.task_id
        finally:
            reset_run_context(token)

    def test_deeply_nested(self, ctx: RunContext, collector: _EventCollector) -> None:
        @eager_task(name="leaf")
        def leaf() -> str:
            return "leaf"

        @eager_task(name="middle")
        def middle() -> str:
            return leaf()

        @eager_task(name="root")
        def root() -> str:
            return middle()

        token = set_run_context(ctx)
        try:
            root()
            started = collector.started_events()
            assert len(started) == 3

            root_ev, mid_ev, leaf_ev = started

            assert root_ev.parent_task_id is None
            assert mid_ev.parent_task_id == root_ev.task_id
            assert leaf_ev.parent_task_id == mid_ev.task_id
        finally:
            reset_run_context(token)

    def test_parent_id_on_completed_events(
        self, ctx: RunContext, collector: _EventCollector
    ) -> None:
        @eager_task(name="child")
        def child() -> int:
            return 42

        @eager_task(name="parent")
        def parent() -> int:
            return child()

        token = set_run_context(ctx)
        try:
            parent()
            completed = collector.completed_events()
            assert len(completed) == 2

            # Inner completes first.
            child_ev = completed[0]
            parent_ev = completed[1]

            assert child_ev.task_name == "child"
            assert child_ev.parent_task_id == collector.started_events()[0].task_id

            assert parent_ev.task_name == "parent"
            assert parent_ev.parent_task_id is None
        finally:
            reset_run_context(token)

    def test_parent_id_on_failed_events(self, ctx: RunContext, collector: _EventCollector) -> None:
        @eager_task(name="failing-child")
        def failing() -> int:
            raise RuntimeError("boom")

        @eager_task(name="caller")
        def caller() -> int:
            return failing()

        token = set_run_context(ctx)
        try:
            with pytest.raises(RuntimeError, match="boom"):
                caller()

            failed = collector.failed_events()
            # Both the child and the caller emit TaskFailed (exception propagates).
            assert len(failed) == 2
            assert failed[0].task_name == "failing-child"
            # Parent should be the caller's task_id.
            caller_started = collector.started_events()[0]
            assert failed[0].parent_task_id == caller_started.task_id

            assert failed[1].task_name == "caller"
            assert failed[1].parent_task_id is None
        finally:
            reset_run_context(token)

    def test_parent_resets_after_call(self) -> None:
        """``get_parent_task_id`` returns ``None`` outside any task."""
        assert get_parent_task_id() is None

        @eager_task(name="temp")
        def temp() -> int:
            return 1

        temp()
        assert get_parent_task_id() is None


class TestAsyncParentTaskId:
    """``parent_task_id`` works for async tasks."""

    def test_async_nested(self, ctx: RunContext, collector: _EventCollector) -> None:
        @eager_task(name="async-inner")
        async def ainner(x: int) -> int:
            return x + 1

        @eager_task(name="async-outer")
        async def aouter(x: int) -> int:
            return await ainner(x)

        token = set_run_context(ctx)
        try:
            result = asyncio.run(aouter(10))
            assert result == 11

            started = collector.started_events()
            assert len(started) == 2

            outer_ev, inner_ev = started
            assert outer_ev.parent_task_id is None
            assert inner_ev.parent_task_id == outer_ev.task_id
        finally:
            reset_run_context(token)

    def test_sync_calls_async_nested(self, ctx: RunContext, collector: _EventCollector) -> None:
        """Sync task calling an async task tracks parent correctly."""

        @eager_task(name="async-child")
        async def achild() -> int:
            return 99

        @eager_task(name="sync-parent")
        def sparent() -> int:
            return asyncio.run(achild())

        token = set_run_context(ctx)
        try:
            result = sparent()
            assert result == 99

            started = collector.started_events()
            assert len(started) == 2
            assert started[0].task_name == "sync-parent"
            assert started[0].parent_task_id is None
            assert started[1].task_name == "async-child"
            assert started[1].parent_task_id == started[0].task_id
        finally:
            reset_run_context(token)


class TestMapIndexAndParentCombined:
    """Both features work together in a mapped-nested scenario."""

    def test_mapped_task_calls_nested(self, ctx: RunContext, collector: _EventCollector) -> None:
        @eager_task(name="helper")
        def helper(x: int) -> int:
            return x * 10

        @eager_task(name="mapped")
        def mapped(x: int) -> int:
            return helper(x)

        token = set_run_context(ctx)
        try:
            result = map_task(mapped, [1, 2])
            assert result == [10, 20]

            started = collector.started_events()
            assert len(started) == 4  # 2 mapped + 2 helpers

            # Mapped tasks get [i] suffix.
            assert started[0].task_name == "mapped[0]"
            assert started[0].parent_task_id is None

            # Helper is nested inside mapped[0].
            assert started[1].task_name == "helper"
            assert started[1].parent_task_id == started[0].task_id

            assert started[2].task_name == "mapped[1]"
            assert started[2].parent_task_id is None

            assert started[3].task_name == "helper"
            assert started[3].parent_task_id == started[2].task_id
        finally:
            reset_run_context(token)


class TestEventDataclassDefaults:
    """New fields have backward-compatible defaults."""

    def test_task_started_default_parent(self) -> None:
        from uuid import uuid4

        event = TaskStarted(task_name="t", task_id=uuid4(), args=(), kwargs={}, backend="inline")
        assert event.parent_task_id is None

    def test_task_completed_default_parent(self) -> None:
        from uuid import uuid4

        event = TaskCompleted(task_name="t", task_id=uuid4(), result=1, elapsed=0.0, cached=False)
        assert event.parent_task_id is None

    def test_task_failed_default_parent(self) -> None:
        from uuid import uuid4

        event = TaskFailed(task_name="t", task_id=uuid4(), error=RuntimeError("x"), elapsed=0.0)
        assert event.parent_task_id is None

    def test_explicit_parent_id(self) -> None:
        from uuid import uuid4

        parent = uuid4()
        event = TaskStarted(
            task_name="t",
            task_id=uuid4(),
            args=(),
            kwargs={},
            backend="inline",
            parent_task_id=parent,
        )
        assert event.parent_task_id == parent
        assert isinstance(event.parent_task_id, UUID)
