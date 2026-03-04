"""Tests for the eager task decorator.

Covers:
- Basic sync / async execution (no context)
- Execution inside an eager context with event reporting
- Cache hit / miss paths
- Retry behaviour
- Error propagation + TaskFailed event
- Hook firing (via mock plugin manager)
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from daglite.cache.core import CACHE_MISS
from daglite.plugins.events import TaskCompleted
from daglite.plugins.events import TaskFailed
from daglite.plugins.events import TaskStarted
from daglite.session import RunContext
from daglite.session import reset_run_context
from daglite.session import set_run_context
from daglite.tasks import Task
from daglite.tasks import task as eager_task


class _FakeReporter:
    """Minimal reporter that records calls for assertions."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def report(self, event_type: str, data: dict[str, Any]) -> None:
        self.events.append((event_type, data))


class _FakeCacheStore:
    """Minimal in-memory cache store for testing."""

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    def get(self, key: str) -> Any:
        return self._store.get(key, CACHE_MISS)

    def put(self, key: str, value: Any, ttl: int | None = None) -> None:
        self._store[key] = value


@pytest.fixture()
def reporter() -> _FakeReporter:
    return _FakeReporter()


@pytest.fixture()
def cache_store() -> _FakeCacheStore:
    return _FakeCacheStore()


@pytest.fixture()
def ctx(reporter: _FakeReporter) -> RunContext:
    return RunContext(event_reporter=reporter, backend_name="inline")


@pytest.fixture()
def ctx_with_cache(reporter: _FakeReporter, cache_store: _FakeCacheStore) -> RunContext:
    return RunContext(event_reporter=reporter, cache_store=cache_store, backend_name="inline")


@eager_task
def add(x: int, y: int) -> int:
    return x + y


@eager_task(name="multiply")
def mul(x: int, y: int) -> int:
    return x * y


@eager_task
async def async_add(x: int, y: int) -> int:
    return x + y


@eager_task(cache=True)
def cached_double(x: int) -> int:
    return x * 2


_call_count = 0


@eager_task(retries=2)
def flaky(x: int) -> int:
    global _call_count
    _call_count += 1
    if _call_count < 3:
        raise ValueError(f"Attempt {_call_count}")
    return x * 10


@eager_task
def broken(x: int) -> int:
    raise RuntimeError("boom")


class TestBareExecution:
    """Tasks work without any execution context — just run the function."""

    def test_sync_call(self) -> None:
        assert add(x=1, y=2) == 3

    def test_sync_positional(self) -> None:
        assert add(1, 2) == 3

    def test_named_decorator(self) -> None:
        assert mul.name == "multiply"
        assert mul(x=3, y=4) == 12

    def test_is_eager_task(self) -> None:
        assert isinstance(add, Task)
        assert add.name == "add"
        assert add.is_async is False

    def test_async_call(self) -> None:
        result = asyncio.run(async_add(x=1, y=2))
        assert result == 3

    def test_async_is_detected(self) -> None:
        assert async_add.is_async is True

    def test_error_propagates(self) -> None:
        with pytest.raises(RuntimeError, match="boom"):
            broken(x=1)


class TestWithContext:
    """Tasks emit events when an eager context is active."""

    def test_sync_emits_events(self, ctx: RunContext, reporter: _FakeReporter) -> None:
        token = set_run_context(ctx)
        try:
            result = add(x=10, y=20)
            assert result == 30

            assert len(reporter.events) == 2
            assert reporter.events[0][0] == "task_started"
            assert reporter.events[1][0] == "task_completed"

            started = reporter.events[0][1]["event"]
            assert isinstance(started, TaskStarted)
            assert started.task_name == "add"
            assert started.backend == "inline"

            completed = reporter.events[1][1]["event"]
            assert isinstance(completed, TaskCompleted)
            assert completed.task_name == "add"
            assert completed.result == 30
            assert completed.cached is False
            assert completed.elapsed >= 0
        finally:
            reset_run_context(token)

    def test_async_emits_events(self, ctx: RunContext, reporter: _FakeReporter) -> None:
        token = set_run_context(ctx)
        try:
            result = asyncio.run(async_add(x=5, y=7))
            assert result == 12

            assert len(reporter.events) == 2
            assert reporter.events[0][0] == "task_started"
            assert reporter.events[1][0] == "task_completed"
        finally:
            reset_run_context(token)

    def test_error_emits_task_failed(self, ctx: RunContext, reporter: _FakeReporter) -> None:
        token = set_run_context(ctx)
        try:
            with pytest.raises(RuntimeError, match="boom"):
                broken(x=1)

            assert len(reporter.events) == 2
            assert reporter.events[0][0] == "task_started"
            assert reporter.events[1][0] == "task_failed"

            failed = reporter.events[1][1]["event"]
            assert isinstance(failed, TaskFailed)
            assert failed.task_name == "broken"
            assert isinstance(failed.error, RuntimeError)
        finally:
            reset_run_context(token)


class TestCaching:
    """Cache hits return stored values without re-executing."""

    def test_cache_miss_then_hit(
        self,
        ctx_with_cache: RunContext,
        reporter: _FakeReporter,
        cache_store: _FakeCacheStore,
    ) -> None:
        token = set_run_context(ctx_with_cache)
        try:
            # First call — cache miss, executes function
            result1 = cached_double(x=5)
            assert result1 == 10
            assert len(reporter.events) == 2  # started + completed

            # Second call — cache hit, no execution
            result2 = cached_double(x=5)
            assert result2 == 10
            assert len(reporter.events) == 3  # + completed(cached=True)

            # The third event should be a cached completion
            cached_event = reporter.events[2][1]["event"]
            assert isinstance(cached_event, TaskCompleted)
            assert cached_event.cached is True
            assert cached_event.elapsed == 0.0
        finally:
            reset_run_context(token)

    def test_different_args_miss(
        self,
        ctx_with_cache: RunContext,
        cache_store: _FakeCacheStore,
    ) -> None:
        token = set_run_context(ctx_with_cache)
        try:
            assert cached_double(x=5) == 10
            assert cached_double(x=6) == 12  # different args → miss
        finally:
            reset_run_context(token)

    def test_no_cache_without_flag(
        self,
        ctx_with_cache: RunContext,
        cache_store: _FakeCacheStore,
    ) -> None:
        """Tasks without `cache=True` don't use the cache even if it's available."""
        token = set_run_context(ctx_with_cache)
        try:
            add(x=1, y=2)
            add(x=1, y=2)
            # Both calls should have executed (no caching)
            assert len(cache_store._store) == 0
        finally:
            reset_run_context(token)


class TestRetries:
    """Tasks with retries re-attempt on failure."""

    def test_retries_succeed_eventually(self, ctx: RunContext, reporter: _FakeReporter) -> None:
        global _call_count
        _call_count = 0

        token = set_run_context(ctx)
        try:
            result = flaky(x=7)
            assert result == 70
            assert _call_count == 3  # 2 failures + 1 success
        finally:
            reset_run_context(token)


class TestDecoratorEdgeCases:
    def test_rejects_class(self) -> None:
        with pytest.raises(TypeError, match="callable functions"):

            @eager_task  # type: ignore[arg-type]
            class NotAFunction:
                pass

    def test_with_parens_no_args(self) -> None:
        @eager_task()
        def noop() -> int:
            return 42

        assert noop() == 42
        assert noop.name == "noop"

    def test_custom_description(self) -> None:
        @eager_task(description="A custom task")
        def my_task() -> None:
            pass

        assert my_task.description == "A custom task"

    def test_signature_preserved(self) -> None:
        params = list(add.signature.parameters.keys())
        assert params == ["x", "y"]


class TestTaskIDs:
    """Each invocation gets a unique UUID."""

    def test_unique_ids(self, ctx: RunContext, reporter: _FakeReporter) -> None:
        token = set_run_context(ctx)
        try:
            add(x=1, y=2)
            add(x=3, y=4)

            ids = [
                ev[1]["event"].task_id
                for ev in reporter.events
                if isinstance(ev[1].get("event"), (TaskStarted, TaskCompleted))
            ]
            # 4 events total (2 started + 2 completed), should be 2 unique invocation IDs
            unique_ids = set(ids)
            assert len(unique_ids) == 2
        finally:
            reset_run_context(token)


class TestNameTemplates:
    """Task name ``{param}`` substitution at call time."""

    def test_resolved_name_in_events(self, ctx: RunContext, reporter: _FakeReporter) -> None:
        """Events use the resolved name when the decorator specifies a template."""

        @eager_task(name="process_{split}")
        def process(split: str) -> str:
            return split.upper()

        token = set_run_context(ctx)
        try:
            result = process(split="train")
            assert result == "TRAIN"

            started = reporter.events[0][1]["event"]
            assert isinstance(started, TaskStarted)
            assert started.task_name == "process_train"

            completed = reporter.events[1][1]["event"]
            assert isinstance(completed, TaskCompleted)
            assert completed.task_name == "process_train"
        finally:
            reset_run_context(token)

    def test_no_template_uses_function_name(self, ctx: RunContext, reporter: _FakeReporter) -> None:
        """Without placeholders the original name is used."""

        @eager_task(name="static_name")
        def my_fn() -> int:
            return 1

        token = set_run_context(ctx)
        try:
            my_fn()
            started = reporter.events[0][1]["event"]
            assert started.task_name == "static_name"
        finally:
            reset_run_context(token)

    def test_async_resolved_name(self, ctx: RunContext, reporter: _FakeReporter) -> None:
        """Async tasks also get resolved names."""

        @eager_task(name="async_{x}")
        async def async_fn(x: int) -> int:
            return x

        token = set_run_context(ctx)
        try:
            asyncio.run(async_fn(x=42))
            started = reporter.events[0][1]["event"]
            assert started.task_name == "async_42"
        finally:
            reset_run_context(token)

    def test_multiple_placeholders(self, ctx: RunContext, reporter: _FakeReporter) -> None:
        """Multiple placeholders are all resolved."""

        @eager_task(name="{model}_{split}")
        def train(model: str, split: str) -> str:
            return f"{model}/{split}"

        token = set_run_context(ctx)
        try:
            train(model="bert", split="val")
            started = reporter.events[0][1]["event"]
            assert started.task_name == "bert_val"
        finally:
            reset_run_context(token)


class TestNameTemplateValidation:
    """Decoration-time validation of name templates."""

    def test_bad_syntax_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid key template"):

            @eager_task(name="bad_{unclosed")
            def fn(x: int) -> int:
                return x

    def test_unknown_placeholder_rejected(self) -> None:
        with pytest.raises(ValueError, match="won't be available"):

            @eager_task(name="process_{missing}")
            def fn(x: int) -> int:
                return x

    def test_valid_placeholder_accepted(self) -> None:
        @eager_task(name="process_{x}")
        def fn(x: int) -> int:
            return x

        assert fn.name == "process_{x}"
