"""
Tests for the session context manager.

Covers:
- Basic session lifecycle (enter / exit / cleanup)
- Eager tasks inside a session emit events and fire hooks
- Cache integration via session(cache=...)
- Async session parity
- Nested sessions
- Error handling and cleanup on exception
- RunContext field population
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from daglite.plugins.events import TaskCompleted
from daglite.plugins.events import TaskFailed
from daglite.plugins.events import TaskStarted
from daglite.session import RunContext
from daglite.session import async_session
from daglite.session import get_run_context
from daglite.session import session
from daglite.tasks import task as eager_task


@eager_task
def add(x: int, y: int) -> int:
    """Adds two numbers."""
    return x + y


@eager_task
async def async_add(x: int, y: int) -> int:
    """Adds two numbers asynchronously."""
    return x + y


@eager_task(cache=True)
def cached_square(x: int) -> int:
    """Squares a number with caching enabled."""
    return x * x


@eager_task
def exploding(x: int) -> int:
    """Always raises."""
    raise RuntimeError("boom")


class TestSessionLifecycle:
    """Tests that session properly creates, activates, and tears down context."""

    def test_context_active_inside_session(self) -> None:
        assert get_run_context() is None
        with session() as ctx:
            assert get_run_context() is ctx
            assert isinstance(ctx, RunContext)
        assert get_run_context() is None

    def test_default_backend_is_inline(self) -> None:
        with session() as ctx:
            assert ctx.backend_name == "inline"

    def test_explicit_backend_name(self) -> None:
        with session(backend="thread") as ctx:
            assert ctx.backend_name == "thread"

    def test_event_processor_started_and_stopped(self) -> None:
        with session() as ctx:
            assert ctx.event_processor is not None
            assert ctx.event_processor._running
        # After exit the processor should be stopped
        assert not ctx.event_processor._running

    def test_plugin_manager_created(self) -> None:
        with session() as ctx:
            assert ctx.plugin_manager is not None

    def test_event_reporter_created(self) -> None:
        with session() as ctx:
            assert ctx.event_reporter is not None
            assert ctx.event_reporter.is_direct

    def test_settings_populated(self) -> None:
        with session() as ctx:
            assert ctx.settings is not None
            assert hasattr(ctx.settings, "default_backend")

    def test_cleanup_on_exception(self) -> None:
        with pytest.raises(RuntimeError, match="test"):
            with session() as ctx:
                processor = ctx.event_processor
                raise RuntimeError("test")
        # Processor should still be stopped
        assert not processor._running  # type: ignore[union-attr]
        assert get_run_context() is None


class TestTasksInsideSession:
    """Tests that eager tasks properly interact with the session context."""

    def test_sync_task_emits_events(self) -> None:
        collected: list[tuple[str, dict[str, Any]]] = []
        with session() as ctx:
            # Intercept events via the reporter
            assert ctx.event_reporter is not None
            original_report = ctx.event_reporter.report

            def spy_report(event_type: str, data: dict[str, Any]) -> None:
                collected.append((event_type, data))
                original_report(event_type, data)

            ctx.event_reporter.report = spy_report  # type: ignore[method-assign]

            result = add(x=1, y=2)

        assert result == 3
        assert len(collected) == 2
        assert collected[0][0] == "task_started"
        assert collected[1][0] == "task_completed"
        assert isinstance(collected[0][1]["event"], TaskStarted)
        assert isinstance(collected[1][1]["event"], TaskCompleted)

    def test_async_task_emits_events(self) -> None:
        collected: list[tuple[str, dict[str, Any]]] = []
        with session() as ctx:
            assert ctx.event_reporter is not None
            original_report = ctx.event_reporter.report

            def spy_report(event_type: str, data: dict[str, Any]) -> None:
                collected.append((event_type, data))
                original_report(event_type, data)

            ctx.event_reporter.report = spy_report  # type: ignore[method-assign]

            result = asyncio.run(async_add(x=3, y=4))

        assert result == 7
        assert len(collected) == 2
        assert collected[0][0] == "task_started"
        assert collected[1][0] == "task_completed"

    def test_error_emits_task_failed(self) -> None:
        collected: list[tuple[str, dict[str, Any]]] = []
        with session() as ctx:
            assert ctx.event_reporter is not None
            original_report = ctx.event_reporter.report

            def spy_report(event_type: str, data: dict[str, Any]) -> None:
                collected.append((event_type, data))
                original_report(event_type, data)

            ctx.event_reporter.report = spy_report  # type: ignore[method-assign]

            with pytest.raises(RuntimeError, match="boom"):
                exploding(x=1)

        assert len(collected) == 2
        assert collected[1][0] == "task_failed"
        assert isinstance(collected[1][1]["event"], TaskFailed)

    def test_backend_name_in_events(self) -> None:
        collected: list[tuple[str, dict[str, Any]]] = []
        with session(backend="thread") as ctx:
            assert ctx.event_reporter is not None
            original_report = ctx.event_reporter.report

            def spy_report(event_type: str, data: dict[str, Any]) -> None:
                collected.append((event_type, data))
                original_report(event_type, data)

            ctx.event_reporter.report = spy_report  # type: ignore[method-assign]

            add(x=1, y=2)

        started_event = collected[0][1]["event"]
        assert started_event.backend == "thread"


class TestSessionCache:
    """Tests cache integration through the session context manager."""

    def test_cache_true_with_settings_path(self, tmp_path: Path) -> None:
        from daglite.settings import DagliteSettings

        settings = DagliteSettings(cache_store=str(tmp_path / "cache"))
        with session(cache=True, settings=settings) as ctx:
            assert ctx.cache_store is not None

            r1 = cached_square(x=5)
            r2 = cached_square(x=5)  # should hit cache

        assert r1 == 25
        assert r2 == 25

    def test_cache_false_disables(self) -> None:
        with session(cache=False) as ctx:
            assert ctx.cache_store is None

    def test_cache_none_disables(self) -> None:
        with session(cache=None) as ctx:
            assert ctx.cache_store is None

    def test_cache_string_path(self, tmp_path: Path) -> None:
        cache_path = str(tmp_path / "my_cache")
        with session(cache=cache_path) as ctx:
            assert ctx.cache_store is not None

    def test_cache_store_instance(self) -> None:
        from daglite.cache.store import CacheStore

        with tempfile.TemporaryDirectory() as td:
            store = CacheStore(td)
            with session(cache=store) as ctx:
                assert ctx.cache_store is store


class TestAsyncSession:
    """Tests the async_session context manager."""

    def test_async_session_lifecycle(self) -> None:
        async def _test() -> None:
            assert get_run_context() is None
            async with async_session() as ctx:
                assert get_run_context() is ctx
                assert ctx.event_processor is not None
            assert get_run_context() is None

        asyncio.run(_test())

    def test_async_session_with_async_task(self) -> None:
        async def _test() -> int:
            async with async_session():
                return await async_add(x=10, y=20)

        result = asyncio.run(_test())
        assert result == 30

    def test_async_session_cleanup_on_error(self) -> None:
        async def _test() -> None:
            async with async_session() as ctx:
                _ = ctx.event_processor
                raise RuntimeError("async boom")

        with pytest.raises(RuntimeError, match="async boom"):
            asyncio.run(_test())


class TestNestedSessions:
    """Tests that sessions can be nested (inner overrides outer)."""

    def test_inner_overrides_outer(self) -> None:
        with session(backend="inline") as outer:
            assert get_run_context() is outer
            assert outer.backend_name == "inline"

            with session(backend="thread") as inner:
                assert get_run_context() is inner
                assert inner.backend_name == "thread"

            # Outer restored
            assert get_run_context() is outer
            assert outer.backend_name == "inline"

    def test_inner_tasks_use_inner_context(self) -> None:
        collected_outer: list[tuple[str, dict[str, Any]]] = []
        collected_inner: list[tuple[str, dict[str, Any]]] = []

        with session(backend="inline") as outer:
            assert outer.event_reporter is not None
            original_outer = outer.event_reporter.report

            def spy_outer(event_type: str, data: dict[str, Any]) -> None:
                collected_outer.append((event_type, data))
                original_outer(event_type, data)

            outer.event_reporter.report = spy_outer  # type: ignore[method-assign]

            add(x=1, y=2)  # goes to outer

            with session(backend="thread") as inner:
                assert inner.event_reporter is not None
                original_inner = inner.event_reporter.report

                def spy_inner(event_type: str, data: dict[str, Any]) -> None:
                    collected_inner.append((event_type, data))
                    original_inner(event_type, data)

                inner.event_reporter.report = spy_inner  # type: ignore[method-assign]

                add(x=3, y=4)  # goes to inner

            add(x=5, y=6)  # goes to outer again

        # Outer saw 2 calls (before and after inner), inner saw 1
        assert len(collected_outer) == 4  # 2 events per call × 2 calls
        assert len(collected_inner) == 2  # 2 events × 1 call


class TestRunContextDefaults:
    """Tests RunContext default field values."""

    def test_defaults(self) -> None:
        ctx = RunContext()
        assert ctx.backend_name == "inline"
        assert ctx.cache_store is None
        assert ctx.event_reporter is None
        assert ctx.event_processor is None
        assert ctx.plugin_manager is None
        assert ctx.backend_manager is None
        assert ctx.settings is None
        assert ctx.started_at > 0

    def test_lightweight_test_context(self) -> None:
        """A minimal context for unit tests — no real infra needed."""
        mock_reporter = MagicMock()
        ctx = RunContext(event_reporter=mock_reporter, backend_name="test")
        assert ctx.backend_name == "test"
        assert ctx.event_reporter is mock_reporter


class TestResolveCacheEdgeCases:
    """Tests for _resolve_cache branches not covered by TestSessionCache."""

    def test_cache_true_settings_cache_none(self) -> None:
        """cache=True with settings.cache_store=None returns None."""
        from daglite.session import _resolve_cache_store
        from daglite.settings import DagliteSettings

        settings = DagliteSettings(cache_store=None)
        assert _resolve_cache_store(True, settings) is None

    def test_cache_true_settings_cache_store_instance(self, tmp_path: Path) -> None:
        """cache=True with settings.cache_store as CacheStore returns the instance."""
        from daglite.cache.store import CacheStore
        from daglite.session import _resolve_cache_store
        from daglite.settings import DagliteSettings

        store = CacheStore(str(tmp_path / "cache"))
        settings = DagliteSettings(cache_store=store)
        assert _resolve_cache_store(True, settings) is store

    def test_cache_invalid_type_raises(self) -> None:
        """Non bool/str/CacheStore/None raises ValueError."""
        from daglite.session import _resolve_cache_store
        from daglite.settings import DagliteSettings

        settings = DagliteSettings()
        with pytest.raises(ValueError, match="Invalid cache argument"):
            _resolve_cache_store(12345, settings)


class TestStopProcessorsErrorHandling:
    """Tests for _stop_processors swallowing exceptions."""

    def test_backend_manager_stop_exception_swallowed(self) -> None:
        from daglite.session import _stop_processors

        ctx = RunContext()
        ctx.backend_manager = MagicMock()
        ctx.backend_manager.stop.side_effect = RuntimeError("backend stop failed")
        # Should not raise
        _stop_processors(ctx)

    def test_event_processor_stop_exception_swallowed(self) -> None:
        from daglite.session import _stop_processors

        ctx = RunContext()
        ctx.event_processor = MagicMock()
        ctx.event_processor.stop.side_effect = RuntimeError("processor stop failed")
        # Should not raise
        _stop_processors(ctx)
