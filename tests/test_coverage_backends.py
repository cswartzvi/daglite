"""Tests targeting uncovered lines in backends/base.py, backends/local.py, and backends/manager.py.

Covers:
- InlineBackend.submit exception path (local.py lines 36-37)
- InlineBackend.async_map sync branch (local.py line 47)
- ThreadBackend._start/_stop/submit (local.py line 93+)
- Default Backend.async_map implementation (base.py lines 136-146)
- _run_async_task helper (base.py line 151)
- BackendManager.register and get unknown backend (manager.py lines 63, 91)
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any

import pytest

from daglite._context import RunContext
from daglite.backends.base import Backend
from daglite.backends.base import _run_async_task
from daglite.backends.local import InlineBackend
from daglite.backends.local import ThreadBackend
from daglite.backends.manager import BackendManager
from daglite.settings import DagliteSettings
from daglite.tasks import task as eager_task

# region InlineBackend


class TestInlineBackendSubmitException:
    """InlineBackend.submit wraps exceptions in the returned Future."""

    def test_exception_captured_in_future(self) -> None:
        backend = InlineBackend()

        def explode() -> None:
            raise ValueError("kaboom")

        fut = backend.submit(explode, ())
        with pytest.raises(ValueError, match="kaboom"):
            fut.result()


class TestInlineBackendAsyncMapSync:
    """InlineBackend.async_map runs sync tasks inline."""

    def test_sync_branch(self) -> None:
        backend = InlineBackend()

        @eager_task
        def double(x: int) -> int:
            return x * 2

        results = asyncio.run(backend.async_map(double, [(2,), (3,), (4,)]))
        assert results == [4, 6, 8]


# region ThreadBackend


class TestThreadBackend:
    """ThreadBackend lifecycle and submission."""

    def test_start_stop_submit(self) -> None:
        backend = ThreadBackend()
        ctx = RunContext(backend_name="thread")
        settings = DagliteSettings()
        backend.start(ctx, settings)

        try:
            fut = backend.submit(lambda x: x * 2, (5,))
            assert fut.result() == 10
        finally:
            backend.stop()

    def test_map(self) -> None:
        backend = ThreadBackend()
        ctx = RunContext(backend_name="thread")
        settings = DagliteSettings()
        backend.start(ctx, settings)

        try:
            results = backend.map(lambda x: x + 1, [(1,), (2,), (3,)])
            assert results == [2, 3, 4]
        finally:
            backend.stop()


# region Default Backend.async_map


class _MinimalBackend(Backend):
    """Minimal Backend subclass using default async_map.

    Uses a thread to execute tasks so that `_run_async_task` (which calls
    `asyncio.run`) works even when the caller is already inside an event loop.
    """

    def submit(self, task: Callable[..., Any], args: tuple[Any, ...]) -> Future[Any]:
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(task, *args)


class TestDefaultAsyncMap:
    """Default Backend.async_map wraps submit futures as asyncio futures."""

    def test_sync_tasks(self) -> None:
        backend = _MinimalBackend()
        ctx = RunContext(backend_name="inline")
        settings = DagliteSettings()
        backend.start(ctx, settings)

        @eager_task
        def add_one(x: int) -> int:
            return x + 1

        results = asyncio.run(backend.async_map(add_one, [(1,), (2,), (3,)]))
        assert results == [2, 3, 4]
        backend.stop()

    def test_async_tasks(self) -> None:
        backend = _MinimalBackend()
        ctx = RunContext(backend_name="inline")
        settings = DagliteSettings()
        backend.start(ctx, settings)

        @eager_task
        async def async_double(x: int) -> int:
            return x * 2

        results = asyncio.run(backend.async_map(async_double, [(1,), (2,), (3,)]))
        assert results == [2, 4, 6]
        backend.stop()


# region _run_async_task


class TestRunAsyncTask:
    """_run_async_task helper runs an async callable in a new event loop."""

    def test_runs_coroutine(self) -> None:
        async def coro(x: int) -> int:
            return x * 3

        assert _run_async_task(coro, 5) == 15


# region BackendManager


class TestBackendManagerEdgeCases:
    """BackendManager: unknown backend and register."""

    def test_unknown_backend_raises(self) -> None:
        from daglite.exceptions import BackendError

        ctx = RunContext(backend_name="inline")
        settings = DagliteSettings()
        mgr = BackendManager(ctx, settings)

        with pytest.raises(BackendError, match="Unknown backend"):
            mgr.get("nonexistent_backend")

    def test_register_custom_backend(self) -> None:
        ctx = RunContext(backend_name="inline")
        settings = DagliteSettings()
        mgr = BackendManager(ctx, settings)

        mgr.register("custom", _MinimalBackend)
        backend = mgr.get("custom")
        assert isinstance(backend, _MinimalBackend)
        mgr.stop()

    def test_get_none_uses_default(self) -> None:
        """get(None) falls back to settings.default_backend."""
        ctx = RunContext(backend_name="inline")
        settings = DagliteSettings(backend="inline")
        mgr = BackendManager(ctx, settings)

        backend = mgr.get(None)
        assert isinstance(backend, InlineBackend)
        mgr.stop()

    def test_get_cached_backend(self) -> None:
        """Second get() for the same name returns the cached instance."""
        ctx = RunContext(backend_name="inline")
        settings = DagliteSettings()
        mgr = BackendManager(ctx, settings)

        b1 = mgr.get("inline")
        b2 = mgr.get("inline")
        assert b1 is b2
        mgr.stop()
