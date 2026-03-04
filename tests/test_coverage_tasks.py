"""Tests targeting uncovered lines in tasks.py.

Covers:
- _bind_args TypeError fallback (lines 107-108)
- Helper exception paths: reporter/plugin_manager raising exceptions
- _on_retry hook firing
- _on_cache_hit hook firing
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

from daglite.cache.core import CACHE_MISS
from daglite.session import RunContext
from daglite.session import reset_run_context
from daglite.session import set_run_context
from daglite.tasks import _on_cache_hit
from daglite.tasks import _on_error
from daglite.tasks import _on_retry
from daglite.tasks import _post_call
from daglite.tasks import _pre_call
from daglite.tasks import task as eager_task

# region Fixtures


class _BrokenReporter:
    """Reporter that raises on every call."""

    def report(self, event_type: str, data: dict[str, Any]) -> None:
        raise RuntimeError("reporter exploded")


class _FakeCacheStore:
    """Minimal in-memory cache store."""

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    def get(self, key: str) -> Any:
        return self._store.get(key, CACHE_MISS)

    def put(self, key: str, value: Any, ttl: int | None = None) -> None:
        self._store[key] = value


# region _bind_args TypeError fallback


class TestBindArgsTypeError:
    """_bind_args returns kwargs dict when signature.bind fails."""

    def test_extra_kwargs_fallback(self) -> None:
        """When args don't match the signature, fallback to returning kwargs."""

        @eager_task
        def strict_fn(a: int) -> int:
            return a

        # Call _bind_args directly with args that won't bind
        result = strict_fn._bind_args((), {"a": 1, "b": 2, "c": 3})
        assert result == {"a": 1, "b": 2, "c": 3}


# region Helper exception paths — reporter.report raises


class TestPreCallReporterException:
    """_pre_call swallows reporter exceptions and logs them."""

    def test_reporter_exception_swallowed(self) -> None:
        reporter = _BrokenReporter()
        ctx = RunContext(event_reporter=reporter, backend_name="inline")

        @eager_task
        def dummy() -> int:
            return 1

        # Should not raise — exception is caught and logged
        _pre_call(dummy, uuid4(), (), {}, ctx)

    def test_plugin_manager_exception_swallowed(self) -> None:
        pm = MagicMock()
        pm.hook.before_node_execute.side_effect = RuntimeError("hook boom")
        ctx = RunContext(plugin_manager=pm, backend_name="inline")

        @eager_task
        def dummy() -> int:
            return 1

        _pre_call(dummy, uuid4(), (), {}, ctx)


class TestPostCallReporterException:
    """_post_call swallows reporter exceptions."""

    def test_reporter_exception_swallowed(self) -> None:
        reporter = _BrokenReporter()
        ctx = RunContext(event_reporter=reporter, backend_name="inline")

        @eager_task
        def dummy() -> int:
            return 1

        _post_call(dummy, uuid4(), 42, 0.1, ctx)

    def test_plugin_manager_exception_swallowed(self) -> None:
        pm = MagicMock()
        pm.hook.after_node_execute.side_effect = RuntimeError("hook boom")
        ctx = RunContext(plugin_manager=pm, backend_name="inline")

        @eager_task
        def dummy() -> int:
            return 1

        _post_call(dummy, uuid4(), 42, 0.1, ctx)


class TestOnErrorReporterException:
    """_on_error swallows reporter exceptions."""

    def test_reporter_exception_swallowed(self) -> None:
        reporter = _BrokenReporter()
        ctx = RunContext(event_reporter=reporter, backend_name="inline")

        @eager_task
        def dummy() -> int:
            return 1

        _on_error(dummy, uuid4(), RuntimeError("task err"), 0.1, ctx)

    def test_plugin_manager_exception_swallowed(self) -> None:
        pm = MagicMock()
        pm.hook.on_node_error.side_effect = RuntimeError("hook boom")
        ctx = RunContext(plugin_manager=pm, backend_name="inline")

        @eager_task
        def dummy() -> int:
            return 1

        _on_error(dummy, uuid4(), RuntimeError("task err"), 0.1, ctx)


class TestOnCacheHitReporterException:
    """_on_cache_hit swallows reporter/hook exceptions."""

    def test_reporter_exception_swallowed(self) -> None:
        reporter = _BrokenReporter()
        ctx = RunContext(event_reporter=reporter, backend_name="inline")

        @eager_task(cache=True)
        def cached_fn(x: int) -> int:
            return x * 2

        _on_cache_hit(cached_fn, uuid4(), {"x": 5}, 10, ctx)

    def test_plugin_manager_exception_swallowed(self) -> None:
        pm = MagicMock()
        pm.hook.on_cache_hit.side_effect = RuntimeError("hook boom")
        ctx = RunContext(plugin_manager=pm, backend_name="inline")

        @eager_task(cache=True)
        def cached_fn(x: int) -> int:
            return x * 2

        _on_cache_hit(cached_fn, uuid4(), {"x": 5}, 10, ctx)


class TestOnRetryException:
    """_on_retry swallows plugin_manager exceptions."""

    def test_plugin_manager_exception_swallowed(self) -> None:
        pm = MagicMock()
        pm.hook.before_node_retry.side_effect = RuntimeError("hook boom")
        ctx = RunContext(plugin_manager=pm, backend_name="inline")

        @eager_task(retries=1)
        def flaky(x: int) -> int:
            return x

        _on_retry(flaky, uuid4(), {"x": 1}, 1, ValueError("err"), ctx)


# region Retry with hooks (integration-level)


class TestRetryWithHooks:
    """Sync and async retry paths exercise _on_retry and after_node_retry hooks."""

    def test_sync_retry_fires_hooks(self) -> None:
        call_count = 0

        @eager_task(retries=1)
        def flaky_fn(x: int) -> int:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("boom")
            return x * 10

        pm = MagicMock()
        reporter = MagicMock()
        ctx = RunContext(plugin_manager=pm, event_reporter=reporter, backend_name="inline")
        token = set_run_context(ctx)
        try:
            result = flaky_fn(x=5)
            assert result == 50
            assert call_count == 2
            pm.hook.before_node_retry.assert_called_once()
        finally:
            reset_run_context(token)

    def test_async_retry_fires_hooks(self) -> None:
        call_count = 0

        @eager_task(retries=1)
        async def async_flaky(x: int) -> int:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("boom")
            return x * 10

        pm = MagicMock()
        reporter = MagicMock()
        ctx = RunContext(plugin_manager=pm, event_reporter=reporter, backend_name="inline")
        token = set_run_context(ctx)
        try:
            result = asyncio.run(async_flaky(x=5))
            assert result == 50
            pm.hook.before_node_retry.assert_called_once()
        finally:
            reset_run_context(token)


# region Async cache paths


class TestAsyncCachePaths:
    """Async cache hit and cache write paths."""

    def test_async_cache_hit(self) -> None:
        @eager_task(cache=True)
        async def cached_async(x: int) -> int:
            return x * 2

        cache = _FakeCacheStore()
        reporter = MagicMock()
        ctx = RunContext(event_reporter=reporter, cache_store=cache, backend_name="inline")
        token = set_run_context(ctx)
        try:
            # First call — miss, writes to cache
            r1 = asyncio.run(cached_async(x=5))
            assert r1 == 10
            # Second call — cache hit
            r2 = asyncio.run(cached_async(x=5))
            assert r2 == 10
        finally:
            reset_run_context(token)

    def test_async_cache_write(self) -> None:
        """Verify cache store is populated after async execution."""

        @eager_task(cache=True)
        async def cached_async(x: int) -> int:
            return x * 3

        cache = _FakeCacheStore()
        ctx = RunContext(cache_store=cache, backend_name="inline")
        token = set_run_context(ctx)
        try:
            asyncio.run(cached_async(x=7))
            assert len(cache._store) == 1
        finally:
            reset_run_context(token)


class TestOnCacheHitCtxNone:
    """_on_cache_hit returns early when ctx is None."""

    def test_ctx_none_returns_early(self) -> None:
        from daglite.tasks import _on_cache_hit

        @eager_task(cache=True)
        def cached_fn(x: int) -> int:
            return x * 2

        # Should return without error when ctx is None
        _on_cache_hit(cached_fn, uuid4(), {"x": 5}, 10, None)
