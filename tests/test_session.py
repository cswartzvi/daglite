"""Tests for the session management subsystem."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from daglite._context import SessionContext
from daglite.cache.store import CacheStore
from daglite.session import _resolve_cache_store
from daglite.session import async_session
from daglite.session import session
from daglite.settings import DagliteSettings

from .examples.tasks import add
from .examples.tasks import async_add


class TestSessionLifecycle:
    """Session properly creates, activates, and tears down context."""

    def test_context_active_inside(self) -> None:
        assert SessionContext._get() is None
        with session() as ctx:
            assert SessionContext._get() is ctx
        assert SessionContext._get() is None

    def test_default_backend_is_inline(self) -> None:
        with session() as ctx:
            assert ctx.backend == "inline"

    def test_explicit_backend_name(self) -> None:
        with session(backend="thread") as ctx:
            assert ctx.backend == "thread"

    def test_cleanup_on_exception(self) -> None:
        with pytest.raises(RuntimeError, match="test"):
            with session():
                raise RuntimeError("test")
        assert SessionContext._get() is None


class TestSessionAttributes:
    """SessionContext fields are populated correctly."""

    def test_event_processor_started(self) -> None:
        with session() as ctx:
            assert ctx.event_processor is not None

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

    def test_session_id_is_uuid(self) -> None:
        from uuid import UUID

        with session() as ctx:
            assert isinstance(ctx.session_id, UUID)


class TestTasksInsideSession:
    """Tasks run correctly inside a session."""

    def test_sync_task_returns_result(self) -> None:
        with session():
            assert add(x=1, y=2) == 3

    def test_async_task_returns_result(self) -> None:
        with session():
            result = asyncio.run(async_add(x=3, y=4))
            assert result == 7


class TestSessionCache:
    """Cache integration through ``session(cache_store=...)``."""

    def test_cache_store_string_path(self, tmp_path: Path) -> None:
        with session(cache_store=str(tmp_path / "my_cache")) as ctx:
            assert ctx.cache_store is not None

    def test_cache_store_instance(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = CacheStore(td)
            with session(cache_store=store) as ctx:
                assert ctx.cache_store is store

    def test_cache_store_none(self) -> None:
        with session(cache_store=None) as ctx:
            assert ctx.cache_store is None


class TestResolveCacheStore:
    """Tests for the ``_resolve_cache_store`` helper."""

    def test_none_returns_none(self) -> None:
        settings = DagliteSettings()
        assert _resolve_cache_store(None, settings) is None

    def test_cache_store_instance_passthrough(self, tmp_path: Path) -> None:
        store = CacheStore(str(tmp_path / "cache"))
        settings = DagliteSettings()
        assert _resolve_cache_store(store, settings) is store

    def test_string_creates_cache_store(self, tmp_path: Path) -> None:
        path = str(tmp_path / "cache")
        settings = DagliteSettings()
        result = _resolve_cache_store(path, settings)
        assert isinstance(result, CacheStore)


class TestAsyncSession:
    """Async session mirrors sync session behaviour."""

    def test_async_session_lifecycle(self) -> None:
        async def _test() -> None:
            assert SessionContext._get() is None
            async with async_session() as ctx:
                assert SessionContext._get() is ctx
            assert SessionContext._get() is None

        asyncio.run(_test())

    def test_async_session_with_async_task(self) -> None:
        async def _test() -> int:
            async with async_session():
                return await async_add(x=10, y=20)

        result = asyncio.run(_test())
        assert result == 30

    def test_async_session_cleanup_on_error(self) -> None:
        async def _test() -> None:
            async with async_session():
                raise RuntimeError("async boom")

        with pytest.raises(RuntimeError, match="async boom"):
            asyncio.run(_test())
        assert SessionContext._get() is None


class TestNestedSessions:
    """Sessions can be nested; inner overrides outer."""

    def test_inner_overrides_outer(self) -> None:
        with session(backend="inline") as outer:
            assert SessionContext._get() is outer
            assert outer.backend == "inline"

            with session(backend="thread") as inner:
                assert SessionContext._get() is inner
                assert inner.backend == "thread"

            assert SessionContext._get() is outer
            assert outer.backend == "inline"


class TestStopProcessorsErrorHandling:
    """``_stop_processors`` swallows exceptions to avoid masking user errors."""

    def test_backend_manager_stop_exception_swallowed(self) -> None:
        from unittest.mock import MagicMock

        from daglite.session import _stop_processors

        ctx = MagicMock()
        mgr = MagicMock()
        mgr.deactivate.side_effect = RuntimeError("backend stop failed")
        # Should not raise
        _stop_processors(ctx, mgr, None)

    def test_event_processor_stop_exception_swallowed(self) -> None:
        from unittest.mock import MagicMock

        from daglite.session import _stop_processors

        ctx = MagicMock()
        ctx.event_processor = MagicMock()
        ctx.event_processor.stop.side_effect = RuntimeError("processor stop failed")
        mgr = MagicMock()
        # Should not raise
        _stop_processors(ctx, mgr, None)
