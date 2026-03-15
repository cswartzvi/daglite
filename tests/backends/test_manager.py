"""Unit tests for BackendManager: lookup, registration, caching, aliases, and lifecycle."""

from __future__ import annotations

from concurrent.futures import Future

import pytest

from daglite.backends.base import Backend
from daglite.backends.local import InlineBackend
from daglite.backends.local import ThreadBackend
from daglite.backends.manager import BackendManager
from daglite.session import session
from daglite.settings import DagliteSettings


class _StubBackend(Backend):
    """Minimal concrete Backend for registration tests."""

    name = "stub"

    def _submit(self, func, args, kwargs, *, context):
        fut: Future = Future()
        fut.set_result(func(*args, **(kwargs or {})))
        return fut


class TestBackendManager:
    """BackendManager lookup, caching, and registration."""

    def test_unknown_backend_raises(self) -> None:
        from daglite.exceptions import BackendError

        with session() as ctx:
            mgr = BackendManager(session=ctx, settings=DagliteSettings())
            with pytest.raises(BackendError, match="Unknown backend"):
                mgr.get("nonexistent")

    def test_register_custom_backend(self) -> None:
        with session() as ctx:
            mgr = BackendManager(session=ctx, settings=DagliteSettings())
            mgr.register("stub", _StubBackend)
            backend = mgr.get("stub")
            assert isinstance(backend, _StubBackend)

    def test_get_none_uses_default(self) -> None:
        with session() as ctx:
            mgr = BackendManager(session=ctx, settings=DagliteSettings())
            backend = mgr.get(None)
            assert isinstance(backend, InlineBackend)

    def test_get_caches_instance(self) -> None:
        with session() as ctx:
            mgr = BackendManager(session=ctx, settings=DagliteSettings())
            first = mgr.get("inline")
            second = mgr.get("inline")
            assert first is second
            mgr.stop()

    def test_aliases_resolve_to_same_type(self) -> None:
        """thread / threading / threads all resolve to ThreadBackend."""
        with session() as ctx:
            mgr = BackendManager(session=ctx, settings=DagliteSettings())
            for alias in ("thread", "threading", "threads"):
                backend = mgr.get(alias)
                assert isinstance(backend, ThreadBackend), (
                    f"{alias} did not resolve to ThreadBackend"
                )
            mgr.stop()

    def test_stop_clears_cache(self) -> None:
        with session() as ctx:
            mgr = BackendManager(session=ctx, settings=DagliteSettings())
            mgr.get("inline")
            assert len(mgr._backends) == 1
            mgr.stop()
            assert len(mgr._backends) == 0

    def test_activate_deactivate(self) -> None:
        with session() as ctx:
            mgr = BackendManager(session=ctx, settings=DagliteSettings())
            token = mgr.activate()
            assert BackendManager.get_active() is mgr
            mgr.deactivate(token)
            # After deactivation the active manager should no longer be ours
            assert BackendManager.get_active() is not mgr
