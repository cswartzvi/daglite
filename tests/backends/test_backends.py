"""Parameterized unit tests for all Backend implementations."""

from __future__ import annotations

from concurrent.futures import Future

import pytest

from daglite._context import SubmitContext
from daglite.backends.local import _process_submit
from daglite.backends.local import _ProcessSubmitContext
from daglite.backends.manager import BackendManager
from daglite.session import session


@pytest.fixture(params=["inline", "thread", "process"])
def started_backend(request):
    """Yield a started Backend inside a properly initialized session."""
    name = request.param
    with session(backend=name) as ctx:
        mgr = BackendManager(session=ctx, settings=ctx.settings)
        backend = mgr.get(name)
        yield backend
        mgr.stop()


# Module-level picklable helpers (required for ProcessBackend)


def _double(x: int) -> int:
    return x * 2


def _boom(x: int) -> int:
    raise ValueError("kaboom")


class TestBackendSubmit:
    """Core submit contract shared by every backend."""

    def test_submit_returns_future(self, started_backend) -> None:
        fut = started_backend.submit(_double, (5,))
        assert isinstance(fut, Future)
        assert fut.result(timeout=10) == 10

    def test_submit_captures_exception(self, started_backend) -> None:
        fut = started_backend.submit(_boom, (1,))
        with pytest.raises(ValueError, match="kaboom"):
            fut.result(timeout=10)


class TestProcessSubmitHelpers:
    """Direct tests for process helper wrappers that run in subprocesses in integration runs."""

    def test_process_submit_enters_submit_context(self) -> None:
        def _read_map_index() -> int | None:
            current = SubmitContext._get()
            return current.map_index if current is not None else None

        result = _process_submit(
            _read_map_index,
            (),
            {},
            _ProcessSubmitContext(map_index=9),
        )

        assert result == 9
        assert SubmitContext._get() is None
