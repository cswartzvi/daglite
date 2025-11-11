"""Thread-local evaluation context used during graph reduction."""
from __future__ import annotations

from contextlib import contextmanager
from threading import local
from typing import Optional


class _ContextState(local):
    def __init__(self) -> None:
        super().__init__()
        self.executor: Optional["Executor"] = None


_state = _ContextState()


@contextmanager
def use_executor(executor: "Executor"):
    prev = _state.executor
    _state.executor = executor
    try:
        yield
    finally:
        _state.executor = prev


def current_executor() -> Optional["Executor"]:
    return _state.executor


# Late import for type checking only.
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .executor import Executor
