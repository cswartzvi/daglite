"""
Eager `@task` decorator — runs the function immediately, returns the real value.

A function decorated with `eager_task` behaves like a normal function cal except it also:

* Emits typed events (`TaskStarted`, `TaskCompleted`, `TaskFailed`).
* Fires pluggy hooks (`before_node_execute`, `after_node_execute`, etc.).
* Checks the cache when `cache=True` and a cache store is in context.

All of this is gated on the execution context — a `ContextVar` populated by the session context
manager or the workflow runner. When there is no active context the decorated function runs inline
with no overhead.

Sync and async tasks produce distinct types (`SyncEagerTask` / `AsyncEagerTask`) so that type
checkers see the correct return type at every call site. Generators will be handled in a later
phase once iteration indexing is designed.
"""

from __future__ import annotations

import functools
import inspect
import logging
import sys
import time
from collections.abc import Callable
from collections.abc import Coroutine
from dataclasses import dataclass
from dataclasses import field
from typing import Any, Generic, ParamSpec, Protocol, TypeVar, overload
from uuid import UUID
from uuid import uuid4

from daglite.cache.core import CACHE_MISS
from daglite.cache.core import default_cache_hash
from daglite.events import TaskCompleted
from daglite.events import TaskFailed
from daglite.events import TaskStarted
from daglite.session import RunContext
from daglite.session import get_run_context

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


# region Task types


@dataclass(frozen=True)
class _BaseEagerTask(Generic[P, R]):
    """Shared fields and helpers for sync and async eager tasks."""

    func: Callable[..., Any]
    """Wrapped user function."""

    name: str
    """Human-readable task name."""

    description: str
    """Task description (from docstring or explicit)."""

    backend_name: str | None = None
    """Preferred backend name for parallel operations."""

    is_async: bool = False
    """Whether `func` is a coroutine function."""

    retries: int = field(default=0, kw_only=True)
    """Number of automatic retries on failure."""

    timeout: float | None = field(default=None, kw_only=True)
    """Max execution time in seconds."""

    cache: bool = field(default=False, kw_only=True)
    """Enable hash-based result caching."""

    cache_ttl: int | None = field(default=None, kw_only=True)
    """Cache TTL in seconds."""

    cache_hash_fn: Callable[..., str] | None = field(default=None, kw_only=True)
    """Custom `(func, bound_args) -> str` hash function."""

    @functools.cached_property
    def signature(self) -> inspect.Signature:
        """Signature of the underlying function."""
        return inspect.signature(self.func)

    def _bind_args(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
        """Resolve positional + keyword args to a flat dict for hashing."""
        try:
            ba = self.signature.bind(*args, **kwargs)
            ba.apply_defaults()
            return dict(ba.arguments)
        except TypeError:
            return kwargs

    def _compute_cache_key(self, bound: dict[str, Any]) -> str:
        hash_fn = self.cache_hash_fn or default_cache_hash
        return hash_fn(self.func, bound)


@dataclass(frozen=True)
class SyncEagerTask(_BaseEagerTask[P, R]):
    """
    Eager task wrapping a synchronous function.

    Calling an instance executes the function immediately and returns its result. When an execution
    context is active, events are emitted and hooks are fired.
    """

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:  # noqa: D102
        task_id = uuid4()
        ctx = get_run_context()
        bound = self._bind_args(args, kwargs)

        # Cache check
        cache_key: str | None = None
        if self.cache and ctx and ctx.cache_store is not None:
            cache_key = self._compute_cache_key(bound)
            cached = ctx.cache_store.get(cache_key)
            if cached is not CACHE_MISS:
                _on_cache_hit(self, task_id, bound, cached, ctx)
                return cached

        _pre_call(self, task_id, args, kwargs, ctx)
        t0 = time.perf_counter()

        last_error: BaseException | None = None
        for attempt in range(1 + self.retries):
            try:
                result = self.func(*args, **kwargs)
                elapsed = time.perf_counter() - t0

                if cache_key is not None and ctx and ctx.cache_store is not None:
                    ctx.cache_store.put(cache_key, result, ttl=self.cache_ttl)

                _post_call(self, task_id, result, elapsed, ctx)
                return result

            except Exception as exc:
                last_error = exc
                if attempt < self.retries:
                    _on_retry(self, task_id, bound, attempt + 1, exc, ctx)
                    continue
                elapsed = time.perf_counter() - t0
                _on_error(self, task_id, exc, elapsed, ctx)
                raise

        assert last_error is not None  # retries >= 0 guarantees at least one iteration
        raise last_error


@dataclass(frozen=True)
class AsyncEagerTask(_BaseEagerTask[P, R]):
    """
    Eager task wrapping an async coroutine function.

    Calling an instance returns a coroutine that the caller must `await`. When an execution context
    is active, events are emitted and hooks are fired.
    """

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Coroutine[Any, Any, R]:  # noqa: D102
        return self._run(*args, **kwargs)

    async def _run(self, *args: Any, **kwargs: Any) -> R:
        task_id = uuid4()
        ctx = get_run_context()
        bound = self._bind_args(args, kwargs)

        # Cache check
        cache_key: str | None = None
        if self.cache and ctx and ctx.cache_store is not None:
            cache_key = self._compute_cache_key(bound)
            cached = ctx.cache_store.get(cache_key)
            if cached is not CACHE_MISS:
                _on_cache_hit(self, task_id, bound, cached, ctx)
                return cached

        _pre_call(self, task_id, args, kwargs, ctx)
        t0 = time.perf_counter()

        last_error: BaseException | None = None
        for attempt in range(1 + self.retries):
            try:
                result = await self.func(*args, **kwargs)
                elapsed = time.perf_counter() - t0

                if cache_key is not None and ctx and ctx.cache_store is not None:
                    ctx.cache_store.put(cache_key, result, ttl=self.cache_ttl)

                _post_call(self, task_id, result, elapsed, ctx)
                return result

            except Exception as exc:
                last_error = exc
                if attempt < self.retries:
                    _on_retry(self, task_id, bound, attempt + 1, exc, ctx)
                    continue
                elapsed = time.perf_counter() - t0
                _on_error(self, task_id, exc, elapsed, ctx)
                raise

        assert last_error is not None  # retries >= 0 guarantees at least one iteration
        raise last_error


EagerTask = SyncEagerTask | AsyncEagerTask
"""Union of sync and async eager task types."""


# region Task decorator


class _EagerTaskDecorator(Protocol):
    """Return type for keyword-args form of ``eager_task()``."""

    @overload
    def __call__(  # type: ignore[overload-overlap]
        self, func: Callable[P, Coroutine[Any, Any, R]], /
    ) -> AsyncEagerTask[P, R]: ...

    @overload
    def __call__(self, func: Callable[P, R], /) -> SyncEagerTask[P, R]: ...

    def __call__(self, func: Any, /) -> Any: ...


@overload
def eager_task(  # type: ignore[overload-overlap]
    func: Callable[P, Coroutine[Any, Any, R]], /
) -> AsyncEagerTask[P, R]: ...


@overload
def eager_task(func: Callable[P, R], /) -> SyncEagerTask[P, R]: ...


@overload
def eager_task(
    *,
    name: str | None = None,
    description: str | None = None,
    backend_name: str | None = None,
    retries: int = 0,
    timeout: float | None = None,
    cache: bool = False,
    cache_ttl: int | None = None,
    cache_hash: Callable[..., str] | None = None,
) -> _EagerTaskDecorator: ...


def eager_task(  # noqa: D417
    func: Any = None,
    *,
    name: str | None = None,
    description: str | None = None,
    backend_name: str | None = None,
    retries: int = 0,
    timeout: float | None = None,
    cache: bool = False,
    cache_ttl: int | None = None,
    cache_hash: Callable[..., str] | None = None,
) -> Any:
    """
    Decorator that wraps a function as an eager task.

    The decorated function executes immediately on call (no futures, no graph). When called inside
    an active execution context it emits events, fires hooks, and participates in caching.

    Can be used bare (`@eager_task`) or with options (`@eager_task(cache=True)`).

    Args:
        name: Custom name. Defaults to `func.__name__`.
        description: Description. Defaults to the function's docstring.
        backend_name: Preferred backend for parallel operations.
        retries: Number of automatic retries on failure.
        timeout: Max execution time in seconds.
        cache: Enable hash-based result caching.
        cache_ttl: Cache time-to-live in seconds.
        cache_hash: Custom `(func, inputs) -> str` hash function.

    Returns:
        A sync or async eager task callable with the original signature.
    """

    def decorator(fn: Callable[..., Any]) -> SyncEagerTask[Any, Any] | AsyncEagerTask[Any, Any]:
        if inspect.isclass(fn) or not callable(fn):
            raise TypeError("`@eager_task` can only be applied to callable functions.")

        _name = name if name is not None else getattr(fn, "__name__", "unnamed_task")
        _description = description if description is not None else getattr(fn, "__doc__", "") or ""
        is_async = inspect.iscoroutinefunction(fn)

        # Store original function in module namespace for pickling (process backend)
        if hasattr(fn, "__module__") and hasattr(fn, "__name__"):
            module = sys.modules.get(fn.__module__)
            if module is not None:
                private_name = f"__{fn.__name__}_func__"
                setattr(module, private_name, fn)
                fn.__qualname__ = private_name

        cls = AsyncEagerTask if is_async else SyncEagerTask
        return cls(
            func=fn,
            name=_name,
            description=_description,
            backend_name=backend_name,
            is_async=is_async,
            retries=retries,
            timeout=timeout,
            cache=cache,
            cache_ttl=cache_ttl,
            cache_hash_fn=cache_hash,
        )

    if func is not None:
        return decorator(func)
    return decorator


# region Helpers


def _pre_call(
    task: _BaseEagerTask[Any, Any],
    task_id: UUID,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    ctx: RunContext | None,
) -> None:
    """Emits a task-started event and fires the pre-execution hook."""
    if ctx is None:
        return

    event = TaskStarted(
        task_name=task.name,
        task_id=task_id,
        args=args,
        kwargs=kwargs,
        backend=ctx.backend_name,
    )

    if ctx.event_reporter is not None:
        try:
            ctx.event_reporter.report("task_started", {"event": event})
        except Exception:
            logger.exception("Failed to report task-started event")

    if ctx.plugin_manager is not None:
        try:
            metadata = _make_metadata(task, task_id)
            ctx.plugin_manager.hook.before_node_execute(
                metadata=metadata,
                inputs=kwargs,
                reporter=ctx.event_reporter,
            )
        except Exception:
            logger.exception("Pre-execution hook failed")


def _post_call(
    task: _BaseEagerTask[Any, Any],
    task_id: UUID,
    result: Any,
    elapsed: float,
    ctx: RunContext | None,
) -> None:
    """Emits a task-completed event and fires the post-execution hook."""
    if ctx is None:
        return

    event = TaskCompleted(
        task_name=task.name,
        task_id=task_id,
        result=result,
        elapsed=elapsed,
        cached=False,
    )

    if ctx.event_reporter is not None:
        try:
            ctx.event_reporter.report("task_completed", {"event": event})
        except Exception:
            logger.exception("Failed to report task-completed event")

    if ctx.plugin_manager is not None:
        try:
            metadata = _make_metadata(task, task_id)
            ctx.plugin_manager.hook.after_node_execute(
                metadata=metadata,
                inputs={},
                result=result,
                duration=elapsed,
                reporter=ctx.event_reporter,
            )
        except Exception:
            logger.exception("Post-execution hook failed")


def _on_error(
    task: _BaseEagerTask[Any, Any],
    task_id: UUID,
    error: BaseException,
    elapsed: float,
    ctx: RunContext | None,
) -> None:
    """Emits a task-failed event and fires the error hook."""
    if ctx is None:
        return

    event = TaskFailed(
        task_name=task.name,
        task_id=task_id,
        error=error,
        elapsed=elapsed,
    )

    if ctx.event_reporter is not None:
        try:
            ctx.event_reporter.report("task_failed", {"event": event})
        except Exception:
            logger.exception("Failed to report task-failed event")

    if ctx.plugin_manager is not None:
        try:
            metadata = _make_metadata(task, task_id)
            ctx.plugin_manager.hook.on_node_error(
                metadata=metadata,
                inputs={},
                error=error if isinstance(error, Exception) else Exception(str(error)),
                duration=elapsed,
                reporter=ctx.event_reporter,
            )
        except Exception:
            logger.exception("Error hook failed")


def _on_cache_hit(
    task: _BaseEagerTask[Any, Any],
    task_id: UUID,
    bound: dict[str, Any],
    result: Any,
    ctx: RunContext | None,
) -> None:
    """Emits a cached task-completed event and fires the cache-hit hook."""
    if ctx is None:
        return

    event = TaskCompleted(
        task_name=task.name,
        task_id=task_id,
        result=result,
        elapsed=0.0,
        cached=True,
    )

    if ctx.event_reporter is not None:
        try:
            ctx.event_reporter.report("task_completed", {"event": event})
        except Exception:
            logger.exception("Failed to report cached task-completed event")

    if ctx.plugin_manager is not None:
        try:
            metadata = _make_metadata(task, task_id)
            ctx.plugin_manager.hook.on_cache_hit(
                func=task.func,
                metadata=metadata,
                inputs=bound,
                result=result,
                reporter=ctx.event_reporter,
            )
        except Exception:
            logger.exception("Cache-hit hook failed")


def _on_retry(
    task: _BaseEagerTask[Any, Any],
    task_id: UUID,
    bound: dict[str, Any],
    attempt: int,
    error: BaseException,
    ctx: RunContext | None,
) -> None:
    """Fires the pre-retry hook."""
    if ctx is None or ctx.plugin_manager is None:
        return
    try:
        metadata = _make_metadata(task, task_id)
        ctx.plugin_manager.hook.before_node_retry(
            metadata=metadata,
            inputs=bound,
            attempt=attempt,
            last_error=error if isinstance(error, Exception) else Exception(str(error)),
            reporter=ctx.event_reporter,
        )
    except Exception:
        logger.exception("Pre-retry hook failed")


def _make_metadata(task: _BaseEagerTask[Any, Any], task_id: UUID) -> Any:
    """Builds a `NodeMetadata` instance for hook compatibility."""
    from daglite._metadata import NodeMetadata

    return NodeMetadata(
        id=task_id,
        name=task.name,
        kind="task",
        description=task.description or None,
        backend_name=task.backend_name,
    )
