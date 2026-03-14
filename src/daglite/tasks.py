"""Defines daglite task decorator, task types, and metadata."""

from __future__ import annotations

import abc
import functools
import inspect
import logging
import sys
import time
from collections.abc import Callable
from collections.abc import Coroutine
from contextlib import AsyncExitStack
from contextlib import ExitStack
from dataclasses import dataclass
from dataclasses import field
from typing import (
    Any,
    AsyncIterator,
    Generic,
    Iterator,
    ParamSpec,
    Protocol,
    TypeVar,
    overload,
    override,
)
from uuid import UUID
from uuid import uuid4

from pluggy import HookRelay

from daglite._context import TaskContext
from daglite._resolvers import resolve_backend
from daglite._resolvers import resolve_cache_store
from daglite._resolvers import resolve_dataset_reporter
from daglite._resolvers import resolve_dataset_store
from daglite._resolvers import resolve_event_reporter
from daglite._resolvers import resolve_hook
from daglite._resolvers import resolve_map_index
from daglite._resolvers import resolve_parent_id
from daglite._templates import parse_template
from daglite._templates import resolve_template
from daglite.cache.core import CACHE_MISS
from daglite.cache.core import CacheHashFn
from daglite.cache.core import default_cache_hash
from daglite.cache.store import CacheStore
from daglite.datasets.store import DatasetStore
from daglite.exceptions import TaskError
from daglite.plugins.reporters import EventReporter

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")

# Built-in template variables available in name and dataset key templates.
_TEMPLATE_VAR_MAP_INDEX = "map_index"
_TEMPLATE_VAR_ITER_INDEX = "iter_index"
_BUILTIN_TEMPLATE_VARS = {_TEMPLATE_VAR_MAP_INDEX, _TEMPLATE_VAR_ITER_INDEX}


# region Task decorator


class _TaskDecorator(Protocol):
    """Return type for keyword-args form of `task()`."""

    @overload
    def __call__(  # type: ignore[overload-overlap]
        self, func: Callable[P, AsyncIterator[R]], /
    ) -> AsyncTaskStream[P, R]: ...

    @overload
    def __call__(  # type: ignore[overload-overlap]
        self, func: Callable[P, Iterator[R]], /
    ) -> SyncTaskStream[P, R]: ...

    @overload
    def __call__(self, func: Callable[P, Coroutine[Any, Any, R]], /) -> AsyncTask[P, R]: ...

    @overload
    def __call__(self, func: Callable[P, R], /) -> SyncTask[P, R]: ...

    def __call__(self, func: Any, /) -> Any: ...


@overload
def task(  # type: ignore[overload-overlap]
    func: Callable[P, AsyncIterator[R]], /
) -> AsyncTaskStream[P, R]: ...


@overload
def task(  # type: ignore[overload-overlap]
    func: Callable[P, Iterator[R]], /
) -> SyncTaskStream[P, R]: ...


@overload
def task(  # type: ignore[overload-overlap]
    func: Callable[P, Coroutine[Any, Any, R]], /
) -> AsyncTask[P, R]: ...


@overload
def task(func: Callable[P, R], /) -> SyncTask[P, R]: ...


@overload
def task(
    *,
    name: str | None = None,
    description: str | None = None,
    backend: str | None = None,
    retries: int = 0,
    timeout: float | None = None,
    cache: bool = False,
    cache_store: CacheStore | str | None = None,
    cache_ttl: int | None = None,
    cache_hash: CacheHashFn | None = None,
    dataset: str | None = None,
    dataset_store: DatasetStore | str | None = None,
    dataset_format: str | None = None,
) -> _TaskDecorator: ...


def task(  # noqa: D417
    func: Any = None,
    *,
    name: str | None = None,
    description: str | None = None,
    backend: str | None = None,
    retries: int = 0,
    timeout: float | None = None,
    cache: bool = False,
    cache_store: CacheStore | str | None = None,
    cache_ttl: int | None = None,
    cache_hash: CacheHashFn | None = None,
    dataset: str | None = None,
    dataset_store: DatasetStore | str | None = None,
    dataset_format: str | None = None,
) -> Any:
    """
    Creates a daglite task from a sync or async function.

    The decorated function executes immediately on call (no futures, no graph).

    When called inside an active session or workflow it emits events, fires hooks, participates
    in caching, and can interact with dataset stores.

    Args:
        name: Custom name of the task. Can include `{param}` placeholders for argument values and
            (`{map_index}` for the current map iteration index (if applicable). Defaults to the
            function's `__name__` if not provided. Cannot contain period (`.`) characters.

        description: Description of the task. Defaults to the function's docstring if not provided.

        backend: Backend name for parallel operations. If `None`, inherits from either the active
            session or the default settings backend.

        retries: Number of automatic task retries on failure.

        timeout: Max execution time in seconds.

        cache: Indicates whether the task participates in caching. Possible values include
            * `False` (default) — no caching.
            * `True` — use the session's cache store (if any).
            * `str` — path to a file based cache store for this task.
            * `CacheStore` instance — use the provided store for this task.

        cache_ttl: Cache time-to-live (TTL) in seconds. This parameter ignored if cache is `False`
            or a store with its own TTL is provided.

        cache_hash: Custom `(func, inputs) -> str` hash function. The default hash function uses
            the function's qualified name and the bound arguments to produce a stable hash. A
            custom hash function can be provided to override this behavior, for example to ignore
            certain arguments or to use a different hashing algorithm. This parameter is ignored if
            caching is disabled.

    Returns:
        A sync or async eager task callable with the original signature.
    """

    def decorator(
        fn: Callable[..., Any],
    ) -> (
        SyncTask[Any, Any]
        | AsyncTask[Any, Any]
        | SyncTaskStream[Any, Any]
        | AsyncTaskStream[Any, Any]
    ):
        if inspect.isclass(fn) or not callable(fn):
            raise TypeError("`@task` can only be applied to callable functions.")

        _name = name if name is not None else getattr(fn, "__name__", "unnamed_task")
        _description = description if description is not None else getattr(fn, "__doc__", "") or ""

        # Validate name template at decoration time.
        name_placeholders = parse_template(_name)
        if name_placeholders:
            param_names = set(inspect.signature(fn).parameters) | _BUILTIN_TEMPLATE_VARS
            unknown = name_placeholders - param_names
            if unknown:
                raise ValueError(
                    f"Name template '{_name}' references {unknown} which won't be available "
                    f"at runtime. Available placeholders: {sorted(param_names)}."
                )

        # Validate dataset key template at decoration time.
        if dataset:
            ds_placeholders = parse_template(dataset)
            if ds_placeholders:
                param_names = set(inspect.signature(fn).parameters) | _BUILTIN_TEMPLATE_VARS
                unknown = ds_placeholders - param_names
                if unknown:
                    raise ValueError(
                        f"Dataset template '{dataset}' references {unknown} which won't be "
                        f"available at runtime. Available placeholders: {sorted(param_names)}."
                    )

        # Store original function in module namespace for pickling (process backend)
        if hasattr(fn, "__module__") and hasattr(fn, "__name__"):
            module = sys.modules.get(fn.__module__)
            if module is not None:
                private_name = f"__{fn.__name__}_func__"
                setattr(module, private_name, fn)
                fn.__qualname__ = private_name

        # Resolve task based on function type
        is_sync_gen = inspect.isgeneratorfunction(fn)
        is_async_gen = inspect.isasyncgenfunction(fn)
        if is_sync_gen or is_async_gen:
            if cache:
                raise ValueError("Caching is not supported for generator tasks.")
            if retries > 0:
                raise ValueError("Retries are not supported for generator tasks.")
            if timeout is not None:
                raise ValueError("Timeouts are not supported for generator tasks.")
            if is_sync_gen:
                cls = SyncTaskStream
            else:
                cls = AsyncTaskStream
        elif inspect.iscoroutinefunction(fn):
            cls = AsyncTask
        else:
            cls = SyncTask

        return cls(
            func=fn,
            name=_name,
            description=_description,
            backend=backend,
            retries=retries,
            timeout=timeout,
            cache=cache,
            cache_store=cache_store,
            cache_ttl=cache_ttl,
            cache_hash_fn=cache_hash,
            dataset=dataset,
            dataset_store=dataset_store,
            dataset_format=dataset_format,
        )

    if func is not None:
        return decorator(func)
    return decorator


# region Task types


@dataclass(frozen=True)
class TaskMetadata:
    """Lightweight metadata describing the currently executing task."""

    id: UUID
    """Unique identifier for this task invocation."""

    name: str
    """Human-readable task name."""

    backend: str
    """Name of the backend used to execute this task."""

    inputs: dict[str, object] = field(default_factory=dict, kw_only=True)
    """Dictionary of input values passed to the task."""

    description: str | None = field(default=None, kw_only=True)
    """Optional description of the task."""

    map_index: int | None = field(default=None, kw_only=True)
    """Current map iteration index, if applicable."""

    parent_id: UUID | None = field(default=None, kw_only=True)
    """Unique identifier of the parent task, if any."""

    is_async: bool = field(default=False, kw_only=True)
    """Whether the task is an async coroutine function."""

    is_generator: bool = field(default=False, kw_only=True)
    """Whether the task is a generator function (sync or async)."""


@dataclass
class _TaskExecutionData:
    """Resolved context for a task call."""

    metadata: TaskMetadata
    hook: HookRelay
    event_reporter: EventReporter | None  # Needed for hook calls
    cache_store: CacheStore | None
    dataset_store: DatasetStore | None


@dataclass(frozen=True)
class _BaseTask(abc.ABC, Generic[P, R]):
    """Shared fields and helpers for sync and async eager tasks."""

    func: Callable[..., Any]
    """Wrapped user function."""

    name: str
    """Human-readable task name."""

    description: str | None = None
    """Task description (from docstring or explicit)."""

    backend: str | None = None
    """Preferred backend for parallel operations."""

    retries: int = field(default=0, kw_only=True)
    """Number of automatic retries on failure."""

    timeout: float | None = field(default=None, kw_only=True)
    """Max execution time in seconds."""

    cache: bool = field(default=False, kw_only=True)
    """Indicates how the task participates in caching."""

    cache_store: CacheStore | str | None = field(default=None, kw_only=True)
    """Optional cache store override for this task."""

    cache_ttl: int | None = field(default=None, kw_only=True)
    """Cache time-to-live (TTL) in seconds."""

    cache_hash_fn: CacheHashFn | None = field(default=None, kw_only=True)
    """Custom `(func, bound_args) -> str` hash function`."""

    dataset: str | None = field(default=None, kw_only=True)
    """Output dataset key. Supports `{param}` and `{map_index}` template placeholders."""

    dataset_store: DatasetStore | str | None = field(default=None, kw_only=True)
    """Per-task dataset store override. String paths create a file-based store."""

    dataset_format: str | None = field(default=None, kw_only=True)
    """Serialization format hint for dataset output (e.g. `"pickle"`, `"pandas/csv"`)."""

    @property
    @abc.abstractmethod
    def is_async(self) -> bool:
        """Whether the wrapped function is a coroutine function."""
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def is_generator(self) -> bool:
        """Whether the wrapped function is a generator function (sync or async)."""
        raise NotImplementedError()

    @functools.cached_property
    def _name_placeholders(self) -> frozenset[str]:
        """Placeholder names found in `name`, or empty frozenset if none."""
        return parse_template(self.name)

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

    def _resolve_name(
        self, bound: dict[str, Any], *, map_index: int | None = None, iter_index: int | None = None
    ) -> str:
        """Returns the task name with resolved placeholders and optional map index."""
        if self._name_placeholders:
            template_vars = self._build_template_vars(
                bound, map_index=map_index, iter_index=iter_index
            )
            resolved = resolve_template(self.name, template_vars)

            # If the template did NOT consume {map_index}, append [index] suffix.
            if map_index is not None and _TEMPLATE_VAR_MAP_INDEX not in self._name_placeholders:
                return f"{resolved}[{map_index}]"

            return resolved
        if map_index is not None:
            return f"{self.name}[{map_index}]"
        return self.name

    def _resolve_dataset_key(
        self,
        bound: dict[str, Any],
        *,
        map_index: int | None = None,
        iter_index: int | None = None,
    ) -> str | None:
        """Resolve the dataset output key with template placeholders, or `None` if unset."""
        if self.dataset is None:
            return None
        template_vars = self._build_template_vars(bound, map_index=map_index, iter_index=iter_index)
        return resolve_template(self.dataset, template_vars)

    def _build_template_vars(
        self, bound: dict[str, Any], *, map_index: int | None = None, iter_index: int | None = None
    ) -> dict[str, Any]:
        """Merge bound args with any non-None built-in template variables."""
        template_vars = {**bound}
        if map_index is not None:
            template_vars[_TEMPLATE_VAR_MAP_INDEX] = map_index
        if iter_index is not None:
            template_vars[_TEMPLATE_VAR_ITER_INDEX] = iter_index
        return template_vars

    def _try_save_dataset(
        self,
        metadata: TaskMetadata,
        result: Any,
        store: DatasetStore | None,
        hook: HookRelay,
    ) -> None:
        """
        Save the task result to the dataset store if a dataset key is configured.

        When a `DatasetReporter` is available (i.e. inside a session), the save is routed through
        the reporter so that process/remote backends push the write back to the coordinator.
        Otherwise falls back to a direct store write with hook dispatch.
        """
        if self.dataset is None or store is None:
            return

        key = self._resolve_dataset_key(metadata.inputs, map_index=metadata.map_index)
        if key is None:
            return

        fmt = self.dataset_format

        # Prefer the reporter (handles coordinator routing + hooks internally).
        reporter = resolve_dataset_reporter()
        if reporter is not None:
            reporter.save(key, result, store, format=fmt, metadata=metadata)
        else:
            # Bare call outside a session — save directly with hook dispatch.
            hook_kw = dict(key=key, value=result, format=fmt, options=None, metadata=metadata)
            hook.before_dataset_save(**hook_kw)
            store.save(key, result, format=fmt)
            hook.after_dataset_save(**hook_kw)

    def _prepare_execution_data(self, *args, **kwargs) -> _TaskExecutionData:
        """
        Prepare the task call by resolving metadata, hooks, and stores.

        Args:
            *args: Positional arguments passed to the task.
            **kwargs: Keyword arguments passed to the task.

        Returns:
            _CallData DTO containing resolved metadata, hook, event reporter, cache store, and
            dataset store.
        """
        task_id = uuid4()
        parent_id = resolve_parent_id()
        map_index = resolve_map_index()

        inputs = self._bind_args(args, kwargs)
        name = self._resolve_name(inputs, map_index=map_index)

        backend = resolve_backend()
        hook = resolve_hook()
        reporter = resolve_event_reporter()
        cache_store = resolve_cache_store(self.cache, self.cache_store)
        ds_store = resolve_dataset_store(self.dataset_store) if self.dataset is not None else None

        metadata = TaskMetadata(
            id=task_id,
            name=name,
            backend=backend,
            description=self.description,
            inputs=inputs,
            parent_id=parent_id,
            map_index=map_index,
            is_async=self.is_async,
            is_generator=self.is_generator,
        )

        return _TaskExecutionData(
            metadata=metadata,
            hook=hook,
            event_reporter=reporter,
            cache_store=cache_store,
            dataset_store=ds_store,
        )


@dataclass(frozen=True)
class SyncTask(_BaseTask[P, R]):
    """
    Eager task wrapping a synchronous function.

    Calling an instance executes the function immediately and returns its result. When an execution
    context is active, events are emitted and hooks are fired.
    """

    @property
    @override
    def is_async(self) -> bool:
        return False

    @property
    @override
    def is_generator(self) -> bool:
        return False

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:  # noqa: D102
        data = self._prepare_execution_data(*args, **kwargs)
        metadata, hook = data.metadata, data.hook
        common = {"metadata": metadata, "reporter": data.event_reporter}

        with ExitStack() as stack:
            t0 = time.perf_counter()
            context = TaskContext(
                metadata=metadata,
                cache_store=data.cache_store,
                dataset_store=data.dataset_store,
                timestamp=t0,
            )
            stack.enter_context(context)

            cache_key, cached = self._try_cache_hit(metadata.inputs, data.cache_store)
            if cached is not CACHE_MISS:
                hook.on_cache_hit(**common, key=cache_key, value=cached)
                return cached

            hook.before_node_execute(**common)

            last_error = None
            for attempt in range(1 + self.retries):
                if attempt > 0:
                    hook.before_node_retry(**common, error=last_error, attempt=attempt)
                try:
                    result = self.func(*args, **kwargs)
                    elapsed = time.perf_counter() - t0
                    self._try_write_cache(cache_key, result, data.cache_store)
                    self._try_save_dataset(metadata, result, data.dataset_store, hook)

                    hook.after_node_execute(**common, duration=elapsed, result=result)
                    if attempt > 0:
                        hook.after_node_retry(**common, attempt=attempt, succeeded=True)
                    return result

                except Exception as exc:
                    if attempt > 0:
                        hook.after_node_retry(**common, attempt=attempt, succeeded=False)
                    last_error = exc

            # All attempts failed; fire the error hook and raise the last exception.
            elapsed = time.perf_counter() - t0
            msg = f"Task {metadata.name!r} failed"
            msg += f" after {1 + self.retries} attempts" if self.retries > 0 else ""
            error = TaskError(msg + " with error: " + str(last_error))
            hook.on_node_error(**common, error=last_error, duration=elapsed)
            raise error from last_error

    def _try_cache_hit(
        self, inputs: dict[str, Any], store: CacheStore | None
    ) -> tuple[str | None, Any]:
        """Attempts lookup cached value; result is (None, CACHE_MISS) on miss."""
        if not self.cache or store is None:
            return None, CACHE_MISS

        cache_key = self._compute_cache_key(inputs)
        cached = store.get(cache_key)
        return cache_key, cached

    def _try_write_cache(self, key: str | None, result: Any, store: CacheStore | None) -> bool:
        """Attempts to write the result to the cache if caching is enabled."""
        if not self.cache or store is None or key is None:
            return False

        store.put(key, result, ttl=self.cache_ttl)
        return True


@dataclass(frozen=True)
class SyncTaskStream(_BaseTask[P, R]):
    """Task wrapping a sync generator; yields items lazily."""

    @property
    @override
    def is_async(self) -> bool:
        return False

    @property
    @override
    def is_generator(self) -> bool:
        return True

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Iterator[R]:  # noqa: D102
        return self._stream(*args, **kwargs)

    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[Any]:
        data = self._prepare_execution_data(*args, **kwargs)
        metadata, hook = data.metadata, data.hook
        common = {"metadata": metadata, "reporter": data.event_reporter}

        with ExitStack() as stack:
            t0 = time.perf_counter()
            context = TaskContext(
                metadata=metadata,
                cache_store=data.cache_store,
                dataset_store=data.dataset_store,
                timestamp=t0,
            )
            stack.enter_context(context)

            hook.before_node_execute(**common)

            try:
                for index, item in enumerate(self.func(*args, **kwargs)):
                    hook.node_iteration(**common, index=index)
                    yield item

                elapsed = time.perf_counter() - t0
                hook.after_node_execute(**common, duration=elapsed, result=None)

            except Exception as exc:
                elapsed = time.perf_counter() - t0
                hook.on_node_error(**common, error=exc, duration=elapsed)
                raise TaskError(f"Task {metadata.name!r} failed with error: {exc}") from exc


@dataclass(frozen=True)
class AsyncTask(_BaseTask[P, R]):
    """
    Eager task wrapping an async coroutine function.

    Calling an instance returns a coroutine that the caller must `await`. When an execution context
    is active, events are emitted and hooks are fired.
    """

    @property
    @override
    def is_async(self) -> bool:
        return True

    @property
    @override
    def is_generator(self) -> bool:
        return False

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Coroutine[Any, Any, R]:  # noqa: D102
        return self._run(*args, **kwargs)

    async def _run(self, *args: Any, **kwargs: Any) -> R:
        data = self._prepare_execution_data(*args, **kwargs)
        metadata, hook = data.metadata, data.hook
        common = {"metadata": metadata, "reporter": data.event_reporter}

        async with AsyncExitStack() as stack:
            t0 = time.perf_counter()
            context = TaskContext(
                metadata=metadata,
                cache_store=data.cache_store,
                dataset_store=data.dataset_store,
                timestamp=t0,
            )
            stack.enter_context(context)

            cache_key, cached = await self._try_cache_hit(metadata.inputs, data.cache_store)
            if cached is not CACHE_MISS:
                hook.on_cache_hit(**common, key=cache_key, value=cached)
                return cached

            hook.before_node_execute(**common)

            last_error = None
            for attempt in range(1 + self.retries):
                if attempt > 0:
                    hook.before_node_retry(**common, error=last_error, attempt=attempt)
                try:
                    result = await self.func(*args, **kwargs)
                    elapsed = time.perf_counter() - t0
                    await self._try_write_cache(cache_key, result, data.cache_store)
                    self._try_save_dataset(metadata, result, data.dataset_store, hook)

                    hook.after_node_execute(**common, duration=elapsed, result=result)
                    if attempt > 0:
                        hook.after_node_retry(**common, attempt=attempt, succeeded=True)
                    return result

                except Exception as exc:
                    if attempt > 0:
                        hook.after_node_retry(**common, attempt=attempt, succeeded=False)
                    last_error = exc

            # All attempts failed; fire the error hook and raise the last exception.
            elapsed = time.perf_counter() - t0
            msg = f"Task {metadata.name!r} failed"
            msg += f" after {1 + self.retries} attempts" if self.retries > 0 else ""
            error = TaskError(msg + " with error: " + str(last_error))
            hook.on_node_error(**common, error=last_error, duration=elapsed)
            raise error from last_error

    async def _try_cache_hit(
        self, inputs: dict[str, Any], store: CacheStore | None
    ) -> tuple[str | None, Any]:
        """Attempts lookup cached value; result is (None, CACHE_MISS) on miss."""
        if not self.cache or store is None:
            return None, CACHE_MISS

        cache_key = self._compute_cache_key(inputs)
        cached = store.get(cache_key)
        return cache_key, cached

    async def _try_write_cache(
        self, key: str | None, result: Any, store: CacheStore | None
    ) -> bool:
        """Attempts to write the result to the cache if caching is enabled."""
        if not self.cache or store is None or key is None:
            return False

        store.put(key, result, ttl=self.cache_ttl)
        return True


@dataclass(frozen=True)
class AsyncTaskStream(_BaseTask[P, R]):
    """Task wrapping an async generator; yields items lazily."""

    @property
    @override
    def is_async(self) -> bool:
        return True

    @property
    @override
    def is_generator(self) -> bool:
        return True

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> AsyncIterator[R]:  # noqa: D102
        return self._stream(*args, **kwargs)

    async def _stream(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        data = self._prepare_execution_data(*args, **kwargs)
        metadata, hook = data.metadata, data.hook
        common = {"metadata": metadata, "reporter": data.event_reporter}

        async with AsyncExitStack() as stack:
            t0 = time.perf_counter()
            context = TaskContext(
                metadata=metadata,
                cache_store=data.cache_store,
                dataset_store=data.dataset_store,
                timestamp=t0,
            )
            stack.enter_context(context)

            hook.before_node_execute(**common)

            try:
                index = 0
                async for item in self.func(*args, **kwargs):
                    hook.node_iteration(**common, index=index)
                    yield item
                    index += 1

                elapsed = time.perf_counter() - t0
                hook.after_node_execute(**common, duration=elapsed, result=None)

            except Exception as exc:
                elapsed = time.perf_counter() - t0
                hook.on_node_error(**common, error=exc, duration=elapsed)
                raise TaskError(f"Task {metadata.name!r} failed with error: {exc}") from exc
