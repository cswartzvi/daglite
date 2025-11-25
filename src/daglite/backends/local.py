import os
import sys
from concurrent.futures import Executor
from concurrent.futures import Future
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from typing import Any, Callable, TypeVar

from typing_extensions import override

from daglite.settings import get_global_settings

from .base import Backend

T = TypeVar("T")


_GLOBAL_THREAD_POOL: ThreadPoolExecutor | None = None
_GLOBAL_PROCESS_POOL: ProcessPoolExecutor | None = None


def _get_global_thread_pool() -> ThreadPoolExecutor:
    global _GLOBAL_THREAD_POOL
    if _GLOBAL_THREAD_POOL is None:
        settings = get_global_settings()
        max_workers = settings.max_backend_threads
        _GLOBAL_THREAD_POOL = ThreadPoolExecutor(max_workers=max_workers)
    return _GLOBAL_THREAD_POOL


def _get_global_process_pool() -> ProcessPoolExecutor:
    global _GLOBAL_PROCESS_POOL
    if _GLOBAL_PROCESS_POOL is not None:  # pragma: no cover
        return _GLOBAL_PROCESS_POOL

    import multiprocessing as mp
    from multiprocessing.context import BaseContext

    settings = get_global_settings()
    max_workers = settings.max_parallel_processes
    mp_context: BaseContext
    if os.name == "nt" or sys.platform == "darwin":
        # Use 'spawn' on Windows (required) and macOS (fork deprecated)
        mp_context = mp.get_context("spawn")
    else:
        # Use 'fork' on Linux (explicit, since Python 3.14 changed default to forkserver)
        mp_context = mp.get_context("fork")
    _GLOBAL_PROCESS_POOL = ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context)
    return _GLOBAL_PROCESS_POOL


def _get_global_thread_pool_size() -> int:
    """Get the actual size of the global thread pool."""
    settings = get_global_settings()
    return settings.max_backend_threads


def _get_global_process_pool_size() -> int:
    """Get the actual size of the global process pool."""
    settings = get_global_settings()
    return settings.max_parallel_processes


def _reset_global_pools() -> None:
    """
    Reset global executor pools.

    Useful for testing or after forking processes where the pools may be in
    an inconsistent state.
    """
    global _GLOBAL_THREAD_POOL, _GLOBAL_PROCESS_POOL
    _GLOBAL_THREAD_POOL = None
    _GLOBAL_PROCESS_POOL = None


class SequentialBackend(Backend):
    """Executes immediately in current thread, returns completed futures."""

    @override
    def submit(self, fn: Callable[..., T], **kwargs: Any) -> Future[T]:
        future: Future[T] = Future()
        try:
            result = fn(**kwargs)
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)
        return future

    @override
    def submit_many(self, fn: Callable[..., T], calls: list[dict[str, Any]]) -> list[Future[T]]:
        futures: list[Future[T]] = []
        for kwargs in calls:
            future: Future[T] = Future()
            try:
                result = fn(**kwargs)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)
            futures.append(future)
        return futures


class ThreadBackend(Backend):
    """Executes in thread pool, returns pending futures."""

    def __init__(self, max_workers: int | None = None):
        self._max_workers = max_workers

    @override
    def submit(self, fn: Callable[..., T], **kwargs: Any) -> Future[T]:
        executor = _get_global_thread_pool()
        return executor.submit(fn, **kwargs)

    @override
    def submit_many(self, fn: Callable[..., T], calls: list[dict[str, Any]]) -> list[Future[T]]:
        executor = _get_global_thread_pool()
        if self._max_workers is None:
            return [executor.submit(fn, **kw) for kw in calls]
        max_concurrent = min(self._max_workers, _get_global_thread_pool_size())
        futures = _submit_many_limited(executor, fn, calls, max_concurrent)
        return futures


class ProcessBackend(Backend):
    """Executes in process pool, returns pending futures."""

    def __init__(self, max_workers: int | None = None):
        self._max_workers = max_workers

    @override
    def submit(self, fn: Callable[..., T], **kwargs: Any) -> Future[T]:
        executor = _get_global_process_pool()
        return executor.submit(fn, **kwargs)

    @override
    def submit_many(self, fn: Callable[..., T], calls: list[dict[str, Any]]) -> list[Future[T]]:
        executor = _get_global_process_pool()
        if self._max_workers is None:
            return [executor.submit(fn, **kw) for kw in calls]
        max_concurrent = min(self._max_workers, _get_global_process_pool_size())
        futures = _submit_many_limited(executor, fn, calls, max_concurrent)
        return futures


def _submit_many_limited(
    executor: Executor,
    fn: Callable[..., T],
    calls: list[dict[str, Any]],
    max_concurrent: int,
) -> list[Future[T]]:
    """Submit calls to executor with concurrency limit."""
    # Pre-create all futures
    futures: list[Future[T]] = [Future() for _ in calls]

    remaining = list(enumerate(calls))
    in_flight: dict[Future, int] = {}  # executor future -> output index

    while remaining or in_flight:
        # Submit up to limit
        while remaining and len(in_flight) < max_concurrent:
            idx, kwargs = remaining.pop(0)
            exec_future = executor.submit(fn, **kwargs)
            in_flight[exec_future] = idx

        # Wait for one to complete
        if in_flight:  # pragma: no branch
            done = next(as_completed(in_flight.keys()))
            idx = in_flight.pop(done)

            # Transfer result to output future
            try:
                result = done.result()
                futures[idx].set_result(result)
            except Exception as e:  # pragma: no cover
                futures[idx].set_exception(e)

    return futures
