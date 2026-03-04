"""Backend implementations for local execution (inline, threading, multiprocessing)."""

import contextvars
import inspect
import os
import sys
from collections.abc import Callable
from concurrent.futures import Future
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from typing_extensions import override

from daglite.backends.base import Backend
from daglite.plugins.manager import deserialize_plugin_manager
from daglite.plugins.manager import serialize_plugin_manager
from daglite.plugins.reporters import ProcessEventReporter


class InlineBackend(Backend):
    """
    Executes tasks sequentially in the calling thread.

    No concurrency — each task call runs inline. This is the default when no
    session is active and the simplest backend for debugging.
    """

    @override
    def submit(self, task: Callable[..., Any], args: tuple[Any, ...]) -> Future[Any]:
        fut: Future[Any] = Future()
        try:
            fut.set_result(task(*args))
        except BaseException as exc:
            fut.set_exception(exc)
        return fut

    @override
    async def async_map(self, task: Callable[..., Any], items: list[tuple[Any, ...]]) -> list[Any]:
        is_async = getattr(
            task, "is_async", inspect.iscoroutinefunction(getattr(task, "func", task))
        )
        if is_async:
            return [await task(*args) for args in items]
        return [task(*args) for args in items]


class ThreadBackend(Backend):
    """
    Executes tasks across a thread pool.

    Each `submit` call dispatches one future to the pool. The calling thread's context variables
    are copied into each worker so that the active `RunContext` is visible inside every thread.
    """

    _executor: ThreadPoolExecutor

    @override
    def _start(self) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=self._settings.max_backend_threads,
        )

    @override
    def _stop(self) -> None:
        self._executor.shutdown(wait=True)

    @override
    def submit(self, task: Callable[..., Any], args: tuple[Any, ...]) -> Future[Any]:
        # ThreadPoolExecutor.submit() does NOT propagate context variables;
        # wrap each call so the active RunContext is visible in every worker.
        return self._executor.submit(contextvars.copy_context().run, task, *args)


class ProcessBackend(Backend):
    """
    Executes tasks across a process pool.

    Worker processes receive a serialized plugin manager and a `ProcessEventReporter` backed by a
    shared multiprocessing queue so that events flow back to the coordinator.
    """

    _executor: ProcessPoolExecutor
    _event_queue: Any
    _source_id: UUID | None
    _mp_context: Any

    @staticmethod
    def _determine_mp_context() -> Any:
        """Determine the appropriate multiprocessing context for this platform."""
        from multiprocessing import get_context

        if os.name == "nt" or sys.platform == "darwin":  # pragma: no cover
            return get_context("spawn")

        if (
            sys.version_info >= (3, 13)
            and sys.version_info < (3, 14)
            and not getattr(sys, "_is_gil_enabled", lambda: True)()
        ):  # pragma: no cover
            return get_context("spawn")

        if sys.version_info >= (3, 14):  # pragma: no cover
            return get_context("forkserver")

        return get_context("fork")  # pragma: no cover

    @override
    def _start(self) -> None:
        self._mp_context = self._determine_mp_context()
        self._event_queue = self._mp_context.Queue()

        # Register queue with the session's event processor so events from worker processes are
        # dispatched on the coordinator side.
        self._source_id = None
        if self._ctx.event_processor is not None:
            self._source_id = self._ctx.event_processor.add_source(self._event_queue)

        # Serialize the plugin manager for cross-process transfer.
        serialized_pm = None
        if self._ctx.plugin_manager is not None:
            serialized_pm = serialize_plugin_manager(self._ctx.plugin_manager)

        # Validate that cache_store is picklable before sending to workers.
        if self._ctx.cache_store is not None:
            import pickle

            try:
                pickle.dumps(self._ctx.cache_store)
            except Exception as exc:
                raise TypeError(
                    "cache_store must be picklable for use with the process backend. "
                    "Use the thread or inline backend instead."
                ) from exc

        # Executor initializer will deserialize the plugin manager and create a ProcessEventReporter
        payload = _WorkerPayload(
            serialized_plugin_manager=serialized_pm,
            event_queue=self._event_queue,
            cache_store=self._ctx.cache_store,
            backend_name=self._ctx.backend_name,
        )
        self._executor = ProcessPoolExecutor(
            max_workers=self._settings.max_parallel_processes,
            mp_context=self._mp_context,
            initializer=_process_initializer,
            initargs=(payload,),
        )

    @override
    def _stop(self) -> None:
        self._executor.shutdown(wait=True)

        if self._ctx.event_processor is not None:
            self._ctx.event_processor.flush()

            if self._source_id is not None:
                self._ctx.event_processor.remove_source(self._source_id)

        self._event_queue.close()

    @override
    def submit(self, task: Callable[..., Any], args: tuple[Any, ...]) -> Future[Any]:
        return self._executor.submit(task, *args)


@dataclass
class _WorkerPayload:
    """
    Serializable bundle of everything a `ProcessPoolExecutor` worker needs to reconstruct
    a `RunContext`.

    Any new per-worker state (e.g. dataset stores, metrics sinks) should be added here rather
    than threaded through positional ``initargs``.
    """

    serialized_plugin_manager: dict[str, Any] | None
    event_queue: Any
    cache_store: Any
    backend_name: str


def _process_initializer(payload: _WorkerPayload) -> None:  # pragma: no cover
    """
    Initializer for `ProcessPoolExecutor` workers.

    Deserializes the plugin manager, creates a `ProcessEventReporter`, and pushes a `RunContext`
    into the worker's context variable so that tasks emit events back to the coordinator.
    """
    from daglite._context import RunContext
    from daglite._context import set_run_context

    pm = (
        deserialize_plugin_manager(payload.serialized_plugin_manager)
        if payload.serialized_plugin_manager
        else None
    )
    reporter = ProcessEventReporter(payload.event_queue)

    ctx = RunContext(
        backend_name=payload.backend_name,
        cache_store=payload.cache_store,
        event_reporter=reporter,
        plugin_manager=pm,
    )
    set_run_context(ctx)
