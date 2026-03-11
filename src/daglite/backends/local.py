"""Backend implementations for local execution (inline, threading, multiprocessing)."""

import contextvars
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

from daglite._context import BackendContext
from daglite.backends.base import Backend


class InlineBackend(Backend):
    """
    Executes tasks sequentially in the calling thread.

    No concurrency — each task call runs inline. This is the default when no
    session is active and the simplest backend for debugging.
    """

    name = "inline"

    @override
    def _submit(
        self,
        func: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        context: BackendContext,
    ) -> Future[Any]:
        fut: Future[Any] = Future()
        try:
            with context:
                fut.set_result(func(*args, **kwargs))
        except BaseException as exc:
            fut.set_exception(exc)
        return fut


class ThreadBackend(Backend):
    """Executes tasks across a thread pool."""

    name = "thread"

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
    def _submit(
        self,
        func: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        context: BackendContext,
    ) -> Future[Any]:
        def _run() -> Any:
            with context:
                return func(*args, **kwargs)

        # copy_context() captures the BackendContext so the worker thread sees it.
        return self._executor.submit(contextvars.copy_context().run, _run)


class ProcessBackend(Backend):
    """Executes tasks across a process pool."""

    name = "process"

    _executor: ProcessPoolExecutor
    _event_queue: Any
    _source_id: UUID | None
    _mp_context: Any
    _has_session: bool

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
        self._has_session = self._session is not None

        if self._has_session:
            self._start_with_session()
        else:
            self._start_bare()

    def _start_with_session(self) -> None:
        """Set up the process pool with full session infrastructure (events, plugins, cache)."""
        from daglite.plugins.manager import serialize_plugin_manager

        assert self._session is not None

        self._event_queue = self._mp_context.Queue()

        # Register queue with the session's event processor so events from worker processes are
        # dispatched on the coordinator side.
        self._source_id = None
        if self._session.event_processor is not None:
            self._source_id = self._session.event_processor.add_source(self._event_queue)

        # Serialize the plugin manager for cross-process transfer.
        serialized_pm = None
        if self._session.plugin_manager is not None:
            serialized_pm = serialize_plugin_manager(self._session.plugin_manager)

        # Validate that cache_store is picklable before sending to workers.
        if self._session.cache_store is not None:
            import pickle

            try:
                pickle.dumps(self._session.cache_store)
            except Exception as exc:
                raise TypeError(
                    "cache_store must be picklable for use with the process backend. "
                    "Use the thread or inline backend instead."
                ) from exc

        payload = _ProcessWorkerPayload(
            serialized_plugin_manager=serialized_pm,
            event_queue=self._event_queue,
            cache_store=self._session.cache_store,
        )

        self._executor = ProcessPoolExecutor(
            max_workers=self._settings.max_parallel_processes,
            mp_context=self._mp_context,
            initializer=_process_initializer,
            initargs=(payload,),
        )

    def _start_bare(self) -> None:
        """Set up the process pool without session infrastructure."""
        self._event_queue = None
        self._source_id = None
        self._executor = ProcessPoolExecutor(
            max_workers=self._settings.max_parallel_processes,
            mp_context=self._mp_context,
        )

    @override
    def _stop(self) -> None:
        self._executor.shutdown(wait=True)

        if self._has_session and self._session is not None:
            if self._session.event_processor is not None:
                self._session.event_processor.flush()

                if self._source_id is not None:
                    self._session.event_processor.remove_source(self._source_id)

        if self._event_queue is not None:
            self._event_queue.close()

    @override
    def _submit(
        self,
        func: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        context: BackendContext,
    ) -> Future[Any]:
        return self._executor.submit(func, *args, **kwargs)


@dataclass
class _ProcessWorkerPayload:
    """Serializable bundle sent to `ProcessPoolExecutor` workers via the initializer."""

    serialized_plugin_manager: dict[str, Any] | None
    event_queue: Any
    cache_store: Any


def _process_initializer(payload: _ProcessWorkerPayload) -> None:  # pragma: no cover
    """
    Initializes the `ProcessPoolExecutor` workers.

    Deserializes the plugin manager, creates a `ProcessEventReporter`, and pushes a `BackendContext`
    into the worker's context variable so that tasks have access to session infrastructure (events,
    plugins, cache).
    """
    from daglite.plugins.manager import deserialize_plugin_manager
    from daglite.plugins.reporters import ProcessEventReporter

    pm = (
        deserialize_plugin_manager(payload.serialized_plugin_manager)
        if payload.serialized_plugin_manager
        else None
    )
    event_reporter = ProcessEventReporter(payload.event_queue)

    ctx = BackendContext(
        backend="process",
        event_reporter=event_reporter,
        plugin_manager=pm,
        cache_store=payload.cache_store,
    )
    ctx.__enter__()
