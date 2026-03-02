"""Backend implementations for local execution (inline, threading, multiprocessing)."""

import contextvars
import os
import sys
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
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
    def map(self, task: Callable[..., Any], items: list[tuple[Any, ...]]) -> list[Any]:
        return [task(*args) for args in items]


class ThreadBackend(Backend):
    """
    Executes tasks across a thread pool.

    Each ``map`` call dispatches one future per item. The calling thread's
    context variables are copied into each worker so that the active
    `RunContext` is visible inside every thread.
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
    def map(self, task: Callable[..., Any], items: list[tuple[Any, ...]]) -> list[Any]:
        # ThreadPoolExecutor.submit() does NOT propagate context variables;
        # wrap each call so the active RunContext is visible in every worker.
        futures = [
            self._executor.submit(contextvars.copy_context().run, task, *args) for args in items
        ]
        return [f.result() for f in futures]


class ProcessBackend(Backend):
    """
    Executes tasks across a process pool.

    Worker processes receive a serialised plugin manager and a
    `ProcessEventReporter` backed by a shared multiprocessing queue so
    that events flow back to the coordinator.
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

        # Register queue with the session's event processor so events from
        # worker processes are dispatched on the coordinator side.
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

        self._executor = ProcessPoolExecutor(
            max_workers=self._settings.max_parallel_processes,
            mp_context=self._mp_context,
            initializer=_process_initializer,
            initargs=(
                serialized_pm,
                self._event_queue,
                self._ctx.cache_store,
                self._ctx.backend_name,
            ),
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
    def map(self, task: Callable[..., Any], items: list[tuple[Any, ...]]) -> list[Any]:
        futures = [self._executor.submit(task, *args) for args in items]
        return [f.result() for f in futures]


# region Worker initializers


def _process_initializer(
    serialized_plugin_manager: dict[str, Any] | None,
    event_queue: Any,
    cache_store: Any,
    backend_name: str,
) -> None:  # pragma: no cover
    """
    Initializer for `ProcessPoolExecutor` workers.

    Deserialises the plugin manager, creates a `ProcessEventReporter`, and
    pushes a `RunContext` into the worker's context variable so that eager
    tasks emit events back to the coordinator.
    """
    from daglite.session import RunContext
    from daglite.session import set_run_context

    pm = (
        deserialize_plugin_manager(serialized_plugin_manager) if serialized_plugin_manager else None
    )
    reporter = ProcessEventReporter(event_queue)

    ctx = RunContext(
        backend_name=backend_name,
        cache_store=cache_store,
        event_reporter=reporter,
        plugin_manager=pm,
    )
    set_run_context(ctx)
