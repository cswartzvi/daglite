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
    _dataset_queue: Any
    _event_source_id: UUID | None
    _dataset_source_id: UUID | None
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

        assert self._session is not None

        self._event_queue = self._mp_context.Queue()
        self._dataset_queue = self._mp_context.Queue()

        # Register queues with the session's processors so events/dataset saves from
        # worker processes are dispatched on the coordinator side.
        self._event_source_id = None
        if self._session.event_processor is not None:
            self._event_source_id = self._session.event_processor.add_source(self._event_queue)

        self._dataset_source_id = None
        if self._session.dataset_processor is not None:
            self._dataset_source_id = self._session.dataset_processor.add_source(
                self._dataset_queue
            )

        # Build a BackendContext from the session and serialize it to a JSON-safe dict.
        context_data = BackendContext.from_session(backend=self.name).to_dict()

        payload = _ProcessPayload(
            context_data=context_data,
            event_queue=self._event_queue,
            dataset_queue=self._dataset_queue,
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
        self._dataset_queue = None
        self._event_source_id = None
        self._dataset_source_id = None
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

                if self._event_source_id is not None:
                    self._session.event_processor.remove_source(self._event_source_id)

            if self._session.dataset_processor is not None:
                self._session.dataset_processor.flush()

                if self._dataset_source_id is not None:
                    self._session.dataset_processor.remove_source(self._dataset_source_id)

        if self._event_queue is not None:
            self._event_queue.close()
        if self._dataset_queue is not None:
            self._dataset_queue.close()

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
class _ProcessPayload:
    """Serializable bundle sent to ``ProcessPoolExecutor`` workers via the initializer."""

    context_data: dict[str, Any]
    event_queue: Any
    dataset_queue: Any


def _process_initializer(payload: _ProcessPayload) -> None:  # pragma: no cover
    """
    Initializes the ``ProcessPoolExecutor`` workers.

    Rehydrates a ``BackendContext`` from the serialized dict and pushes it into the worker's
    context variable so that tasks have access to session infrastructure (events, plugins,
    cache, datasets).
    """
    from daglite.datasets.reporters import ProcessDatasetReporter
    from daglite.plugins.reporters import ProcessEventReporter

    event_reporter = ProcessEventReporter(payload.event_queue)
    dataset_reporter = ProcessDatasetReporter(payload.dataset_queue)

    ctx = BackendContext.from_dict(
        payload.context_data,
        event_reporter=event_reporter,
        dataset_reporter=dataset_reporter,
    )
    ctx.__enter__()
