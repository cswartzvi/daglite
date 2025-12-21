"""
Backend implementations for local execution (direct, threading, multiprocessing).

Warning: This module is intended for internal use only.
"""

import os
import sys
from concurrent.futures import Future
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable
from uuid import UUID

from pluggy import PluginManager
from typing_extensions import override

from daglite.backends.base import Backend
from daglite.backends.context import reset_execution_context
from daglite.backends.context import set_execution_context
from daglite.plugins.manager import deserialize_plugin_manager
from daglite.plugins.manager import serialize_plugin_manager
from daglite.plugins.reporters import DirectReporter
from daglite.plugins.reporters import EventReporter
from daglite.plugins.reporters import ProcessReporter
from daglite.plugins.reporters import ThreadReporter
from daglite.settings import get_global_settings


class SequentialBackend(Backend):
    """Executes immediately in current thread/process, returns completed futures."""

    @override
    def _get_reporter(self) -> DirectReporter:
        return DirectReporter(self._event_processor.dispatch)

    @override
    def _submit_impl(
        self, func: Callable[[dict[str, Any]], Any], inputs: dict[str, Any], **kwargs: Any
    ) -> Future[Any]:
        future: Future[Any] = Future()

        # Set execution context for immediate execution (runs in main thread)
        # Context cleanup happens when backend stops, not per-task
        set_execution_context(self._plugin_manager, self._reporter)

        try:
            result = func(inputs, **kwargs)
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)

        return future


class ThreadBackend(Backend):
    """Executes in thread pool, returns pending futures."""

    _executor: ThreadPoolExecutor
    _reporter_id: UUID

    @override
    def _get_reporter(self) -> ThreadReporter:
        from queue import Queue

        queue: Queue[Any] = Queue()
        return ThreadReporter(queue)

    @override
    def _start(self) -> None:
        settings = get_global_settings()
        max_workers = settings.max_backend_threads

        assert isinstance(self._reporter, ThreadReporter)
        self._reporter_id = self._event_processor.add_source(self._reporter.queue)
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            initializer=_thread_initializer,
            initargs=(self._plugin_manager, self._reporter),
        )

    @override
    def _stop(self) -> None:
        self._executor.shutdown(wait=True)
        self._event_processor.remove_source(self._reporter_id)

    @override
    def _submit_impl(
        self, func: Callable[[dict[str, Any]], Any], inputs: dict[str, Any], **kwargs: Any
    ) -> Future[Any]:
        return self._executor.submit(func, inputs, **kwargs)


class ProcessBackend(Backend):
    """Executes in process pool, returns pending futures."""

    _executor: ProcessPoolExecutor
    _reporter_id: UUID

    @override
    def _get_reporter(self) -> ProcessReporter:
        from multiprocessing import Queue

        queue: Queue[Any] = Queue()
        return ProcessReporter(queue)

    @override
    def _start(self) -> None:
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

        assert isinstance(self._reporter, ProcessReporter)
        self._reporter_id = self._event_processor.add_source(self._reporter.queue)
        serialized_pm = serialize_plugin_manager(self._plugin_manager)
        self._executor = ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=mp_context,
            initializer=_process_initilizer,
            initargs=(serialized_pm, self._reporter.queue),
        )

    @override
    def _stop(self) -> None:
        self._executor.shutdown(wait=True)
        self._event_processor.remove_source(self._reporter_id)

        assert isinstance(self._reporter, ProcessReporter)
        self._reporter.queue.close()

    @override
    def _submit_impl(self, func, inputs: dict[str, Any], **kwargs: Any) -> Future[Any]:
        return self._executor.submit(func, inputs, **kwargs)


def _thread_initializer(plugin_manager: PluginManager, reporter: EventReporter) -> None:
    """Initializer for thread pool workers to set execution context."""
    set_execution_context(plugin_manager, reporter)


def _process_initilizer(serialized_plugin_manager: dict, queue) -> None:
    """Initializer for process pool workers to set execution context."""
    plugin_manager = deserialize_plugin_manager(serialized_plugin_manager)
    reporter = ProcessReporter(queue)
    set_execution_context(plugin_manager, reporter)
