"""
Backend implementations for local execution (direct, threading, multiprocessing).

Warning: This module is intended for internal use only.
"""

import asyncio
import inspect
import os
import sys
from concurrent.futures import Future
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as CFTimeoutError
from typing import Any, Callable
from uuid import UUID

from pluggy import PluginManager
from typing_extensions import override

from daglite.backends.base import Backend
from daglite.backends.context import set_execution_context
from daglite.plugins.manager import deserialize_plugin_manager
from daglite.plugins.manager import serialize_plugin_manager
from daglite.plugins.reporters import DirectReporter
from daglite.plugins.reporters import EventReporter
from daglite.plugins.reporters import ProcessReporter
from daglite.settings import get_global_settings


class SequentialBackend(Backend):
    """Executes immediately in current thread/process, returns completed futures."""

    _timeout_executor: ThreadPoolExecutor

    @override
    def _get_reporter(self) -> DirectReporter:
        return DirectReporter(self.event_processor.dispatch)

    @override
    def _start(self) -> None:
        # Small thread pool for timeout enforcement on sync tasks
        self._timeout_executor = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="seq-timeout-",
            initializer=_thread_initializer,
            initargs=(self.plugin_manager, self.reporter),
        )

    @override
    def _stop(self) -> None:
        self._timeout_executor.shutdown(wait=True)

    @override
    def submit(
        self,
        func: Callable[[dict[str, Any]], Any],
        inputs: dict[str, Any],
        timeout: float | None = None,
        **kwargs: Any,
    ) -> Future[Any]:
        future: Future[Any] = Future()

        # Set execution context for immediate execution (runs in main thread)
        # Context cleanup happens when backend stops, not per-task
        set_execution_context(self.plugin_manager, self.reporter)

        try:
            if inspect.iscoroutinefunction(func):
                coro = func(inputs, **kwargs)

                if timeout is not None:
                    coro = asyncio.wait_for(coro, timeout=timeout)

                def _on_done(f):
                    try:
                        future.set_result(f.result())
                    except asyncio.TimeoutError:
                        future.set_exception(TimeoutError(f"Task exceeded timeout of {timeout}s"))
                    except Exception as e:
                        future.set_exception(e)

                asyncio_future = asyncio.ensure_future(coro)
                asyncio_future.add_done_callback(_on_done)
            else:
                if timeout is not None:
                    executor_future = self._timeout_executor.submit(func, inputs, **kwargs)
                    wrapped_future: Future[Any] = Future()
                    self._timeout_executor.submit(
                        _wait_with_timeout, executor_future, wrapped_future, timeout
                    )
                    return wrapped_future
                else:
                    result = func(inputs, **kwargs)
                    future.set_result(result)

        except Exception as e:
            future.set_exception(e)

        return future


class ThreadBackend(Backend):
    """Executes in thread pool, returns pending futures."""

    _executor: ThreadPoolExecutor
    _timeout_executor: ThreadPoolExecutor

    @override
    def _get_reporter(self) -> DirectReporter:
        # Threads run in same process - use DirectReporter with dispatcher
        return DirectReporter(self.event_processor.dispatch)

    @override
    def _start(self) -> None:
        settings = get_global_settings()
        max_workers = settings.max_backend_threads

        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            initializer=_thread_initializer,
            initargs=(self.plugin_manager, self.reporter),
        )
        self._timeout_executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="thread-timeout-"
        )

    @override
    def _stop(self) -> None:
        self._executor.shutdown(wait=True)
        self._timeout_executor.shutdown(wait=True)

    @override
    def submit(
        self,
        func: Callable[[dict[str, Any]], Any],
        inputs: dict[str, Any],
        timeout: float | None = None,
        **kwargs: Any,
    ) -> Future[Any]:
        # Submit to executor first
        if inspect.iscoroutinefunction(func):
            executor_future = self._executor.submit(_run_coro_in_worker, func, inputs, **kwargs)
        else:
            executor_future = self._executor.submit(func, inputs, **kwargs)

        if timeout is None:
            return executor_future

        # Use dedicated timeout executor to avoid consuming task execution threads
        wrapped_future: Future[Any] = Future()
        self._timeout_executor.submit(_wait_with_timeout, executor_future, wrapped_future, timeout)

        return wrapped_future


class ProcessBackend(Backend):
    """Executes in process pool, returns pending futures."""

    _executor: ProcessPoolExecutor
    _timeout_executor: ThreadPoolExecutor
    _reporter_id: UUID
    _mp_context: Any  # BaseContext, but we can't import it at class level

    @override
    def _get_reporter(self) -> ProcessReporter:
        from multiprocessing import Queue
        from multiprocessing import get_context

        if os.name == "nt" or sys.platform == "darwin":  # pragma: no cover
            # Use 'spawn' on Windows (required) and macOS (fork deprecated)
            self._mp_context = get_context("spawn")

        elif (
            sys.version_info >= (3, 13)
            and sys.version_info < (3, 14)
            and not getattr(sys, "_is_gil_enabled", lambda: True)()
        ):  # pragma: no cover
            # Use 'spawn' for Python 3.13t (free-threaded builds with GIL disabled). Fork is
            # incompatible with free-threading in 3.13t, causing hangs. Python 3.14 defaults to
            # 'forkserver', so this workaround is only needed for 3.13t.
            self._mp_context = get_context("spawn")

        elif sys.version_info >= (3, 14):  # pragma: no cover
            # Use 'forkserver' on Python 3.14+ (safe with event loops, and it's the new default)
            self._mp_context = get_context("forkserver")

        else:  # pragma: no cover
            # Use 'fork' on Linux with Python < 3.14 (fast startup, safe without event loops)
            # This path is not covered in CI which runs Python 3.14+
            self._mp_context = get_context("fork")

        # We need to defer Queue creation until we know the context
        queue: Queue[Any] = self._mp_context.Queue()
        return ProcessReporter(queue)

    @override
    def _start(self) -> None:
        settings = get_global_settings()
        max_workers = settings.max_parallel_processes

        assert isinstance(self.reporter, ProcessReporter)
        self._reporter_id = self.event_processor.add_source(self.reporter.queue)
        serialized_pm = serialize_plugin_manager(self.plugin_manager)
        self._executor = ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=self._mp_context,
            initializer=_process_initializer,
            initargs=(serialized_pm, self.reporter.queue),
        )
        self._timeout_executor = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="proc-timeout-",
            initializer=_thread_initializer,
            initargs=(self.plugin_manager, self.reporter),
        )

    @override
    def _stop(self) -> None:
        self._executor.shutdown(wait=True)
        self._timeout_executor.shutdown(wait=True)
        self.event_processor.flush()  # Before removing source
        self.event_processor.remove_source(self._reporter_id)

        assert isinstance(self.reporter, ProcessReporter)
        self.reporter.queue.close()

    @override
    def submit(
        self,
        func: Callable[[dict[str, Any]], Any],
        inputs: dict[str, Any],
        timeout: float | None = None,
        **kwargs: Any,
    ) -> Future[Any]:
        # Submit to executor first
        if inspect.iscoroutinefunction(func):
            executor_future = self._executor.submit(_run_coro_in_worker, func, inputs, **kwargs)
        else:
            executor_future = self._executor.submit(func, inputs, **kwargs)

        if timeout is None:
            return executor_future

        # Use dedicated timeout executor to avoid unbounded thread creation
        wrapped_future: Future[Any] = Future()
        self._timeout_executor.submit(_wait_with_timeout, executor_future, wrapped_future, timeout)

        return wrapped_future


def _run_coro_in_worker(func: Callable, inputs: dict[str, Any], **kwargs: Any) -> Any:
    """
    Run an async function to completion in a worker thread/process.

    Uses asyncio.run() to create an isolated event loop for each task. This works correctly in both
    sync and async contexts because the worker thread/process has no running event loop of its own.

    Args:
        func: Async function to run.
        inputs: Inputs to pass to the function.
        **kwargs: Additional keyword arguments to pass to the function.

    Returns:
        The result of the async function.
    """
    return asyncio.run(func(inputs, **kwargs))


def _wait_with_timeout(
    executor_future: Future[Any], wrapped_future: Future[Any], timeout: float
) -> None:
    """
    Wait for executor future with timeout and propagate result to wrapped future.

    Args:
        executor_future: The future returned by the executor.
        wrapped_future: The future to set the result or exception on.
        timeout: Timeout in seconds.
    """
    try:
        result = executor_future.result(timeout=timeout)
        wrapped_future.set_result(result)
    except (TimeoutError, CFTimeoutError):
        # Explicitly catch both built-in TimeoutError and concurrent.futures.TimeoutError
        # In Python 3.10+, they should be aliased, but exception handling may differ
        wrapped_future.set_exception(TimeoutError(f"Task exceeded timeout of {timeout}s"))
    except Exception as e:  # pragma: no cover
        wrapped_future.set_exception(e)


def _thread_initializer(plugin_manager: PluginManager, reporter: EventReporter) -> None:
    """Initializer for thread pool workers to set execution context."""
    set_execution_context(plugin_manager, reporter)


def _process_initializer(serialized_plugin_manager: dict, queue) -> None:  # pragma: no cover
    """Initializer for process pool workers to set execution context."""
    plugin_manager = deserialize_plugin_manager(serialized_plugin_manager)
    reporter = ProcessReporter(queue)
    set_execution_context(plugin_manager, reporter)
