"""
Logging plugin for cross-process/thread execution.

This module provides logging that works seamlessly across different execution backends
(threading, multiprocessing, distributed) by leveraging the event reporter system to
send log records from workers back to the coordinator/main process.
"""

import logging
import logging.config
import threading
from collections.abc import MutableMapping
from typing import Any
from uuid import UUID

from daglite._resolvers import resolve_event_reporter
from daglite._resolvers import resolve_task_metadata
from daglite.plugins.reporters import EventReporter

LOGGER_EVENT = "daglite.logging"  # Event name for log records sent via reporter

DEFAULT_LOGGER_NAME_COORD = "daglite.lifecycle"  # Coordinator-side default logger
DEFAULT_LOGGER_NAME_TASKS = "daglite.tasks"  # Worker-side default logger

TASK_EXTRA_ID = "daglite_task_id"  # LogRecord extra field for task ID
TASK_EXTRA_NAME = "daglite_task_name"  # LogRecord extra field for task name
TASK_EXTRA_KEY = "daglite_task_key"  # LogRecord extra field for task key (name + map index)


# Lock to prevent race conditions when adding handlers (critical for free-threaded Python)
_logger_lock = threading.Lock()


def get_logger(name: str | None = None) -> logging.LoggerAdapter:
    """
    Get a logger instance that works across process/thread/machine boundaries.

    This is the main entry point into daglite logging for user code. It returns a standard
    Python `logging.LoggerAdapter` that automatically:
    - Injects task context (`daglite_task_name`, `daglite_task_id`, and `daglite_task_key`) into
      all log records
    - Uses the reporter system when available for centralized logging (requires
      CentralizedLoggingPlugin on coordinator side)
    - Works with standard Python logging when no reporter is available (Inline execution)

    Args:
        name: Logger name for code organization. If None, uses "daglite.tasks". Typically use
            `__name__` for module-based naming. Note: Task context (daglite_task_name,
            daglite_task_id, daglite_task_key) is automatically added to log records
            regardless of logger name and can be used in formatters.

    Returns:
        LoggerAdapter instance configured with current execution context and
        automatic task context injection

    Examples:
        >>> from daglite import task
        >>> from daglite.logging import get_logger

        Simple usage - automatic task context in logs
        >>> @task
        ... def my_task(x):
        ...     logger = get_logger()  # Uses "daglite.tasks" logger
        ...     logger.info(f"Processing {x}")  # Output: "Node: my_task - ..."
        ...     return x * 2

        Module-based naming for code organization
        >>> @task
        ... def custom_logging(x):
        ...     logger = get_logger(__name__)  # Uses module name
        ...     logger.info(f"Custom log for {x}")  # Still has task_name in output
        ...     return x

        Configure logging with custom format
        >>> import logging
        >>> logging.basicConfig(
        ...     format="%(daglite_task_name)s [%(levelname)s] %(message)s", level=logging.INFO
        ... )
    """
    if resolve_task_metadata() is not None:
        name = DEFAULT_LOGGER_NAME_TASKS

    if name is None:
        name = DEFAULT_LOGGER_NAME_COORD

    base_logger = logging.getLogger(name)

    # Add ReporterHandler only for all non-direct reporters to route logs to coordinator.
    reporter = resolve_event_reporter()
    if reporter and not reporter.is_direct:
        with _logger_lock:
            if not any(isinstance(hlr, _ReporterHandler) for hlr in base_logger.handlers):
                # In worker processes, remove all existing handlers and only use ReporterHandler
                # The coordinator will receive logs via the reporter and re-emit them through
                # its own handlers (console, file, etc.)
                base_logger.handlers.clear()

                handler = _ReporterHandler(reporter)
                base_logger.addHandler(handler)

                # Set logger to DEBUG to prevent filtering before handler. Actual filtering happens
                # on coordinator side via CentralizedLoggingPlugin level.
                if base_logger.getEffectiveLevel() > logging.DEBUG:  # pragma: no branch
                    base_logger.setLevel(logging.DEBUG)

                # Disable propagation to prevent duplicate logging from inherited handlers.
                # Worker processes send logs ONLY via ReporterHandler; coordinator re-emits.
                base_logger.propagate = False

    return _TaskLoggerAdapter(base_logger, {})


class _ReporterHandler(logging.Handler):
    """
    Logging handler that sends log records via EventReporter to the coordinator.

    This handler integrates with Python's standard logging system to transparently route log
    records across process/thread boundaries using the reporter system.

    Note: This handler is automatically added to loggers returned by `get_logger()` when a reporter
    is available in the execution context.
    """

    def __init__(self, reporter: EventReporter):
        """
        Initialize handler with event reporter.

        Args:
            reporter: PluginEvent reporter for sending logs to coordinator
        """
        super().__init__()
        self._reporter = reporter

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a log record by sending it via the reporter system.

        Args:
            record: Log record to emit
        """
        # Skip records that were already forwarded (re-emitted by coordinator)
        if getattr(record, "_daglite_already_forwarded", False):
            return

        try:
            # Build log event payload
            payload: dict[str, Any] = {
                "name": record.name,
                "level": record.levelname,
                "message": record.getMessage(),
            }

            if record.exc_info:
                import traceback

                payload["exc_info"] = "".join(traceback.format_exception(*record.exc_info))

            # Add all LogRecord attributes to payload (skip only internal fields)
            # This allows users to use standard format strings like %(filename)s:%(lineno)d
            extra = {}
            for key, value in record.__dict__.items():
                if key not in [
                    "name",  # Sent separately
                    "msg",  # Internal - we send formatted message
                    "args",  # Internal - we send formatted message
                    "levelname",  # Sent separately as 'level'
                    "levelno",  # Internal int - levelname is the string version
                    "message",  # Sent separately
                    "exc_info",  # Handled separately
                    "exc_text",  # Internal formatting cache
                    "stack_info",  # Handled via exc_info
                ]:
                    extra[key] = value

            if extra:  # pragma: no branch
                payload["extra"] = extra

            self._reporter.report(LOGGER_EVENT, payload)
        except Exception:  # pragma: no cover
            self.handleError(record)


class _TaskLoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that automatically injects task context into log records.

    The task context is automatically derived from the current execution context when available,
    requiring no manual setup from users.
    """

    def process(self, msg: Any, kwargs: MutableMapping[str, Any]) -> tuple[Any, dict[str, Any]]:
        """
        Process log call to inject task context.

        Injects ``daglite_task_id``, ``daglite_task_name``, and ``daglite_task_key`` into the log
        record's extra dict. When a :class:`TaskContext` is active (inside a task), values are
        derived automatically. Otherwise, safe empty defaults are set so formatters using
        ``%(daglite_task_key)s`` never crash. Callers can always override via explicit
        ``extra=build_task_extra(...)`` — ``setdefault`` ensures explicit values take priority.

        Args:
            msg: Log message
            kwargs: Keyword arguments from log call

        Returns:
            Tuple of (message, modified kwargs with task context)
        """
        extra = kwargs.get("extra", {})

        meta = resolve_task_metadata()
        if meta is not None:
            extra.setdefault(TASK_EXTRA_ID, str(meta.id))
            extra.setdefault(TASK_EXTRA_NAME, meta.name)
            extra.setdefault(TASK_EXTRA_KEY, meta.name)
        else:
            extra.setdefault(TASK_EXTRA_ID, "")
            extra.setdefault(TASK_EXTRA_NAME, "")
            extra.setdefault(TASK_EXTRA_KEY, "")

        kwargs["extra"] = extra
        return msg, dict(kwargs)


def build_task_extra(task_id: UUID, task_name: str) -> dict[str, str]:
    """Build task context dict for logging extra fields."""
    return {TASK_EXTRA_ID: str(task_id), TASK_EXTRA_NAME: task_name, TASK_EXTRA_KEY: task_name}


def format_duration(duration: float) -> str:
    """Format duration in seconds to human-readable string."""
    if duration < 1:
        return f"{duration * 1000:.0f} ms"
    elif duration < 60:
        return f"{duration:.2f} s"
    else:
        minutes = int(duration // 60)
        seconds = duration % 60
        return f"{minutes} min {seconds:.2f} s"
