"""
Logging plugin for cross-process/thread execution.

This module provides logging that works seamlessly across different execution backends
(threading, multiprocessing, distributed) by leveraging the event reporter system to
send log records from workers back to the coordinator/main process.

Example:
    >>> from daglite import evaluate, task
    >>> from daglite.plugins.default.logging import CentralizedLoggingPlugin, get_logger
    >>>
    >>> @task
    ... def my_task(x: int) -> int:
    ...     logger = get_logger(__name__)
    ...     logger.info(f"Processing {x}")
    ...     return x * 2
    >>> evaluate(my_task(x=10), plugins=[CentralizedLoggingPlugin()])
    20
"""

import logging
import threading
from typing import Any, MutableMapping

from daglite.backends.context import get_current_task
from daglite.backends.context import get_reporter
from daglite.plugins.base import BidirectionalPlugin
from daglite.plugins.events import EventRegistry
from daglite.plugins.reporters import EventReporter

LOGGER_EVENT = "daglite-log"
DEFAULT_LOGGER_NAME = "daglite"
DEFAULT_LOGGER_NAME_TASKS = "daglite.tasks"

# Lock to prevent race conditions when adding handlers (critical for free-threaded Python)
_logger_lock = threading.Lock()


def get_logger(name: str | None = None) -> logging.LoggerAdapter:
    """
    Get a logger instance that works across process/thread/machine boundaries.

    This is the main entry point into daglite logging for user code. It returns a standard
    Python `logging.LoggerAdapter` that automatically:
    - Injects task context (`daglite_task_name`, `daglite_task_id`, and `daglite_node_key`) into
      all log records
    - Uses the reporter system when available for centralized logging (requires
      CentralizedLoggingPlugin on coordinator side)
    - Works with standard Python logging when no reporter is available (sequential execution)

    Args:
        name: Logger name for code organization. If None, uses "daglite.tasks". Typically use
            `__name__` for module-based naming. Note: Task context (daglite_task_name,
            daglite_task_id, daglite_node_key) is automatically added to log records
            regardless of logger name and can be used in formatters.

    Returns:
        LoggerAdapter instance configured with current execution context and
        automatic task context injection

    Examples:
        >>> from daglite import task
        >>> from daglite.plugins.default import get_logger

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
    if get_current_task() is not None:
        name = DEFAULT_LOGGER_NAME_TASKS  # Called within a worker task context

    if name is None:
        name = DEFAULT_LOGGER_NAME

    base_logger = logging.getLogger(name)

    # Add ReporterHandler only for remote reporters
    reporter = get_reporter()
    if reporter and reporter.is_remote:  # pragma: no branch
        with _logger_lock:
            if not any(isinstance(hlr, _ReporterHandler) for hlr in base_logger.handlers):
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


class CentralizedLoggingPlugin(BidirectionalPlugin):
    """
    Plugin that enables centralized logging via the reporter system.

    This plugin centralizes logs from out-of-process or distributed workers to the coordinator. On
    the worker side, log records are sent to the coordinator using the reporter system. On the
    coordinator side, log records are reconstructed and emitted through the standard logging
    framework.

    Args:
        level: Minimum log level to handle on coordinator side (default: WARNING).

    Examples:
        >>> from daglite import evaluate, task
        >>> from daglite.plugins.default import CentralizedLoggingPlugin
        >>> import logging
        >>> @task
        ... def my_task(x: int) -> int:
        ...     logger = get_logger(__name__)
        ...     logger.info(f"Processing {x}")
        ...     return x * 2

        Configure logging format to include task context
        >>> logging.basicConfig(
        ...     format="%(daglite_task_name)s [%(levelname)s] %(message)s", level=logging.INFO
        ... )

        Add plugin to enable centralized logging
        >>> plugin = CentralizedLoggingPlugin(level=logging.INFO)
        >>> evaluate(my_task(x=10), plugins=[plugin])
        20
    """

    def __init__(self, level: int = logging.WARNING):
        self._level = level

    def register_event_handlers(self, registry: EventRegistry) -> None:
        """
        Register coordinator-side handler for log events.

        Args:
            registry: Event registry for registering handlers
        """
        registry.register(LOGGER_EVENT, self._handle_log_event)

    def _handle_log_event(self, event: dict[str, Any]) -> None:
        """
        Handle log event from worker.

        Reconstructs a log record and dispatches it through Python's logging system
        on the coordinator side.

        Args:
            event: Log event dict with name, level, message, and optional extras
        """
        logger_name = event.get("name", "daglite")
        level = event.get("level", "INFO")
        message = event.get("message", "")
        exc_info_str = event.get("exc_info")
        all_extra = event.get("extra", {})

        # Filter based on the plugin's configured minimum level
        log_level = getattr(logging, level, logging.INFO)
        if log_level < self._level:
            return

        # Format message with exception info if present
        if exc_info_str:
            message = f"{message}\n{exc_info_str}"

        # Separate standard LogRecord fields from custom extra fields
        # Standard fields must be passed as makeRecord parameters, not in extra dict
        standard_fields = {
            "filename",
            "pathname",
            "module",
            "funcName",
            "lineno",
            "created",
            "msecs",
            "relativeCreated",
            "process",
            "processName",
            "thread",
            "threadName",
            "taskName",
        }

        extra = {k: v for k, v in all_extra.items() if k not in standard_fields}

        # Emit record to coordinator-side logger (excluding ReporterHandler to avoid loops)
        base_logger = logging.getLogger(logger_name or DEFAULT_LOGGER_NAME_TASKS)
        record = base_logger.makeRecord(
            name=base_logger.name,
            level=log_level,
            fn=all_extra.get("filename", ""),
            lno=all_extra.get("lineno", 0),
            msg=message,
            args=(),
            exc_info=None,
            extra=extra,
        )

        # Restore standard fields that makeRecord doesn't set via parameters
        for field in [
            "pathname",
            "module",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "process",
            "processName",
            "thread",
            "threadName",
            "taskName",
        ]:
            if field in all_extra:
                setattr(record, field, all_extra[field])

        # Mark record to prevent re-emission by ReporterHandler (avoid infinite loops)
        setattr(record, "_daglite_already_forwarded", True)

        # Temporarily restore propagation for coordinator re-emission
        # (worker processes may have disabled it to prevent duplicates)
        original_propagate = base_logger.propagate
        base_logger.propagate = True
        try:
            # Use normal logging flow (includes propagation to parent loggers)
            base_logger.handle(record)
        finally:
            base_logger.propagate = original_propagate


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
            reporter: Event reporter for sending logs to coordinator
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

        Args:
            msg: Log message
            kwargs: Keyword arguments from log call

        Returns:
            Tuple of (message, modified kwargs with task context)
        """
        from daglite.backends.context import get_current_task

        extra = kwargs.get("extra", {})
        task = get_current_task()
        if task:
            extra.update(
                {
                    "daglite_task_id": str(task.id),
                    "daglite_task_name": task.name,
                    "daglite_node_key": task.key or "unknown",
                }
            )

        kwargs["extra"] = extra
        return msg, dict(kwargs)
