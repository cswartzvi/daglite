"""
Logging plugin for cross-process/thread execution.

This module provides logging that works seamlessly across different execution backends
(threading, multiprocessing, distributed) by leveraging the event reporter system to
send log records from workers back to the coordinator/main process.

Example:
    >>> from daglite.plugins.default import LoggingPlugin, get_logger
    >>> from daglite import evaluate, task
    >>>
    >>> @task
    >>> def my_task(x):
    ...     logger = get_logger(__name__)
    ...     logger.info(f"Processing {x}")
    ...     return x * 2
    >>>
    >>> evaluate(my_task(10), plugins=[LoggingPlugin()])
"""

import logging
from typing import Any, MutableMapping

from daglite.backends.context import get_reporter
from daglite.plugins.base import BidirectionalPlugin
from daglite.plugins.events import EventRegistry
from daglite.plugins.reporters import EventReporter

LOGGER_EVENT = "daglite-log"
DEFAULT_LOGGER_NAME = "daglite.tasks"
DEFAULT_LOGGER_FORMAT = "%(asctime)s - Node: %(daglite_task_name)s - %(levelname)s - %(message)s"


class ReporterHandler(logging.Handler):
    """
    Logging handler that sends log records via EventReporter to the coordinator.

    This handler integrates with Python's standard logging system to transparently
    route log records across process/thread boundaries using the reporter system.

    Note: This handler is automatically added to loggers returned by get_logger()
    when a reporter is available in the execution context.
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
        try:
            # Build log event payload
            payload: dict[str, Any] = {
                "name": record.name,
                "level": record.levelname,
                "message": record.getMessage(),
            }

            # Add exception info if present
            if record.exc_info:
                import traceback

                payload["exc_info"] = "".join(traceback.format_exception(*record.exc_info))

            # Add extra context from LogRecord
            extra = {}
            for key, value in record.__dict__.items():
                # Skip standard LogRecord attributes
                if key not in [
                    "name",
                    "msg",
                    "args",
                    "created",
                    "filename",
                    "funcName",
                    "levelname",
                    "levelno",
                    "lineno",
                    "module",
                    "msecs",
                    "message",
                    "pathname",
                    "process",
                    "processName",
                    "relativeCreated",
                    "thread",
                    "threadName",
                    "taskName",  # Thread task name (asyncio)
                    "exc_info",
                    "exc_text",
                    "stack_info",
                ]:
                    extra[key] = value

            if extra:
                payload["extra"] = extra

            # Send via reporter
            self._reporter.report(LOGGER_EVENT, payload)
        except Exception:
            self.handleError(record)


class TaskLoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that automatically injects task context into log records.

    This adapter adds task_id and task_name to the 'extra' dict of all log records,
    making them available for use in log formatters (e.g., "%(task_name)s").

    The task context is automatically derived from the current execution context
    when available, requiring no manual setup from users.
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


class LoggingPlugin(BidirectionalPlugin):
    """
    Plugin that enables cross-process logging via the reporter system.

    Worker side - Logs are sent via EventReporter
    Coordinator side - Logs are received and dispatched to Python's logging system

    Args:
        level: Minimum log level to handle on coordinator side.
        format: Log format string for coordinator-side formatting. If None, uses a default
            format with task name and timestamp. Use `%(daglite_task_name)s`,
            `%(daglite_node_key)s`, and `%(daglite_task_id)s` to include task context in output.

    Examples:
        >>> from daglite.plugins.default import LoggingPlugin
        >>> plugin = LoggingPlugin(level=logging.INFO)
        >>> pipeline = Pipeline(plugins=[plugin])
    """

    def __init__(self, level: int = logging.WARNING, format: str | None = None):
        self._level = level
        self._format = format or DEFAULT_LOGGER_FORMAT
        self._formatter = logging.Formatter(self._format, defaults={"daglite_task_name": "unknown"})

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
        extra = event.get("extra", {})

        # Check level against plugin's configured minimum level
        log_level = getattr(logging, level, logging.INFO)

        # Filter based on the plugin's configured minimum level
        if log_level < self._level:
            return

        # Format message with exception info if present
        if exc_info_str:
            message = f"{message}\n{exc_info_str}"

        # Create a LogRecord directly and emit to the base logger's handlers
        # We bypass the normal logging path to avoid triggering ReporterHandler
        # which would create an infinite loop (log -> reporter -> log -> ...)
        base_logger = logging.getLogger(logger_name or DEFAULT_LOGGER_NAME)

        # Create record manually
        record = base_logger.makeRecord(
            name=base_logger.name,
            level=log_level,
            fn="",
            lno=0,
            msg=message,
            args=(),
            exc_info=None,
            extra=extra,
        )

        # Emit to all handlers EXCEPT ReporterHandler
        for handler in base_logger.handlers:
            if not isinstance(handler, ReporterHandler):
                handler.handle(record)  # pragma: no cover


def get_logger(name: str | None = None) -> logging.LoggerAdapter:
    """
    Get a logger instance that works across process/thread boundaries.

    This is the main entry point for user code. It returns a standard python
    `logging.LoggerAdapter` that automatically:
    - Uses the reporter system when available for cross-process logging
    - Injects task context (task_id, task_name) into all log records
    - Falls back to standard logging when no reporter is available


    Args:
        name: Logger name for code organization. If None, uses "daglite.tasks". Typically use
            `__name__` for module-based naming. Note: Task context (daglite_task_name,
            daglite_task_id, daglite_node_key) is automatically added to log records
            regardless of logger name and can be used in formatters.

    Returns:
        LoggerAdapter instance configured with current execution context and
        automatic task context injection

    Examples:
        >>> from daglite.plugins.default import get_logger
        >>>

        Simple usage - automatic task context in logs
        >>> @task
        >>> def my_task(x):
        ...     logger = get_logger()  # Uses "daglite.tasks" logger
        ...     logger.info(f"Processing {x}")  # Output: "Node: my_task - ..."
        ...     return x * 2

        Module-based naming for code organization
        >>> @task
        >>> def custom_logging(x):
        ...     logger = get_logger(__name__)  # Uses module name
        ...     logger.info(f"Custom log for {x}")  # Still has task_name in output
        ...     return x

        Configure logging with custom format
        >>> import logging
        >>> logging.basicConfig(
        ...     format="%(daglite_task_name)s [%(levelname)s] %(message)s", level=logging.INFO
        ... )
    """
    if name is None:
        name = DEFAULT_LOGGER_NAME

    base_logger = logging.getLogger(name)

    # Add ReporterHandler if reporter available and not already added
    reporter = get_reporter()
    if reporter:  # pragma: no branch
        if not any(isinstance(hlr, ReporterHandler) for hlr in base_logger.handlers):
            handler = ReporterHandler(reporter)
            base_logger.addHandler(handler)

    return TaskLoggerAdapter(base_logger, {})
