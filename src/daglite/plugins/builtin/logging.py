"""
Logging plugin for cross-process/thread execution.

This module provides logging that works seamlessly across different execution backends
(threading, multiprocessing, distributed) by leveraging the event reporter system to
send log records from workers back to the coordinator/main process.
"""

import json
import logging
import logging.config
import threading
from pathlib import Path
from typing import Any, MutableMapping
from uuid import UUID

from typing_extensions import override

from daglite.backends.context import get_current_task
from daglite.backends.context import get_event_reporter
from daglite.graph.nodes.base import NodeMetadata
from daglite.plugins.base import BidirectionalPlugin
from daglite.plugins.base import SerializablePlugin
from daglite.plugins.hooks.markers import hook_impl
from daglite.plugins.registry import EventRegistry
from daglite.plugins.reporters import EventReporter

LOGGER_EVENT = "daglite-log"
DEFAULT_LOGGER_NAME_COORD = "daglite.lifecycle"  # Coordinator-side default logger
DEFAULT_LOGGER_NAME_TASKS = "daglite.tasks"  # Worker-side default logger

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
        >>> from daglite.plugins.builtin.logging import get_logger

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
        name = DEFAULT_LOGGER_NAME_COORD

    base_logger = logging.getLogger(name)

    # Add ReporterHandler only for all non-direct reporters to route logs to coordinator.
    reporter = get_event_reporter()
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


class CentralizedLoggingPlugin(BidirectionalPlugin):
    """
    Plugin that enables centralized logging via the reporter system.

    This plugin centralizes logs from out-of-process or distributed workers to the coordinator. On
    the worker side, log records are sent to the coordinator using the reporter system. On the
    coordinator side, log records are reconstructed and emitted through the standard logging
    framework.

    Args:
        level: Minimum log level to handle on coordinator side (default: WARNING).
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
            "asctime",  # Generated by formatters, not allowed in extra
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

        # Use normal logging flow with configured propagation settings
        base_logger.handle(record)


class LifecycleLoggingPlugin(CentralizedLoggingPlugin, SerializablePlugin):
    """
    Plugin that logs node lifecycle events (start, completion, failure).

    This plugin extends CentralizedLoggingPlugin to log key task lifecycle events such as
    task start, successful completion, and failure. It provides visibility into task execution
    flow in the logs.

    Args:
        name: Optional logger name to use (default: "daglite").
        level: Optional minimum log level to handle on coordinator side (default: INFO).
    """

    __config_attrs__: list[str] = ["mapped_nodes"]

    def __init__(
        self,
        name: str | None = None,
        level: int = logging.INFO,
        config: dict[str, Any] | None = None,
    ):
        super().__init__(level=level)
        self._logger = get_logger(name)
        self._mapped_nodes: set[UUID] = set()

        # Set logger level to ensure debug messages aren't filtered out.
        self._logger.logger.setLevel(level)

        # Load logging config if not provided
        config = config if config is not None else self._load_default_config()
        self._apply_logging_config(config)

    def _load_default_config(self) -> dict[str, Any]:
        """Load default logging configuration from logging.json."""
        config_path = Path(__file__).parent / "logging.json"
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)
        else:  # pragma: no cover
            raise FileNotFoundError(f"Default logging configuration not found at {config_path}")

    def _apply_logging_config(self, config: dict[str, Any]) -> None:
        """Apply logging configuration using dictConfig."""
        logging.config.dictConfig(config)

    @override
    def to_config(self) -> dict[str, Any]:
        return {"mapped_nodes": list(self._mapped_nodes)}

    @classmethod
    @override
    def from_config(cls, config: dict[str, Any]) -> "LifecycleLoggingPlugin":
        instance = cls()  # Use defaults when deserializing on workers
        instance._mapped_nodes = set(config.get("mapped_nodes", []))
        return instance

    def register_event_handlers(self, registry: EventRegistry) -> None:
        """
        Register event handlers for task lifecycle events.

        Args:
            registry: Event registry for registering handlers
        """
        super().register_event_handlers(registry)
        registry.register("daglite-logging-node-start", self._handle_node_start)
        registry.register("daglite-logging-node-complete", self._handle_node_complete)
        registry.register("daglite-logging-node-fail", self._handle_task_fail)
        registry.register("daglite-logging-node-retry", self._handle_node_retry)
        registry.register("daglite-logging-node-retry-result", self._handle_node_retry_result)

    @hook_impl
    def before_graph_execute(
        self,
        graph_id: UUID,
        root_id: UUID,
        node_count: int,
    ) -> None:
        self._logger.info(f"Starting evaluation {graph_id}")
        self._logger.debug(f"Evaluation {graph_id}: Computing {node_count} tasks total")
        self._logger.debug(f"Evaluation {graph_id}: Root task ID is {root_id}")

    @hook_impl
    def after_graph_execute(
        self,
        graph_id: UUID,
        root_id: UUID,
        result: Any,
        duration: float,
    ) -> None:
        self._logger.info(
            f"Completed evaluation {graph_id} successfully in {_format_duration(duration)}"
        )

    @hook_impl
    def on_graph_error(
        self,
        graph_id: UUID,
        root_id: UUID,
        error: Exception,
        duration: float,
    ) -> None:
        self._logger.error(
            f"Evaluation {graph_id} failed after {_format_duration(duration)} with error: {error}"
        )

    @hook_impl
    def before_mapped_node_execute(
        self,
        metadata: NodeMetadata,
        iteration_count: int,
    ) -> None:
        self._mapped_nodes.add(metadata.id)
        # Coordinator-side hooks need manual task context since get_current_task() returns None.
        # This enables format strings like %(daglite_task_name)s to work in log output.
        node_key = metadata.key or metadata.name
        backend_name = metadata.backend_name or "inline"
        self._logger.info(
            f"Task '{node_key}' - Starting task with {iteration_count} iterations using "
            f"{backend_name} backend",
            extra=_build_task_context(metadata.id, metadata.name, metadata.key),
        )

    @hook_impl
    def before_node_execute(
        self,
        metadata: NodeMetadata,
        inputs: dict[str, Any],
        reporter: EventReporter | None,
    ) -> None:
        data = {
            "node_id": metadata.id,
            "node_key": metadata.key,
            "backend_name": metadata.backend_name,
        }
        if reporter:
            reporter.report("daglite-logging-node-start", data=data)
        else:  # pragma: no cover
            self._handle_node_start(data)  # Fallback if no reporter is available

    @hook_impl
    def after_node_execute(
        self,
        metadata: NodeMetadata,
        inputs: dict[str, Any],
        result: Any,
        duration: float,
        reporter: EventReporter | None,
    ) -> None:
        data = {"node_id": metadata.id, "node_key": metadata.key, "duration": duration}
        if reporter:
            reporter.report("daglite-logging-node-complete", data=data)
        else:  # pragma: no cover
            self._handle_node_complete(data)  # Fallback if no reporter is available

    @hook_impl
    def on_node_error(
        self,
        metadata: NodeMetadata,
        inputs: dict[str, Any],
        error: Exception,
        duration: float,
        reporter: EventReporter | None,
    ) -> None:
        data = {
            "node_id": metadata.id,
            "node_key": metadata.key,
            "error": str(error),
            "error_type": type(error).__name__,
            "duration": duration,
        }
        if reporter:
            reporter.report("daglite-logging-node-fail", data=data)
        else:  # pragma: no cover
            self._handle_task_fail(data)  # Fallback if no reporter is available

    @hook_impl
    def before_node_retry(
        self,
        metadata: NodeMetadata,
        inputs: dict[str, Any],
        attempt: int,
        last_error: Exception,
        reporter: EventReporter | None,
    ) -> None:
        data = {
            "node_id": metadata.id,
            "node_key": metadata.key,
            "attempt": attempt,
            "error": str(last_error),
            "error_type": type(last_error).__name__,
        }
        if reporter:
            reporter.report("daglite-logging-node-retry", data=data)
        else:  # pragma: no cover
            self._handle_node_retry(data)  # Fallback if no reporter is available

    @hook_impl
    def after_node_retry(
        self,
        metadata: NodeMetadata,
        inputs: dict[str, Any],
        attempt: int,
        succeeded: bool,
        reporter: EventReporter | None,
    ) -> None:
        data = {
            "node_id": metadata.id,
            "node_key": metadata.key,
            "attempt": attempt,
            "succeeded": succeeded,
        }
        if reporter:
            reporter.report("daglite-logging-node-retry-result", data=data)
        else:  # pragma: no cover
            self._handle_node_retry_result(data)  # Fallback if no reporter is available

    @hook_impl
    def on_cache_hit(
        self,
        func: Any,
        metadata: NodeMetadata,
        inputs: dict[str, Any],
        result: Any,
        reporter: EventReporter | None,
    ) -> None:
        node_key = metadata.key or metadata.name
        self._logger.info(
            f"Task '{node_key}' - Using cached result",
            extra=_build_task_context(metadata.id, metadata.name, metadata.key),
        )

    @hook_impl
    def after_mapped_node_execute(
        self,
        metadata: NodeMetadata,
        iteration_count: int,
        duration: float,
    ) -> None:
        node_key = metadata.key or metadata.name
        self._logger.info(
            f"Task '{node_key}' - Completed task successfully in {_format_duration(duration)}",
            extra=_build_task_context(metadata.id, metadata.name, metadata.key),
        )

    @hook_impl
    def before_dataset_save(
        self,
        key: str,
        value: Any,
        format: str | None,
        options: dict[str, Any] | None,
    ) -> None:
        suffix = f" (format={format})" if format else ""
        self._logger.debug(f"Saving dataset to '{key}'{suffix}")

    @hook_impl
    def after_dataset_save(
        self,
        key: str,
        value: Any,
        format: str | None,
        options: dict[str, Any] | None,
    ) -> None:
        suffix = f" (format={format})" if format else ""
        self._logger.info(f"Saved dataset to '{key}'{suffix}")

    @hook_impl
    def before_dataset_load(
        self,
        key: str,
        return_type: type | None,
        format: str | None,
        options: dict[str, Any] | None,
    ) -> None:
        suffix = f" (format={format})" if format else ""
        self._logger.debug(f"Loading dataset from '{key}'{suffix}")

    @hook_impl
    def after_dataset_load(
        self,
        key: str,
        return_type: type | None,
        format: str | None,
        options: dict[str, Any] | None,
        result: Any,
        duration: float,
    ) -> None:
        suffix = f" (format={format})" if format else ""
        self._logger.info(f"Loaded dataset from '{key}'{suffix} in {_format_duration(duration)}")

    def _handle_node_start(self, event: dict[str, Any]) -> None:
        node_id = event["node_id"]
        node_key = event["node_key"]
        backend_name = event.get("backend_name") or "inline"
        if node_id in self._mapped_nodes:
            self._logger.debug(
                f"Task '{node_key}' - Starting iteration using {backend_name} backend"
            )
        else:
            self._logger.info(f"Task '{node_key}' - Starting task using {backend_name} backend")

    def _handle_node_complete(self, event: dict[str, Any]) -> None:
        node_id = event["node_id"]
        node_key = event["node_key"]
        duration = event.get("duration", 0)
        if node_id in self._mapped_nodes:
            self._logger.debug(
                f"Task '{node_key}' - Completed iteration successfully in "
                f"{_format_duration(duration)}"
            )
        else:
            self._logger.info(
                f"Task '{node_key}' - Completed task successfully in {_format_duration(duration)}"
            )

    def _handle_task_fail(self, event: dict[str, Any]) -> None:
        node_id = event["node_id"]
        node_key = event["node_key"]
        error = event.get("error", "unknown error")
        error_type = event.get("error_type", "Exception")
        duration = event.get("duration", 0)
        if node_id in self._mapped_nodes:
            self._logger.error(
                f"Task '{node_key}' - Mapped iteration failed after "
                f"{_format_duration(duration)}: {error_type}: {error}"
            )
        else:
            self._logger.error(
                f"Task '{node_key}' - Failed after {_format_duration(duration)}: "
                f"{error_type}: {error}"
            )

    def _handle_node_retry(self, event: dict[str, Any]) -> None:
        node_key = event["node_key"]
        attempt = event["attempt"]
        error_type = event.get("error_type", "Exception")
        error = event.get("error", "unknown error")
        self._logger.warning(
            f"Task '{node_key}' - Retrying after failure (attempt {attempt}): {error_type}: {error}"
        )

    def _handle_node_retry_result(self, event: dict[str, Any]) -> None:
        node_key = event["node_key"]
        attempt = event["attempt"]
        succeeded = event["succeeded"]
        if succeeded:
            self._logger.info(f"Task '{node_key}' - Retry succeeded on attempt {attempt}")
        else:
            self._logger.debug(f"Task '{node_key}' - Retry attempt {attempt} failed")


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
            extra.update(_build_task_context(task.id, task.name, task.key))

        kwargs["extra"] = extra
        return msg, dict(kwargs)


def _build_task_context(task_id: UUID, task_name: str, task_key: str | None) -> dict[str, str]:
    """
    Build task context dict for logging extra fields.

    Helper to construct the standard daglite task context fields.
    Used when automatic context injection isn't available (coordinator-side hooks)
    or when merging with existing extra dicts.

    Args:
        task_id: Task UUID
        task_name: Task name
        task_key: Task key (optional)

    Returns:
        Dict with daglite_task_id, daglite_task_name, and daglite_task_key
    """
    return {
        "daglite_task_id": str(task_id),
        "daglite_task_name": task_name,
        "daglite_task_key": task_key or task_name,
    }


def _format_duration(duration: float) -> str:
    """Format duration in seconds to human-readable string."""
    if duration < 1:
        return f"{duration * 1000:.0f} ms"
    elif duration < 60:
        return f"{duration:.2f} s"
    else:
        minutes = int(duration // 60)
        seconds = duration % 60
        return f"{minutes} min {seconds:.2f} s"
