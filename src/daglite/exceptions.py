"""
Centralized exception classes for the daglite library.

All daglite-specific exceptions inherit from DagliteError for easy catching.
"""


class DagliteError(Exception):
    """Base exception for all daglite errors."""


class TaskError(DagliteError):
    """Raised when a task is configured incorrectly."""


class GraphError(DagliteError):
    """Raised when a IR graph construction or validation error occurs."""


class ParameterError(TaskError):
    """Raised when task parameters are invalid or incorrectly specified."""


class BackendError(DagliteError):
    """Raised when there's an error with backend configuration or execution."""


class DatasetError(DagliteError):
    """Raised when there's an error with dataset serialization or deserialization."""


class ExecutionError(DagliteError):
    """Raised when there's an error during task graph execution."""


class AmbiguousResultError(DagliteError):
    """Raised when a WorkflowResult is indexed by a name that matches multiple sink nodes."""
