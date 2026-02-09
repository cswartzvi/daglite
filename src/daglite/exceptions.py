"""
Centralized exception classes for the daglite library.

All daglite-specific exceptions inherit from DagliteError for easy catching.
"""


class DagliteError(Exception):
    """Base exception for all daglite errors."""


class TaskConfigurationError(DagliteError):
    """Raised when a task is configured incorrectly."""


class GraphConstructionError(DagliteError):
    """Raised when there's an error constructing the task graph."""


class ParameterError(TaskConfigurationError):
    """Raised when task parameters are invalid or incorrectly specified."""


class BackendError(DagliteError):
    """Raised when there's an error with backend configuration or execution."""


class DatasetError(DagliteError):
    """Raised when there's an error with dataset serialization or deserialization."""


class ExecutionError(DagliteError):
    """Raised when there's an error during task graph execution."""
