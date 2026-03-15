"""
Centralized exception classes for the daglite library.

All daglite-specific exceptions inherit from DagliteError for easy catching.
"""


class DagliteError(Exception):
    """Base exception for all daglite errors."""


class TaskError(DagliteError):
    """Raised when a task is configured incorrectly."""


class ParameterError(TaskError):
    """Raised when task parameters are invalid or incorrectly specified."""


class BackendError(DagliteError):
    """Raised when there's an error with backend configuration or execution."""


class DatasetError(DagliteError):
    """Raised when there's an error with dataset serialization or deserialization."""
