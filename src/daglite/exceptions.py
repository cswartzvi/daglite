"""
Centralized exception classes for the daglite library.

All daglite-specific exceptions inherit from DagliteError for easy catching.
"""


class DagliteError(Exception):
    """Base exception for all daglite errors."""


class TaskConfigurationError(DagliteError):
    """
    Raised when a task is configured incorrectly.

    Examples:
        - Missing required parameters
        - Invalid parameter combinations
        - Conflicting backend specifications
    """


class GraphConstructionError(DagliteError):
    """
    Raised when there's an error constructing the task graph.

    Examples:
        - Invalid parameter binding
        - Circular dependencies
        - Invalid graph structure
    """


class ParameterError(TaskConfigurationError):
    """
    Raised when task parameters are invalid or incorrectly specified.

    Examples:
        - Empty sequences in extend/zip operations
        - Mismatched sequence lengths in zip
        - Parameter already bound in PartialTask
        - Missing required parameters
    """


class BackendError(DagliteError):
    """
    Raised when there's an error with backend configuration or execution.

    Examples:
        - Unknown backend name
        - Backend initialization failure
        - Backend execution error
    """


class ExecutionError(DagliteError):
    """
    Raised when there's an error during task graph execution.

    Examples:
        - Task function raises an exception
        - Node execution failure
        - Type resolution errors
    """
