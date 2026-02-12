"""Internal shared validation functions for tasks and futures."""

import string
from collections.abc import Iterable
from inspect import Signature
from typing import Literal

from daglite.exceptions import ParameterError

MapMode = Literal["product", "zip"]


def check_invalid_params(signature: Signature, kwargs: dict, task_name: str) -> None:
    """
    Checks that all provided parameters are valid for the given task function signature.

    Args:
        signature: Signature of the task function.
        kwargs: Task function arguments to validate.
        task_name: Name of the task (used for error messages).

    Raises:
        ParameterError: If any provided parameters are not in the task's function signature.
    """
    if invalid_params := sorted(kwargs.keys() - signature.parameters.keys()):
        raise ParameterError(f"Invalid parameters for task '{task_name}': {invalid_params}")


def check_missing_params(signature: Signature, kwargs: dict, task_name: str) -> None:
    """
    Checks that all required parameters for the given task function signature are provided.

    Args:
        signature: Signature of the task function.
        kwargs: Task function arguments to validate.
        task_name: Name of the task (used for error messages).

    Raises:
        ParameterError: If any required parameters are missing.
    """
    if missing_params := sorted(signature.parameters.keys() - kwargs.keys()):
        raise ParameterError(f"Missing parameters for task '{task_name}': {missing_params}")


def check_overlap_params(fixed_kwargs: dict, new_kwargs: dict, task_name: str) -> None:
    """
    Checks that no provided parameters overlap with already fixed parameters.

    Args:
        fixed_kwargs: Dictionary of already fixed parameters (e.g., from `.partial()`).
        new_kwargs: Dictionary of new parameters to validate.
        task_name: Name of the task (used for error messages).

    Raises:
        ParameterError: If any provided parameters overlap with already fixed parameters.
    """
    fixed = fixed_kwargs.keys()
    if overlap_params := sorted(fixed & new_kwargs.keys()):
        raise ParameterError(
            f"Overlapping parameters for task '{task_name}', specified parameters "
            f"were previously bound in `.partial()`: {overlap_params}"
        )


def check_invalid_map_params(signature: Signature, kwargs: dict, task_name: str) -> None:
    """
    Checks that all provided parameters for a map task are iterable or TaskFuture.

    Note that this function is intended to be used for validating parameters passed to map tasks,
    where all parameters must be iterable (or TaskFuture that produces an iterable).

    Args:
        signature: Signature of the map task function.
        kwargs: Task function arguments to validate.
        task_name: Name of the task (used for error messages).

    Raises:
        ParameterError: If any provided parameters are not iterable.
    """
    from daglite.futures.base import BaseTaskFuture

    non_sequences = []
    parameters = signature.parameters.keys()
    for key, value in kwargs.items():
        if key in parameters and not isinstance(value, (Iterable, BaseTaskFuture)):
            non_sequences.append(key)
    if non_sequences := sorted(non_sequences):
        raise ParameterError(
            f"Non-iterable parameters for task '{task_name}', "
            f"all parameters must be Iterable or TaskFuture[Iterable] "
            f"(use `.partial()` to set scalar parameters): {non_sequences}"
        )


def get_unbound_param(signature: Signature, kwargs: dict, task_name: str) -> str:
    """
    Returns the single unbound parameter name for the given task and provided arguments.

    Args:
        signature: Signature of the task function.
        kwargs: Task function arguments to validate.
        task_name: Name of the task (used for error messages).

    Raises:
        ParameterError: If there are zero or multiple unbound parameters.
    """
    unbound = [p for p in signature.parameters if p not in kwargs]
    if len(unbound) == 0:
        raise ParameterError(
            f"Task '{task_name}' has no unbound parameters for "
            f"upstream value. All parameters already provided: {list(kwargs.keys())}"
        )
    if len(unbound) > 1:
        raise ParameterError(
            f"Task '{task_name}' must have exactly one "
            f"unbound parameter for upstream value, found {len(unbound)}: {unbound} "
            f"(use `.partial()` to set scalar parameters): {unbound[1:]}"
        )
    return unbound[0]


def check_key_template(key: str) -> None:
    """Validates a key template string for well-formed {placeholder} syntax."""
    try:
        parsed = list(string.Formatter().parse(key))
    except (ValueError, IndexError) as e:
        raise ValueError(
            f"Invalid key template '{key}': {e}. Key templates use {{name}} placeholders for "
            f"parameter substitution."
        ) from e

    for _, field_name, _, _ in parsed:
        if field_name is None:
            continue
        if field_name == "":
            raise ValueError(
                f"Invalid key template '{key}': empty placeholder '{{}}' is not allowed. "
                f"Use named placeholders like '{{param_name}}' instead."
            )


def check_key_placeholders(key: str, allowed: set[str]) -> None:
    """
    Checks that all placeholders in a key template string are present in the available variable set.

    Args:
        key: Template string containing {placeholders} to validate.
        allowed: Set of names that are allowed to be used as placeholders in the key template.

    Raises:
        ValueError: If any placeholder in the key template is not in the allowed names set.
    """
    import string

    placeholders = {
        field_name
        for _, field_name, _, _ in string.Formatter().parse(key)
        if field_name is not None
    }
    unknown = placeholders - allowed
    if unknown:
        raise ValueError(
            f"Key template '{key}' references {unknown} which won't be available "
            f"at runtime. Available placeholders: {sorted(allowed)}. "
        )
