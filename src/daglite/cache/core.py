"""Default cache hashing strategy using cloudpickle."""

from __future__ import annotations

import hashlib
import inspect
from typing import Any, Callable

import cloudpickle


def default_cache_hash(func: Callable[..., Any], bound_args: dict[str, Any]) -> str:
    """
    Generate cache key from function source and cloudpickle'd parameter values.

    Uses cloudpickle to serialize parameter values for hashing, which handles
    lambdas, closures, and most Python types automatically.

    Args:
        func: The function being cached.
        bound_args: Bound parameter values as a dictionary.

    Returns:
        SHA256 hex digest string.

    Examples:
        >>> def add(x: int, y: int) -> int:
        ...     return x + y
        >>> hash1 = default_cache_hash(add, {"x": 1, "y": 2})
        >>> hash2 = default_cache_hash(add, {"x": 1, "y": 2})
        >>> hash1 == hash2
        True
        >>> hash3 = default_cache_hash(add, {"x": 1, "y": 3})
        >>> hash1 == hash3
        False
    """
    h = hashlib.sha256()

    # Hash function source
    try:
        source = inspect.getsource(func)
        h.update(source.encode())
    except (OSError, TypeError):  # pragma: no cover
        h.update(func.__qualname__.encode())

    # Hash each parameter via cloudpickle
    for name, value in sorted(bound_args.items()):
        h.update(name.encode())
        h.update(cloudpickle.dumps(value))

    return h.hexdigest()


__all__ = ["default_cache_hash"]
