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

    # Hash each parameter via cloudpickle, normalizing order-sensitive containers first
    for name, value in sorted(bound_args.items()):
        h.update(name.encode())
        h.update(cloudpickle.dumps(_canonical(value)))

    return h.hexdigest()


def _canonical(value: Any) -> Any:
    """
    Recursively normalize order-sensitive containers for stable hashing.

    Dicts and sets have no guaranteed iteration order (dict insertion order aside,
    equal dicts built in different orders compare equal but may serialize differently
    with cloudpickle). This converts them to sorted structures so that logically
    equal values always produce the same bytes.
    """
    if isinstance(value, dict):
        return sorted((_canonical(k), _canonical(v)) for k, v in value.items())
    if isinstance(value, (set, frozenset)):
        return sorted(_canonical(v) for v in value)
    if isinstance(value, list):
        return [_canonical(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_canonical(v) for v in value)
    return value


__all__ = ["default_cache_hash"]
