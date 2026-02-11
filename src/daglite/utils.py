"""Shared utility helpers for Daglite."""

from __future__ import annotations

import reprlib
from collections.abc import Mapping
from typing import Any


def build_repr(class_name: str, *leading: str, kwargs: Mapping[str, Any] | None = None) -> str:
    """Build a concise repr string: ``ClassName(leading…, k=v, …)``."""
    parts = list(leading)
    if kwargs:
        parts.extend(f"{k}={reprlib.Repr().repr(v)}" for k, v in kwargs.items())
    return f"{class_name}({', '.join(parts)})"


def infer_tuple_size(task_func: Any) -> int | None:
    """Try to infer tuple size from type annotations of a task function."""
    # Import here to avoid issues with circular imports
    from typing import get_args, get_type_hints

    try:
        hints = get_type_hints(task_func)
    except Exception:  # pragma: no cover
        return None

    return_type = hints.get("return")
    if return_type is None:
        return None
    args = get_args(return_type)
    if args and (len(args) < 2 or args[-1] is not Ellipsis):  # Skip tuple[int, ...]
        return len(args)
    return None
