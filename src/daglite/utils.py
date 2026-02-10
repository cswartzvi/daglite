"""Shared utility helpers for Daglite."""

from __future__ import annotations

import reprlib
from collections.abc import Mapping
from typing import Any


def truncate_repr(value: Any, max_len: int = 50) -> str:
    """Return a truncated repr of *value*, appending '...' if it exceeds *max_len*."""
    r = reprlib.Repr()
    r.maxstring = max_len
    r.maxother = max_len
    return r.repr(value)


def build_repr(class_name: str, *leading: str, kwargs: Mapping[str, Any] | None = None) -> str:
    """Build a concise repr string: ``ClassName(leading…, k=v, …)``."""
    parts = list(leading)
    if kwargs:
        parts.extend(f"{k}={truncate_repr(v)}" for k, v in kwargs.items())
    return f"{class_name}({', '.join(parts)})"
