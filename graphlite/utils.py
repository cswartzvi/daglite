"""Helper utilities for working with expressions."""
from __future__ import annotations

from typing import Any

from .expr import Expr, ValueExpr


def to_expr(value: Any) -> Expr:
    """Convert raw values or expressions into :class:`Expr` instances."""

    if isinstance(value, Expr):
        return value
    return ValueExpr(value)
