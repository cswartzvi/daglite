"""High level helpers for building composite expressions."""
from __future__ import annotations

from typing import Iterable, TypeVar

from .expr import ConditionalExpr, Expr, MapExpr
from .task import Task
from .utils import to_expr

T = TypeVar("T")
R = TypeVar("R")


def fanout(task: Task[..., R], items: Iterable[T] | Expr) -> Expr:
    """Map ``task`` across ``items`` with parallel evaluation."""

    return MapExpr(task, to_expr(items))


def conditional(condition: bool | Expr, if_true: Expr, if_false: Expr) -> Expr:
    """Select between two expressions based on ``condition`` lazily."""

    return ConditionalExpr(to_expr(condition), to_expr(if_true), to_expr(if_false))
