"""Core expression data structures used for graph construction and reduction."""
from __future__ import annotations

from itertools import count
from typing import Any, Iterable, Mapping


_expr_ids = count()


def _next_id() -> int:
    return next(_expr_ids)


class Expr:
    """Base class for all expressions in the evaluation graph."""

    id: int

    def __init__(self) -> None:
        self.id = _next_id()

    def iter_children(self) -> Iterable["Expr"]:
        return ()


class ValueExpr(Expr):
    """Expression representing a concrete Python value."""

    def __init__(self, value: Any) -> None:
        super().__init__()
        self.value = value


class CallExpr(Expr):
    """Expression representing invocation of a task with arguments."""

    def __init__(self, task: "Task[Any, Any]", args: tuple[Expr, ...], kwargs: Mapping[str, Expr]) -> None:
        super().__init__()
        self.task = task
        self.args = args
        self.kwargs = dict(kwargs)

    def iter_children(self) -> Iterable[Expr]:
        yield from self.args
        yield from self.kwargs.values()


class MapExpr(Expr):
    """Expression that maps a task over an iterable of values in parallel."""

    def __init__(self, task: "Task[Any, Any]", items: Expr) -> None:
        super().__init__()
        self.task = task
        self.items = items

    def iter_children(self) -> Iterable[Expr]:
        yield self.items


class ConditionalExpr(Expr):
    """Expression encoding a branch based on the result of a condition expression."""

    def __init__(self, condition: Expr, if_true: Expr, if_false: Expr) -> None:
        super().__init__()
        self.condition = condition
        self.if_true = if_true
        self.if_false = if_false

    def iter_children(self) -> Iterable[Expr]:
        yield self.condition
        yield self.if_true
        yield self.if_false


# Late import for type checking only.
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .task import Task
