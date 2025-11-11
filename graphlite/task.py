"""Task decorator and runtime wrapper implementing the typing illusion."""
from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Generic, Mapping, ParamSpec, TypeVar, cast, overload

from . import context
from .expr import CallExpr
from .utils import to_expr

P = ParamSpec("P")
R = TypeVar("R")


class Task(Generic[P, R]):
    """Wrap a callable so the type checker sees a normal function while runtime is lazy.

    The class masquerades as the original callable by reusing its ``__call__`` signature
    (thanks to :class:`typing.ParamSpec`). Static type checkers therefore treat instances
    exactly like the wrapped function. At runtime the wrapper either constructs a lazy
    expression (when no executor is active) or immediately evaluates the expression via
    the current evaluation context. This mirrors the "typing illusion" popularized by
    the redun project.
    """

    def __init__(self, func: Callable[P, R], name: str | None = None) -> None:
        self.func = func
        self.name = name or func.__name__
        wraps(func)(self)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        executor = context.current_executor()
        expr = CallExpr(self, tuple(to_expr(arg) for arg in args), {k: to_expr(v) for k, v in kwargs.items()})
        if executor is not None:
            return cast(R, executor.evaluate(expr))
        return cast(R, expr)

    def __get__(self, obj: Any, objtype: type[Any]) -> Callable[P, R]:  # pragma: no cover - descriptor delegation
        return cast(Callable[P, R], self.__call__.__get__(obj, objtype))

    def execute(self, executor: "Executor", args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> R:
        """Execute the underlying function within an evaluation context."""

        with context.use_executor(executor):
            return self.func(*args, **kwargs)


@overload
def task(func: Callable[P, R]) -> Task[P, R]:
    ...


def task(func: Callable[P, R]) -> Task[P, R]:
    """Decorator turning a function into a :class:`Task` with lazy semantics."""

    return Task(func)


# Late import for type checking only.
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .executor import Executor
