"""Graph reduction executor evaluating expressions produced by tasks."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import Any, Dict, Iterable
from .expr import CallExpr, ConditionalExpr, Expr, MapExpr, ValueExpr


class Executor:
    """Evaluates expressions by recursively reducing the dependency graph.

    Graph reduction evaluates the DAG from the leaves upward: every dependency of an
    expression is resolved before the expression itself is computed. Because expressions
    capture both tasks and concrete values, the executor can transparently traverse the
    graph, execute tasks, and reuse cached results as it goes.
    """

    def __init__(self, max_workers: int | None = None) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._cache: Dict[int, Any] = {}

    def close(self) -> None:
        self._executor.shutdown(wait=True)

    def evaluate(self, expr: Expr) -> Any:
        if isinstance(expr, ValueExpr):
            return expr.value
        if expr.id in self._cache:
            return self._cache[expr.id]
        result = self._evaluate(expr)
        self._cache[expr.id] = result
        return result

    def _evaluate(self, expr: Expr) -> Any:
        if isinstance(expr, ValueExpr):
            return expr.value
        if isinstance(expr, CallExpr):
            args = tuple(self.evaluate(arg) for arg in expr.args)
            kwargs = {name: self.evaluate(val) for name, val in expr.kwargs.items()}
            result = expr.task.execute(self, args, kwargs)
            if isinstance(result, Expr):
                return self.evaluate(result)
            return result
        if isinstance(expr, MapExpr):
            items = list(self.evaluate(expr.items))
            futures: list[Future[Any]] = []
            for item in items:
                call_expr = CallExpr(expr.task, (ValueExpr(item),), {})
                futures.append(self._executor.submit(self.evaluate, call_expr))
            wait(futures)
            return [future.result() for future in futures]
        if isinstance(expr, ConditionalExpr):
            condition = bool(self.evaluate(expr.condition))
            branch = expr.if_true if condition else expr.if_false
            return self.evaluate(branch)
        raise TypeError(f"Unknown expression type: {type(expr)!r}")

    def map(self, task: "Task[Any, Any]", items: Iterable[Any]) -> list[Any]:
        futures = [self._executor.submit(task.execute, self, (item,), {}) for item in items]
        wait(futures)
        return [fut.result() for fut in futures]

    def __enter__(self) -> "Executor":  # pragma: no cover - convenience
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - convenience
        self.close()


# Late import for type checking only.
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .task import Task
