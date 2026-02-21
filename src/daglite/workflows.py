"""@workflow decorator and Workflow class for multi-sink task networks."""

from __future__ import annotations

import importlib
import inspect
import sys
import typing
from collections.abc import Callable
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import ValuesView
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, ItemsView, ParamSpec, overload
from uuid import UUID

from daglite.exceptions import AmbiguousResultError
from daglite.futures.base import BaseTaskFuture

P = ParamSpec("P")


@overload
def workflow(func: Callable[P, Any]) -> Workflow[P]: ...


@overload
def workflow(
    *,
    name: str | None = None,
    description: str | None = None,
) -> Callable[[Callable[P, Any]], Workflow[P]]: ...


def workflow(
    func: Callable[P, Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Workflow[P] | Callable[[Callable[P, Any]], Workflow[P]]:
    """
    Decorator to convert a Python function into a daglite `Workflow`.

    Workflows are **multi-sink entry points** for DAG execution. They wrap a function that builds a
    task graph returning one or more `BaseTaskFuture` objects (as a single future, tuple, or list)
    and provide convenient methods to evaluate the entire graph in one pass:

    - Call the workflow to invoke the underlying function (returns the raw futures).
    - Use `.run()` / `.run_async()` to build and evaluate in one step, returning a `WorkflowResult`
      indexable by task name or UUID.

    Args:
        func: The function to wrap. When used without parentheses (`@workflow`), this is
            automatically passed.  When used with parentheses (`@workflow()`), this is `None`.
        name: Custom name for the workflow.  Defaults to the function's `__name__`.
        description: Workflow description.  Defaults to the function's docstring.

    Returns:
        Either a `Workflow` (when used as `@workflow`) or a decorator function (when used as
        `@workflow()`).

    Examples:
        >>> from daglite import task, workflow
        >>> @task
        ... def add(x: int, y: int) -> int:
        ...     return x + y
        >>> @task
        ... def mul(x: int, y: int) -> int:
        ...     return x * y

        Single-sink workflow
        >>> @workflow
        ... def my_workflow(x: int, y: int):
        ...     return add(x=x, y=y)
        >>> result = my_workflow.run(2, 3)
        >>> result["add"]
        5

        Multi-sink workflow
        >>> @workflow
        ... def dual_workflow(x: int, y: int):
        ...     return add(x=x, y=y), mul(x=x, y=y)
        >>> result = dual_workflow.run(2, 3)
        >>> result["add"], result["mul"]
        (5, 6)
    """

    def decorator(fn: Callable[P, Any]) -> Workflow[P]:
        if inspect.isclass(fn) or not callable(fn):
            raise TypeError("`@workflow` can only be applied to callable functions.")

        return Workflow(
            func=fn,
            name=name if name is not None else getattr(fn, "__name__", "unnamed_workflow"),
            description=description
            if description is not None
            else getattr(fn, "__doc__", "") or "",
        )

    if func is not None:
        return decorator(func)

    return decorator


def load_workflow(workflow_path: str) -> Workflow[Any]:
    """
    Load a workflow from a module path.

    Args:
        workflow_path: Dotted path to the workflow (e.g., 'mymodule.my_workflow').

    Returns:
        The loaded Workflow instance.

    Raises:
        ValueError: If the workflow path is invalid.
        ModuleNotFoundError: If the module cannot be found.
        AttributeError: If the workflow attribute does not exist in the module.
        TypeError: If the loaded object is not a Workflow instance.
    """
    if "." not in workflow_path:
        raise ValueError(
            f"Invalid workflow path: '{workflow_path}'. Expected format: 'module.workflow_name'"
        )

    module_path, attr_name = workflow_path.rsplit(".", 1)

    # Add current directory to Python path if not already there
    cwd = str(Path.cwd())
    if cwd not in sys.path:  # pragma: no cover
        sys.path.insert(0, cwd)

    module = importlib.import_module(module_path)

    if not hasattr(module, attr_name):
        raise AttributeError(f"Workflow '{attr_name}' not found in module '{module_path}'")

    workflow_obj = getattr(module, attr_name)

    if not isinstance(workflow_obj, Workflow):
        raise TypeError(
            f"'{workflow_path}' is not a Workflow. Did you forget to use the @workflow decorator?"
        )

    return workflow_obj


@dataclass(frozen=True)
class Workflow(Generic[P]):
    """
    Entry point for building and running a multi-sink task graph.

    Users should **not** directly instantiate this class — use the `@workflow` decorator instead.

    A workflow wraps a function that accepts parameters and returns one or more `BaseTaskFuture`
    objects (single future, tuple, or list).  Workflows can be invoked two ways:

    1. **Call** — `workflow(...)` returns the raw future(s) for manual handling.
    2. **Run** — `workflow.run(...)` / `workflow.run_async(...)` builds and evaluates the DAG in a
       single step, returning a `WorkflowResult`.
    """

    func: Callable[P, Any]
    """Workflow function to be wrapped."""

    name: str
    """Name of the workflow."""

    description: str
    """Description of the workflow."""

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Any:
        """Call the underlying workflow function, returning the raw future(s)."""
        return self.func(*args, **kwargs)

    def _collect_futures(self, raw: Any) -> list[BaseTaskFuture]:
        """Normalize the decorated function's return value to a list of futures."""
        if isinstance(raw, BaseTaskFuture):
            return [raw]
        if isinstance(raw, (tuple, list)):
            futures = list(raw)
            if not futures:
                raise TypeError(
                    "@workflow function must return one or more BaseTaskFuture "
                    "objects; received an empty tuple/list."
                )
            for idx, item in enumerate(futures):
                if not isinstance(item, BaseTaskFuture):
                    raise TypeError(
                        "@workflow function must return only BaseTaskFuture objects; "
                        f"element at index {idx} has invalid type "
                        f"{type(item).__name__!r}."
                    )
            return futures
        raise TypeError(
            "@workflow function must return a BaseTaskFuture or a non-empty "
            f"tuple/list of them, got {type(raw).__name__!r}."
        )

    def run(self, *args: P.args, **kwargs: P.kwargs) -> WorkflowResult:
        """
        Build and evaluate the workflow synchronously.

        Cannot be called from within an async context. Use `.run_async()` instead.

        Args:
            *args: Positional arguments forwarded to the workflow function.
            **kwargs: Keyword arguments forwarded to the workflow function.

        Returns:
            A `WorkflowResult` containing the evaluated outputs of all sink nodes.
        """
        from daglite.engine import evaluate_workflow

        futures = self._collect_futures(self.func(*args, **kwargs))
        return evaluate_workflow(futures)

    async def run_async(self, *args: P.args, **kwargs: P.kwargs) -> WorkflowResult:
        """
        Build and evaluate the workflow asynchronously.

        Args:
            *args: Positional arguments forwarded to the workflow function.
            **kwargs: Keyword arguments forwarded to the workflow function.

        Returns:
            A `WorkflowResult` containing the evaluated outputs of all sink nodes.
        """
        from daglite.engine import evaluate_workflow_async

        futures = self._collect_futures(self.func(*args, **kwargs))
        return await evaluate_workflow_async(futures)

    @property
    def signature(self) -> inspect.Signature:
        """Get the signature of the underlying workflow function."""
        return inspect.signature(self.func)

    def get_typed_params(self) -> dict[str, type | None]:
        """
        Extract parameter names and their type annotations from the workflow function.

        Uses ``typing.get_type_hints`` rather than the raw ``inspect.Signature``
        so that stringified annotations (produced by ``from __future__ import
        annotations``) are resolved to their actual types.

        Returns:
            Dictionary mapping parameter names to their type annotations. If a parameter has no
            annotation, the value is None.
        """
        try:
            hints = typing.get_type_hints(self.func)
        except Exception:
            hints = {}

        typed_params: dict[str, type | None] = {}
        for param_name in self.signature.parameters:
            typed_params[param_name] = hints.get(param_name)

        return typed_params

    def has_typed_params(self) -> bool:
        """
        Check if all parameters have type annotations.

        Returns:
            True if all parameters are typed, False otherwise.
        """
        return all(t is not None for t in self.get_typed_params().values())


@dataclass(frozen=True)
class WorkflowResult(Mapping[str, Any]):
    """
    Holds the evaluated outputs of a `@workflow`, indexable by task name or UUID.

    Access results by name (`result["task_name"]`) or by UUID (`result[uuid]`). If two sink nodes
    share the same name, name-based lookup raises `AmbiguousResultError`; UUID-based lookup always
    works.
    """

    _results: dict[UUID, Any]
    """Primary store: UUID → result value."""

    _by_name: dict[str, list[UUID]]
    """Secondary index: task name → list of UUIDs (typically one, but may be many)."""

    @classmethod
    def build(cls, results: dict[UUID, Any], name_for: dict[UUID, str]) -> WorkflowResult:
        """Build a WorkflowResult from raw results and a uuid→name mapping."""
        by_name: dict[str, list[UUID]] = {}
        for uid, name in name_for.items():
            by_name.setdefault(name, []).append(uid)
        return cls(_results=results, _by_name=by_name)

    def __getitem__(self, key: str | UUID) -> Any:
        if isinstance(key, UUID):
            try:
                return self._results[key]
            except KeyError:
                raise KeyError(f"No workflow output with UUID {key!r}")
        uuids = self._by_name.get(key)
        if not uuids:
            raise KeyError(f"No workflow output named {key!r}")
        if len(uuids) > 1:
            raise AmbiguousResultError(
                f"Multiple sink nodes named {key!r}: {uuids}. "
                f"Use a UUID to disambiguate: result[uuid]"
            )
        return self._results[uuids[0]]

    def single(self, name: str) -> Any:
        """
        Return the result for `name`, asserting there is exactly one.

        Equivalent to `result[name]` — raises `AmbiguousResultError` if multiple sinks share this
        name, or `KeyError` if none do. The method name makes the intent explicit when reading code.

        Args:
            name: Task name or alias to look up.

        Returns:
            The evaluated output of the single matching sink node.

        Raises:
            KeyError: If no sink with this name exists.
            AmbiguousResultError: If multiple sinks share this name.
        """
        return self[name]

    def all(self, name: str) -> list[Any]:
        """
        Return all results for `name` as a list.

        Unlike `result[name]`, this never raises `AmbiguousResultError`. Useful for fan-out
        patterns where multiple sink nodes intentionally share the same name and you want all of
        their values.

        Args:
            name: Task name or alias to look up.

        Returns:
            A list of evaluated outputs for all matching sink nodes. Returns an empty list if no
            sink with this name exists.
        """
        return [self._results[uid] for uid in self._by_name.get(name, [])]

    def values(self) -> ValuesView[Any]:
        """Iterate over all results in name-index order, expanding duplicate-named sinks."""
        return _WorkflowValuesView(self)

    def items(self) -> ItemsView[str, Any]:
        """
        Iterate over (name, value) pairs, expanding duplicate-named sinks.

        Unlike `result[name]`, this never raises `AmbiguousResultError`. Duplicate-named sinks each
        appear as a separate `(name, value)` pair.
        """
        return _WorkflowItemsView(self)

    def __iter__(self) -> Iterator[str]:
        return iter(self._by_name)

    def __len__(self) -> int:
        return len(self._by_name)

    def __repr__(self) -> str:
        names = list(self._by_name)
        return f"WorkflowResult({names})"


class _WorkflowValuesView(ValuesView[Any]):
    """ValuesView that expands duplicate-named sinks instead of raising."""

    _mapping: WorkflowResult  # type: ignore[assignment]

    def __iter__(self) -> Iterator[Any]:
        for uuids in self._mapping._by_name.values():
            for uid in uuids:
                yield self._mapping._results[uid]

    def __len__(self) -> int:
        return len(self._mapping._results)


class _WorkflowItemsView(ItemsView[str, Any]):
    """ItemsView that expands duplicate-named sinks instead of raising."""

    _mapping: WorkflowResult  # type: ignore[assignment]

    def __iter__(self) -> Iterator[tuple[str, Any]]:
        for name, uuids in self._mapping._by_name.items():
            for uid in uuids:
                yield name, self._mapping._results[uid]

    def __len__(self) -> int:
        return len(self._mapping._results)
