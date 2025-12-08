"""Task composers for advanced control flow patterns."""

from __future__ import annotations

from typing import Any, TypeVar, get_args, get_type_hints, overload

from daglite.exceptions import DagliteError
from daglite.futures import ConditionalFuture
from daglite.futures import LoopFuture
from daglite.tasks import BaseTaskFuture
from daglite.tasks import Task
from daglite.tasks import TaskFuture
from daglite.tasks import task

R = TypeVar("R")
T = TypeVar("T")

S1 = TypeVar("S1")
S2 = TypeVar("S2")
S3 = TypeVar("S3")
S4 = TypeVar("S4")
S5 = TypeVar("S5")
S6 = TypeVar("S6")


# region split


@overload
def split(future: TaskFuture[tuple[S1]]) -> tuple[TaskFuture[S1]]: ...


@overload
def split(future: TaskFuture[tuple[S1, S2]]) -> tuple[TaskFuture[S1], TaskFuture[S2]]: ...


@overload
def split(
    future: TaskFuture[tuple[S1, S2, S3]],
) -> tuple[TaskFuture[S1], TaskFuture[S2], TaskFuture[S3]]: ...


@overload
def split(
    future: TaskFuture[tuple[S1, S2, S3, S4]],
) -> tuple[TaskFuture[S1], TaskFuture[S2], TaskFuture[S3], TaskFuture[S4]]: ...


@overload
def split(
    future: TaskFuture[tuple[S1, S2, S3, S4, S5]],
) -> tuple[
    TaskFuture[S1],
    TaskFuture[S2],
    TaskFuture[S3],
    TaskFuture[S4],
    TaskFuture[S5],
]: ...


@overload
def split(
    future: TaskFuture[tuple[S1, S2, S3, S4, S5, S6]],
) -> tuple[
    TaskFuture[S1],
    TaskFuture[S2],
    TaskFuture[S3],
    TaskFuture[S4],
    TaskFuture[S5],
    TaskFuture[S6],
]: ...


@overload
def split(
    future: TaskFuture[tuple[Any, ...]],
    *,
    size: int,
) -> tuple[TaskFuture[Any], ...]: ...


def split(
    future: TaskFuture[tuple[Any, ...]],
    *,
    size: int | None = None,
) -> tuple[TaskFuture[Any], ...]:
    """
    Split a tuple-producing TaskFuture into individual TaskFutures for each element.

    Args:
        future: A `TaskFuture` that produces a tuple.
        size: Optional explicit size. Required if type annotations don't specify tuple size.

    Returns:
        A tuple of TaskFutures, one for each element of the input tuple.

    Raises:
        DagliteError: If size cannot be inferred from type hints and size parameter is not provided.

    Examples:
        >>> # With type annotations (size inferred)
        >>> @task
        >>> def make_pair() -> tuple[int, str]:
        >>>     return (42, "hello")
        >>>
        >>> result = make_pair.bind()
        >>> num, text = split(result)  # 42, "hello"
        >>>
        >>> # With explicit size parameter:
        >>> @task
        >>> def make_triple():
        >>>     return (1, 2, 3)
        >>>
        >>> result = make_triple.bind()
        >>> a, b, c = split(result, size=3)  # 1, 2, 3
    """
    # NOTE: This composer creates independent accessor tasks for each tuple element,
    # enabling parallel processing of tuple components. Type information is preserved
    # when the tuple has explicit type annotations.

    def _infer_size() -> int | None:
        """Try to infer tuple size from type annotations."""
        try:
            task_obj = future.task
            hints = get_type_hints(task_obj.func)
        except Exception:  # pragma: no cover
            return None

        return_type = hints.get("return")
        if return_type is None:
            return None
        args = get_args(return_type)
        if args and (len(args) < 2 or args[-1] is not Ellipsis):  # Skip tuple[int, ...]
            return len(args)
        return None

    final_size = _infer_size() if size is None else size
    if final_size is None:
        raise DagliteError(
            f"Cannot infer tuple size from type annotations for future {future.task.name}. "
            f"Please provide an explicit size parameter to split()."
        )

    # Create index accessor tasks for each position
    @task
    def _get_index(tup: tuple[Any, ...], index: int) -> Any:
        return tup[index]

    # Bind the accessor task for each index
    return tuple(_get_index.bind(tup=future, index=i) for i in range(final_size))


# region when


def when(
    condition: TaskFuture[bool],
    then_branch: BaseTaskFuture[R],
    else_branch: BaseTaskFuture[R],
) -> ConditionalFuture[R]:
    """
    Conditional execution: select one of two branches based on a boolean condition.

    The condition TaskFuture is evaluated first. Based on its result:
    - If True: return the result of `then_branch`
    - If False: return the result of `else_branch`

    **Current Limitation**: Both branches are currently evaluated eagerly before the
    condition is checked. Only the selected branch's result is returned, but both
    branches execute. This means expensive operations or side effects in both branches
    will occur regardless of the condition. See GitHub issue for planned lazy evaluation
    support.

    Args:
        condition: A `TaskFuture` that produces a boolean value.
        then_branch: Task future to execute if condition is True.
        else_branch: Task future to execute if condition is False.

    Returns:
        A ConditionalFuture representing the conditional execution.

    Examples:
        >>> @task
        >>> def is_large(x: int) -> bool:
        >>>     return x > 100
        >>>
        >>> @task
        >>> def process_large(x: int) -> str:
        >>>     return f"Large value: {x}"
        >>>
        >>> @task
        >>> def process_small(x: int) -> str:
        >>>     return f"Small value: {x}"
        >>>
        >>> value = compute.bind(x=50)
        >>> check = is_large.bind(x=value)
        >>> result = when(
        >>>     check,
        >>>     then_branch=process_large.bind(x=value),
        >>>     else_branch=process_small.bind(x=value)
        >>> )
        >>> evaluate(result)  # "Small value: 50"
    """
    return ConditionalFuture(
        condition=condition,
        then_branch=then_branch,
        else_branch=else_branch,
    )


# region loop


def loop(
    *,
    initial: TaskFuture[R] | R,
    body: Task[Any, tuple[R, bool]],
    max_iterations: int = 1000,
    **body_kwargs: Any,
) -> LoopFuture[R]:
    """
    Iterative execution: repeatedly execute a body task until a condition is met.

    The loop maintains state across iterations. The body task receives the current
    state and returns a tuple of (new_state, should_continue). The loop continues
    while should_continue is True.

    Args:
        initial: Initial state value or TaskFuture producing the initial state.
        body: Task that takes current state and returns (new_state, should_continue).
            Must have exactly one unbound parameter for the state.
        max_iterations: Maximum number of iterations to prevent infinite loops.
            Defaults to 1000.
        **body_kwargs: Additional fixed parameters to pass to the body task.
            Can include TaskFutures as dependencies.

    Returns:
        A LoopFuture representing the iterative execution.

    Raises:
        ValueError: If body task doesn't have exactly one unbound parameter.

    Examples:
        Simple counter:
        >>> @task
        >>> def increment(count: int) -> tuple[int, bool]:
        >>>     new_count = count + 1
        >>>     should_continue = new_count < 5
        >>>     return (new_count, should_continue)
        >>>
        >>> result = loop(initial=0, body=increment)
        >>> evaluate(result)  # 5

        With additional parameters:
        >>> @task
        >>> def add_factor(state: int, factor: int) -> tuple[int, bool]:
        >>>     new_state = state + factor
        >>>     return (new_state, new_state < 50)
        >>>
        >>> result = loop(initial=0, body=add_factor, factor=7)
        >>> evaluate(result)  # 56

        With TaskFuture initial state:
        >>> @task
        >>> def get_start() -> int:
        >>>     return 10
        >>>
        >>> @task
        >>> def double(n: int) -> tuple[int, bool]:
        >>>     return (n * 2, n < 100)
        >>>
        >>> start = get_start.bind()
        >>> result = loop(initial=start, body=double)
        >>> evaluate(result)  # 128
    """
    import inspect

    # Validate body task has exactly one unbound parameter
    # (unbound = not provided in body_kwargs)
    sig = inspect.signature(body.func)
    all_params = list(sig.parameters.keys())
    unbound_params = [name for name in all_params if name not in body_kwargs]

    if len(unbound_params) == 0:
        msg = (
            f"Loop body task '{body.name}' must have at least one unbound parameter "
            f"for the state. All parameters are bound in body_kwargs: {list(body_kwargs.keys())}"
        )
        raise ValueError(msg)

    if len(unbound_params) > 1:
        msg = (
            f"Loop body task '{body.name}' has multiple unbound parameters: {unbound_params}. "
            f"Only one unbound parameter is allowed for the loop state. "
            f"Bind additional parameters using body_kwargs."
        )
        raise ValueError(msg)

    return LoopFuture(
        initial_state=initial,
        body=body,
        body_kwargs=body_kwargs,
        max_iterations=max_iterations,
    )


# endregion
