from __future__ import annotations

import abc
import inspect
from collections.abc import Callable
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, Generic, TypeVar, overload
from uuid import UUID
from uuid import uuid4

from typing_extensions import override

from daglite.exceptions import ParameterError
from daglite.graph.base import GraphBuilder
from daglite.graph.base import ParamInput

# NOTE: Import types only for type checking to avoid circular imports, if you need
# to use them at runtime, import them within methods.
if TYPE_CHECKING:
    from daglite.engine import Backend
    from daglite.graph.nodes import ChooseNode
    from daglite.graph.nodes import MapTaskNode
    from daglite.graph.nodes import SwitchNode
    from daglite.graph.nodes import TaskNode
    from daglite.graph.nodes import WhileLoopNode
    from daglite.tasks import FixedParamTask
    from daglite.tasks import Task
else:
    Backend = object
    ChooseNode = object
    MapTaskNode = object
    SwitchNode = object
    TaskNode = object
    WhileLoopNode = object
    FixedParamTask = object
    Task = object

R = TypeVar("R")
S = TypeVar("S")


# region Task Futures


class BaseTaskFuture(abc.ABC, GraphBuilder, Generic[R]):
    """Base class for all task futures, representing unevaluated task invocations."""

    @cached_property
    def id(self) -> UUID:
        """Unique identifier for this lazy value instance."""
        return uuid4()

    # NOTE: The following methods are to prevent accidental usage of unevaluated nodes.

    def __bool__(self) -> bool:
        raise TypeError(
            "TaskFutures cannot be used in boolean context. Did you mean to call evaluate() first?"
        )

    def __len__(self) -> int:
        raise TypeError("TaskFutures do not support len(). Did you mean to call evaluate() first?")

    def __repr__(self) -> str:  # pragma : no cover
        return f"<Lazy {id(self):#x}>"


@dataclass(frozen=True)
class TaskFuture(BaseTaskFuture[R]):
    """
    Represents a single task invocation that will produce a value of type R.

    All parameters to the task can be either concrete values or other TaskFutures.
    """

    task: Task[Any, R]
    """Underlying task to be called."""

    kwargs: Mapping[str, Any]
    """Parameters to be passed to the task during execution, can contain other task futures."""

    backend: Backend
    """Engine backend override for this task, if `None`, uses the default engine backend."""

    @overload
    def then(self, next_task: Task[Any, S]) -> "TaskFuture[S]": ...

    @overload
    def then(
        self, next_task: Task[Any, S] | FixedParamTask[Any, S], **kwargs: Any
    ) -> "TaskFuture[S]": ...

    def then(
        self,
        next_task: Task[Any, S] | FixedParamTask[Any, S],
        **kwargs: Any,
    ) -> "TaskFuture[S]":
        """
        Chain this task's output as input to another task.

        Additional kwargs are passed to the next task. The upstream output
        is automatically bound to the single remaining unbound parameter.

        Args:
            next_task: Task or FixedParamTask to chain with this task's output.
            **kwargs: Additional parameters to pass to the next task.

        Returns:
            A new TaskFuture representing the chained computation.

        Raises:
            ParameterError: If the next task has no unbound parameters, multiple
                unbound parameters, or if kwargs overlap with fixed parameters.

        Examples:
            Basic chaining:
            >>> @task
            >>> def fetch(url: str) -> str:
            >>>     return requests.get(url).text
            >>>
            >>> @task
            >>> def parse(html: str) -> dict:
            >>>     return parse_html(html)
            >>>
            >>> result = fetch.bind(url="example.com").then(parse)

            With inline parameters:
            >>> @task
            >>> def transform(data: str, format: str) -> dict:
            >>>     return convert(data, format)
            >>>
            >>> result = fetch.bind(url="example.com").then(transform, format="json")

            With pre-fixed tasks:
            >>> transform_json = transform.fix(format="json")
            >>> result = fetch.bind(url="example.com").then(transform_json)

            Chaining multiple operations:
            >>> result = (
            >>>     fetch.bind(url="example.com")
            >>>     .then(parse)
            >>>     .then(transform, format="json")
            >>>     .then(save, path="output.json")
            >>> )
        """
        # NOTE: Import at runtime to avoid circular import issues
        from daglite.tasks import FixedParamTask

        if isinstance(next_task, FixedParamTask):
            # Check for overlapping parameters
            if overlap_params := set(next_task.fixed_kwargs.keys()) & set(kwargs.keys()):
                raise ParameterError(
                    f"Overlapping parameters for task '{next_task.name}' in `.then()`, "
                    f"specified parameters were previously bound in `.fix()`: "
                    f"{sorted(overlap_params)}"
                )
            # Merge existing fixed kwargs with new ones
            all_fixed = {**next_task.fixed_kwargs, **kwargs}
            sig = inspect.signature(next_task.task.func)
            actual_task = next_task.task
        else:
            all_fixed = kwargs
            sig = inspect.signature(next_task.func)
            actual_task = next_task

        # Find the one unbound parameter
        unbound = [name for name in sig.parameters if name not in all_fixed]

        if len(unbound) == 0:
            raise ParameterError(
                f"Task '{next_task.name}' in `.then()` has no unbound parameters for "
                f"upstream value. All parameters already provided: {list(all_fixed.keys())}"
            )
        if len(unbound) > 1:
            raise ParameterError(
                f"Task '{next_task.name}' in `.then()` must have exactly one "
                f"unbound parameter for upstream value, found {len(unbound)}: {unbound} "
                f"(use `.fix()` to set scalar parameters): {unbound[1:]}"
            )

        target_param = unbound[0]
        return actual_task.bind(**{target_param: self}, **all_fixed)

    @overload
    def choose(
        self, *, condition: Callable[[R], bool], if_true: Task[Any, S], if_false: Task[Any, S]
    ) -> "ChooseFuture[S]": ...

    @overload
    def choose(
        self,
        *,
        condition: Callable[[R], bool],
        if_true: Task[Any, S] | FixedParamTask[Any, S],
        if_false: Task[Any, S] | FixedParamTask[Any, S],
        true_kwargs: dict[str, Any] | None = None,
        false_kwargs: dict[str, Any] | None = None,
    ) -> "ChooseFuture[S]": ...

    def choose(
        self,
        *,
        condition: Callable[[R], bool],
        if_true: Task[Any, S] | FixedParamTask[Any, S],
        if_false: Task[Any, S] | FixedParamTask[Any, S],
        true_kwargs: dict[str, Any] | None = None,
        false_kwargs: dict[str, Any] | None = None,
    ) -> "ChooseFuture[S]":
        """
        Create a conditional branch based on a condition function.

        Evaluates the condition on this task's output and executes either
        if_true or if_false task based on the result.

        Args:
            condition: Function that takes this task's output and returns bool.
            if_true: Task to execute when condition returns True.
            if_false: Task to execute when condition returns False.
            true_kwargs: Additional kwargs to pass to if_true task.
            false_kwargs: Additional kwargs to pass to if_false task.

        Returns:
            A ChooseFuture representing the conditional operation.

        Examples:
            Simple conditional:
            >>> @task
            >>> def get_value(x: int) -> int:
            >>>     return x
            >>>
            >>> @task
            >>> def positive_handler(x: int) -> str:
            >>>     return f"positive: {x}"
            >>>
            >>> @task
            >>> def negative_handler(x: int) -> str:
            >>>     return f"negative: {x}"
            >>>
            >>> result = get_value.bind(x=5).choose(
            >>>     condition=lambda x: x > 0,
            >>>     if_true=positive_handler,
            >>>     if_false=negative_handler
            >>> )
            >>> # Result: "positive: 5"

            With additional kwargs:
            >>> @task
            >>> def scale(x: int, factor: int) -> int:
            >>>     return x * factor
            >>>
            >>> result = get_value.bind(x=5).choose(
            >>>     condition=lambda x: x > 0,
            >>>     if_true=scale,
            >>>     if_false=scale,
            >>>     true_kwargs={"factor": 2},
            >>>     false_kwargs={"factor": -2}
            >>> )
        """
        from daglite.tasks import FixedParamTask

        # Extract actual tasks and merge kwargs
        if isinstance(if_true, FixedParamTask):
            true_task = if_true.task
            true_kw = {**if_true.fixed_kwargs, **(true_kwargs or {})}
        else:
            true_task = if_true
            true_kw = true_kwargs or {}

        if isinstance(if_false, FixedParamTask):
            false_task = if_false.task
            false_kw = {**if_false.fixed_kwargs, **(false_kwargs or {})}
        else:
            false_task = if_false
            false_kw = false_kwargs or {}

        # Validate that both tasks have exactly one unbound parameter for the input value
        true_sig = inspect.signature(true_task.func)
        false_sig = inspect.signature(false_task.func)

        true_unbound = [p for p in true_sig.parameters if p not in true_kw]
        false_unbound = [p for p in false_sig.parameters if p not in false_kw]

        if len(true_unbound) != 1:
            raise ParameterError(
                f"if_true task '{true_task.name}' must have exactly one "
                f"unbound parameter for the input value, found {len(true_unbound)}: {true_unbound}"
            )

        if len(false_unbound) != 1:
            raise ParameterError(
                f"if_false task '{false_task.name}' must have exactly one "
                f"unbound parameter for the input value, found {len(false_unbound)}: {false_unbound}"
            )

        return ChooseFuture(
            input_future=self,
            condition=condition,
            if_true=true_task,
            if_false=false_task,
            true_kwargs=true_kw,
            false_kwargs=false_kw,
            backend=self.backend,
        )

    def switch(
        self,
        cases: Mapping[Any, Task[Any, S] | FixedParamTask[Any, S]],
        *,
        default: Task[Any, S] | FixedParamTask[Any, S] | None = None,
        key: Callable[[R], Any] | None = None,
        case_kwargs: Mapping[Any, dict[str, Any]] | None = None,
        default_kwargs: dict[str, Any] | None = None,
    ) -> "SwitchFuture[S]":
        """
        Create a switch/case operation based on a key function.

        Evaluates the key function on this task's output and executes the
        corresponding task from the cases mapping.

        Args:
            cases: Mapping from case keys to tasks to execute.
            default: Optional task to execute when no case matches.
            key: Optional function to extract the case key from input.
                If None, uses the input value directly as the key.
            case_kwargs: Additional kwargs for each case task.
            default_kwargs: Additional kwargs for default task.

        Returns:
            A SwitchFuture representing the switch operation.

        Raises:
            ParameterError: If no default is provided and key is not found.

        Examples:
            Simple switch on value:
            >>> @task
            >>> def get_code(x: str) -> int:
            >>>     return {"red": 1, "green": 2, "blue": 3}[x]
            >>>
            >>> @task
            >>> def handle_red(code: int) -> str:
            >>>     return f"Red handler: {code}"
            >>>
            >>> @task
            >>> def handle_green(code: int) -> str:
            >>>     return f"Green handler: {code}"
            >>>
            >>> @task
            >>> def handle_default(code: int) -> str:
            >>>     return f"Default handler: {code}"
            >>>
            >>> result = get_code.bind(x="red").switch(
            >>>     {1: handle_red, 2: handle_green},
            >>>     default=handle_default
            >>> )

            Switch with key function:
            >>> @task
            >>> def get_value(x: int) -> int:
            >>>     return x
            >>>
            >>> result = get_value.bind(x=5).switch(
            >>>     {"small": small_handler, "large": large_handler},
            >>>     key=lambda x: "small" if x < 10 else "large"
            >>> )
        """
        from daglite.tasks import FixedParamTask

        # Extract actual tasks and merge kwargs
        actual_cases: dict[Any, Task[Any, S]] = {}
        actual_case_kwargs: dict[Any, dict[str, Any]] = {}

        for case_key, case_task in cases.items():
            if isinstance(case_task, FixedParamTask):
                actual_cases[case_key] = case_task.task
                base_kw = dict(case_task.fixed_kwargs)
                if case_kwargs and case_key in case_kwargs:
                    base_kw.update(case_kwargs[case_key])
                actual_case_kwargs[case_key] = base_kw
            else:
                actual_cases[case_key] = case_task
                actual_case_kwargs[case_key] = (
                    case_kwargs[case_key] if case_kwargs and case_key in case_kwargs else {}
                )

        # Handle default task
        actual_default: Task[Any, S] | None = None
        actual_default_kwargs: dict[str, Any] = {}

        if default is not None:
            if isinstance(default, FixedParamTask):
                actual_default = default.task
                actual_default_kwargs = {**default.fixed_kwargs, **(default_kwargs or {})}
            else:
                actual_default = default
                actual_default_kwargs = default_kwargs or {}

        # Validate that all tasks have exactly one unbound parameter
        for case_key, case_task in actual_cases.items():
            sig = inspect.signature(case_task.func)
            case_kw = actual_case_kwargs[case_key]
            unbound = [p for p in sig.parameters if p not in case_kw]

            if len(unbound) != 1:
                raise ParameterError(
                    f"Case task '{case_task.name}' (key={case_key}) must have exactly one "
                    f"unbound parameter for the input value, found {len(unbound)}: {unbound}"
                )

        if actual_default is not None:
            sig = inspect.signature(actual_default.func)
            unbound = [p for p in sig.parameters if p not in actual_default_kwargs]

            if len(unbound) != 1:
                raise ParameterError(
                    f"Default task '{actual_default.name}' must have exactly one "
                    f"unbound parameter for the input value, found {len(unbound)}: {unbound}"
                )

        return SwitchFuture(
            input_future=self,
            cases=actual_cases,
            default=actual_default,
            key_func=key,
            case_kwargs=actual_case_kwargs,
            default_kwargs=actual_default_kwargs,
            backend=self.backend,
        )

    def while_loop(
        self,
        *,
        condition: Callable[[R], bool],
        body: Task[Any, R] | FixedParamTask[Any, R],
        max_iterations: int = 1000,
        body_kwargs: dict[str, Any] | None = None,
    ) -> "WhileLoopFuture[R]":
        """
        Create a while loop that repeatedly executes a body task.

        The loop continues as long as the condition returns True, starting
        with this task's output as the initial value.

        Args:
            condition: Function to determine whether to continue looping.
            body: Task to execute on each iteration. Must return the same
                type as the input to enable chaining.
            max_iterations: Maximum number of iterations to prevent infinite loops.
            body_kwargs: Additional kwargs to pass to body task.

        Returns:
            A WhileLoopFuture representing the loop operation.

        Examples:
            Simple countdown:
            >>> @task
            >>> def start(x: int) -> int:
            >>>     return x
            >>>
            >>> @task
            >>> def decrement(x: int) -> int:
            >>>     return x - 1
            >>>
            >>> result = start.bind(x=10).while_loop(
            >>>     condition=lambda x: x > 0,
            >>>     body=decrement
            >>> )
            >>> # Result: 0

            Accumulator pattern:
            >>> @task
            >>> def accumulate(state: dict, increment: int) -> dict:
            >>>     return {"value": state["value"] + increment, "count": state["count"] + 1}
            >>>
            >>> result = start_state.bind().while_loop(
            >>>     condition=lambda s: s["count"] < 10,
            >>>     body=accumulate,
            >>>     body_kwargs={"increment": 5}
            >>> )
        """
        from daglite.tasks import FixedParamTask

        # Extract actual task and merge kwargs
        if isinstance(body, FixedParamTask):
            actual_body = body.task
            actual_body_kwargs = {**body.fixed_kwargs, **(body_kwargs or {})}
        else:
            actual_body = body
            actual_body_kwargs = body_kwargs or {}

        # Validate that body has exactly one unbound parameter
        sig = inspect.signature(actual_body.func)
        unbound = [p for p in sig.parameters if p not in actual_body_kwargs]

        if len(unbound) != 1:
            raise ParameterError(
                f"Body task '{actual_body.name}' must have exactly one "
                f"unbound parameter for the loop value, found {len(unbound)}: {unbound}"
            )

        return WhileLoopFuture(
            initial_value=self,
            condition=condition,
            body=actual_body,
            body_kwargs=actual_body_kwargs,
            max_iterations=max_iterations,
            backend=self.backend,
        )

    @override
    def get_dependencies(self) -> list[GraphBuilder]:
        deps: list[GraphBuilder] = []
        for value in self.kwargs.values():
            if isinstance(value, BaseTaskFuture):
                deps.append(value)
        return deps

    @override
    def to_graph(self) -> TaskNode:
        from daglite.graph.nodes import TaskNode

        kwargs: dict[str, ParamInput] = {}
        for name, value in self.kwargs.items():
            if isinstance(value, BaseTaskFuture):
                kwargs[name] = ParamInput.from_ref(value.id)
            else:
                kwargs[name] = ParamInput.from_value(value)
        return TaskNode(
            id=self.id,
            name=self.task.name,
            description=self.task.description,
            func=self.task.func,
            kwargs=kwargs,
            backend=self.backend,
        )


@dataclass(frozen=True)
class MapTaskFuture(BaseTaskFuture[R]):
    """
    Represents a fan-out task invocation producing a sequence of values of type R.

    Fan-out can be done in either 'extend' (Cartesian product) or 'zip' (pairwise) mode. In both
    modes, some parameters are fixed across all calls, while others are parameters will be iterated
    over. All iterable parameters must be of type `Iterable[Any]` or `TaskFuture[Iterable[Any]]`.
    """

    task: Task[Any, R]
    """Underlying task to be called."""

    mode: str  # "product" or "zip"
    """Mode of operation ('product' for Cartesian product, 'zip' for pairwise)."""

    fixed_kwargs: Mapping[str, Any]
    """
    Mapping of parameter names to fixed values applied to every call.

    Note that fixed parameters can be a combination of concrete values and TaskFutures.
    """

    mapped_kwargs: Mapping[str, Any]
    """
    Mapping of parameter names to sequences to be iterated over during calls.

    Note that sequence parameters can be a combination of concrete values and TaskFutures.
    """

    backend: Backend
    """Engine backend override for this task, if `None`, uses the default engine backend."""

    @overload
    def map(self, mapped_task: Task[Any, S]) -> "MapTaskFuture[S]": ...

    @overload
    def map(
        self, mapped_task: Task[Any, S] | FixedParamTask[Any, S], **kwargs: Any
    ) -> "MapTaskFuture[S]": ...

    def map(
        self, mapped_task: Task[Any, S] | FixedParamTask[Any, S], **kwargs: Any
    ) -> MapTaskFuture[S]:
        """
        Apply a task to each element of this sequence.

        Args:
            mapped_task: Task with exactly ONE parameter, or a FixedParamTask where
                one parameter remains unbound.
            **kwargs: Additional parameters to pass to the task. The sequence element
                will be bound to the single remaining unbound parameter.

        Returns:
            A new MapTaskFuture with the mapped task applied to each element.

        Raises:
            ParameterError: If the provided task does not have exactly one unbound
                parameter, or if kwargs overlap with fixed parameters.

        Examples:
            Single-parameter task:
            >>> @task
            >>> def double(x: int) -> int:
            >>>     return x * 2
            >>>
            >>> numbers = identity.product(x=[1, 2, 3])
            >>> doubled = numbers.map(double)
            >>> # Result: [2, 4, 6]

            Multi-parameter task with inline kwargs:
            >>> @task
            >>> def scale(x: int, factor: int) -> int:
            >>>     return x * factor
            >>>
            >>> scaled = numbers.map(scale, factor=10)
            >>> # Result: [10, 20, 30]

            Chaining map operations:
            >>> result = (
            >>>     numbers
            >>>     .map(scale, factor=2)
            >>>     .map(add, offset=10)
            >>>     .map(square)
            >>> )

            With pre-fixed tasks (still supported):
            >>> scaled = numbers.map(scale.fix(factor=10))
        """
        # NOTE: Import at runtime to avoid circular import issues
        from daglite.tasks import FixedParamTask

        if isinstance(mapped_task, FixedParamTask):
            # Check for overlapping parameters
            if overlap_params := set(mapped_task.fixed_kwargs.keys()) & set(kwargs.keys()):
                raise ParameterError(
                    f"Overlapping parameters for task '{mapped_task.name}' in `.map()`, "
                    f"specified parameters were previously bound in `.fix()`: "
                    f"{sorted(overlap_params)}"
                )
            # Merge existing fixed kwargs with new ones
            all_fixed = {**mapped_task.fixed_kwargs, **kwargs}
            sig = inspect.signature(mapped_task.task.func)
            actual_task = mapped_task.task
        else:
            all_fixed = kwargs
            sig = inspect.signature(mapped_task.func)
            actual_task = mapped_task

        # Find the one unbound parameter
        unbound_params = [p for p in sig.parameters if p not in all_fixed]
        if len(unbound_params) != 1:
            raise ParameterError(
                f"Task '{mapped_task.name}' in `.map()` must have exactly one "
                f"unbound parameter, found {len(unbound_params)} "
                f"(use `.fix()` to set scalar parameters): {unbound_params}"
            )
        param_name = unbound_params[0]

        return MapTaskFuture(
            task=actual_task,
            mode="product",
            fixed_kwargs=all_fixed,
            mapped_kwargs={param_name: self},
            backend=self.backend,
        )

    @overload
    def join(self, reducer_task: Task[Any, S]) -> "TaskFuture[S]": ...

    @overload
    def join(
        self, reducer_task: Task[Any, S] | FixedParamTask[Any, S], **kwargs: Any
    ) -> "TaskFuture[S]": ...

    def join(
        self, reducer_task: Task[Any, S] | FixedParamTask[Any, S], **kwargs: Any
    ) -> TaskFuture[S]:
        """
        Reduce this sequence to a single value.

        Args:
            reducer_task: Task with exactly ONE parameter (the sequence), or a
                FixedParamTask with one unbound parameter.
            **kwargs: Additional parameters to pass to the reducer task. The sequence
                will be bound to the single remaining unbound parameter.

        Returns:
            A TaskFuture representing the reduced single value.

        Raises:
            ParameterError: If the provided task does not have exactly one unbound
                parameter, or if kwargs overlap with fixed parameters.

        Examples:
            Simple reducer:
            >>> @task
            >>> def sum_all(xs: list[int]) -> int:
            >>>     return sum(xs)
            >>>
            >>> numbers = square.product(x=[1, 2, 3, 4])
            >>> total = numbers.join(sum_all)
            >>> # Result: 30 (1 + 4 + 9 + 16)

            Reducer with inline kwargs:
            >>> @task
            >>> def weighted_sum(xs: list[int], weight: float) -> float:
            >>>     return sum(xs) * weight
            >>>
            >>> total = numbers.join(weighted_sum, weight=0.5)
            >>> # Result: 15.0

            Complete map-reduce pipeline:
            >>> result = (
            >>>     identity.product(x=[1, 2, 3])
            >>>     .map(scale, factor=2)
            >>>     .map(add, offset=10)
            >>>     .join(sum_with_bonus, bonus=100)
            >>> )

            With pre-fixed tasks (still supported):
            >>> total = numbers.join(weighted_sum.fix(weight=0.5))
        """
        # NOTE: Import at runtime to avoid circular import issues
        from daglite.tasks import FixedParamTask

        if isinstance(reducer_task, FixedParamTask):
            # Check for overlapping parameters
            if overlap_params := set(reducer_task.fixed_kwargs.keys()) & set(kwargs.keys()):
                raise ParameterError(
                    f"Overlapping parameters for task '{reducer_task.name}' in `.join()`, "
                    f"specified parameters were previously bound in `.fix()`: "
                    f"{sorted(overlap_params)}"
                )
            # Merge existing fixed kwargs with new ones
            all_fixed = {**reducer_task.fixed_kwargs, **kwargs}
            sig = inspect.signature(reducer_task.task.func)
            actual_task = reducer_task.task
        else:
            all_fixed = kwargs
            sig = inspect.signature(reducer_task.func)
            actual_task = reducer_task

        # Find the one unbound parameter
        unbound_params = [p for p in sig.parameters if p not in all_fixed]
        if len(unbound_params) != 1:
            raise ParameterError(
                f"Task '{reducer_task.name}' in `.join()` must have exactly one "
                f"unbound parameter, found {len(unbound_params)} "
                f"(use `.fix()` to set scalar parameters): {unbound_params}"
            )
        param_name = unbound_params[0]

        merged_kwargs = dict(all_fixed)
        merged_kwargs[param_name] = self

        return TaskFuture(
            task=actual_task,
            kwargs=merged_kwargs,
            backend=self.backend,
        )

    @override
    def get_dependencies(self) -> list[GraphBuilder]:
        deps: list[GraphBuilder] = []
        for value in self.fixed_kwargs.values():
            if isinstance(value, BaseTaskFuture):
                deps.append(value)
        for seq in self.mapped_kwargs.values():
            if isinstance(seq, BaseTaskFuture):
                deps.append(seq)
        return deps

    @override
    def to_graph(self) -> MapTaskNode:
        from daglite.graph.nodes import MapTaskNode

        fixed_kwargs: dict[str, ParamInput] = {}
        mapped_kwargs: dict[str, ParamInput] = {}

        for name, value in self.fixed_kwargs.items():
            if isinstance(value, BaseTaskFuture):
                fixed_kwargs[name] = ParamInput.from_ref(value.id)
            else:
                fixed_kwargs[name] = ParamInput.from_value(value)

        for name, seq in self.mapped_kwargs.items():
            if isinstance(seq, BaseTaskFuture):
                mapped_kwargs[name] = ParamInput.from_sequence_ref(seq.id)
            else:
                mapped_kwargs[name] = ParamInput.from_sequence(seq)

        return MapTaskNode(
            id=self.id,
            name=self.task.name,
            description=self.task.description,
            func=self.task.func,
            mode=self.mode,
            fixed_kwargs=fixed_kwargs,
            mapped_kwargs=mapped_kwargs,
            backend=self.backend,
        )


@dataclass(frozen=True)
class ChooseFuture(BaseTaskFuture[S]):
    """
    Represents a conditional branching operation (if/else).

    Evaluates a condition on the upstream value and executes either the
    if_true task or if_false task based on the result.
    """

    input_future: TaskFuture[R]
    """Upstream future providing the input value for the condition."""

    condition: Callable[[R], bool]
    """Condition function to determine which branch to execute."""

    if_true: Task[Any, S]
    """Task to execute when condition returns True."""

    if_false: Task[Any, S]
    """Task to execute when condition returns False."""

    true_kwargs: Mapping[str, Any]
    """Additional kwargs to pass to if_true task."""

    false_kwargs: Mapping[str, Any]
    """Additional kwargs to pass to if_false task."""

    backend: Backend
    """Engine backend override for this operation."""

    def then(
        self,
        next_task: Task[Any, Any] | FixedParamTask[Any, Any],
        **kwargs: Any,
    ) -> "TaskFuture[Any]":
        """Chain this choose result as input to another task."""
        from daglite.tasks import FixedParamTask

        if isinstance(next_task, FixedParamTask):
            # Check for overlapping parameters
            if overlap_params := set(next_task.fixed_kwargs.keys()) & set(kwargs.keys()):
                raise ParameterError(
                    f"Overlapping parameters for task '{next_task.name}' in `.then()`, "
                    f"specified parameters were previously bound in `.fix()`: "
                    f"{sorted(overlap_params)}"
                )
            # Merge existing fixed kwargs with new ones
            all_fixed = {**next_task.fixed_kwargs, **kwargs}
            sig = inspect.signature(next_task.task.func)
            actual_task = next_task.task
        else:
            all_fixed = kwargs
            sig = inspect.signature(next_task.func)
            actual_task = next_task

        # Find the one unbound parameter
        unbound = [name for name in sig.parameters if name not in all_fixed]

        if len(unbound) == 0:
            raise ParameterError(
                f"Task '{next_task.name}' in `.then()` has no unbound parameters for "
                f"upstream value. All parameters already provided: {list(all_fixed.keys())}"
            )
        if len(unbound) > 1:
            raise ParameterError(
                f"Task '{next_task.name}' in `.then()` must have exactly one "
                f"unbound parameter for upstream value, found {len(unbound)}: {unbound} "
                f"(use `.fix()` to set scalar parameters): {unbound[1:]}"
            )

        target_param = unbound[0]
        return actual_task.bind(**{target_param: self}, **all_fixed)

    def choose(
        self,
        *,
        condition: Callable[[S], bool],
        if_true: Task[Any, Any] | FixedParamTask[Any, Any],
        if_false: Task[Any, Any] | FixedParamTask[Any, Any],
        true_kwargs: dict[str, Any] | None = None,
        false_kwargs: dict[str, Any] | None = None,
    ) -> "ChooseFuture[Any]":
        """Create a conditional branch on this choose result."""
        from daglite.tasks import task

        # Convert to TaskFuture first, then call choose
        return self.then(task(lambda x: x)).choose(
            condition=condition,
            if_true=if_true,
            if_false=if_false,
            true_kwargs=true_kwargs,
            false_kwargs=false_kwargs,
        )

    def switch(
        self,
        cases: Mapping[Any, Task[Any, Any] | FixedParamTask[Any, Any]],
        *,
        default: Task[Any, Any] | FixedParamTask[Any, Any] | None = None,
        key: Callable[[S], Any] | None = None,
        case_kwargs: Mapping[Any, dict[str, Any]] | None = None,
        default_kwargs: dict[str, Any] | None = None,
    ) -> "SwitchFuture[Any]":
        """Create a switch operation on this choose result."""
        from daglite.tasks import task

        # Convert to TaskFuture first, then call switch
        return self.then(task(lambda x: x)).switch(
            cases, default=default, key=key, case_kwargs=case_kwargs, default_kwargs=default_kwargs
        )

    def while_loop(
        self,
        *,
        condition: Callable[[S], bool],
        body: Task[Any, S] | FixedParamTask[Any, S],
        max_iterations: int = 1000,
        body_kwargs: dict[str, Any] | None = None,
    ) -> "WhileLoopFuture[S]":
        """Create a while loop on this choose result."""
        from daglite.tasks import task

        # Convert to TaskFuture first, then call while_loop
        return self.then(task(lambda x: x)).while_loop(
            condition=condition, body=body, max_iterations=max_iterations, body_kwargs=body_kwargs
        )

    @override
    def get_dependencies(self) -> list[GraphBuilder]:
        return [self.input_future]

    @override
    def to_graph(self) -> ChooseNode:
        from daglite.graph.nodes import ChooseNode

        # Convert kwargs to ParamInput
        true_kwargs_inputs: dict[str, ParamInput] = {}
        false_kwargs_inputs: dict[str, ParamInput] = {}

        for name, value in self.true_kwargs.items():
            if isinstance(value, BaseTaskFuture):
                true_kwargs_inputs[name] = ParamInput.from_ref(value.id)
            else:
                true_kwargs_inputs[name] = ParamInput.from_value(value)

        for name, value in self.false_kwargs.items():
            if isinstance(value, BaseTaskFuture):
                false_kwargs_inputs[name] = ParamInput.from_ref(value.id)
            else:
                false_kwargs_inputs[name] = ParamInput.from_value(value)

        return ChooseNode(
            id=self.id,
            name=f"choose({self.if_true.name}, {self.if_false.name})",
            description=f"Conditional: {self.if_true.name} or {self.if_false.name}",
            input_ref=self.input_future.id,
            condition=self.condition,
            if_true_func=self.if_true.func,
            if_false_func=self.if_false.func,
            true_kwargs=true_kwargs_inputs,
            false_kwargs=false_kwargs_inputs,
            backend=self.backend,
        )


@dataclass(frozen=True)
class SwitchFuture(BaseTaskFuture[S]):
    """
    Represents a switch/case operation.

    Evaluates a key function on the upstream value and executes the
    corresponding task from the cases mapping.
    """

    input_future: TaskFuture[R]
    """Upstream future providing the input value."""

    cases: Mapping[Any, Task[Any, S]]
    """Mapping from case keys to tasks."""

    default: Task[Any, S] | None
    """Default task to execute when no case matches."""

    key_func: Callable[[R], Any] | None
    """Optional function to extract the case key from input."""

    case_kwargs: Mapping[Any, Mapping[str, Any]]
    """Additional kwargs for each case task."""

    default_kwargs: Mapping[str, Any]
    """Additional kwargs for default task."""

    backend: Backend
    """Engine backend override for this operation."""

    def then(
        self,
        next_task: Task[Any, Any] | FixedParamTask[Any, Any],
        **kwargs: Any,
    ) -> "TaskFuture[Any]":
        """Chain this switch result as input to another task."""
        from daglite.tasks import FixedParamTask

        if isinstance(next_task, FixedParamTask):
            # Check for overlapping parameters
            if overlap_params := set(next_task.fixed_kwargs.keys()) & set(kwargs.keys()):
                raise ParameterError(
                    f"Overlapping parameters for task '{next_task.name}' in `.then()`, "
                    f"specified parameters were previously bound in `.fix()`: "
                    f"{sorted(overlap_params)}"
                )
            # Merge existing fixed kwargs with new ones
            all_fixed = {**next_task.fixed_kwargs, **kwargs}
            sig = inspect.signature(next_task.task.func)
            actual_task = next_task.task
        else:
            all_fixed = kwargs
            sig = inspect.signature(next_task.func)
            actual_task = next_task

        # Find the one unbound parameter
        unbound = [name for name in sig.parameters if name not in all_fixed]

        if len(unbound) == 0:
            raise ParameterError(
                f"Task '{next_task.name}' in `.then()` has no unbound parameters for "
                f"upstream value. All parameters already provided: {list(all_fixed.keys())}"
            )
        if len(unbound) > 1:
            raise ParameterError(
                f"Task '{next_task.name}' in `.then()` must have exactly one "
                f"unbound parameter for upstream value, found {len(unbound)}: {unbound} "
                f"(use `.fix()` to set scalar parameters): {unbound[1:]}"
            )

        target_param = unbound[0]
        return actual_task.bind(**{target_param: self}, **all_fixed)

    def choose(
        self,
        *,
        condition: Callable[[S], bool],
        if_true: Task[Any, Any] | FixedParamTask[Any, Any],
        if_false: Task[Any, Any] | FixedParamTask[Any, Any],
        true_kwargs: dict[str, Any] | None = None,
        false_kwargs: dict[str, Any] | None = None,
    ) -> "ChooseFuture[Any]":
        """Create a conditional branch on this switch result."""
        from daglite.tasks import task

        # Convert to TaskFuture first, then call choose
        return self.then(task(lambda x: x)).choose(
            condition=condition,
            if_true=if_true,
            if_false=if_false,
            true_kwargs=true_kwargs,
            false_kwargs=false_kwargs,
        )

    def switch(
        self,
        cases: Mapping[Any, Task[Any, Any] | FixedParamTask[Any, Any]],
        *,
        default: Task[Any, Any] | FixedParamTask[Any, Any] | None = None,
        key: Callable[[S], Any] | None = None,
        case_kwargs: Mapping[Any, dict[str, Any]] | None = None,
        default_kwargs: dict[str, Any] | None = None,
    ) -> "SwitchFuture[Any]":
        """Create a switch operation on this switch result."""
        from daglite.tasks import task

        # Convert to TaskFuture first, then call switch
        return self.then(task(lambda x: x)).switch(
            cases, default=default, key=key, case_kwargs=case_kwargs, default_kwargs=default_kwargs
        )

    def while_loop(
        self,
        *,
        condition: Callable[[S], bool],
        body: Task[Any, S] | FixedParamTask[Any, S],
        max_iterations: int = 1000,
        body_kwargs: dict[str, Any] | None = None,
    ) -> "WhileLoopFuture[S]":
        """Create a while loop on this switch result."""
        from daglite.tasks import task

        # Convert to TaskFuture first, then call while_loop
        return self.then(task(lambda x: x)).while_loop(
            condition=condition, body=body, max_iterations=max_iterations, body_kwargs=body_kwargs
        )

    @override
    def get_dependencies(self) -> list[GraphBuilder]:
        deps: list[GraphBuilder] = [self.input_future]

        # Add dependencies from kwargs
        for kwargs in self.case_kwargs.values():
            for value in kwargs.values():
                if isinstance(value, BaseTaskFuture):
                    deps.append(value)

        for value in self.default_kwargs.values():
            if isinstance(value, BaseTaskFuture):
                deps.append(value)

        return deps

    @override
    def to_graph(self) -> SwitchNode:
        from daglite.graph.nodes import SwitchNode

        # Convert cases to use funcs instead of tasks
        case_funcs: dict[Any, Callable] = {
            key: task.func for key, task in self.cases.items()
        }

        # Convert kwargs to ParamInput
        case_kwargs_inputs: dict[Any, dict[str, ParamInput]] = {}
        for key, kwargs in self.case_kwargs.items():
            case_kwargs_inputs[key] = {}
            for name, value in kwargs.items():
                if isinstance(value, BaseTaskFuture):
                    case_kwargs_inputs[key][name] = ParamInput.from_ref(value.id)
                else:
                    case_kwargs_inputs[key][name] = ParamInput.from_value(value)

        default_kwargs_inputs: dict[str, ParamInput] = {}
        for name, value in self.default_kwargs.items():
            if isinstance(value, BaseTaskFuture):
                default_kwargs_inputs[name] = ParamInput.from_ref(value.id)
            else:
                default_kwargs_inputs[name] = ParamInput.from_value(value)

        return SwitchNode(
            id=self.id,
            name="switch",
            description=f"Switch with {len(self.cases)} cases",
            input_ref=self.input_future.id,
            case_funcs=case_funcs,
            default_func=self.default.func if self.default else None,
            key_func=self.key_func,
            case_kwargs=case_kwargs_inputs,
            default_kwargs=default_kwargs_inputs,
            backend=self.backend,
        )


@dataclass(frozen=True)
class WhileLoopFuture(BaseTaskFuture[R]):
    """
    Represents a while loop operation.

    Repeatedly executes a body task while a condition remains true.
    """

    initial_value: TaskFuture[R] | R
    """Initial value for the loop."""

    condition: Callable[[R], bool]
    """Condition function to determine whether to continue looping."""

    body: Task[Any, R]
    """Task to execute on each iteration."""

    body_kwargs: Mapping[str, Any]
    """Additional kwargs to pass to body task."""

    max_iterations: int
    """Maximum number of iterations to prevent infinite loops."""

    backend: Backend
    """Engine backend override for this operation."""

    def then(
        self,
        next_task: Task[Any, Any] | FixedParamTask[Any, Any],
        **kwargs: Any,
    ) -> "TaskFuture[Any]":
        """Chain this while_loop result as input to another task."""
        from daglite.tasks import FixedParamTask

        if isinstance(next_task, FixedParamTask):
            # Check for overlapping parameters
            if overlap_params := set(next_task.fixed_kwargs.keys()) & set(kwargs.keys()):
                raise ParameterError(
                    f"Overlapping parameters for task '{next_task.name}' in `.then()`, "
                    f"specified parameters were previously bound in `.fix()`: "
                    f"{sorted(overlap_params)}"
                )
            # Merge existing fixed kwargs with new ones
            all_fixed = {**next_task.fixed_kwargs, **kwargs}
            sig = inspect.signature(next_task.task.func)
            actual_task = next_task.task
        else:
            all_fixed = kwargs
            sig = inspect.signature(next_task.func)
            actual_task = next_task

        # Find the one unbound parameter
        unbound = [name for name in sig.parameters if name not in all_fixed]

        if len(unbound) == 0:
            raise ParameterError(
                f"Task '{next_task.name}' in `.then()` has no unbound parameters for "
                f"upstream value. All parameters already provided: {list(all_fixed.keys())}"
            )
        if len(unbound) > 1:
            raise ParameterError(
                f"Task '{next_task.name}' in `.then()` must have exactly one "
                f"unbound parameter for upstream value, found {len(unbound)}: {unbound} "
                f"(use `.fix()` to set scalar parameters): {unbound[1:]}"
            )

        target_param = unbound[0]
        return actual_task.bind(**{target_param: self}, **all_fixed)

    def choose(
        self,
        *,
        condition: Callable[[R], bool],
        if_true: Task[Any, Any] | FixedParamTask[Any, Any],
        if_false: Task[Any, Any] | FixedParamTask[Any, Any],
        true_kwargs: dict[str, Any] | None = None,
        false_kwargs: dict[str, Any] | None = None,
    ) -> "ChooseFuture[Any]":
        """Create a conditional branch on this while_loop result."""
        from daglite.tasks import task

        # Convert to TaskFuture first, then call choose
        return self.then(task(lambda x: x)).choose(
            condition=condition,
            if_true=if_true,
            if_false=if_false,
            true_kwargs=true_kwargs,
            false_kwargs=false_kwargs,
        )

    def switch(
        self,
        cases: Mapping[Any, Task[Any, Any] | FixedParamTask[Any, Any]],
        *,
        default: Task[Any, Any] | FixedParamTask[Any, Any] | None = None,
        key: Callable[[R], Any] | None = None,
        case_kwargs: Mapping[Any, dict[str, Any]] | None = None,
        default_kwargs: dict[str, Any] | None = None,
    ) -> "SwitchFuture[Any]":
        """Create a switch operation on this while_loop result."""
        from daglite.tasks import task

        # Convert to TaskFuture first, then call switch
        return self.then(task(lambda x: x)).switch(
            cases, default=default, key=key, case_kwargs=case_kwargs, default_kwargs=default_kwargs
        )

    def while_loop(
        self,
        *,
        condition: Callable[[R], bool],
        body: Task[Any, R] | FixedParamTask[Any, R],
        max_iterations: int = 1000,
        body_kwargs: dict[str, Any] | None = None,
    ) -> "WhileLoopFuture[R]":
        """Create a while loop on this while_loop result."""
        from daglite.tasks import task

        # Convert to TaskFuture first, then call while_loop
        return self.then(task(lambda x: x)).while_loop(
            condition=condition, body=body, max_iterations=max_iterations, body_kwargs=body_kwargs
        )

    @override
    def get_dependencies(self) -> list[GraphBuilder]:
        deps: list[GraphBuilder] = []

        if isinstance(self.initial_value, BaseTaskFuture):
            deps.append(self.initial_value)

        for value in self.body_kwargs.values():
            if isinstance(value, BaseTaskFuture):
                deps.append(value)

        return deps

    @override
    def to_graph(self) -> WhileLoopNode:
        from daglite.graph.nodes import WhileLoopNode

        # Convert initial_value to ParamInput
        if isinstance(self.initial_value, BaseTaskFuture):
            initial_input = ParamInput.from_ref(self.initial_value.id)
        else:
            initial_input = ParamInput.from_value(self.initial_value)

        # Convert kwargs to ParamInput
        body_kwargs_inputs: dict[str, ParamInput] = {}
        for name, value in self.body_kwargs.items():
            if isinstance(value, BaseTaskFuture):
                body_kwargs_inputs[name] = ParamInput.from_ref(value.id)
            else:
                body_kwargs_inputs[name] = ParamInput.from_value(value)

        return WhileLoopNode(
            id=self.id,
            name=f"while_loop({self.body.name})",
            description=f"While loop with max {self.max_iterations} iterations",
            initial_value=initial_input,
            condition=self.condition,
            body_func=self.body.func,
            body_kwargs=body_kwargs_inputs,
            max_iterations=self.max_iterations,
            backend=self.backend,
        )
