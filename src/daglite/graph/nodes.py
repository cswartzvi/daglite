"""Defines graph nodes for the daglite Intermediate Representation (IR)."""

from collections.abc import Callable
from collections.abc import Mapping
from concurrent.futures import Future
from dataclasses import dataclass
from functools import cached_property
from typing import Any, TypeVar
from uuid import UUID

from typing_extensions import override

from daglite.engine import Backend
from daglite.graph.base import GraphNode
from daglite.graph.base import NodeKind
from daglite.graph.base import ParamInput

T_co = TypeVar("T_co", covariant=True)


@dataclass(frozen=True)
class TaskNode(GraphNode):
    """Basic function task node representation within the graph IR."""

    func: Callable
    """Function to be executed for this task node."""

    kwargs: Mapping[str, ParamInput]
    """Keyword parameters for the task function."""

    @cached_property
    @override
    def kind(self) -> NodeKind:
        return "task"

    @override
    def inputs(self) -> list[tuple[str, ParamInput]]:
        return list(self.kwargs.items())

    @override
    def dependencies(self) -> set[UUID]:
        return {p.ref for p in self.kwargs.values() if p.is_ref and p.ref is not None}

    @override
    def submit(self, backend: Backend, values: Mapping[UUID, Any]) -> Future[Any]:
        """Submit single task execution."""
        kwargs = {name: p.resolve(values) for name, p in self.kwargs.items()}
        return backend.submit(self.func, **kwargs)


@dataclass(frozen=True)
class MapTaskNode(GraphNode):
    """Map function task node representation within the graph IR."""

    func: Callable
    """Function to be executed for each map iteration."""

    mode: str
    """Mapping mode: 'extend' for Cartesian product, 'zip' for parallel iteration."""

    fixed_kwargs: Mapping[str, ParamInput]
    """Fixed keyword arguments for the mapped function."""

    mapped_kwargs: Mapping[str, ParamInput]
    """Mapped keyword arguments for the mapped function."""

    @cached_property
    @override
    def kind(self) -> NodeKind:
        return "map"

    @override
    def inputs(self) -> list[tuple[str, ParamInput]]:
        return [*self.fixed_kwargs.items(), *self.mapped_kwargs.items()]

    @override
    def dependencies(self) -> set[UUID]:
        out: set[UUID] = set()
        # Add fixed and mapped kwargs refs
        for p in self.fixed_kwargs.values():
            if p.is_ref and p.ref is not None:
                out.add(p.ref)
        for p in self.mapped_kwargs.values():
            if p.is_ref and p.ref is not None:
                out.add(p.ref)
        return out

    @override
    def get_iteration_count(self, values: Mapping[UUID, Any]) -> int:
        """Return the number of iterations this map will execute."""
        mapped = {k: p.resolve_sequence(values) for k, p in self.mapped_kwargs.items()}

        if self.mode == "product":
            from math import prod

            return prod(len(v) for v in mapped.values()) if mapped else 0
        elif self.mode == "zip":
            lengths = {len(v) for v in mapped.values()}
            return lengths.pop() if lengths else 0
        else:
            return 0

    @override
    def submit(self, backend: Backend, values: Mapping[UUID, Any]) -> list[Future[Any]]:
        """Submit multiple task executions."""
        fixed = {k: p.resolve(values) for k, p in self.fixed_kwargs.items()}
        mapped = {k: p.resolve_sequence(values) for k, p in self.mapped_kwargs.items()}

        # Build calls list (same logic as before)
        from itertools import product

        calls: list[dict[str, Any]] = []

        if self.mode == "product":
            items = list(mapped.items())
            names, lists = zip(*items) if items else ([], [])
            for combo in product(*lists):
                kw = dict(fixed)
                for name, val in zip(names, combo):
                    kw[name] = val
                calls.append(kw)
        elif self.mode == "zip":
            lengths = {len(v) for v in mapped.values()}
            if len(lengths) > 1:
                from daglite.exceptions import ParameterError

                length_details = {name: len(vals) for name, vals in mapped.items()}
                raise ParameterError(
                    f"zip() requires all sequences to have the same length. "
                    f"Got mismatched lengths: {length_details}. "
                    f"Consider using extend() if you want a Cartesian product instead."
                )
            n = lengths.pop() if lengths else 0
            for i in range(n):
                kw = dict(fixed)
                for name, vs in mapped.items():
                    kw[name] = vs[i]
                calls.append(kw)
        else:
            from daglite.exceptions import ExecutionError

            raise ExecutionError(
                f"Unknown map mode '{self.mode}'. Expected 'extend' or 'zip'. "
                f"This indicates an internal error in graph construction."
            )

        return backend.submit_many(self.func, calls)


@dataclass(frozen=True)
class ChooseNode(GraphNode):
    """Conditional branching node (if/else) in the graph IR."""

    input_ref: UUID
    """Reference to the node providing the input value."""

    condition: Callable[[Any], bool]
    """Condition function to evaluate on the input."""

    if_true_func: Callable
    """Function to execute when condition is True."""

    if_false_func: Callable
    """Function to execute when condition is False."""

    true_kwargs: Mapping[str, ParamInput]
    """Parameters for the true branch function."""

    false_kwargs: Mapping[str, ParamInput]
    """Parameters for the false branch function."""

    @cached_property
    @override
    def kind(self) -> NodeKind:
        return "choose"

    @override
    def inputs(self) -> list[tuple[str, ParamInput]]:
        # Include both branches' inputs
        result = [("__input__", ParamInput.from_ref(self.input_ref))]
        result.extend((f"true_{k}", v) for k, v in self.true_kwargs.items())
        result.extend((f"false_{k}", v) for k, v in self.false_kwargs.items())
        return result

    @override
    def dependencies(self) -> set[UUID]:
        deps = {self.input_ref}
        for p in self.true_kwargs.values():
            if p.is_ref and p.ref is not None:
                deps.add(p.ref)
        for p in self.false_kwargs.values():
            if p.is_ref and p.ref is not None:
                deps.add(p.ref)
        return deps

    @override
    def submit(self, backend: Backend, values: Mapping[UUID, Any]) -> Future[Any]:
        """Submit conditional execution."""
        input_value = values[self.input_ref]
        condition_result = self.condition(input_value)

        if condition_result:
            # Execute true branch
            kwargs = {name: p.resolve(values) for name, p in self.true_kwargs.items()}
            # Find the unbound parameter and bind the input value
            sig_params = self.if_true_func.__code__.co_varnames[
                : self.if_true_func.__code__.co_argcount
            ]
            unbound = [p for p in sig_params if p not in kwargs]
            if unbound:
                kwargs[unbound[0]] = input_value
            return backend.submit(self.if_true_func, **kwargs)
        else:
            # Execute false branch
            kwargs = {name: p.resolve(values) for name, p in self.false_kwargs.items()}
            # Find the unbound parameter and bind the input value
            sig_params = self.if_false_func.__code__.co_varnames[
                : self.if_false_func.__code__.co_argcount
            ]
            unbound = [p for p in sig_params if p not in kwargs]
            if unbound:
                kwargs[unbound[0]] = input_value
            return backend.submit(self.if_false_func, **kwargs)


@dataclass(frozen=True)
class SwitchNode(GraphNode):
    """Switch/case node in the graph IR."""

    input_ref: UUID
    """Reference to the node providing the input value."""

    case_funcs: Mapping[Any, Callable]
    """Mapping from case keys to functions."""

    default_func: Callable | None
    """Default function when no case matches."""

    key_func: Callable[[Any], Any] | None
    """Optional function to extract the key from input."""

    case_kwargs: Mapping[Any, Mapping[str, ParamInput]]
    """Parameters for each case function."""

    default_kwargs: Mapping[str, ParamInput]
    """Parameters for the default function."""

    @cached_property
    @override
    def kind(self) -> NodeKind:
        return "choose"  # Reuse "choose" for switch nodes

    @override
    def inputs(self) -> list[tuple[str, ParamInput]]:
        result = [("__input__", ParamInput.from_ref(self.input_ref))]
        for case_key, case_kw in self.case_kwargs.items():
            result.extend((f"case_{case_key}_{k}", v) for k, v in case_kw.items())
        result.extend((f"default_{k}", v) for k, v in self.default_kwargs.items())
        return result

    @override
    def dependencies(self) -> set[UUID]:
        deps = {self.input_ref}
        for case_kw in self.case_kwargs.values():
            for p in case_kw.values():
                if p.is_ref and p.ref is not None:
                    deps.add(p.ref)
        for p in self.default_kwargs.values():
            if p.is_ref and p.ref is not None:
                deps.add(p.ref)
        return deps

    @override
    def submit(self, backend: Backend, values: Mapping[UUID, Any]) -> Future[Any]:
        """Submit switch execution."""
        input_value = values[self.input_ref]

        # Determine the case key
        if self.key_func is not None:
            case_key = self.key_func(input_value)
        else:
            case_key = input_value

        # Find the matching case
        if case_key in self.case_funcs:
            func = self.case_funcs[case_key]
            kwargs_params = self.case_kwargs.get(case_key, {})
            kwargs = {name: p.resolve(values) for name, p in kwargs_params.items()}
        elif self.default_func is not None:
            func = self.default_func
            kwargs = {name: p.resolve(values) for name, p in self.default_kwargs.items()}
        else:
            from daglite.exceptions import ExecutionError

            raise ExecutionError(
                f"No matching case for key '{case_key}' and no default provided. "
                f"Available cases: {list(self.case_funcs.keys())}"
            )

        # Bind the input value to the unbound parameter
        sig_params = func.__code__.co_varnames[: func.__code__.co_argcount]
        unbound = [p for p in sig_params if p not in kwargs]
        if unbound:
            kwargs[unbound[0]] = input_value

        return backend.submit(func, **kwargs)


@dataclass(frozen=True)
class WhileLoopNode(GraphNode):
    """While loop node in the graph IR."""

    initial_value: ParamInput
    """Initial value for the loop."""

    condition: Callable[[Any], bool]
    """Condition function to determine whether to continue."""

    body_func: Callable
    """Function to execute on each iteration."""

    body_kwargs: Mapping[str, ParamInput]
    """Additional parameters for the body function."""

    max_iterations: int
    """Maximum iterations to prevent infinite loops."""

    @cached_property
    @override
    def kind(self) -> NodeKind:
        return "loop"

    @override
    def inputs(self) -> list[tuple[str, ParamInput]]:
        result = [("__initial__", self.initial_value)]
        result.extend(self.body_kwargs.items())
        return result

    @override
    def dependencies(self) -> set[UUID]:
        deps: set[UUID] = set()
        if self.initial_value.is_ref and self.initial_value.ref is not None:
            deps.add(self.initial_value.ref)
        for p in self.body_kwargs.values():
            if p.is_ref and p.ref is not None:
                deps.add(p.ref)
        return deps

    @override
    def get_iteration_count(self, values: Mapping[UUID, Any]) -> int:
        """
        Return the maximum possible iteration count for hooks.

        Note: The actual iteration count may be less if the condition becomes false.
        """
        return self.max_iterations

    @override
    def submit(self, backend: Backend, values: Mapping[UUID, Any]) -> Future[Any]:
        """Submit while loop execution."""
        # Resolve the initial value and body kwargs
        current_value = self.initial_value.resolve(values)
        body_kwargs = {name: p.resolve(values) for name, p in self.body_kwargs.items()}

        # Execute the loop synchronously within a single backend submission
        def loop_body() -> Any:
            nonlocal current_value
            iteration = 0

            while self.condition(current_value) and iteration < self.max_iterations:
                # Find the unbound parameter and bind the current value
                sig_params = self.body_func.__code__.co_varnames[
                    : self.body_func.__code__.co_argcount
                ]
                unbound = [p for p in sig_params if p not in body_kwargs]

                kwargs = dict(body_kwargs)
                if unbound:
                    kwargs[unbound[0]] = current_value

                # Execute the body function directly (synchronously)
                current_value = self.body_func(**kwargs)
                iteration += 1

            return current_value

        return backend.submit(loop_body)
