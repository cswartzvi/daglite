"""Defines graph nodes for the daglite Intermediate Representation (IR)."""

from collections.abc import Mapping
from concurrent.futures import Future
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable, TypeVar

from typing_extensions import override

from daglite.backends import Backend
from daglite.exceptions import ExecutionError
from daglite.exceptions import ParameterError
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
    def submit(self, resolved_backend: Backend, resolved_inputs: dict[str, Any]) -> Future[Any]:
        return resolved_backend.submit(self.func, **resolved_inputs)


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

    def _build_map_calls(
        self, fixed: Mapping[str, Any], mapped: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        """Build the list of function calls for map execution."""
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
                length_details = {name: len(vals) for name, vals in mapped.items()}
                raise ParameterError(
                    f"Map task '{self.name}' with `.zip()` requires all sequences to have the "
                    f"same length. Got mismatched lengths: {length_details}. "
                    f"Consider using `.extend()` if you want a Cartesian product instead."
                )
            n = lengths.pop() if lengths else 0
            for i in range(n):
                kw = dict(fixed)
                for name, vs in mapped.items():
                    kw[name] = vs[i]
                calls.append(kw)
        else:
            raise ExecutionError(
                f"Unknown map mode '{self.mode}'. Expected 'extend' or 'zip'. "
                f"This indicates an internal error in graph construction."
            )

        return calls

    @override
    def submit(
        self, resolved_backend: Backend, resolved_inputs: dict[str, Any]
    ) -> list[Future[Any]]:
        """Submit multiple task executions."""
        fixed = {k: v for k, v in resolved_inputs.items() if k in self.fixed_kwargs}
        mapped = {k: v for k, v in resolved_inputs.items() if k in self.mapped_kwargs}
        calls = self._build_map_calls(fixed, mapped)
        return resolved_backend.submit_many(self.func, calls)


@dataclass(frozen=True)
class ConditionalNode(GraphNode):
    """Conditional execution node: execute one of two branches based on a condition."""

    condition_ref: ParamInput
    """Reference to the condition TaskFuture (must resolve to bool)."""

    then_ref: ParamInput
    """Reference to the then branch TaskFuture."""

    else_ref: ParamInput
    """Reference to the else branch TaskFuture."""

    @cached_property
    @override
    def kind(self) -> NodeKind:
        return "conditional"

    @override
    def inputs(self) -> list[tuple[str, ParamInput]]:
        return [
            ("condition", self.condition_ref),
            ("then_value", self.then_ref),
            ("else_value", self.else_ref),
        ]

    @override
    def submit(self, resolved_backend: Backend, resolved_inputs: dict[str, Any]) -> Future[Any]:
        """Execute the selected branch based on condition."""
        condition = resolved_inputs["condition"]
        then_value = resolved_inputs["then_value"]
        else_value = resolved_inputs["else_value"]

        # Select the appropriate value based on condition
        selected_value = then_value if condition else else_value

        # Return a completed future with the selected value
        future: Future[Any] = Future()
        future.set_result(selected_value)
        return future


@dataclass(frozen=True)
class LoopNode(GraphNode):
    """Loop execution node: repeatedly execute body until condition is met."""

    initial_state: ParamInput
    """Initial state value or reference."""

    body_func: Callable
    """Function to execute in loop body. Takes state and returns (new_state, should_continue)."""

    body_kwargs: Mapping[str, ParamInput]
    """Additional keyword arguments for body function."""

    max_iterations: int
    """Maximum number of iterations to prevent infinite loops."""

    @cached_property
    @override
    def kind(self) -> NodeKind:
        return "loop"

    @override
    def inputs(self) -> list[tuple[str, ParamInput]]:
        return [("initial_state", self.initial_state), *self.body_kwargs.items()]

    @override
    def submit(self, resolved_backend: Backend, resolved_inputs: dict[str, Any]) -> Future[Any]:
        """Execute the loop."""
        import inspect

        state = resolved_inputs["initial_state"]
        body_kwargs = {k: v for k, v in resolved_inputs.items() if k != "initial_state"}

        # Determine the state parameter name
        signature = inspect.signature(self.body_func)
        unbound_params = [name for name in signature.parameters if name not in self.body_kwargs]
        if len(unbound_params) != 1:
            raise ParameterError(
                f"Loop body function must have exactly one unbound parameter (the state), "
                f"but found: {unbound_params}"
            )
        state_param = unbound_params[0]

        # Execute loop iterations
        iteration = 0
        while iteration < self.max_iterations:
            # Execute body with current state
            call_kwargs = {state_param: state, **body_kwargs}
            result_future = resolved_backend.submit(self.body_func, **call_kwargs)
            new_state, should_continue = result_future.result()

            state = new_state
            iteration += 1

            if not should_continue:
                break

        # Return a completed future with the final state
        future: Future[Any] = Future()
        future.set_result(state)
        return future
