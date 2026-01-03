"""Defines graph nodes for the daglite Intermediate Representation (IR)."""

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, TypeVar
from uuid import UUID

from typing_extensions import override

from daglite.exceptions import ExecutionError
from daglite.exceptions import ParameterError
from daglite.graph.base import BaseGraphNode
from daglite.graph.base import GraphMetadata
from daglite.graph.base import ParamInput

T_co = TypeVar("T_co", covariant=True)


@dataclass(frozen=True)
class TaskNode(BaseGraphNode):
    """Basic function task node representation within the graph IR."""

    func: Callable
    """Function to be executed for this task node."""

    kwargs: Mapping[str, ParamInput]
    """Keyword parameters for the task function."""

    retries: int = 0
    """Number of times to retry the task on failure."""

    @override
    def dependencies(self) -> set[UUID]:
        return {p.ref for p in self.kwargs.values() if p.is_ref and p.ref is not None}

    @override
    def resolve_inputs(self, completed_nodes: Mapping[UUID, Any]) -> dict[str, Any]:
        inputs = {}
        for name, param in self.kwargs.items():
            if param.kind in ("sequence", "sequence_ref"):  # pragma: no cover
                # Defensive: TaskNode kwargs are always "value" or "ref", never sequence types
                inputs[name] = param.resolve_sequence(completed_nodes)
            else:
                inputs[name] = param.resolve(completed_nodes)
        return inputs

    @override
    def to_metadata(self) -> "GraphMetadata":
        return GraphMetadata(
            id=self.id,
            name=self.name,
            kind="task",
            description=self.description,
            backend_name=self.backend_name,
            key=self.key,
        )

    @override
    def run(self, resolved_inputs: dict[str, Any], **kwargs: Any) -> Any:
        from dataclasses import replace

        metadata = replace(self.to_metadata(), key=self.name)

        return _run_sync_impl(
            func=self.func,
            metadata=metadata,
            resolved_inputs=resolved_inputs,
            retries=self.retries,
        )

    @override
    async def run_async(self, resolved_inputs: dict[str, Any], **kwargs: Any) -> Any:
        from dataclasses import replace

        metadata = replace(self.to_metadata(), key=self.name)

        return await _run_async_impl(
            func=self.func,
            metadata=metadata,
            resolved_inputs=resolved_inputs,
            retries=self.retries,
        )


@dataclass(frozen=True)
class MapTaskNode(BaseGraphNode):
    """Map function task node representation within the graph IR."""

    func: Callable
    """Function to be executed for each map iteration."""

    mode: str
    """Mapping mode: 'extend' for Cartesian product, 'zip' for parallel iteration."""

    fixed_kwargs: Mapping[str, ParamInput]
    """Fixed keyword arguments for the mapped function."""

    mapped_kwargs: Mapping[str, ParamInput]
    """Mapped keyword arguments for the mapped function."""

    retries: int = 0
    """Number of times to retry the task on failure."""

    @override
    def dependencies(self) -> set[UUID]:
        deps = set()
        for param in self.fixed_kwargs.values():
            if param.is_ref and param.ref is not None:
                deps.add(param.ref)
        for param in self.mapped_kwargs.values():
            if param.is_ref and param.ref is not None:
                deps.add(param.ref)
        return deps

    @override
    def resolve_inputs(self, completed_nodes: Mapping[UUID, Any]) -> dict[str, Any]:
        inputs = {}

        # Resolve fixed kwargs
        for name, param in self.fixed_kwargs.items():
            if param.kind in ("sequence", "sequence_ref"):  # pragma: no cover
                # Defensive: fixed_kwargs are always "value" or "ref", never sequence types
                inputs[name] = param.resolve_sequence(completed_nodes)
            else:
                inputs[name] = param.resolve(completed_nodes)

        # Resolve mapped kwargs
        for name, param in self.mapped_kwargs.items():
            if param.kind in ("sequence", "sequence_ref"):
                inputs[name] = param.resolve_sequence(completed_nodes)
            else:  # pragma: no cover
                # Defensive: mapped_kwargs are always "sequence" or "sequence_ref", never value/ref
                inputs[name] = param.resolve(completed_nodes)

        return inputs

    def build_iteration_calls(self, resolved_inputs: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Build the list of input dictionaries for each iteration of the mapped node.

        Args:
            resolved_inputs: Pre-resolved parameter inputs for this node.
        """

        from itertools import product

        fixed = {k: v for k, v in resolved_inputs.items() if k in self.fixed_kwargs}
        mapped = {k: v for k, v in resolved_inputs.items() if k in self.mapped_kwargs}

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
    def to_metadata(self) -> "GraphMetadata":
        """Returns a metadata object for this graph node."""
        return GraphMetadata(
            id=self.id,
            name=self.name,
            kind="map",
            description=self.description,
            backend_name=self.backend_name,
            key=self.key,
        )

    @override
    def run(self, resolved_inputs: dict[str, Any], **kwargs: Any) -> Any:
        from dataclasses import replace

        node_key = f"{self.name}[{kwargs['iteration_index']}]"
        metadata = replace(self.to_metadata(), key=node_key)

        return _run_sync_impl(
            func=self.func,
            metadata=metadata,
            resolved_inputs=resolved_inputs,
            retries=self.retries,
        )

    @override
    async def run_async(self, resolved_inputs: dict[str, Any], **kwargs: Any) -> Any:
        from dataclasses import replace

        node_key = f"{self.name}[{kwargs['iteration_index']}]"
        metadata = replace(self.to_metadata(), key=node_key)

        return await _run_async_impl(
            func=self.func,
            metadata=metadata,
            resolved_inputs=resolved_inputs,
            retries=self.retries,
        )


# region Helpers


def _run_sync_impl(
    func: Callable[..., Any],
    metadata: GraphMetadata,
    resolved_inputs: dict[str, Any],
    retries: int = 0,
) -> Any:
    """Helper to run a node synchronously with context setup and retries."""
    from daglite.backends.context import get_plugin_manager
    from daglite.backends.context import get_reporter
    from daglite.backends.context import reset_current_task
    from daglite.backends.context import set_current_task

    task_token = set_current_task(metadata)
    plugin_manager = get_plugin_manager()
    reporter = get_reporter()

    plugin_manager.hook.before_node_execute(
        metadata=metadata,
        inputs=resolved_inputs,
        reporter=reporter,
    )

    last_error: Exception | None = None
    attempt = 0
    max_attempts = retries + 1

    start_time = time.time()
    try:
        while attempt < max_attempts:
            attempt += 1
            try:
                # Notify plugins before retry (skip for first attempt)
                if attempt > 1:
                    assert last_error is not None
                    plugin_manager.hook.before_node_retry(
                        metadata=metadata,
                        inputs=resolved_inputs,
                        attempt=attempt,
                        last_error=last_error,
                        reporter=reporter,
                    )

                result = func(**resolved_inputs)

                # Notify plugins after successful retry (skip for first attempt)
                if attempt > 1:
                    plugin_manager.hook.after_node_retry(
                        metadata=metadata,
                        inputs=resolved_inputs,
                        attempt=attempt,
                        succeeded=True,
                        reporter=reporter,
                    )

                duration = time.time() - start_time

                plugin_manager.hook.after_node_execute(
                    metadata=metadata,
                    inputs=resolved_inputs,
                    result=result,
                    duration=duration,
                    reporter=reporter,
                )

                return result

            except Exception as error:
                last_error = error

                # Notify plugins after failed retry (skip for last attempt which will error)
                if attempt > 1 and attempt < max_attempts:
                    plugin_manager.hook.after_node_retry(
                        metadata=metadata,
                        inputs=resolved_inputs,
                        attempt=attempt,
                        succeeded=False,
                        reporter=reporter,
                    )

                if attempt >= max_attempts:
                    break  # No more retries left

        # All attempts exhausted
        duration = time.time() - start_time
        assert last_error is not None
        plugin_manager.hook.on_node_error(
            metadata=metadata,
            inputs=resolved_inputs,
            error=last_error,
            duration=duration,
            reporter=reporter,
        )
        raise last_error

    finally:
        reset_current_task(task_token)


async def _run_async_impl(
    func: Callable[..., Any],
    metadata: GraphMetadata,
    resolved_inputs: dict[str, Any],
    retries: int = 0,
) -> Any:
    """Helper to run a node asynchronously with context setup and retries."""
    import inspect

    from daglite.backends.context import get_plugin_manager
    from daglite.backends.context import get_reporter
    from daglite.backends.context import reset_current_task
    from daglite.backends.context import set_current_task

    task_token = set_current_task(metadata)
    plugin_manager = get_plugin_manager()
    reporter = get_reporter()

    plugin_manager.hook.before_node_execute(
        metadata=metadata,
        inputs=resolved_inputs,
        reporter=reporter,
    )

    last_error: Exception | None = None
    attempt = 0
    max_attempts = retries + 1

    start_time = time.time()
    try:
        while attempt < max_attempts:
            attempt += 1
            try:
                # Notify plugins before retry (skip for first attempt)
                if attempt > 1:
                    assert last_error is not None
                    plugin_manager.hook.before_node_retry(
                        metadata=metadata,
                        inputs=resolved_inputs,
                        attempt=attempt,
                        last_error=last_error,
                        reporter=reporter,
                    )

                # Handle both async and sync functions
                if inspect.iscoroutinefunction(func):
                    result = await func(**resolved_inputs)
                else:
                    assert metadata.backend_name != "sequential"
                    result = func(**resolved_inputs)

                # Notify plugins after successful retry (skip for first attempt)
                if attempt > 1:
                    plugin_manager.hook.after_node_retry(
                        metadata=metadata,
                        inputs=resolved_inputs,
                        attempt=attempt,
                        succeeded=True,
                        reporter=reporter,
                    )

                duration = time.time() - start_time

                plugin_manager.hook.after_node_execute(
                    metadata=metadata,
                    inputs=resolved_inputs,
                    result=result,
                    duration=duration,
                    reporter=reporter,
                )

                return result

            except Exception as error:
                last_error = error

                # Notify plugins after failed retry (skip for last attempt which will error)
                if attempt > 1 and attempt < max_attempts:
                    plugin_manager.hook.after_node_retry(
                        metadata=metadata,
                        inputs=resolved_inputs,
                        attempt=attempt,
                        succeeded=False,
                        reporter=reporter,
                    )

                if attempt >= max_attempts:
                    break  # No more retries left

        # All attempts exhausted
        duration = time.time() - start_time
        assert last_error is not None
        plugin_manager.hook.on_node_error(
            metadata=metadata,
            inputs=resolved_inputs,
            error=last_error,
            duration=duration,
            reporter=reporter,
        )
        raise last_error

    finally:
        reset_current_task(task_token)
