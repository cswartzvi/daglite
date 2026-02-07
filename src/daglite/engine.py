"""Evaluation engine for Daglite task graphs."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator
from collections.abc import AsyncIterator
from collections.abc import Coroutine
from collections.abc import Generator
from collections.abc import Iterator
from dataclasses import dataclass
from types import CoroutineType
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, overload
from uuid import UUID
from uuid import uuid4

from pluggy import PluginManager

if TYPE_CHECKING:
    from daglite.plugins.events import EventProcessor
else:
    EventProcessor = Any


from daglite.backends import BackendManager
from daglite.exceptions import ExecutionError
from daglite.graph.base import BaseGraphNode
from daglite.graph.builder import build_graph
from daglite.tasks import MapTaskFuture
from daglite.tasks import TaskFuture

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")


# region Evaluation


# NOTE: Due to limitations in Python's type system, we need to repeat all overloads for both
# evaluate_sync() and evaluate_async(). Changes to overloads in one function should be mirrored in
# the other.


# Coroutine/Generator/Iterator overloads must come first (most specific)
@overload
async def evaluate(
    future: TaskFuture[CoroutineType[Any, Any, T]],
    *,
    plugins: list[Any] | None = None,
) -> T: ...


@overload
async def evaluate(
    future: TaskFuture[Coroutine[Any, Any, T]],
    *,
    plugins: list[Any] | None = None,
) -> T: ...


@overload
async def evaluate(
    future: TaskFuture[AsyncGenerator[T, Any]],
    *,
    plugins: list[Any] | None = None,
) -> list[T]: ...


@overload
async def evaluate(
    future: TaskFuture[AsyncIterator[T]],
    *,
    plugins: list[Any] | None = None,
) -> list[T]: ...


# Overloads for Generators and iterators are materialized to lists
@overload
def evaluate(
    future: TaskFuture[Generator[T, Any, Any]],
    *,
    plugins: list[Any] | None = None,
) -> list[T]: ...


@overload
def evaluate(
    future: TaskFuture[Iterator[T]],
    *,
    plugins: list[Any] | None = None,
) -> list[T]: ...


# General overloads
@overload
def evaluate(
    future: MapTaskFuture[T],
    *,
    plugins: list[Any] | None = None,
) -> list[T]: ...


@overload
def evaluate(
    future: TaskFuture[T],
    *,
    plugins: list[Any] | None = None,
) -> T: ...


def evaluate(future: Any, *, plugins: list[Any] | None = None) -> Any:
    """
    Evaluate the results of a task future synchronously.

    **NOTE**: This is a convenience wrapper around `evaluate_async()` that creates an event loop
    internally via `asyncio.run()`. Because of this, it **cannot** be called from within an async
    context (e.g., inside an `async def` or running event loop). In those cases, use
    `evaluate_async()` directly.

    Args:
        future: Task future that will be evaluated.
        plugins: Additional plugins to included with globally registered plugins.

    Returns:
        The result of evaluating the root task

    Raises:
        RuntimeError: If called from within an async context with a running event loop

    Examples:
        >>> from daglite import task, evaluate
        >>> @task
        ... def my_task(x: int, y: int) -> int:
        ...     return x + y
        >>> future = my_task(x=1, y=2)

        Standard evaluation
        >>> evaluate(future)
        3

        Evaluation with plugins
        >>> from daglite.plugins.builtin.logging import CentralizedLoggingPlugin
        >>> evaluate(future, plugins=[CentralizedLoggingPlugin()])
        3

        Sibling parallelism with threading backend
        >>> @task(backend_name="threading")
        ... def concurrent_task(x: int) -> int:
        ...     return x * 2
        >>> t1, t2 = concurrent_task(x=1), concurrent_task(x=2)
        >>> @task
        ... def combine(a: int, b: int) -> int:
        ...     return a + b
        >>> evaluate(combine(a=t1, b=t2))  # t1 and t2 run in parallel
        6
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass  # No loop running, safe to proceed
    else:
        raise RuntimeError(
            "Cannot call evaluate() from an async context. Use evaluate_async() instead."
        )
    return asyncio.run(evaluate_async(future, plugins=plugins))


# Coroutine/Generator/Iterator overloads must come first (most specific)
@overload
async def evaluate_async(
    future: TaskFuture[CoroutineType[Any, Any, T]],
    *,
    plugins: list[Any] | None = None,
) -> T: ...


@overload
async def evaluate_async(
    future: TaskFuture[Coroutine[Any, Any, T]],
    *,
    plugins: list[Any] | None = None,
) -> T: ...


@overload
async def evaluate_async(
    future: TaskFuture[AsyncGenerator[T, Any]],
    *,
    plugins: list[Any] | None = None,
) -> list[T]: ...


@overload
async def evaluate_async(
    future: TaskFuture[AsyncIterator[T]],
    *,
    plugins: list[Any] | None = None,
) -> list[T]: ...


# Overloads for Generators and iterators are materialized to lists
@overload
async def evaluate_async(
    future: TaskFuture[Generator[T, Any, Any]],
    *,
    plugins: list[Any] | None = None,
) -> list[T]: ...


@overload
async def evaluate_async(
    future: TaskFuture[Iterator[T]],
    *,
    plugins: list[Any] | None = None,
) -> list[T]: ...


# General overloads
@overload
async def evaluate_async(
    future: MapTaskFuture[T],
    *,
    plugins: list[Any] | None = None,
) -> list[T]: ...


@overload
async def evaluate_async(
    future: TaskFuture[T],
    *,
    plugins: list[Any] | None = None,
) -> T: ...


async def evaluate_async(future: Any, *, plugins: list[Any] | None = None) -> Any:
    """
    Evaluate the results of a task future via asynchronous execution.

    The future to be evaluated can contain any combination of sync and async task futures, and the
    engine will execute them in an async-first manner. This means that sibling tasks (tasks nodes
    at the same level of the DAG) can be executed concurrently if they are defined with an async
    coroutine function and/or if their backend supports async execution (e.g., threading or
    process). Tasks defined with synchronous functions are executed in a blocking manner.

    Args:
        future: Task future that will be evaluated.
        plugins: Additional plugins to included with globally registered plugins.

    Returns:
        The result of evaluating the root task

    Examples:
        >>> import asyncio
        >>> from daglite import task, evaluate_async
        >>> @task
        ... async def my_task(x: int, y: int) -> int:
        ...     return x + y
        >>> future = my_task(x=1, y=2)

        Standard evaluation
        >>> asyncio.run(evaluate_async(future))
        3

        With execution-specific plugins
        >>> import asyncio
        >>> from daglite.plugins.builtin.logging import CentralizedLoggingPlugin
        >>> asyncio.run(evaluate_async(future, plugins=[CentralizedLoggingPlugin()]))
        3
    """
    from daglite.backends.context import get_current_task

    if get_current_task():
        raise RuntimeError("Cannot call evaluate()/evaluate_async() from within another task.")

    graph_id = uuid4()
    nodes = build_graph(future)
    state = _ExecutionState.from_nodes(nodes)

    plugin_manager, event_processor = _setup_plugin_system(plugins=plugins or [])
    backend_manager = BackendManager(plugin_manager, event_processor)

    hook_ids = {"graph_id": graph_id, "root_id": future.id}
    plugin_manager.hook.before_graph_execute(**hook_ids, node_count=len(nodes))

    start_time = time.perf_counter()
    try:
        backend_manager.start()
        event_processor.start()

        nodes_to_process = state.get_source_nodes()

        while nodes_to_process:
            # Submit all ready siblings
            tasks: dict[asyncio.Task[Any], UUID] = {
                asyncio.create_task(_submit_node(state.nodes[nid], state, backend_manager)): nid
                for nid in nodes_to_process
            }

            # Wait for completion, returning early on first exception for fail-fast
            done, pending = await asyncio.wait(tasks.keys(), return_when=asyncio.FIRST_EXCEPTION)

            # Collect results and mark complete; cancel pending tasks on exception
            nodes_to_process = []
            try:
                for task in done:
                    nid = tasks[task]
                    result = task.result()
                    nodes_to_process.extend(state.mark_complete(nid, result))
            except Exception:
                for t in pending:  # pragma: no cover
                    t.cancel()
                await asyncio.gather(*tasks.keys(), return_exceptions=True)
                raise

        state.check_complete()
        result = state.completed_nodes[future.id]
        duration = time.perf_counter() - start_time

        event_processor.flush()  # Drain event queue before after_graph_execute
        plugin_manager.hook.after_graph_execute(**hook_ids, result=result, duration=duration)

    except Exception as e:
        duration = time.perf_counter() - start_time
        plugin_manager.hook.on_graph_error(**hook_ids, error=e, duration=duration)
        raise

    finally:
        event_processor.stop()
        backend_manager.stop()

    return result


async def _submit_node(
    node: BaseGraphNode, state: _ExecutionState, backend_manager: BackendManager
) -> Any:
    """
    Execute a node asynchronously and return its result.

    Wraps backend futures as asyncio-compatible futures to enable concurrent
    execution of independent nodes.

    Returns:
        The node's execution result (single value or list)
    """

    from daglite.graph.nodes import MapTaskNode

    backend = backend_manager.get(node.backend_name)
    plugin_manager = backend.plugin_manager
    completed_nodes = state.completed_nodes
    resolved_inputs = node.resolve_inputs(completed_nodes)
    resolved_output_extras = node.resolve_output_extras(completed_nodes)

    if isinstance(node, MapTaskNode):
        # Submit multiple calls for map tasks, one per iteration, and gather results
        futures = []
        mapped_inputs = node.build_iteration_calls(resolved_inputs)

        start_time = time.perf_counter()
        plugin_manager.hook.before_mapped_node_execute(
            metadata=node.to_metadata(), inputs_list=mapped_inputs
        )

        for idx, call in enumerate(mapped_inputs):
            kwargs = {"iteration_index": idx, "resolved_output_extras": resolved_output_extras}
            future = backend.submit(node.run, call, timeout=node.timeout, **kwargs)
            futures.append(future)

        result = await asyncio.gather(*futures)
        duration = time.perf_counter() - start_time

        plugin_manager.hook.after_mapped_node_execute(
            metadata=node.to_metadata(),
            inputs_list=mapped_inputs,
            results=result,
            duration=duration,
        )
    else:
        # Submit a single call for regular tasks
        future = backend.submit(
            node.run,
            resolved_inputs,
            timeout=node.timeout,
            resolved_output_extras=resolved_output_extras,
        )
        result = await future

    result = await _materialize_result(result)

    return result


def _setup_plugin_system(plugins: list[Any]) -> tuple[PluginManager, EventProcessor]:
    """
    Sets up plugin system (manager, processor, registry) for this engine.

    Args:
        plugins: List of plugin implementations to use for this execution. These are combined
            with any globally registered plugins.

    Returns:
        Tuple of (PluginManager, EventProcessor) initialized with the appropriate registry.
    """
    from daglite.plugins.events import EventProcessor
    from daglite.plugins.events import EventRegistry
    from daglite.plugins.manager import build_plugin_manager

    registry = EventRegistry()
    plugin_manager = build_plugin_manager(plugins or [], registry)
    event_processor = EventProcessor(registry)
    return plugin_manager, event_processor


async def _materialize_result(result: Any) -> Any:
    """
    Materialize mapped tasks, coroutines, and generators in asynchronous execution context.

    Args:
        result: The raw result from a node execution, which may be a list of mapped results, a
            function, coroutine, or a sync/async generator.
    """
    if isinstance(result, list):  # From map tasks
        return await asyncio.gather(*[_materialize_result(item) for item in result])

    if isinstance(result, (AsyncGenerator, AsyncIterator)):
        items = []
        async for item in result:
            items.append(item)
        return items

    if isinstance(result, (Generator, Iterator)) and not isinstance(result, (str, bytes)):
        return list(result)

    return result


# region State


@dataclass
class _ExecutionState:
    """
    Tracks graph topology and execution progress.

    Combines immutable graph structure (nodes, successors) with mutable execution
    state (indegree, completed_nodes) to manage topological execution of a DAG.
    """

    nodes: dict[UUID, BaseGraphNode]
    """All nodes in the graph."""

    indegree: dict[UUID, int]
    """Current number of unresolved dependencies for each node."""

    successors: dict[UUID, set[UUID]]
    """Mapping from node ID to its dependent nodes."""

    completed_nodes: dict[UUID, Any]
    """Results of completed node executions."""

    @classmethod
    def from_nodes(cls, nodes: dict[UUID, BaseGraphNode]) -> _ExecutionState:
        """
        Build execution state from a graph node dictionary.

        Computes the dependency graph (indegree and successors) needed for
        topological execution.

        Args:
            nodes: Mapping from node IDs to GraphNode instances.

        Returns:
            Initialized ExecutionState instance.
        """
        from collections import defaultdict

        indegree: dict[UUID, int] = {nid: 0 for nid in nodes}
        successors: dict[UUID, set[UUID]] = defaultdict(set)

        for nid, node in nodes.items():
            for dep in node.dependencies():
                indegree[nid] += 1
                successors[dep].add(nid)

        return cls(
            nodes=nodes,
            indegree=indegree,
            successors=dict(successors),
            completed_nodes={},
        )

    def get_source_nodes(self) -> list[UUID]:
        """Gets all nodes with no remaining dependencies (source nodes)."""
        return [nid for nid, deg in self.indegree.items() if deg == 0]

    def mark_complete(self, nid: UUID, result: Any) -> list[UUID]:
        """
        Mark a node complete and return newly ready successors.

        Args:
            nid: ID of the completed node
            result: Execution result to store

        Returns:
            List of node IDs that are now ready to execute
        """
        self.completed_nodes[nid] = result
        del self.indegree[nid]  # Remove from tracking
        newly_ready = []

        for succ in self.successors.get(nid, ()):
            self.indegree[succ] -= 1
            if self.indegree[succ] == 0:
                newly_ready.append(succ)

        return newly_ready

    def check_complete(self) -> None:
        """
        Check if graph execution is complete.

        Raises:
            ExecutionError: If there are remaining nodes with unresolved dependencies (cycle
            detected).
        """
        if self.indegree:
            remaining = list(self.indegree.keys())
            raise ExecutionError(
                f"Cycle detected in task graph. {len(remaining)} node(s) have unresolved "
                f"dependencies and cannot execute. This indicates a circular dependency. "
                f"Remaining node IDs: {remaining[:5]}{'...' if len(remaining) > 5 else ''}"
            )
