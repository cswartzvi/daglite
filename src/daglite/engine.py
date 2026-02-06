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
from dataclasses import field
from types import CoroutineType
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, overload
from uuid import UUID
from uuid import uuid4

from pluggy import PluginManager

if TYPE_CHECKING:
    from daglite.plugins.events import EventProcessor
    from daglite.plugins.events import EventRegistry
else:
    EventProcessor = Any
    EventRegistry = Any


from daglite.backends import BackendManager
from daglite.exceptions import ExecutionError
from daglite.graph.base import BaseGraphNode
from daglite.graph.base import GraphBuilder
from daglite.graph.builder import build_graph
from daglite.tasks import BaseTaskFuture
from daglite.tasks import MapTaskFuture
from daglite.tasks import TaskFuture

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")


# region API


@overload
def evaluate(
    expr: TaskFuture[Generator[T, Any, Any]],
    *,
    plugins: list[Any] | None = None,
) -> list[T]: ...


@overload
def evaluate(
    expr: TaskFuture[Iterator[T]],
    *,
    plugins: list[Any] | None = None,
) -> list[T]: ...


# General overloads
@overload
def evaluate(
    expr: TaskFuture[T],
    *,
    plugins: list[Any] | None = None,
) -> T: ...


@overload
def evaluate(
    expr: MapTaskFuture[T],
    *,
    plugins: list[Any] | None = None,
) -> list[T]: ...


def evaluate(
    expr: BaseTaskFuture[Any],
    *,
    plugins: list[Any] | None = None,
) -> Any:
    """
    Evaluate the results of a task future synchronously.

    This is a convenience wrapper around `evaluate_async()` that creates an event loop
    internally via `asyncio.run()`. All execution uses the same async-first engine,
    giving sync callers automatic sibling concurrency.

    **Important**: Cannot be called from within an async context (e.g., inside an
    `async def` or running event loop). Use `evaluate_async()` directly in those cases.

    Args:
        expr: Task graph object to evaluate, typically a `TaskFuture` or `MapTaskFuture`.
        plugins: Optional list of plugin implementations for this execution only.
            These are combined with any globally registered plugins.

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
    engine = Engine(plugins=plugins)
    return engine.evaluate(expr)


# Coroutine/Generator/Iterator overloads must come first (most specific)
@overload  # some type checkers need this overload for compatibility
async def evaluate_async(
    expr: TaskFuture[CoroutineType[Any, Any, T]],
    *,
    plugins: list[Any] | None = None,
) -> T: ...


@overload
async def evaluate_async(
    expr: TaskFuture[Coroutine[Any, Any, T]],
    *,
    plugins: list[Any] | None = None,
) -> T: ...


@overload
async def evaluate_async(
    expr: TaskFuture[AsyncGenerator[T, Any]],
    *,
    plugins: list[Any] | None = None,
) -> list[T]: ...


@overload
async def evaluate_async(
    expr: TaskFuture[AsyncIterator[T]],
    *,
    plugins: list[Any] | None = None,
) -> list[T]: ...


@overload
async def evaluate_async(
    expr: TaskFuture[Generator[T, Any, Any]],
    *,
    plugins: list[Any] | None = None,
) -> list[T]: ...


@overload
async def evaluate_async(
    expr: TaskFuture[Iterator[T]],
    *,
    plugins: list[Any] | None = None,
) -> list[T]: ...


# General overloads
@overload
async def evaluate_async(
    expr: TaskFuture[T],
    *,
    plugins: list[Any] | None = None,
) -> T: ...


@overload
async def evaluate_async(
    expr: MapTaskFuture[T],
    *,
    plugins: list[Any] | None = None,
) -> list[T]: ...


async def evaluate_async(
    expr: BaseTaskFuture[Any],
    *,
    plugins: list[Any] | None = None,
) -> Any:
    """
    Evaluate the results of a task future via asynchronous execution.

    Sibling tasks execute concurrently using asyncio. Both sync and async task functions
    are supported — sync functions are called directly within the async execution context.

    Use this when:
    - Your tasks are async (defined with `async def`)
    - You need to integrate with existing async code
    - You want to avoid blocking the event loop

    For synchronous callers, `evaluate()` provides a simpler interface that wraps this
    function with `asyncio.run()`.

    Args:
        expr: Task graph to evaluate, typically a `TaskFuture` or `MapTaskFuture`.
        plugins: Optional list of plugin implementations for this execution only. These are
            combined with any globally registered plugins.

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
    engine = Engine(plugins=plugins)
    return await engine.evaluate_async(expr)


# region Engine


@dataclass
class Engine:
    """
    Engine to evaluate a `GraphBuilder` (or more commonly, a `TaskFuture`).

    The Engine compiles a `GraphBuilder` into a `GraphNode` dict, then executes it in topological
    order using an async-first execution model. Individual nodes are executed via backends
    managed by a `BackendManager`.

    Sibling tasks (tasks at the same level of the DAG) execute concurrently via asyncio when
    using appropriate backends (threading, processes). Both sync and async task functions are
    supported.
    """

    plugins: list[Any] | None = None
    """Optional list of plugins implementations to be used during execution."""

    _registry: EventRegistry | None = field(default=None, init=False, repr=False)
    _backend_manager: BackendManager | None = field(default=None, init=False, repr=False)
    _plugin_manager: PluginManager | None = field(default=None, init=False, repr=False)
    _event_processor: EventProcessor | None = field(default=None, init=False, repr=False)

    def evaluate(self, root: GraphBuilder) -> Any:
        """
        Builds and evaluates a graph synchronously.

        Wraps `evaluate_async()` with `asyncio.run()`. Cannot be called from within
        an async context.

        Args:
            root: Root `GraphBuilder` to evaluate, typically a `TaskFuture`.

        Returns:
            The result of evaluating the root node.

        Raises:
            RuntimeError: If called from within an async context with a running event loop.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass  # No loop running, safe to proceed
        else:
            raise RuntimeError(
                "Cannot call Engine.evaluate() from an async context. "
                "Use Engine.evaluate_async() instead."
            )
        return asyncio.run(self.evaluate_async(root))

    async def evaluate_async(self, root: GraphBuilder) -> Any:
        """
        Builds and evaluates a graph using asynchronous execution.

        Args:
            root: Root `GraphBuilder` to evaluate, typically a `TaskFuture`.

        Returns:
            The result of evaluating the root node.
        """
        nodes = build_graph(root)
        return await self._run_async(nodes, root.id)

    def _setup_plugin_system(self) -> tuple[PluginManager, EventProcessor]:
        """Sets up plugin system (manager, processor, registry) for this engine."""
        from daglite.plugins.events import EventProcessor
        from daglite.plugins.events import EventRegistry
        from daglite.plugins.manager import build_plugin_manager

        if self._registry is None:  # pragma: no branch
            self._registry = EventRegistry()

        if self._plugin_manager is None:  # pragma: no branch
            self._plugin_manager = build_plugin_manager(self.plugins or [], self._registry)

        if self._event_processor is None:  # pragma: no branch
            self._event_processor = EventProcessor(self._registry)

        return self._plugin_manager, self._event_processor

    async def _run_async(self, nodes: dict[UUID, BaseGraphNode], root_id: UUID) -> Any:
        """Async execution with sibling parallelism."""
        graph_id = uuid4()
        plugin_manager, event_processor = self._setup_plugin_system()
        backend_manager = BackendManager(plugin_manager, event_processor)

        plugin_manager.hook.before_graph_execute(
            graph_id=graph_id, root_id=root_id, node_count=len(nodes)
        )

        start_time = time.perf_counter()
        try:
            backend_manager.start()
            event_processor.start()
            state = _ExecutionState.from_nodes(nodes)
            ready = state.get_ready()

            while ready:
                # Submit all ready siblings
                tasks: dict[asyncio.Task[Any], UUID] = {
                    asyncio.create_task(
                        self._execute_node_async(state.nodes[nid], state, backend_manager)
                    ): nid
                    for nid in ready
                }

                # Wait for any sibling to complete
                done, _ = await asyncio.wait(tasks.keys())

                # Collect results and mark complete
                ready = []
                for task in done:
                    nid = tasks[task]
                    try:
                        result = task.result()
                        ready.extend(state.mark_complete(nid, result))
                    except Exception:
                        # Cancel all remaining tasks before propagating
                        for t in tasks.keys():
                            if not t.done():  # pragma: no cover
                                # Defensive: Cancels concurrent siblings on error. Requires
                                # contrived timing to test where one task fails while others still
                                # running
                                t.cancel()
                        await asyncio.gather(*tasks.keys(), return_exceptions=True)
                        raise

            state.check_complete()
            result = state.completed_nodes[root_id]
            duration = time.perf_counter() - start_time

            event_processor.flush()  # Drain event queue before after_graph_execute
            plugin_manager.hook.after_graph_execute(
                graph_id=graph_id, root_id=root_id, result=result, duration=duration
            )

            return result
        except Exception as e:
            duration = time.perf_counter() - start_time
            plugin_manager.hook.on_graph_error(
                graph_id=graph_id, root_id=root_id, error=e, duration=duration
            )
            raise
        finally:
            event_processor.stop()
            backend_manager.stop()

    async def _execute_node_async(
        self, node: BaseGraphNode, state: _ExecutionState, backend_manager: BackendManager
    ) -> Any:
        """
        Execute a node asynchronously and return its result.

        Wraps backend futures as asyncio-compatible futures to enable concurrent
        execution of independent nodes.

        Returns:
            The node's execution result (single value or list)
        """
        from asyncio import wrap_future

        from daglite.graph.nodes import MapTaskNode

        backend = backend_manager.get(node.backend_name)
        completed_nodes = state.completed_nodes
        resolved_inputs = node.resolve_inputs(completed_nodes)
        resolved_output_extras = node.resolve_output_extras(completed_nodes)

        # Determine how to submit to backend based on node type
        if isinstance(node, MapTaskNode):
            # For mapped nodes, submit each iteration separately
            futures = []
            mapped_inputs = node.build_iteration_calls(resolved_inputs)

            start_time = time.perf_counter()
            backend.plugin_manager.hook.before_mapped_node_execute(
                metadata=node.to_metadata(), inputs_list=mapped_inputs
            )

            for idx, call in enumerate(mapped_inputs):
                kwargs = {"iteration_index": idx, "resolved_output_extras": resolved_output_extras}
                future = wrap_future(
                    backend.submit(node.run_async, call, timeout=node.timeout, **kwargs)
                )
                futures.append(future)

            result = await asyncio.gather(*futures)
            duration = time.perf_counter() - start_time

            backend.plugin_manager.hook.after_mapped_node_execute(
                metadata=node.to_metadata(),
                inputs_list=mapped_inputs,
                results=result,
                duration=duration,
            )
        else:
            future = wrap_future(
                backend.submit(
                    node.run_async,
                    resolved_inputs,
                    timeout=node.timeout,
                    resolved_output_extras=resolved_output_extras,
                )
            )
            result = await future

        result = await _materialize_async(result)

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

        return cls(nodes=nodes, indegree=indegree, successors=dict(successors), completed_nodes={})

    def get_ready(self) -> list[UUID]:
        """Get all nodes with no remaining dependencies."""
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


# region Helpers


async def _materialize_async(result: Any) -> Any:
    """Materialize coroutines and generators in asynchronous execution context."""
    if isinstance(result, list):  # From map tasks
        return await asyncio.gather(*[_materialize_async(item) for item in result])

    if isinstance(result, (AsyncGenerator, AsyncIterator)):
        items = []
        async for item in result:
            items.append(item)
        return items

    if isinstance(result, (Generator, Iterator)) and not isinstance(result, (str, bytes)):
        return list(result)

    return result
