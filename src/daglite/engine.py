"""Evaluation engine for Daglite task graphs."""

from __future__ import annotations

import asyncio
import inspect
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


# Coroutine/Generator/Iterator overloads must come first (most specific)
@overload  # some type checkers need this overload for compatibility
async def evaluate(
    expr: TaskFuture[CoroutineType[Any, Any, T]],
    *,
    plugins: list[Any] | None = None,
) -> T: ...


@overload
def evaluate(
    expr: TaskFuture[Coroutine[Any, Any, T]],
    *,
    plugins: list[Any] | None = None,
) -> T: ...


@overload
def evaluate(
    expr: TaskFuture[AsyncGenerator[T, Any]],
    *,
    plugins: list[Any] | None = None,
) -> list[T]: ...


@overload
def evaluate(
    expr: TaskFuture[AsyncIterator[T]],
    *,
    plugins: list[Any] | None = None,
) -> list[T]: ...


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
    Evaluate a task graph synchronously.

    For concurrent execution of independent tasks (sibling parallelism), use
    evaluate_async() instead.

    Args:
        expr: The task graph to evaluate.
        plugins: Optional list of plugin implementations for this execution only.
            These are combined with any globally registered plugins.

    Returns:
        The result of evaluating the root task

    Examples:
        >>> # Sequential execution
        >>> result = evaluate(my_task)
        >>>
        >>> # With custom backend
        >>> result = evaluate(my_task, default_backend="threading")
        >>>
        >>> # With execution-specific plugins
        >>> from daglite.plugins.examples import ProgressTracker
        >>> result = evaluate(my_task, plugins=[ProgressTracker()])
        >>>
        >>> # For async execution with sibling parallelism
        >>> import asyncio
        >>> result = asyncio.run(evaluate_async(my_task))
    """
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
    Evaluate a task graph asynchronously.

    This function is for use within async contexts. It always uses async execution
    with sibling parallelism. For sync code, wrap this in asyncio.run().

    Args:
        expr: The task graph to evaluate.
        plugins: Optional list of plugin implementations for this execution only.
            These are combined with any globally registered plugins.

    Returns:
        The result of evaluating the root task

    Examples:
        >>> async def workflow():
        ...     result = await evaluate_async(my_task)
        ...     return result
        >>>
        >>> # With execution-specific plugins
        >>> from daglite.plugins.examples import PerformanceProfiler
        >>> result = await evaluate_async(my_task, plugins=[PerformanceProfiler()])
    """
    engine = Engine(plugins=plugins)
    return await engine.evaluate_async(expr)


# region Engine


@dataclass
class Engine:
    """
    Engine to evaluate a GraphBuilder.

    The Engine compiles a GraphBuilder into a GraphNode dict, then executes
    it in topological order.

    Execution Modes:
        - evaluate(): Sequential execution (single-threaded)
        - evaluate_async(): Async execution with sibling parallelism

    Sibling Parallelism:
        When using evaluate_async(), independent nodes at the same level of the DAG
        execute concurrently using asyncio. This is particularly beneficial for
        I/O-bound tasks (network requests, file operations).

        Tasks using SequentialBackend are automatically wrapped with asyncio.to_thread()
        to avoid blocking the event loop. ThreadBackend and ProcessBackend tasks manage
        their own parallelism.

    Backend Resolution Priority:
        1. Node-specific backend from task/task-future operations (bind, product, ...)
        2. Default task backend from `@task` decorator
        3. Engine's default_backend
    """

    plugins: list[Any] | None = None
    """Optional list of hook implementations for this execution only."""

    # cache: MutableMapping[UUID, Any] = field(default_factory=dict)
    # """Optional cache keyed by TaskFuture UUID (not used yet, but ready)."""

    _registry: EventRegistry | None = field(default=None, init=False, repr=False)
    _backend_manager: BackendManager | None = field(default=None, init=False, repr=False)
    _plugin_manager: PluginManager | None = field(default=None, init=False, repr=False)
    _event_processor: EventProcessor | None = field(default=None, init=False, repr=False)

    def evaluate(self, root: GraphBuilder) -> Any:
        """Evaluate the graph using sequential execution."""
        nodes = build_graph(root)
        return self._run_sequential(nodes, root.id)

    async def evaluate_async(self, root: GraphBuilder) -> Any:
        """Evaluate the graph using async execution with sibling parallelism."""
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

    def _run_sequential(self, nodes: dict[UUID, BaseGraphNode], root_id: UUID) -> Any:
        """Sequential blocking execution."""
        plugin_manager, event_processor = self._setup_plugin_system()
        backend_manager = BackendManager(plugin_manager, event_processor)

        plugin_manager.hook.before_graph_execute(
            root_id=root_id, node_count=len(nodes), is_async=False
        )

        start_time = time.perf_counter()
        try:
            backend_manager.start()
            event_processor.start()
            state = ExecutionState.from_nodes(nodes)
            ready = state.get_ready()

            while ready:
                nid = ready.pop()
                node = state.nodes[nid]
                result = self._execute_node_sync(node, state, backend_manager)
                ready.extend(state.mark_complete(nid, result))

            state.check_complete()
            result = state.completed_nodes[root_id]
            duration = time.perf_counter() - start_time

            plugin_manager.hook.after_graph_execute(
                root_id=root_id, result=result, duration=duration, is_async=False
            )

            return result
        except Exception as e:
            duration = time.perf_counter() - start_time
            plugin_manager.hook.on_graph_error(
                root_id=root_id, error=e, duration=duration, is_async=False
            )
            raise
        finally:
            event_processor.stop()
            backend_manager.stop()

    async def _run_async(self, nodes: dict[UUID, BaseGraphNode], root_id: UUID) -> Any:
        """Async execution with sibling parallelism."""
        plugin_manager, event_processor = self._setup_plugin_system()
        backend_manager = BackendManager(plugin_manager, event_processor)

        plugin_manager.hook.before_graph_execute(
            root_id=root_id, node_count=len(nodes), is_async=True
        )

        start_time = time.perf_counter()
        try:
            backend_manager.start()
            event_processor.start()
            state = ExecutionState.from_nodes(nodes)
            ready = state.get_ready()

            while ready:
                tasks: dict[asyncio.Task[Any], UUID] = {
                    asyncio.create_task(
                        self._execute_node_async(state.nodes[nid], state, backend_manager)
                    ): nid
                    for nid in ready
                }

                done, _ = await asyncio.wait(tasks.keys())

                ready = []
                for task in done:
                    nid = tasks[task]
                    try:
                        result = task.result()
                        ready.extend(state.mark_complete(nid, result))
                    except Exception:
                        # Cancel all remaining tasks before propagating
                        for t in tasks.keys():
                            if not t.done():
                                t.cancel()
                        await asyncio.gather(*tasks.keys(), return_exceptions=True)
                        raise

            state.check_complete()
            result = state.completed_nodes[root_id]
            duration = time.perf_counter() - start_time
            plugin_manager.hook.after_graph_execute(
                root_id=root_id, result=result, duration=duration, is_async=True
            )

            return result
        except Exception as e:
            duration = time.perf_counter() - start_time
            plugin_manager.hook.on_graph_error(
                root_id=root_id, error=e, duration=duration, is_async=True
            )
            raise
        finally:
            event_processor.stop()
            backend_manager.stop()

    def _execute_node_sync(
        self, node: BaseGraphNode, state: ExecutionState, backend_manager: BackendManager
    ) -> Any:
        """
        Execute a node synchronously and return its result.

        Returns:
            The node's execution result (single value or list)
        """
        backend = backend_manager.get(node.backend_name)
        completed_nodes = state.completed_nodes
        resolved_inputs = node.resolve_inputs(completed_nodes)

        future_or_futures = backend.submit_node(node, resolved_inputs)
        if isinstance(future_or_futures, list):
            result = [f.result() for f in future_or_futures]
        else:
            result = future_or_futures.result()

        result = _materialize_sync(result)
        return result

    async def _execute_node_async(
        self, node: BaseGraphNode, state: ExecutionState, backend_manager: BackendManager
    ) -> Any:
        """
        Execute a node asynchronously and return its result.

        Wraps backend futures as asyncio-compatible futures to enable concurrent
        execution of independent nodes.

        Returns:
            The node's execution result (single value or list)
        """
        backend = backend_manager.get(node.backend_name)
        completed_nodes = state.completed_nodes
        resolved_inputs = node.resolve_inputs(completed_nodes)
        future_or_futures = await backend.submit_node_async(node, resolved_inputs)

        if isinstance(future_or_futures, list):
            result = await asyncio.gather(*future_or_futures)
        else:
            result = await future_or_futures

        result = await _materialize_async(result)

        return result


# region State


@dataclass
class ExecutionState:
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
    def from_nodes(cls, nodes: dict[UUID, BaseGraphNode]) -> ExecutionState:
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


def _materialize_sync(result: Any) -> Any:
    """Materialize coroutines and generators in synchronous execution context."""
    # Handle lists (from map operations)
    if isinstance(result, list):
        return [_materialize_sync(item) for item in result]

    if inspect.iscoroutine(result):
        result = asyncio.run(result)

    if isinstance(result, (AsyncGenerator, AsyncIterator)):

        async def _collect():
            items = []
            async for item in result:
                items.append(item)
            return items

        return asyncio.run(_collect())

    if isinstance(result, (Generator, Iterator)) and not isinstance(result, (str, bytes)):
        return list(result)
    return result


async def _materialize_async(result: Any) -> Any:
    """Materialize coroutines and generators in asynchronous execution context."""
    # Handle lists (from map operations)
    if isinstance(result, list):
        return await asyncio.gather(*[_materialize_async(item) for item in result])

    if inspect.iscoroutine(result):
        result = await result

    if isinstance(result, (AsyncGenerator, AsyncIterator)):
        items = []
        async for item in result:
            items.append(item)
        return items

    if isinstance(result, (Generator, Iterator)) and not isinstance(result, (str, bytes)):
        return list(result)

    return result
