# daglite/engine.py
from __future__ import annotations

import asyncio
from collections.abc import Generator
from collections.abc import Iterator
from dataclasses import dataclass
from dataclasses import field
from typing import Any, ParamSpec, TypeVar, overload
from uuid import UUID

from daglite.backends.base import Backend
from daglite.futures import BaseTaskFuture
from daglite.futures import MapTaskFuture
from daglite.futures import TaskFuture
from daglite.graph.base import GraphBuilder
from daglite.graph.base import GraphNode
from daglite.graph.builder import build_graph
from daglite.settings import DagliteSettings

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")


# region API


# Generator/Iterator overloads must come first (most specific)
@overload
def evaluate(
    expr: TaskFuture[Iterator[T]], *, default_backend: str | Backend = "sequential"
) -> list[T]: ...


@overload
def evaluate(
    expr: TaskFuture[Generator[T, Any, Any]], *, default_backend: str | Backend = "sequential"
) -> list[T]: ...


# General overloads
@overload
def evaluate(expr: TaskFuture[T], *, default_backend: str | Backend = "sequential") -> T: ...


@overload
def evaluate(
    expr: MapTaskFuture[T], *, default_backend: str | Backend = "sequential"
) -> list[T]: ...


def evaluate(
    expr: BaseTaskFuture[Any],
    default_backend: str | Backend = "sequential",
) -> Any:
    """
    Evaluate a task graph synchronously.

    For concurrent execution of independent tasks (sibling parallelism), use
    evaluate_async() instead.

    Args:
        expr: The task graph to evaluate.
        default_backend: Default backend for task execution. If a node does not have
            a specific backend assigned, this backend will be used. Defaults to "sequential".

    Returns:
        The result of evaluating the root task

    Examples:
        >>> # Sequential execution
        >>> result = evaluate(my_task)

        >>> # With custom backend
        >>> result = evaluate(my_task, default_backend="threading")

        >>> # For async execution with sibling parallelism
        >>> import asyncio
        >>> result = asyncio.run(evaluate_async(my_task))
    """
    engine = Engine(default_backend=default_backend)
    return engine.evaluate(expr)


# Generator/Iterator overloads must come first (most specific)
@overload
async def evaluate_async(
    expr: TaskFuture[Iterator[T]], *, default_backend: str | Backend = "sequential"
) -> list[T]: ...


@overload
async def evaluate_async(
    expr: TaskFuture[Generator[T, Any, Any]], *, default_backend: str | Backend = "sequential"
) -> list[T]: ...


# General overloads
@overload
async def evaluate_async(
    expr: TaskFuture[T], *, default_backend: str | Backend = "sequential"
) -> T: ...


@overload
async def evaluate_async(
    expr: MapTaskFuture[T], *, default_backend: str | Backend = "sequential"
) -> list[T]: ...


async def evaluate_async(
    expr: BaseTaskFuture[Any],
    default_backend: str | Backend = "sequential",
) -> Any:
    """
    Evaluate a task graph asynchronously.

    This function is for use within async contexts. It always uses async execution
    with sibling parallelism. For sync code, wrap this in asyncio.run().

    Args:
        expr: The task graph to evaluate.
        default_backend: Default backend for task execution. Defaults to "sequential".

    Returns:
        The result of evaluating the root task

    Examples:
        >>> async def workflow():
        ...     result = await evaluate_async(my_task)
        ...     return result

        >>> # Use with custom backend
        >>> async def workflow():
        ...     result = await evaluate_async(my_task, default_backend="threading")
    """
    engine = Engine(default_backend=default_backend)
    return await engine.evaluate_async(expr)


# endregion


# region Internal


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

    default_backend: str | Backend
    """Default backend name or instance for nodes without a specific backend."""

    settings: DagliteSettings = field(default_factory=DagliteSettings)
    """Daglite configuration settings."""

    # cache: MutableMapping[UUID, Any] = field(default_factory=dict)
    # """Optional cache keyed by TaskFuture UUID (not used yet, but ready)."""

    _backend_cache: dict[str | Backend, Backend] = field(default_factory=dict, init=False)

    def evaluate(self, root: GraphBuilder) -> Any:
        """Evaluate the graph using sequential execution."""
        nodes = build_graph(root)
        return self._run_sequential(nodes, root.id)

    async def evaluate_async(self, root: GraphBuilder) -> Any:
        """Evaluate the graph using async execution with sibling parallelism."""
        nodes = build_graph(root)
        return await self._run_async(nodes, root.id)

    def _resolve_node_backend(self, node: GraphNode) -> Backend:
        """Decide which Backend instance to use for this node's *internal* work."""
        from daglite.backends import find_backend

        backend_key = node.backend or self.default_backend
        if backend_key not in self._backend_cache:
            backend = find_backend(backend_key)
            self._backend_cache[backend_key] = backend
        return self._backend_cache[backend_key]

    def _execute_node_sync(self, node: GraphNode, values: dict[UUID, Any]) -> Any:
        """
        Execute a node synchronously and return its result.

        Handles both single tasks and map tasks, blocking until completion.

        Args:
            node: The graph node to execute
            values: Completed results from dependencies

        Returns:
            The node's execution result (single value or list)
        """
        backend = self._resolve_node_backend(node)
        future_or_futures = node.submit(backend, values)

        if isinstance(future_or_futures, list):
            # MapTaskNode - gather all results
            results = [f.result() for f in future_or_futures]
            # Materialize any generators in the list
            return [_materialize_generators(r) for r in results]
        else:
            # TaskNode - single result
            result = future_or_futures.result()
            # Materialize generator if needed
            return _materialize_generators(result)

    async def _execute_node_async(self, node: GraphNode, values: dict[UUID, Any]) -> Any:
        """
        Execute a node asynchronously and return its result.

        Wraps backend futures as asyncio-compatible futures to enable concurrent
        execution of independent nodes. SequentialBackend tasks are wrapped in
        asyncio.to_thread() to prevent blocking the event loop.

        Args:
            node: The graph node to execute
            values: Completed results from dependencies

        Returns:
            The node's execution result (single value or list)
        """
        from daglite.backends.local import SequentialBackend

        backend = self._resolve_node_backend(node)

        # Special case: Sequential backend executes synchronously and would block
        # the event loop, so wrap in thread
        if isinstance(backend, SequentialBackend):
            return await asyncio.to_thread(self._execute_node_sync, node, values)

        # All other backends: wrap their futures
        future_or_futures = node.submit(backend, values)

        if isinstance(future_or_futures, list):
            # MapTaskNode - wrap all futures and gather
            wrapped = [asyncio.wrap_future(f) for f in future_or_futures]
            results = await asyncio.gather(*wrapped)
            # Materialize any generators in the list
            return [_materialize_generators(r) for r in results]
        else:
            # TaskNode - wrap single future
            result = await asyncio.wrap_future(future_or_futures)
            # Materialize generator if needed
            return _materialize_generators(result)

    def _run_sequential(self, nodes: dict[UUID, GraphNode], root_id: UUID) -> Any:
        """Sequential blocking execution."""
        state = ExecutionState.from_nodes(nodes)
        ready = state.get_ready()

        while ready:
            nid = ready.pop()
            node = state.nodes[nid]
            result = self._execute_node_sync(node, state.values)
            ready.extend(state.mark_complete(nid, result))

        return state.values[root_id]

    async def _run_async(self, nodes: dict[UUID, GraphNode], root_id: UUID) -> Any:
        """Async execution with sibling parallelism."""
        state = ExecutionState.from_nodes(nodes)
        ready = state.get_ready()

        while ready:
            tasks: dict[asyncio.Task[Any], UUID] = {
                asyncio.create_task(self._execute_node_async(state.nodes[nid], state.values)): nid
                for nid in ready
            }

            done, _ = await asyncio.wait(tasks.keys())

            ready = []
            for task in done:
                nid = tasks[task]
                result = task.result()
                ready.extend(state.mark_complete(nid, result))

        return state.values[root_id]


@dataclass
class ExecutionState:
    """
    Tracks graph topology and execution progress.

    Combines immutable graph structure (nodes, successors) with mutable execution
    state (indegree, values) to manage topological execution of a DAG.
    """

    nodes: dict[UUID, GraphNode]
    """All nodes in the graph."""

    indegree: dict[UUID, int]
    """Current number of unresolved dependencies for each node."""

    successors: dict[UUID, set[UUID]]
    """Mapping from node ID to its dependent nodes."""

    values: dict[UUID, Any]
    """Results of completed node executions."""

    @classmethod
    def from_nodes(cls, nodes: dict[UUID, GraphNode]) -> ExecutionState:
        """
        Build execution state from a graph node dictionary.

        Computes the dependency graph (indegree and successors) needed for
        topological execution.
        """
        indegree: dict[UUID, int] = {nid: 0 for nid in nodes}
        successors: dict[UUID, set[UUID]] = {nid: set() for nid in nodes}

        for nid, node in nodes.items():
            for dep in node.dependencies():
                indegree[nid] += 1
                successors.setdefault(dep, set()).add(nid)

        return cls(
            nodes=nodes,
            indegree=indegree,
            successors=successors,
            values={},
        )

    def get_ready(self) -> list[UUID]:
        """Get all nodes with no remaining dependencies."""
        return [nid for nid, deg in self.indegree.items() if deg == 0]

    def mark_complete(self, nid: UUID, result: Any) -> list[UUID]:
        """
        Mark a node complete and return newly-ready successors.

        Args:
            nid: ID of the completed node
            result: Execution result to store

        Returns:
            List of node IDs that are now ready to execute
        """
        self.values[nid] = result
        del self.indegree[nid]  # Remove from tracking
        newly_ready = []

        for succ in self.successors.get(nid, ()):
            self.indegree[succ] -= 1
            if self.indegree[succ] == 0:
                newly_ready.append(succ)

        return newly_ready


def _materialize_generators(result: Any) -> Any:
    """
    Materialize generators and iterators to lists.

    When a task returns a generator or iterator, we consume it into a list
    to prevent single-use issues and enable proper caching. This ensures
    that multiple downstream tasks can safely use the result.

    Args:
        result: The result to potentially materialize

    Returns:
        A list if result was a generator/iterator, otherwise unchanged

    Note:
        This consumes the generator immediately. For streaming support in the
        future, we may add a task option like @task(stream=True) to opt-in to
        streaming behavior.
    """
    # Check if it's a generator or iterator (but not string/bytes which are iterable)
    if isinstance(result, (Generator, Iterator)) and not isinstance(result, (str, bytes)):
        return list(result)
    return result
