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
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, overload
from uuid import UUID

from typing_extensions import TypeIs

if TYPE_CHECKING:
    from pluggy import PluginManager

from daglite.backends.base import Backend
from daglite.graph.base import GraphBuilder
from daglite.graph.base import GraphNode
from daglite.graph.builder import build_graph
from daglite.settings import DagliteSettings
from daglite.tasks import BaseTaskFuture
from daglite.tasks import MapTaskFuture
from daglite.tasks import TaskFuture

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")


# region API


# Coroutine/Generator/Iterator overloads must come first (most specific)
@overload
def evaluate(
    expr: TaskFuture[Coroutine[Any, Any, T]],
    *,
    default_backend: str | Backend = "sequential",
    hooks: list[Any] | None = None,
) -> T: ...


@overload
def evaluate(
    expr: TaskFuture[AsyncGenerator[T, Any]],
    *,
    default_backend: str | Backend = "sequential",
    hooks: list[Any] | None = None,
) -> list[T]: ...


@overload
def evaluate(
    expr: TaskFuture[AsyncIterator[T]],
    *,
    default_backend: str | Backend = "sequential",
    hooks: list[Any] | None = None,
) -> list[T]: ...


@overload
def evaluate(
    expr: TaskFuture[Generator[T, Any, Any]],
    *,
    default_backend: str | Backend = "sequential",
    hooks: list[Any] | None = None,
) -> list[T]: ...


@overload
def evaluate(
    expr: TaskFuture[Iterator[T]],
    *,
    default_backend: str | Backend = "sequential",
    hooks: list[Any] | None = None,
) -> list[T]: ...


# General overloads
@overload
def evaluate(
    expr: TaskFuture[T],
    *,
    default_backend: str | Backend = "sequential",
    hooks: list[Any] | None = None,
) -> T: ...


@overload
def evaluate(
    expr: MapTaskFuture[T],
    *,
    default_backend: str | Backend = "sequential",
    hooks: list[Any] | None = None,
) -> list[T]: ...


def evaluate(
    expr: BaseTaskFuture[Any],
    *,
    default_backend: str | Backend = "sequential",
    hooks: list[Any] | None = None,
) -> Any:
    """
    Evaluate a task graph synchronously.

    For concurrent execution of independent tasks (sibling parallelism), use
    evaluate_async() instead.

    Args:
        expr: The task graph to evaluate.
        default_backend: Default backend for task execution. If a node does not have
            a specific backend assigned, this backend will be used. Defaults to "sequential".
        hooks: Optional list of hook implementations for this execution only.
            These are combined with any globally registered hooks.

    Returns:
        The result of evaluating the root task

    Examples:
        >>> # Sequential execution
        >>> result = evaluate(my_task)

        >>> # With custom backend
        >>> result = evaluate(my_task, default_backend="threading")

        >>> # With execution-specific hooks
        >>> from daglite.hooks.examples import ProgressTracker
        >>> result = evaluate(my_task, hooks=[ProgressTracker()])

        >>> # For async execution with sibling parallelism
        >>> import asyncio
        >>> result = asyncio.run(evaluate_async(my_task))
    """
    engine = Engine(default_backend=default_backend, hooks=hooks)
    return engine.evaluate(expr)


# Coroutine/Generator/Iterator overloads must come first (most specific)
@overload
async def evaluate_async(
    expr: TaskFuture[Coroutine[Any, Any, T]],
    *,
    default_backend: str | Backend = "sequential",
    hooks: list[Any] | None = None,
) -> T: ...


@overload
async def evaluate_async(
    expr: TaskFuture[AsyncGenerator[T, Any]],
    *,
    default_backend: str | Backend = "sequential",
    hooks: list[Any] | None = None,
) -> list[T]: ...


@overload
async def evaluate_async(
    expr: TaskFuture[AsyncIterator[T]],
    *,
    default_backend: str | Backend = "sequential",
    hooks: list[Any] | None = None,
) -> list[T]: ...


@overload
async def evaluate_async(
    expr: TaskFuture[Generator[T, Any, Any]],
    *,
    default_backend: str | Backend = "sequential",
    hooks: list[Any] | None = None,
) -> list[T]: ...


@overload
async def evaluate_async(
    expr: TaskFuture[Iterator[T]],
    *,
    default_backend: str | Backend = "sequential",
    hooks: list[Any] | None = None,
) -> list[T]: ...


# General overloads
@overload
async def evaluate_async(
    expr: TaskFuture[T],
    *,
    default_backend: str | Backend = "sequential",
    hooks: list[Any] | None = None,
) -> T: ...


@overload
async def evaluate_async(
    expr: MapTaskFuture[T],
    *,
    default_backend: str | Backend = "sequential",
    hooks: list[Any] | None = None,
) -> list[T]: ...


async def evaluate_async(
    expr: BaseTaskFuture[Any],
    *,
    default_backend: str | Backend = "sequential",
    hooks: list[Any] | None = None,
) -> Any:
    """
    Evaluate a task graph asynchronously.

    This function is for use within async contexts. It always uses async execution
    with sibling parallelism. For sync code, wrap this in asyncio.run().

    Args:
        expr: The task graph to evaluate.
        default_backend: Default backend for task execution. Defaults to "sequential".
        hooks: Optional list of hook implementations for this execution only.
            These are combined with any globally registered hooks.

    Returns:
        The result of evaluating the root task

    Examples:
        >>> async def workflow():
        ...     result = await evaluate_async(my_task)
        ...     return result

        >>> # Use with custom backend
        >>> async def workflow():
        ...     result = await evaluate_async(my_task, default_backend="threading")

        >>> # With execution-specific hooks
        >>> from daglite.hooks.examples import PerformanceProfiler
        >>> result = await evaluate_async(my_task, hooks=[PerformanceProfiler()])
    """
    engine = Engine(default_backend=default_backend, hooks=hooks)
    return await engine.evaluate_async(expr)


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

    hooks: list[Any] | None = None
    """Optional list of hook implementations for this execution only."""

    # cache: MutableMapping[UUID, Any] = field(default_factory=dict)
    # """Optional cache keyed by TaskFuture UUID (not used yet, but ready)."""

    _backend_cache: dict[str | Backend, Backend] = field(default_factory=dict, init=False)
    _hook_manager: "PluginManager | None" = field(default=None, init=False, repr=False)

    def evaluate(self, root: GraphBuilder) -> Any:
        """Evaluate the graph using sequential execution."""
        nodes = build_graph(root)
        return self._run_sequential(nodes, root.id)

    async def evaluate_async(self, root: GraphBuilder) -> Any:
        """Evaluate the graph using async execution with sibling parallelism."""
        nodes = build_graph(root)
        return await self._run_async(nodes, root.id)

    def _get_hook_manager(self) -> "PluginManager":
        """Get hook manager for this execution."""
        from daglite.hooks.manager import create_hook_manager_with_plugins
        from daglite.hooks.manager import get_hook_manager

        if self._hook_manager is None:
            if self.hooks:
                self._hook_manager = create_hook_manager_with_plugins(self.hooks)
            else:
                self._hook_manager = get_hook_manager()
        return self._hook_manager

    def _resolve_node_backend(self, node: GraphNode) -> Backend:
        """Decide which Backend instance to use for this node's *internal* work."""
        from daglite.backends import find_backend

        backend_key = node.backend or self.default_backend
        if backend_key not in self._backend_cache:
            backend = find_backend(backend_key)
            self._backend_cache[backend_key] = backend
        return self._backend_cache[backend_key]

    def _run_sequential(self, nodes: dict[UUID, GraphNode], root_id: UUID) -> Any:
        """Sequential blocking execution."""
        hook_manager = self._get_hook_manager()
        hook_manager.hook.before_graph_execute(
            root_id=root_id, node_count=len(nodes), is_async=False
        )

        start_time = time.perf_counter()
        try:
            state = ExecutionState.from_nodes(nodes)
            ready = state.get_ready()

            while ready:
                nid = ready.pop()
                node = state.nodes[nid]
                result = self._execute_node_sync(node, state.completed_nodes)
                ready.extend(state.mark_complete(nid, result))

            result = state.completed_nodes[root_id]
            duration = time.perf_counter() - start_time

            hook_manager.hook.after_graph_execute(
                root_id=root_id, result=result, duration=duration, is_async=False
            )

            return result
        except Exception as e:
            duration = time.perf_counter() - start_time
            hook_manager.hook.on_graph_error(
                root_id=root_id, error=e, duration=duration, is_async=False
            )
            raise

    async def _run_async(self, nodes: dict[UUID, GraphNode], root_id: UUID) -> Any:
        """Async execution with sibling parallelism."""
        hook_manager = self._get_hook_manager()
        hook_manager.hook.before_graph_execute(
            root_id=root_id, node_count=len(nodes), is_async=True
        )

        start_time = time.perf_counter()
        try:
            state = ExecutionState.from_nodes(nodes)
            ready = state.get_ready()

            while ready:
                tasks: dict[asyncio.Task[Any], UUID] = {
                    asyncio.create_task(
                        self._execute_node_async(state.nodes[nid], state.completed_nodes)
                    ): nid
                    for nid in ready
                }

                done, _ = await asyncio.wait(tasks.keys())

                ready = []
                for task in done:
                    nid = tasks[task]
                    result = task.result()
                    ready.extend(state.mark_complete(nid, result))

            result = state.completed_nodes[root_id]
            duration = time.perf_counter() - start_time
            hook_manager.hook.after_graph_execute(
                root_id=root_id, result=result, duration=duration, is_async=True
            )

            return result
        except Exception as e:
            duration = time.perf_counter() - start_time
            hook_manager.hook.on_graph_error(
                root_id=root_id, error=e, duration=duration, is_async=True
            )
            raise

    def _execute_node_sync(self, node: GraphNode, completed_nodes: dict[UUID, Any]) -> Any:
        """
        Execute a node synchronously and return its result.

        Handles both single tasks and map tasks, blocking until completion.

        Args:
            node: The graph node to execute.
            completed_nodes: Results from all completed dependency nodes.

        Returns:
            The node's execution result (single value or list)
        """
        hook_manager = self._get_hook_manager()
        backend = self._resolve_node_backend(node)
        resolved_inputs = _resolve_inputs(node, completed_nodes)

        start_time = time.perf_counter()
        iteration_total = None
        try:
            future_or_futures = node.submit(backend, resolved_inputs)

            if _is_map_future(future_or_futures):
                iteration_total = len(future_or_futures)
                hook_manager.hook.before_node_execute(
                    node_id=node.id,
                    node=node,
                    backend=backend,
                    inputs=resolved_inputs,
                    iteration_count=iteration_total,
                )
                results = []
                for idx, future_or_futures in enumerate(future_or_futures):
                    hook_manager.hook.before_iteration_execute(
                        node_id=node.id,
                        node=node,
                        backend=backend,
                        iteration_index=idx,
                        iteration_total=iteration_total,
                    )

                    iter_start = time.perf_counter()
                    iter_result = future_or_futures.result()
                    iter_result = _materialize_sync(iter_result)
                    iter_duration = time.perf_counter() - iter_start

                    hook_manager.hook.after_iteration_execute(
                        node_id=node.id,
                        node=node,
                        backend=backend,
                        iteration_index=idx,
                        iteration_total=iteration_total,
                        result=iter_result,
                        duration=iter_duration,
                    )
                    results.append(iter_result)
                result = results
            else:
                hook_manager.hook.before_node_execute(
                    node_id=node.id,
                    node=node,
                    backend=backend,
                    inputs=resolved_inputs,
                    iteration_count=None,
                )
                result = future_or_futures.result()
                result = _materialize_sync(result)

            duration = time.perf_counter() - start_time
            hook_manager.hook.after_node_execute(
                node_id=node.id,
                node=node,
                backend=backend,
                result=result,
                duration=duration,
                iteration_count=iteration_total,
            )

            return result
        except Exception as e:
            duration = time.perf_counter() - start_time
            hook_manager.hook.on_node_error(
                node_id=node.id,
                node=node,
                backend=backend,
                error=e,
                duration=duration,
                iteration_count=iteration_total,
            )
            raise

    async def _execute_node_async(self, node: GraphNode, completed_nodes: dict[UUID, Any]) -> Any:
        """
        Execute a node asynchronously and return its result.

        Wraps backend futures as asyncio-compatible futures to enable concurrent
        execution of independent nodes. SequentialBackend tasks are wrapped in
        asyncio.to_thread() to prevent blocking the event loop.

        Args:
            node: The graph node to execute.
            completed_nodes: Results from all completed dependency nodes.

        Returns:
            The node's execution result (single value or list)
        """
        from daglite.backends.local import SequentialBackend

        backend = self._resolve_node_backend(node)

        # Special case: Sequential backend executes synchronously and would block
        # the event loop, so wrap in thread
        if isinstance(backend, SequentialBackend):
            return await asyncio.to_thread(self._execute_node_sync, node, completed_nodes)

        hook_manager = self._get_hook_manager()
        resolved_inputs = _resolve_inputs(node, completed_nodes)

        start_time = time.perf_counter()
        iteration_total = None
        try:
            future_or_futures = node.submit(backend, resolved_inputs)

            if _is_map_future(future_or_futures):
                iteration_total = len(future_or_futures)
                hook_manager.hook.before_node_execute(
                    node_id=node.id,
                    node=node,
                    backend=backend,
                    inputs=resolved_inputs,
                    iteration_count=iteration_total,
                )
                results = []
                for idx, future in enumerate(future_or_futures):
                    hook_manager.hook.before_iteration_execute(
                        node_id=node.id,
                        node=node,
                        backend=backend,
                        iteration_index=idx,
                        iteration_total=iteration_total,
                    )

                    iter_start = time.perf_counter()
                    wrapped = asyncio.wrap_future(future)
                    iter_result = await wrapped
                    iter_result = await _materialize_async(iter_result)
                    iter_duration = time.perf_counter() - iter_start

                    hook_manager.hook.after_iteration_execute(
                        node_id=node.id,
                        node=node,
                        backend=backend,
                        iteration_index=idx,
                        iteration_total=iteration_total,
                        result=iter_result,
                        duration=iter_duration,
                    )
                    results.append(iter_result)
                result = results
            else:
                hook_manager.hook.before_node_execute(
                    node_id=node.id,
                    node=node,
                    backend=backend,
                    inputs=resolved_inputs,
                    iteration_count=None,
                )
                result = await asyncio.wrap_future(future_or_futures)
                result = await _materialize_async(result)

            duration = time.perf_counter() - start_time
            hook_manager.hook.after_node_execute(
                node_id=node.id,
                node=node,
                backend=backend,
                result=result,
                duration=duration,
                iteration_count=iteration_total,
            )

            return result
        except Exception as e:
            duration = time.perf_counter() - start_time
            hook_manager.hook.on_node_error(
                node_id=node.id,
                node=node,
                backend=backend,
                error=e,
                duration=duration,
                iteration_count=iteration_total,
            )
            raise


@dataclass
class ExecutionState:
    """
    Tracks graph topology and execution progress.

    Combines immutable graph structure (nodes, successors) with mutable execution
    state (indegree, completed_nodes) to manage topological execution of a DAG.
    """

    nodes: dict[UUID, GraphNode]
    """All nodes in the graph."""

    indegree: dict[UUID, int]
    """Current number of unresolved dependencies for each node."""

    successors: dict[UUID, set[UUID]]
    """Mapping from node ID to its dependent nodes."""

    completed_nodes: dict[UUID, Any]
    """Results of completed node executions."""

    @classmethod
    def from_nodes(cls, nodes: dict[UUID, GraphNode]) -> ExecutionState:
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


def _is_map_future(future: Any) -> TypeIs[list[Any]]:
    """Check if the future is a list of futures (MapTaskFuture)."""
    return isinstance(future, list)


def _resolve_inputs(node: GraphNode, completed_nodes: dict[UUID, Any]) -> dict[str, Any]:
    """Resolve all input parameters for a node using completed node results."""
    inputs = {}
    for name, param in node.inputs():
        if param.kind in ("sequence", "sequence_ref"):
            inputs[name] = param.resolve_sequence(completed_nodes)
        else:
            inputs[name] = param.resolve(completed_nodes)
    return inputs


def _materialize_sync(result: Any) -> Any:
    """Materialize coroutines and generators in synchronous execution context."""
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
