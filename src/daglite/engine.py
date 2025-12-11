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
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, overload
from uuid import UUID

if TYPE_CHECKING:
    from pluggy import PluginManager

from daglite.backends.base import Backend
from daglite.graph.base import GraphBuilder
from daglite.graph.base import GraphNode
from daglite.graph.builder import build_graph
from daglite.graph.utils import is_composite_node
from daglite.graph.utils import materialize_async
from daglite.graph.utils import materialize_sync
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
        id_mapping: dict[UUID, UUID] = {}

        if self.settings.enable_optimization:
            from daglite.graph.optimizer import optimize_graph

            nodes, id_mapping = optimize_graph(nodes, root.id)

        result = self._run_sequential(nodes, root.id, id_mapping)
        return result

    async def evaluate_async(self, root: GraphBuilder) -> Any:
        """Evaluate the graph using async execution with sibling parallelism."""
        nodes = build_graph(root)
        id_mapping: dict[UUID, UUID] = {}

        if self.settings.enable_optimization:
            from daglite.graph.optimizer import optimize_graph

            nodes, id_mapping = optimize_graph(nodes, root.id)

        result = await self._run_async(nodes, root.id, id_mapping)
        return result

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

    def _resolve_node_backend(self, node: GraphNode, all_nodes: dict[UUID, GraphNode]) -> Backend:
        """
        Decide which Backend instance to use for this node's *internal* work.

        Automatically forces SequentialBackend for nested MapTaskNodes to prevent
        exponential resource usage from nested fan-outs.

        Args:
            node: The node to resolve backend for
            all_nodes: All nodes in the graph (needed to check dependencies)

        Returns:
            Resolved backend instance
        """
        from daglite.backends import find_backend
        from daglite.backends.local import SequentialBackend

        # Check for nested fan-out: MapTaskNode depending on another MapTaskNode (or composite)
        if node.is_mapped:
            for dep_id in node.dependencies():
                dep_node = all_nodes.get(dep_id)
                if dep_node and dep_node.is_mapped:
                    # Nested fan-out detected - force sequential execution
                    if "sequential" not in self._backend_cache:  # pragma: no branch
                        self._backend_cache["sequential"] = SequentialBackend()
                    return self._backend_cache["sequential"]

        # Normal backend resolution
        backend_key = node.backend or self.default_backend
        if backend_key not in self._backend_cache:
            backend = find_backend(backend_key)
            self._backend_cache[backend_key] = backend
        return self._backend_cache[backend_key]

    def _run_sequential(
        self, nodes: dict[UUID, GraphNode], root_id: UUID, id_mapping: dict[UUID, UUID]
    ) -> Any:
        """Sequential blocking execution."""
        hook_manager = self._get_hook_manager()
        hook_manager.hook.before_graph_execute(
            root_id=root_id, node_count=len(nodes), is_async=False
        )

        start_time = time.perf_counter()
        try:
            state = ExecutionState.from_nodes(nodes, id_mapping)
            ready = state.get_ready()

            while ready:
                nid = ready.pop()
                node = state.nodes[nid]
                result = self._execute_node_sync(node, state.completed_nodes, state.nodes)
                ready.extend(state.mark_complete(nid, result))

            actual_root_id = id_mapping.get(root_id, root_id)
            result = state.completed_nodes[actual_root_id]
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

    async def _run_async(
        self, nodes: dict[UUID, GraphNode], root_id: UUID, id_mapping: dict[UUID, UUID]
    ) -> Any:
        """Async execution with sibling parallelism."""
        hook_manager = self._get_hook_manager()
        hook_manager.hook.before_graph_execute(
            root_id=root_id, node_count=len(nodes), is_async=True
        )

        start_time = time.perf_counter()
        try:
            state = ExecutionState.from_nodes(nodes, id_mapping)
            ready = state.get_ready()

            while ready:
                tasks: dict[asyncio.Task[Any], UUID] = {
                    asyncio.create_task(
                        self._execute_node_async(
                            state.nodes[nid], state.completed_nodes, state.nodes
                        )
                    ): nid
                    for nid in ready
                }

                done, _ = await asyncio.wait(tasks.keys())

                ready = []
                for task in done:
                    nid = tasks[task]
                    result = task.result()
                    ready.extend(state.mark_complete(nid, result))

            actual_root_id = id_mapping.get(root_id, root_id)
            result = state.completed_nodes[actual_root_id]
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

    def _execute_node_sync(
        self, node: GraphNode, completed_nodes: dict[UUID, Any], all_nodes: dict[UUID, GraphNode]
    ) -> Any:
        """
        Execute a node synchronously and return its result.

        Delegates to node.execute(), which handles task execution, hook firing,
        and result materialization.

        Args:
            node: The graph node to execute.
            completed_nodes: Results from all completed dependency nodes.
            all_nodes: All nodes in the graph (for backend resolution).

        Returns:
            The node's execution result (single value or list)
        """
        hook_manager = self._get_hook_manager()
        backend = self._resolve_node_backend(node, all_nodes)
        resolved_inputs = node.resolve_inputs(completed_nodes)

        start_time = time.perf_counter()
        try:
            if is_composite_node(node):
                hook_manager.hook.before_group_execute(
                    group_id=node.id,
                    node_count=len(node.chain),
                    backend=backend,
                )

            result = node.execute(backend, resolved_inputs, hook_manager)

            if is_composite_node(node):
                duration = time.perf_counter() - start_time
                hook_manager.hook.after_group_execute(
                    group_id=node.id,
                    node_count=len(node.chain),
                    backend=backend,
                    result=result,
                    duration=duration,
                )

            return materialize_sync(result)

        except Exception as e:
            if is_composite_node(node):  # pragma: no cover
                duration = time.perf_counter() - start_time
                hook_manager.hook.on_group_error(
                    group_id=node.id,
                    node_count=len(node.chain),
                    backend=backend,
                    error=e,
                    duration=duration,
                )
            raise

    async def _execute_node_async(
        self, node: GraphNode, completed_nodes: dict[UUID, Any], all_nodes: dict[UUID, GraphNode]
    ) -> Any:
        """
        Execute a node asynchronously and return its result.

        For SequentialBackend, wraps in asyncio.to_thread() to prevent blocking.
        For other backends, delegates to node.execute_async() which handles
        execution, hook firing, and result materialization.

        Args:
            node: The graph node to execute.
            completed_nodes: Results from all completed dependency nodes.
            all_nodes: All nodes in the graph (for backend resolution).

        Returns:
            The node's execution result (single value or list)
        """
        from daglite.backends.local import SequentialBackend

        backend = self._resolve_node_backend(node, all_nodes)

        # Special case: Sequential backend executes synchronously and would block
        # the event loop, so wrap in thread
        if isinstance(backend, SequentialBackend):
            return await asyncio.to_thread(
                self._execute_node_sync, node, completed_nodes, all_nodes
            )

        hook_manager = self._get_hook_manager()
        resolved_inputs = node.resolve_inputs(completed_nodes)

        start_time = time.perf_counter()
        try:
            if is_composite_node(node):
                hook_manager.hook.before_group_execute(
                    group_id=node.id,
                    node_count=len(node.chain),
                    backend=backend,
                )

            result = await node.execute_async(backend, resolved_inputs, hook_manager)

            if is_composite_node(node):
                duration = time.perf_counter() - start_time
                hook_manager.hook.after_group_execute(
                    group_id=node.id,
                    node_count=len(node.chain),
                    backend=backend,
                    result=result,
                    duration=duration,
                )

            return await materialize_async(result)

        except Exception as e:
            if is_composite_node(node):  # pragma: no cover
                duration = time.perf_counter() - start_time
                hook_manager.hook.on_group_error(
                    group_id=node.id,
                    node_count=len(node.chain),
                    backend=backend,
                    error=e,
                    duration=duration,
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

    id_mapping: dict[UUID, UUID]
    """Mapping from grouped node IDs to their composite IDs (for optimizer)."""

    @classmethod
    def from_nodes(
        cls, nodes: dict[UUID, GraphNode], id_mapping: dict[UUID, UUID] | None = None
    ) -> ExecutionState:
        """
        Build execution state from a graph node dictionary.

        Computes the dependency graph (indegree and successors) needed for
        topological execution.

        Args:
            nodes: Mapping from node IDs to GraphNode instances.
            id_mapping: Optional mapping from grouped node IDs to composite IDs.

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
            id_mapping=id_mapping or {},
        )

    def get_ready(self) -> list[UUID]:
        """Get all nodes with no remaining dependencies."""
        return [nid for nid, deg in self.indegree.items() if deg == 0]

    def mark_complete(self, nid: UUID, result: Any) -> list[UUID]:
        """
        Mark a node complete and return newly ready successors.

        When a composite node completes, also creates aliases for all grouped nodes
        it contains, allowing dependencies to resolve correctly.

        Args:
            nid: ID of the completed node
            result: Execution result to store

        Returns:
            List of node IDs that are now ready to execute
        """
        self.completed_nodes[nid] = result
        del self.indegree[nid]  # Remove from tracking
        newly_ready = []

        # Mark successors of the completed node as ready
        for succ in self.successors.get(nid, ()):
            self.indegree[succ] -= 1
            if self.indegree[succ] == 0:
                newly_ready.append(succ)

        # Handle grouped nodes: if this is a composite, add aliases for all nodes it contains
        for grouped_id, composite_id in self.id_mapping.items():
            if composite_id == nid:
                self.completed_nodes[grouped_id] = result
                # Mark successors of the grouped node as ready
                for succ in self.successors.get(grouped_id, ()):
                    self.indegree[succ] -= 1
                    if self.indegree[succ] == 0:
                        newly_ready.append(succ)

        return newly_ready
