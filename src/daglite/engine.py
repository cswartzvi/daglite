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
    Evaluate the results of a task future via synchronous execution.

    Sibling tasks (tasks at the same level of the DAG) are submitted concurrently when using
    non-sequential backends (e.g., `threading`, `processes`). This enables parallel execution
    without requiring async/await syntax.

    Args:
        expr: Task graph object to evaluate, typically a `TaskFuture` or `MapTaskFuture`.
        plugins: Optional list of plugin implementations for this execution only.
            These are combined with any globally registered plugins.

    Returns:
        The result of evaluating the root task

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
        >>> from daglite.plugins.default import CentralizedLoggingPlugin
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

    This function is designed for async contexts and async tasks. Sibling tasks execute
    concurrently using asyncio. Use this when:
    - Your tasks are async (defined with `async def`)
    - You need to integrate with existing async code
    - You want to avoid blocking the event loop

    For synchronous tasks with parallel execution, `evaluate()` with a threading/processes
    backend is often simpler.

    **Important**: Synchronous tasks cannot be executed with the `sequential` backend in
    `evaluate_async()`. Use `threading` or `processes` backend for parallel execution, or
    `evaluate()` for synchronous evaluation.

    Args:
        expr: Task graph to evaluate, typically a `TaskFuture` or `MapTaskFuture`.
        plugins: Optional list of plugin implementations for this execution only. These are
            combined with any globally registered plugins.

    Returns:
        The result of evaluating the root task

    Raises:
        ValueError: If synchronous tasks are used with the sequential backend

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
        >>> from daglite.plugins.default import CentralizedLoggingPlugin
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
    order in either synchronous or asynchronous mode. Individual nodes are executed via backends
    managed by a `BackendManager`.

    Sibling tasks (tasks at the same level of the DAG) execute concurrently in both sync and async
    modes when using appropriate backends:
    - **Sync mode**: Siblings using `threading` or `processes` backends run in parallel
    - **Async mode**: All sibling tasks run concurrently via asyncio

    **Backend compatibility**:
    - Async mode (`evaluate_async()`): Sequential backend cannot be used with synchronous tasks;
      use threading/processes backend instead, or use sync mode (`evaluate()`).
    """

    plugins: list[Any] | None = None
    """Optional list of plugins implementations to be used during execution."""

    _registry: EventRegistry | None = field(default=None, init=False, repr=False)
    _backend_manager: BackendManager | None = field(default=None, init=False, repr=False)
    _plugin_manager: PluginManager | None = field(default=None, init=False, repr=False)
    _event_processor: EventProcessor | None = field(default=None, init=False, repr=False)

    def evaluate(self, root: GraphBuilder) -> Any:
        """
        Builds and evaluates a graph using synchronous execution.

        Args:
            root: Root `GraphBuilder` to evaluate, typically a `TaskFuture`.

        Returns:
            The result of evaluating the root node.
        """
        nodes = build_graph(root)
        return self._run_sequential(nodes, root.id)

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

    def _validate_async_compatibility(self, nodes: dict[UUID, BaseGraphNode]) -> None:
        """
        Validate that nodes are compatible with async execution.

        Raises ValueError if any node uses sequential backend with a synchronous task function.
        """
        from daglite.graph.nodes import MapTaskNode
        from daglite.graph.nodes import TaskNode

        for node in nodes.values():
            if node.backend_name == "sequential":
                if isinstance(node, (TaskNode, MapTaskNode)):  # pragma: no branch
                    if not inspect.iscoroutinefunction(node.func):  # pragma: no branch
                        raise ValueError(
                            f"Sequential backend cannot execute synchronous task '{node.name}' "
                            "with evaluate_async(). Use threading/processes backend for parallel "
                            "execution, or evaluate() for sync tasks."
                        )

    def _run_sequential(self, nodes: dict[UUID, BaseGraphNode], root_id: UUID) -> Any:
        """Sequential blocking execution."""
        from concurrent.futures import wait

        graph_id = uuid4()
        plugin_manager, event_processor = self._setup_plugin_system()
        backend_manager = BackendManager(plugin_manager, event_processor)

        plugin_manager.hook.before_graph_execute(
            graph_id=graph_id, root_id=root_id, node_count=len(nodes), is_async=False
        )

        start_time = time.perf_counter()
        try:
            backend_manager.start()
            event_processor.start()
            state = _ExecutionState.from_nodes(nodes)
            ready = state.get_ready()

            while ready:
                # Submit all ready siblings (non-blocking)
                wrappers_map: dict[UUID, _NodeFutureWrapper | _MapFutureWrapper] = {
                    nid: self._submit_node_sync(state.nodes[nid], state, backend_manager)
                    for nid in ready
                }

                # Wait for all ready siblings to complete (blocking)
                all_futures = []
                for wrapper in wrappers_map.values():
                    if isinstance(wrapper, _MapFutureWrapper):
                        all_futures.extend(wrapper.futures)
                    else:
                        all_futures.append(wrapper.future)
                wait(all_futures)

                # Collect results and mark complete
                ready = []
                for nid, wrapper in wrappers_map.items():
                    result = self._collect_result_sync(wrapper)
                    ready.extend(state.mark_complete(nid, result))

            state.check_complete()
            result = state.completed_nodes[root_id]
            duration = time.perf_counter() - start_time

            event_processor.flush()  # Drain event queue before after_graph_execute
            plugin_manager.hook.after_graph_execute(
                graph_id=graph_id, root_id=root_id, result=result, duration=duration, is_async=False
            )

            return result
        except Exception as e:
            duration = time.perf_counter() - start_time
            plugin_manager.hook.on_graph_error(
                graph_id=graph_id, root_id=root_id, error=e, duration=duration, is_async=False
            )
            raise
        finally:
            event_processor.stop()
            backend_manager.stop()

    async def _run_async(self, nodes: dict[UUID, BaseGraphNode], root_id: UUID) -> Any:
        """Async execution with sibling parallelism."""
        # Validate that sequential backend is not used with sync tasks
        self._validate_async_compatibility(nodes)

        graph_id = uuid4()
        plugin_manager, event_processor = self._setup_plugin_system()
        backend_manager = BackendManager(plugin_manager, event_processor)

        plugin_manager.hook.before_graph_execute(
            graph_id=graph_id, root_id=root_id, node_count=len(nodes), is_async=True
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
                            if not t.done():
                                t.cancel()
                        await asyncio.gather(*tasks.keys(), return_exceptions=True)
                        raise

            state.check_complete()
            result = state.completed_nodes[root_id]
            duration = time.perf_counter() - start_time

            event_processor.flush()  # Drain event queue before after_graph_execute
            plugin_manager.hook.after_graph_execute(
                graph_id=graph_id, root_id=root_id, result=result, duration=duration, is_async=True
            )

            return result
        except Exception as e:
            duration = time.perf_counter() - start_time
            plugin_manager.hook.on_graph_error(
                graph_id=graph_id, root_id=root_id, error=e, duration=duration, is_async=True
            )
            raise
        finally:
            event_processor.stop()
            backend_manager.stop()

    def _submit_node_sync(
        self, node: BaseGraphNode, state: _ExecutionState, backend_manager: BackendManager
    ) -> _NodeFutureWrapper | _MapFutureWrapper:
        """
        Submit a node for execution without blocking.

        Returns:
            Wrapper containing future(s) and context for later collection.
        """
        from daglite.graph.nodes import MapTaskNode

        backend = backend_manager.get(node.backend_name)
        resolved_inputs = node.resolve_inputs(state.completed_nodes)

        if isinstance(node, MapTaskNode):
            # For mapped nodes, submit each iteration separately
            calls = node.build_iteration_calls(resolved_inputs)
            start_time = time.perf_counter()

            # Hook fires before submission
            backend.plugin_manager.hook.before_mapped_node_execute(
                metadata=node.to_metadata(), inputs_list=calls
            )

            # Submit all iterations (non-blocking)
            futures = []
            for idx, call in enumerate(calls):
                kwargs = {"iteration_index": idx}
                future = backend.submit(node.run, call, **kwargs)
                futures.append(future)

            return _MapFutureWrapper(
                futures=futures,
                node=node,
                calls=calls,
                start_time=start_time,
                backend=backend,
            )
        else:
            future = backend.submit(node.run, resolved_inputs)
            return _NodeFutureWrapper(future=future, node=node)

    def _collect_result_sync(self, wrapper: _NodeFutureWrapper | _MapFutureWrapper) -> Any:
        """
        Wait for node completion, fire after hooks, and materialize result.

        Returns:
            The node's execution result (single value or list).
        """
        if isinstance(wrapper, _MapFutureWrapper):
            # Wait for all iterations (blocking here)
            results = [future.result() for future in wrapper.futures]
            duration = time.perf_counter() - wrapper.start_time

            # Hook fires after all iterations complete
            wrapper.backend.plugin_manager.hook.after_mapped_node_execute(
                metadata=wrapper.node.to_metadata(),
                inputs_list=wrapper.calls,
                results=results,
                duration=duration,
            )

            return _materialize_sync(results)
        else:
            result = wrapper.future.result()
            return _materialize_sync(result)

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

        # Determine how to submit to backend based on node type
        if isinstance(node, MapTaskNode):
            # For mapped nodes, submit each iteration separately
            futures = []
            calls = node.build_iteration_calls(resolved_inputs)

            start_time = time.perf_counter()
            backend.plugin_manager.hook.before_mapped_node_execute(
                metadata=node.to_metadata(), inputs_list=calls
            )

            for idx, call in enumerate(calls):
                kwargs = {"iteration_index": idx}
                future = wrap_future(backend.submit(node.run_async, call, **kwargs))
                futures.append(future)

            result = await asyncio.gather(*futures)
            duration = time.perf_counter() - start_time

            backend.plugin_manager.hook.after_mapped_node_execute(
                metadata=node.to_metadata(), inputs_list=calls, results=result, duration=duration
            )
        else:
            future = wrap_future(backend.submit(node.run_async, resolved_inputs))
            result = await future

        result = await _materialize_async(result)

        return result


# region Wrappers


@dataclass
class _NodeFutureWrapper:
    """Wrapper for a single node's backend future."""

    future: Any  # concurrent.futures.Future
    node: BaseGraphNode


@dataclass
class _MapFutureWrapper:
    """Wrapper for a mapped node's multiple backend futures."""

    futures: list[Any]  # list[concurrent.futures.Future]
    node: BaseGraphNode  # MapTaskNode
    calls: list[dict[str, Any]]
    start_time: float
    backend: Any  # Backend instance


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
