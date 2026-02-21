"""Evaluation engine for Daglite task graphs."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import UUID
from uuid import uuid4

from pluggy import HookRelay
from pluggy import PluginManager

if TYPE_CHECKING:
    from daglite.datasets.processor import DatasetProcessor
    from daglite.graph.builder import NodeBuilder
    from daglite.plugins.processor import EventProcessor
else:
    NodeBuilder = Any
    DatasetProcessor = Any
    EventProcessor = Any


from daglite.backends import BackendManager
from daglite.exceptions import ExecutionError
from daglite.graph.builder import build_graph
from daglite.graph.nodes.base import BaseGraphNode

# region Evaluation


def evaluate(future: Any, *, plugins: list[Any] | None = None) -> Any:
    """
    Evaluate the results of a task future synchronously.

    Important: This function creates an event loop internally via `asyncio.run()`, so it **cannot**
    be called from within an async context. In those cases, use `evaluate_async()` instead.

    Args:
        future: Task future that will be evaluated.
        plugins: Additional plugins to included with globally registered plugins.

    Returns:
        The result of evaluating the root task

    Raises:
        RuntimeError: If called from within an async context with a running event loop

    Examples:
        >>> from daglite import task
        >>> @task
        ... def my_task(x: int, y: int) -> int:
        ...     return x + y
        >>> future = my_task(x=1, y=2)

        Standard evaluation
        >>> future.run()
        3

        Evaluation with plugins
        >>> from daglite.plugins.builtin.logging import CentralizedLoggingPlugin
        >>> future.run(plugins=[CentralizedLoggingPlugin()])
        3

        Sibling parallelism with threading backend
        >>> @task(backend_name="threading")
        ... def concurrent_task(x: int) -> int:
        ...     return x * 2
        >>> t1, t2 = concurrent_task(x=1), concurrent_task(x=2)
        >>> @task
        ... def combine(a: int, b: int) -> int:
        ...     return a + b
        >>> combine(a=t1, b=t2).run()  # t1 and t2 run in parallel
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


async def evaluate_async(future: Any, *, plugins: list[Any] | None = None) -> Any:
    """
    Evaluate the results of a task future via asynchronous execution.

    The task future can contain any combination of sync and async task futures, and the engine will
    execute them in an async-first manner. Sibling tasks can be executed concurrently if they use
    an async coroutine and/or an async-capable backend (e.g., threading or process).

    Args:
        future: Task future that will be evaluated.
        plugins: Additional plugins to included with globally registered plugins.

    Returns:
        The result of evaluating the root task

    Examples:
        >>> import asyncio
        >>> from daglite import task
        >>> @task
        ... async def my_task(x: int, y: int) -> int:
        ...     return x + y
        >>> future = my_task(x=1, y=2)

        Standard evaluation
        >>> asyncio.run(future.run_async())
        3

        With execution-specific plugins
        >>> import asyncio
        >>> from daglite.plugins.builtin.logging import CentralizedLoggingPlugin
        >>> asyncio.run(future.run_async(plugins=[CentralizedLoggingPlugin()]))
        3
    """
    from daglite.backends.context import get_current_task

    if get_current_task():
        raise RuntimeError("Cannot call evaluate()/evaluate_async() from within another task.")

    graph_id = uuid4()
    state = _setup_graph_execution_state(future)

    plugin_manager, event_processor = _setup_plugin_system(plugins=plugins or [])
    dataset_processor = _setup_dataset_processor(hook=plugin_manager.hook)
    backend_manager = BackendManager(plugin_manager, event_processor, dataset_processor)

    hook_ids = {"graph_id": graph_id, "root_id": future.id}
    plugin_manager.hook.before_graph_execute(**hook_ids, node_count=len(state.nodes))

    start_time = time.perf_counter()
    try:
        backend_manager.start()
        event_processor.start()
        dataset_processor.start()

        nodes_to_process = state.get_source_nodes()

        while nodes_to_process:
            # Submit all ready siblings
            tasks: dict[asyncio.Task[Any], UUID] = {}
            for nid in nodes_to_process:
                node = state.nodes[nid]
                backend = backend_manager.get(node.backend_name)
                coro = node.execute(backend, state.completed_nodes, plugin_manager.hook)
                task = asyncio.create_task(coro)
                tasks[task] = nid

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
                for t in pending:  # pragma: no cover - difficult to simulate
                    t.cancel()
                await asyncio.gather(*tasks.keys(), return_exceptions=True)
                raise

        # Finalize state and results
        state.check_complete()
        result = state.get_result(future.id)
        duration = time.perf_counter() - start_time

        # Flush any remaining events/dataset processors
        event_processor.flush()
        dataset_processor.flush()

        plugin_manager.hook.after_graph_execute(**hook_ids, result=result, duration=duration)

    except Exception as e:
        duration = time.perf_counter() - start_time
        plugin_manager.hook.on_graph_error(**hook_ids, error=e, duration=duration)
        raise

    finally:
        event_processor.stop()
        dataset_processor.stop()
        backend_manager.stop()

    return result


def _setup_graph_execution_state(future: NodeBuilder) -> _ExecutionState:
    """Builds the graph execution state, including graph optimization if enabled."""
    from daglite.graph.optimizer import optimize_graph
    from daglite.settings import get_global_settings

    nodes = build_graph(future)
    settings = get_global_settings()

    id_mapping: dict[UUID, UUID] = {}
    if settings.enable_graph_optimization:
        nodes, id_mapping = optimize_graph(nodes)

    state = _ExecutionState.from_nodes(nodes, id_mapping)
    return state


def _setup_plugin_system(plugins: list[Any]) -> tuple[PluginManager, EventProcessor]:
    """Sets up plugin system (manager, processor, registry) for this engine."""
    from daglite.plugins.manager import build_plugin_manager
    from daglite.plugins.processor import EventProcessor
    from daglite.plugins.registry import EventRegistry

    registry = EventRegistry()
    plugin_manager = build_plugin_manager(plugins or [], registry)
    event_processor = EventProcessor(registry)
    return plugin_manager, event_processor


def _setup_dataset_processor(hook: HookRelay | None = None) -> DatasetProcessor:
    """Creates a `DatasetProcessor` for draining queue-based dataset saves."""
    from daglite.datasets.processor import DatasetProcessor

    # Always created regardless of whether any nodes have output configs.
    # An idle processor is just a sleeping daemon thread with zero overhead,
    # matching how the `EventProcessor` is unconditionally started.
    return DatasetProcessor(hook=hook)


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

    id_mapping: dict[UUID, UUID]
    """Maps original folded node IDs to their composite node IDs."""

    @classmethod
    def from_nodes(
        cls, nodes: dict[UUID, BaseGraphNode], id_mapping: dict[UUID, UUID] | None = None
    ) -> _ExecutionState:
        """
        Build execution state from a graph node dictionary.

        Computes the dependency graph (indegree and successors) needed for
        topological execution.

        Args:
            nodes: Mapping from node IDs to GraphNode instances.
            id_mapping: Optional mapping from original folded node IDs to
                composite node IDs (from the graph optimizer).

        Returns:
            Initialized ExecutionState instance.
        """
        from collections import defaultdict

        indegree: dict[UUID, int] = {nid: 0 for nid in nodes}
        successors: dict[UUID, set[UUID]] = defaultdict(set)

        for nid, node in nodes.items():
            for dep in node.get_dependencies():
                indegree[nid] += 1
                successors[dep].add(nid)

        return cls(
            nodes=nodes,
            indegree=indegree,
            successors=dict(successors),
            completed_nodes={},
            id_mapping=id_mapping or {},
        )

    def get_result(self, node_id: UUID) -> Any:
        """Resolves a node result, accounting for any ID remapping from graph optimization."""
        resolved = self.id_mapping.get(node_id, node_id)
        return self.completed_nodes[resolved]

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
