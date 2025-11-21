# daglite/engine.py
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from dataclasses import field
from typing import Any, ParamSpec, TypeVar, overload
from uuid import UUID

from daglite.backends.base import Backend
from daglite.backends.local import ThreadBackend
from daglite.graph.base import GraphBuilder
from daglite.graph.base import GraphNode
from daglite.graph.builder import build_graph
from daglite.settings import DagliteSettings
from daglite.tasks import MapTaskFuture
from daglite.tasks import TaskFuture

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")


# region API


@overload
def evaluate(
    expr: TaskFuture[T], *, default_backend: str | Backend = "sequential", use_async: bool = False
) -> T: ...


@overload
def evaluate(
    expr: MapTaskFuture[T],
    *,
    default_backend: str | Backend = "sequential",
    use_async: bool = False,
) -> list[T]: ...


def evaluate(
    expr: GraphBuilder,
    default_backend: str | Backend = "sequential",
    use_async: bool = False,
) -> Any:
    """
    Evaluate a task graph.

    Args:
        expr (daglite.graph.base.GraphBuilder):
            The task graph to evaluate.
        default_backend (str | daglite.backends.Backend, optional):
            Default backend for task execution. If a node does not have a specific backend
            assigned, this backend will be used. Defaults to "sequential".
        use_async (bool, optional):
            If True, use async execution for sibling parallelism. This enables concurrent execution
            of independent nodes using asyncio. Best for I/O-bound workloads (network, disk
            operations).

    Note:
        Global settings (thread pool size, etc.) are configured via set_global_settings().
        These must be set before the first task execution.

    Returns:
        The result of evaluating the root task

    Examples:
        >>> # Sequential execution (default)
        >>> result = evaluate(my_task)

        >>> # Async execution with sibling parallelism
        >>> result = evaluate(my_task, use_async=True)

        >>> # With custom backend
        >>> result = evaluate(my_task, default_backend_name="threading", use_async=True)
    """
    engine = Engine(default_backend=default_backend, use_async=use_async)
    return engine.evaluate(expr)


# endregion


# region Engine


@dataclass
class Engine:
    """
    Engine to evaluate a GraphBuilder.

    The Engine compiles a GraphBuilder into a GraphNode dict, then executes
    it in topological order.

    Execution Modes:
        - use_async=False: Sequential execution (single-threaded)
        - use_async=True: Async execution with sibling parallelism

    Sibling Parallelism:
        When use_async=True, independent nodes at the same level of the DAG
        execute concurrently using asyncio. This is particularly beneficial for
        I/O-bound tasks (network requests, file operations).

        Sync task functions are automatically wrapped with asyncio.to_thread()
        to avoid blocking the event loop. ThreadBackend tasks run directly
        since they manage their own parallelism.

    Backend Resolution Priority:
        1. Node-specific backend from task/task-future operations (bind, extend, ...)
        2. Default task backend from `@task` decorator
        3. Engine's default_backend_name
    """

    default_backend: str | Backend
    """Default backend name or instance for nodes without a specific backend."""

    use_async: bool = False
    """If True, use async/await for sibling parallelism."""

    settings: DagliteSettings = field(default_factory=DagliteSettings)
    """Daglite configuration settings."""

    # cache: MutableMapping[UUID, Any] = field(default_factory=dict)
    # """Optional cache keyed by TaskFuture UUID (not used yet, but ready)."""

    _backend_cache: dict[str | Backend, Backend] = field(default_factory=dict, init=False)

    def evaluate(self, root: GraphBuilder) -> Any:
        """Compiles the lazy expression into a graph and execute it."""
        nodes = build_graph(root)

        if self.use_async:
            return asyncio.run(self._run_async(nodes, root.id))
        else:
            return self._run_sequential(nodes, root.id)

    def _resolve_node_backend(self, node: GraphNode) -> Backend:
        """Decide which Backend instance to use for this node's *internal* work."""
        from daglite.backends import find_backend

        backend_key = node.backend or self.default_backend
        if backend_key not in self._backend_cache:
            backend = find_backend(backend_key)
            self._backend_cache[backend_key] = backend
        return self._backend_cache[backend_key]

    def _run_sequential(self, nodes: dict[UUID, GraphNode], root_id: UUID) -> Any:
        """Sequential execution (original implementation)."""
        indegree: dict[UUID, int] = {nid: 0 for nid in nodes}
        successors: dict[UUID, set[UUID]] = {nid: set() for nid in nodes}

        for nid, node in nodes.items():
            for dep in node.deps():
                indegree[nid] += 1
                successors.setdefault(dep, set()).add(nid)

        values: dict[UUID, Any] = {}
        ready: list[UUID] = [nid for nid, d in indegree.items() if d == 0]

        # Fast path: single-node graph
        if len(nodes) == 1 and ready == [root_id]:
            node = nodes[root_id]
            backend = self._resolve_node_backend(node)
            result = node.run(backend, values)
            values[root_id] = result
            return result

        # Sequential execution
        while ready:
            nid = ready.pop()
            node = nodes[nid]
            backend = self._resolve_node_backend(node)
            result = node.run(backend, values)
            values[nid] = result

            for succ in successors.get(nid, ()):
                indegree[succ] -= 1
                if indegree[succ] == 0:
                    ready.append(succ)

        return values[root_id]

    async def _run_async(self, nodes: dict[UUID, GraphNode], root_id: UUID) -> Any:
        """
        Async execution with sibling parallelism.

        Runs independent sibling nodes concurrently using asyncio.
        Sync task functions are executed in threads via asyncio.to_thread().
        """
        indegree: dict[UUID, int] = {nid: 0 for nid in nodes}
        successors: dict[UUID, set[UUID]] = {nid: set() for nid in nodes}

        for nid, node in nodes.items():
            for dep in node.deps():
                indegree[nid] += 1
                successors.setdefault(dep, set()).add(nid)

        values: dict[UUID, Any] = {}
        ready: list[UUID] = [nid for nid, d in indegree.items() if d == 0]

        # Fast path: single-node graph
        if len(nodes) == 1 and ready == [root_id]:
            node = nodes[root_id]
            backend = self._resolve_node_backend(node)
            result = await self._run_node_async(root_id, node, backend, values)
            values[root_id] = result
            return result

        # Async execution: run siblings concurrently
        while ready:
            task_map: dict[asyncio.Task, UUID] = {}
            for nid in ready:
                node = nodes[nid]
                backend = self._resolve_node_backend(node)
                task = asyncio.create_task(self._run_node_async(nid, node, backend, values))
                task_map[task] = nid

            ready = []

            try:
                # Wait for all siblings to complete concurrently
                done, _ = await asyncio.wait(task_map.keys())

                for task in done:
                    nid = task_map[task]
                    try:
                        result = task.result()
                        values[nid] = result

                        for succ in successors.get(nid, ()):
                            indegree[succ] -= 1
                            if indegree[succ] == 0:
                                ready.append(succ)
                    except Exception:
                        # Cancel all remaining tasks on first failure
                        for t in task_map.keys():
                            if not t.done():
                                t.cancel()
                        raise

            except asyncio.CancelledError:
                # Propagate cancellation to all tasks
                for task in task_map.keys():
                    if not task.done():
                        task.cancel()
                raise

        return values[root_id]

    async def _run_node_async(
        self, nid: UUID, node: GraphNode, backend: Backend, values: dict[UUID, Any]
    ) -> Any:
        """
        Run a single node asynchronously.

        If the backend is ThreadBackend, runs directly since it handles
        parallelism internally. Otherwise, wraps in asyncio.to_thread()
        to avoid blocking the event loop.
        """
        # If backend is already threaded, we don't need to_thread()
        if isinstance(backend, ThreadBackend):
            # Run directly - backend handles parallelism internally
            return node.run(backend, values)
        else:
            # Wrap sync operation in thread for non-blocking execution
            return await asyncio.to_thread(node.run, backend, values)
