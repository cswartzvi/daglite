from __future__ import annotations

import abc
from asyncio import wrap_future
from asyncio.futures import Future as AsyncioFuture
from concurrent.futures import Future as ConcurrentFuture
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from pluggy import PluginManager
from typing_extensions import final

if TYPE_CHECKING:
    from daglite.graph.base import BaseGraphNode
    from daglite.plugins.events import EventProcessor
    from daglite.plugins.reporters import EventReporter

T = TypeVar("T")


class Backend(abc.ABC):
    """Abstract base class for task execution backends."""

    _plugin_manager: PluginManager
    _event_processor: EventProcessor
    _reporter: EventReporter

    @abc.abstractmethod
    def _get_reporter(self) -> EventReporter:
        """Gets the event reporter for this backend."""
        raise NotImplementedError()

    @final
    def start(self, plugin_manager: PluginManager, event_processor: EventProcessor) -> None:
        """
        Start any global backend resources.

        Subclasses should NOT override this method. Instead, override `_start()`.

        Args:
            plugin_manager: Plugin manager for hook execution
            event_processor: Event processor for event handling
        """
        self._plugin_manager = plugin_manager
        self._event_processor = event_processor
        self._reporter = self._get_reporter()
        self._start()

    def _start(self) -> None:
        """
        Set up any per-execution-context resources.

        Subclasses may override this to set up context-specific resources.
        """
        pass

    @final
    def stop(self) -> None:
        """
        Clean up any global backend resources.

        Subclasses should NOT override this method. Instead, override `_end()`.
        """
        self._stop()
        delattr(self, "_plugin_manager")
        delattr(self, "_event_processor")
        if hasattr(self, "_reporter"):
            del self._reporter

    def _stop(self) -> None:
        """
        Clean up any per-execution-context resources.

        Subclasses may override this to clean up context-specific resources.
        """
        pass

    @final
    def submit_node(
        self, node: BaseGraphNode, resolved_inputs: dict[str, Any]
    ) -> ConcurrentFuture[Any] | list[ConcurrentFuture[Any]]:
        """
        Execute a graph node with resolved inputs.

        Args:
            node: Graph node to execute
            resolved_inputs: Pre-resolved parameter inputs

        Returns:
            `concurrent.futures.Future` representing the execution result, or a list of futures
            for mapped nodes.
        """
        from daglite.graph.base import BaseMapGraphNode

        if isinstance(node, BaseMapGraphNode):
            calls = node.build_iteration_calls(resolved_inputs)
            return [self._submit_impl(node.run, call) for call in calls]
        else:
            return self._submit_impl(node.run, resolved_inputs)

    @final
    async def submit_node_async(
        self, node: BaseGraphNode, resolved_inputs: dict[str, Any]
    ) -> AsyncioFuture[Any] | list[AsyncioFuture[Any]]:
        """
        Execute a graph node asynchronously with resolved inputs.

        Similar to execute_node but for async execution contexts. Returns the actual result value
        after awaiting completion.

        Args:
            node: Graph node to execute
            resolved_inputs: Pre-resolved parameter inputs

        Returns:
            `asyncio.Future` representing the execution result, or a list of futures for mapped
            nodes.
        """
        from daglite.graph.base import BaseMapGraphNode

        if isinstance(node, BaseMapGraphNode):
            calls = node.build_iteration_calls(resolved_inputs)
            futures = [wrap_future(self._submit_impl(node.run_async, call)) for call in calls]
            return futures
        else:
            future = wrap_future(self._submit_impl(node.run_async, resolved_inputs))
            return future

    @abc.abstractmethod
    def _submit_impl(
        self, func: Callable[[dict[str, Any]], Any], inputs: dict[str, Any]
    ) -> ConcurrentFuture[Any]:
        """
        Submit a callable for execution in the backend.

        Args:
            func: Callable to execute
            inputs: Pre-resolved parameter inputs

        Returns:
            Future representing the execution
        """
        raise NotImplementedError()
