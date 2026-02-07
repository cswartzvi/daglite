from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, Awaitable, Callable, TypeVar

from pluggy import PluginManager
from typing_extensions import final

if TYPE_CHECKING:
    from daglite.plugins.events import EventProcessor
    from daglite.plugins.reporters import EventReporter
else:
    EventProcessor = object
    EventReporter = object

T = TypeVar("T")


class Backend(abc.ABC):
    """Abstract base class for task execution backends."""

    plugin_manager: PluginManager
    event_processor: EventProcessor
    reporter: EventReporter

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
        self.plugin_manager = plugin_manager
        self.event_processor = event_processor
        self.reporter = self._get_reporter()
        self._start()

    def _start(self) -> None:
        """
        Set up any per-execution-context resources.

        Subclasses may override this to set up context-specific resources.
        """
        pass  # pragma: no cover

    @final
    def stop(self) -> None:
        """
        Clean up any global backend resources.

        Subclasses should NOT override this method. Instead, override `_stop()`.
        """
        self._stop()
        delattr(self, "plugin_manager")
        delattr(self, "event_processor")
        if hasattr(self, "reporter"):  # pragma: no branch
            del self.reporter

    def _stop(self) -> None:
        """
        Clean up any per-execution-context resources.

        Subclasses may override this to clean up context-specific resources.
        """
        pass  # pragma: no cover

    @abc.abstractmethod
    def submit(
        self,
        func: Callable[[dict[str, Any]], Any],
        inputs: dict[str, Any],
        timeout: float | None = None,
        **kwargs: Any,
    ) -> Awaitable[Any]:
        """
        Submit a callable for execution in the backend.

        Args:
            func: Callable to execute
            inputs: Pre-resolved parameter inputs
            timeout: Maximum execution time in seconds. If None, no timeout is enforced.
            **kwargs: Additional backend-specific execution parameters

        Returns:
            An awaitable that resolves to the result of the callable.
        """
        raise NotImplementedError()
