from __future__ import annotations

import abc
from concurrent.futures import Future as ConcurrentFuture
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from pluggy import PluginManager
from typing_extensions import final

if TYPE_CHECKING:
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

        Subclasses should NOT override this method. Instead, override `_stop()`.
        """
        self._stop()
        delattr(self, "_plugin_manager")
        delattr(self, "_event_processor")
        if hasattr(self, "_reporter"):  # pragma: no branch
            del self._reporter

    def _stop(self) -> None:
        """
        Clean up any per-execution-context resources.

        Subclasses may override this to clean up context-specific resources.
        """
        pass

    @abc.abstractmethod
    def submit(
        self, func: Callable[[dict[str, Any]], Any], inputs: dict[str, Any], **kwargs: Any
    ) -> ConcurrentFuture[Any]:
        """
        Submit a callable for execution in the backend.

        Args:
            func: Callable to execute
            inputs: Pre-resolved parameter inputs
            **kwargs: Additional backend-specific execution parameters

        Returns:
            Future representing the execution
        """
        raise NotImplementedError()
