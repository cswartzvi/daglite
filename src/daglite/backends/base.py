from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, Awaitable

from pluggy import PluginManager
from typing_extensions import final

from daglite._typing import Submission

if TYPE_CHECKING:
    from daglite.datasets.processor import DatasetProcessor
    from daglite.datasets.reporters import DatasetReporter
    from daglite.plugins.processor import EventProcessor
    from daglite.plugins.reporters import EventReporter
else:
    DatasetProcessor = object
    DatasetReporter = object
    EventProcessor = object
    EventReporter = object


class Backend(abc.ABC):
    """Abstract base class for task execution backends."""

    plugin_manager: PluginManager
    event_processor: EventProcessor
    event_reporter: EventReporter
    dataset_processor: DatasetProcessor
    dataset_reporter: DatasetReporter | None
    _started: bool = False

    @abc.abstractmethod
    def _get_event_reporter(self) -> EventReporter:
        """Gets the event reporter for this backend."""
        raise NotImplementedError()

    @abc.abstractmethod
    def _get_dataset_reporter(self) -> DatasetReporter | None:
        """Gets the dataset reporter for this backend."""
        return None

    @final
    def start(
        self,
        plugin_manager: PluginManager,
        event_processor: EventProcessor,
        dataset_processor: DatasetProcessor,
    ) -> None:
        """
        Start any global backend resources.

        Subclasses should NOT override this method. Instead, override ``_start()``.

        Args:
            plugin_manager: Plugin manager for hook execution
            event_processor: Event processor for event handling
            dataset_processor: Dataset processor for persisting outputs
        """
        if self._started:  # pragma: no cover
            raise RuntimeError("Backend is already started.")

        self.plugin_manager = plugin_manager
        self.event_processor = event_processor
        self.dataset_processor = dataset_processor
        self.event_reporter = self._get_event_reporter()
        self.dataset_reporter = self._get_dataset_reporter()
        self._start()
        self._started = True

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

        Subclasses should NOT override this method. Instead, override ``_stop()``.
        """
        if not self._started:  # pragma: no cover
            return

        self._stop()
        self._started = False

    def _stop(self) -> None:
        """
        Clean up any per-execution-context resources.

        Subclasses may override this to clean up context-specific resources.
        """
        pass  # pragma: no cover

    @abc.abstractmethod
    def submit(self, func: Submission, timeout: float | None = None) -> Awaitable[Any]:
        """
        Submit a callable for execution in the backend.

        Args:
            func: A parameterless async coroutine to execute.
            timeout: Maximum execution time in seconds. If None, no timeout is enforced.

        Returns:
            An awaitable that resolves to the result of the callable.
        """
        raise NotImplementedError()
