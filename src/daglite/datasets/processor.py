"""
Background processor for dataset save requests from workers.

The ``DatasetProcessor`` extends ``BackgroundQueueProcessor`` to drain save
requests from IPC sources (e.g. ``multiprocessing.Queue`` instances used by
``ProcessDatasetReporter``) and writes them using the ``DatasetStore`` carried
within each request.

For direct reporters the save happens inline, so the processor only needs to
handle queue-based sources from process/remote backends.
"""

from __future__ import annotations

import logging
from typing import Any

from daglite._processor import BackgroundQueueProcessor
from daglite.datasets.events import DatasetSaveRequest

logger = logging.getLogger(__name__)


class DatasetProcessor(BackgroundQueueProcessor):
    """
    Background processor for draining worker → coordinator dataset save requests.

    Unlike the initial implementation, the processor does **not** own a single
    ``DatasetStore``.  Each save request arriving through the queue carries its
    own store reference, so different output configs can target different stores
    without ambiguity.

    Args:
        hook: Optional pluggy `HookRelay` for firing `before_dataset_save`
            and ``after_dataset_save`` hooks.  Passed from the engine since the
            processor's daemon thread does not inherit the worker's context vars.
    """

    def __init__(self, hook: Any | None = None) -> None:
        super().__init__(name="DatasetProcessor")
        self._hook = hook

    # -- abstract implementation -------------------------------------------

    def _handle_item(self, item: Any) -> None:
        """Process a single save request received from a worker."""
        self._handle_request(item)

    # -- internal helpers --------------------------------------------------

    def _handle_request(self, request: DatasetSaveRequest) -> None:
        """
        Process a single save request received from a worker.

        Args:
            request: `DatasetSaveRequest` containing the key, value, store,
                optional format, and options.
        """
        try:
            hook_kwargs = dict(
                key=request.key,
                value=request.value,
                format=request.format,
                options=request.options,
            )
            if self._hook:
                self._hook.before_dataset_save(**hook_kwargs)
            request.store.save(
                request.key, request.value, format=request.format, options=request.options
            )
            if self._hook:
                self._hook.after_dataset_save(**hook_kwargs)
        except Exception as e:
            logger.exception(f"Error processing dataset save request: {e}")
