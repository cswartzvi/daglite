"""Typed request container for the dataset save system."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from daglite.datasets.store import DatasetStore


@dataclass(frozen=True)
class DatasetSaveRequest:
    """
    Typed save request for worker → coordinator dataset communication.


    Attributes:
        key: Storage key/path for the output.
        value: The Python object to persist.
        store: The ``DatasetStore`` to write to.
        format: Optional serialization format hint (e.g. ``"pickle"``).
        options: Additional options passed to the Dataset's save method.

    Examples:
        >>> from daglite.datasets.store import DatasetStore
        >>> request = DatasetSaveRequest(
        ...     key="output.pkl",
        ...     value={"data": 1},
        ...     store=DatasetStore("/tmp/store"),
        ...     format="pickle",
        ... )
        >>> request.key
        'output.pkl'
    """

    key: str
    value: Any
    store: DatasetStore
    format: str | None = None
    options: dict[str, Any] | None = None
