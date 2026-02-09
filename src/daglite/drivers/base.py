"""Abstract base class for byte-level storage backends."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod


class Driver(ABC):
    """
    Abstract base class for low-level byte storage driver.

    Drivers are responsible for persisting raw bytes to a storage location. They do NOT handle
    serialization or metadata.
    """

    @abstractmethod
    def save(self, key: str, data: bytes) -> str:
        """
        Store raw bytes.

        Args:
            key: Storage key/path (the exact filename to use).
            data: Raw bytes to store.

        Returns:
            The actual path/key where data was stored.
        """
        ...

    @abstractmethod
    def load(self, key: str) -> bytes:
        """
        Load raw bytes.

        Args:
            key: Storage key/path.

        Returns:
            The raw bytes.

        Raises:
            KeyError: If key not found.
        """
        ...

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if a key exists."""
        ...

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete stored data. Safe to call on non-existent keys."""
        ...

    @abstractmethod
    def list_keys(self) -> list[str]:
        """List all stored keys."""
        ...

    @property
    def is_local(self) -> bool:
        """
        Whether this driver accesses local storage.

        Returns ``True`` for drivers that persist to the local filesystem (or
        other resources only reachable from the coordinator), ``False`` for
        remote/network storage (S3, GCS, …) accessible from any worker.

        Subclasses for remote drivers should override this to return ``False``.
        """
        return True  # pragma: no cover

    def get_format_hint(self, key: str) -> str | None:  # pragma: no cover
        """
        Return a format hint for the given key, or None if unknown.

        The format hint helps stakeholders determine how to deserialize data.

        Args:
            key: Storage key/path.

        Returns:
            Format hint string (e.g., "csv", "json") or None.
        """
        return None
