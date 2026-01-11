"""Storage protocols and implementations for task outputs."""

from __future__ import annotations

from typing import Any, Protocol, TypeVar

T = TypeVar("T")


class OutputStore(Protocol):
    """Protocol for storing task outputs."""

    def save(self, key: str, value: Any) -> None:
        """
        Save an output artifact.

        Args:
            key: Unique identifier for this output
            value: The value to store
        """
        ...

    def load(self, key: str, return_type: type[T] | None = None) -> T:
        """
        Load an output artifact.

        Args:
            key: The key to load
            return_type: Optional type hint for deserialization

        Returns:
            The loaded value

        Raises:
            KeyError: If key doesn't exist
        """
        ...

    def exists(self, key: str) -> bool:
        """Check if an output exists."""
        ...

    def delete(self, key: str) -> None:
        """Delete an output."""
        ...

    def list_keys(self) -> list[str]:
        """List all stored output keys."""
        ...
