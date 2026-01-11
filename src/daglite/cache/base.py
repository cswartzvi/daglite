"""Cache storage protocols and implementations."""

from __future__ import annotations

from typing import Any, Protocol, TypeVar

T = TypeVar("T")


class CacheStore(Protocol):
    """
    Protocol for hash-based task result caching.

    Used by @task(cache=True) to store results keyed by content hash
    (function source + parameter values).
    """

    def get(self, hash_key: str, return_type: type[T]) -> T | None:
        """
        Retrieve cached value by hash key.

        Args:
            hash_key: SHA256 hash of (function source, parameters)
            return_type: Type hint for deserialization

        Returns:
            Cached value if found, None otherwise
        """
        ...

    def put(self, hash_key: str, value: Any, ttl: int | None = None) -> None:
        """
        Store value with hash key.

        Args:
            hash_key: SHA256 hash of (function source, parameters)
            value: The value to cache
            ttl: Time-to-live in seconds (None = no expiration)
        """
        ...

    def invalidate(self, hash_key: str) -> None:
        """Remove cached entry."""
        ...

    def clear(self) -> None:
        """Clear entire cache."""
        ...
