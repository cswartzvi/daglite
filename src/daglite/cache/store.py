"""Hash-based cache store using Driver for byte storage and cloudpickle for serialization."""

from __future__ import annotations

import json
import pickle
import time
from typing import Any

import cloudpickle

from daglite.cache.core import CACHE_MISS
from daglite.cache.core import CacheMiss
from daglite.drivers.base import Driver


class CacheStore:
    """
    Hash-based cache store using Driver for byte storage and cloudpickle for serialization.

    Mimics the DatasetStore pattern (uses a Driver under the hood) but handles cache-specific
    concerns: git-style sharded keys, TTL metadata, and cloudpickle serialization.

    Uses a two-level directory structure (XX/YYYYYY...) to avoid filesystem performance issues with
    many files in one directory.

    Layout::

        cache_dir/
            ab/
                cdef1234...           # Data file (cloudpickle serialized)
                cdef1234....meta.json # Metadata (timestamp, ttl)

    Examples:
        >>> import tempfile
        >>> store = CacheStore(tempfile.mkdtemp())
        >>> store.put("abc123", 42, ttl=3600)
        >>> store.get("abc123")
        42
        >>> store.invalidate("abc123")
        >>> from daglite.cache.core import CACHE_MISS
        >>> store.get("abc123") is CACHE_MISS
        True
    """

    def __init__(self, driver: Driver | str) -> None:
        """
        Initialize cache store.

        Args:
            driver: A Driver instance or string path (creates FileDriver).

        Examples:
            >>> import tempfile
            >>> store = CacheStore(tempfile.mkdtemp())
        """
        if isinstance(driver, str):
            from daglite.drivers import FileDriver

            self._driver = FileDriver(driver)
        else:
            self._driver = driver

    @property
    def is_local(self) -> bool:
        """Whether the underlying driver accesses local storage."""
        return getattr(self._driver, "is_local", True)

    def get(self, hash_key: str) -> Any | CacheMiss:
        """
        Retrieve cached value by hash key.

        Returns ``CACHE_MISS`` on a cache miss or TTL expiry. Expired entries are
        automatically cleaned up.

        Args:
            hash_key: SHA256 hash digest string.

        Returns:
            Cached value if found and not expired, ``CACHE_MISS`` otherwise.
        """
        data_key = self._hash_to_key(hash_key)
        meta_key = f"{data_key}.meta.json"

        if not self._driver.exists(data_key):
            return CACHE_MISS

        # Check TTL from metadata sidecar
        if self._driver.exists(meta_key):
            try:
                meta_bytes = self._driver.load(meta_key)
                metadata = json.loads(meta_bytes.decode("utf-8"))

                ttl = metadata.get("ttl")
                if ttl is not None:
                    timestamp = metadata["timestamp"]
                    if time.time() - timestamp > ttl:
                        # Expired — clean up and return miss
                        self.invalidate(hash_key)
                        return CACHE_MISS
            except (json.JSONDecodeError, KeyError, OSError):
                # Corrupted metadata — treat as miss
                return CACHE_MISS

        # Load and deserialize cached data
        try:
            data = self._driver.load(data_key)
            return cloudpickle.loads(data)
        except (OSError, pickle.UnpicklingError):
            return CACHE_MISS

    def put(self, hash_key: str, value: Any, ttl: int | None = None) -> None:
        """
        Store value in cache with optional TTL expiration.

        Args:
            hash_key: SHA256 hash digest string.
            value: The value to cache (must be cloudpickle-serializable).
            ttl: Time-to-live in seconds. None means no expiration.
        """
        data_key = self._hash_to_key(hash_key)
        meta_key = f"{data_key}.meta.json"

        # Serialize and store data
        data = cloudpickle.dumps(value)
        self._driver.save(data_key, data)

        # Write metadata sidecar
        metadata = {"timestamp": time.time(), "ttl": ttl}
        meta_bytes = json.dumps(metadata).encode("utf-8")
        self._driver.save(meta_key, meta_bytes)

    def invalidate(self, hash_key: str) -> None:
        """
        Remove a cached entry (data + metadata).

        Safe to call on non-existent entries.

        Args:
            hash_key: SHA256 hash digest string.
        """
        data_key = self._hash_to_key(hash_key)
        meta_key = f"{data_key}.meta.json"

        self._driver.delete(data_key)
        self._driver.delete(meta_key)

    def clear(self) -> None:
        """Remove all cached entries."""
        for key in self._driver.list_keys():
            self._driver.delete(key)

    def _hash_to_key(self, hash_key: str) -> str:
        """
        Convert hash digest to git-style sharded storage key.

        Uses first 2 characters as shard directory prefix.

        Args:
            hash_key: SHA256 hash digest string (e.g., "abcdef1234...").

        Returns:
            Sharded key path (e.g., "ab/cdef1234...").

        Raises:
            ValueError: If ``hash_key`` is too short to shard safely or
                contains non-hex characters.
        """
        # Validate that hash_key is long enough so that the suffix is non-empty.
        if len(hash_key) < 3:
            raise ValueError(
                f"hash_key must be at least 3 characters long to form a sharded "
                f"path, got {len(hash_key)!r}: {hash_key!r}"
            )

        # Validate that the key looks like a hex digest.
        lowered = hash_key.lower()
        if any(c not in "0123456789abcdef" for c in lowered):
            raise ValueError(f"hash_key must be a hexadecimal digest string, got: {hash_key!r}")

        prefix = hash_key[:2]
        suffix = hash_key[2:]
        return f"{prefix}/{suffix}"

    def __getstate__(self) -> dict[str, Any]:
        """Serialize for pickling (needed for process backends)."""
        return {"driver": self._driver}

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Reconstruct from pickled state."""
        self._driver = state["driver"]
