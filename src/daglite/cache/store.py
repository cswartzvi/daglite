"""Cache storage protocols and implementations."""

from __future__ import annotations

import json
import pickle
import time
from typing import TYPE_CHECKING, Any, TypeVar

from daglite.cache.base import CacheStore

if TYPE_CHECKING:
    from fsspec import AbstractFileSystem
else:
    AbstractFileSystem = object

T = TypeVar("T")


class FileCacheStore(CacheStore):
    """
    File-based cache store with fsspec support and git-style sharded layout.

    Supports both local and remote filesystems (s3://, gcs://, etc.) via fsspec.

    Uses a two-level directory structure (XX/YYYYYY...) to avoid
    filesystem performance issues with many files in one directory.

    Layout:
        cache_dir/
            ab/
                cdef1234...  # Data file (serialized)
                cdef1234.meta.json  # Metadata (timestamp, ttl)
            12/
                3456789a...
                3456789a.meta.json

    Metadata format:
        {
            "timestamp": <unix_time>,
            "ttl": <seconds_or_null>
        }

    Examples:
        >>> # Local filesystem
        >>> store = FileCacheStore("/tmp/cache")
        >>> store = FileCacheStore("file:///tmp/cache")  # Equivalent
        >>>
        >>> # S3
        >>> store = FileCacheStore("s3://my-bucket/cache")  # doctest: +SKIP
        >>>
        >>> # Google Cloud Storage
        >>> store = FileCacheStore("gcs://my-bucket/cache")  # doctest: +SKIP
    """

    def __init__(
        self,
        base_path: str,
        fs: AbstractFileSystem | None = None,
    ) -> None:
        """
        Initialize file cache store.

        Args:
            base_path: Root directory/prefix for cache storage (can be fsspec URL)
            fs: fsspec filesystem instance (auto-detected from base_path if None)
        """
        from fsspec import filesystem
        from fsspec.utils import get_protocol

        self.base_path = base_path.rstrip("/")
        self.fs = fs if fs is not None else filesystem(get_protocol(base_path))
        self.fs.mkdirs(self.base_path, exist_ok=True)

    def _get_paths(self, hash_key: str) -> tuple[str, str]:
        """
        Get data and metadata paths for a cache key.

        Args:
            hash_key: Hash digest (e.g., "abcdef1234...")

        Returns:
            Tuple of (data_path, metadata_path)
        """
        # Use first 2 chars as directory, rest as filename (git-style)
        prefix = hash_key[:2]
        suffix = hash_key[2:]

        data_path = f"{self.base_path}/{prefix}/{suffix}"
        meta_path = f"{self.base_path}/{prefix}/{suffix}.meta.json"

        return data_path, meta_path

    def get(self, hash_key: str, return_type: type[T]) -> T | None:
        """Retrieve cached value by hash key."""
        data_path, meta_path = self._get_paths(hash_key)

        if not self.fs.exists(data_path):
            return None

        # Check TTL if metadata exists
        if self.fs.exists(meta_path):  # pragma: no branch
            try:
                with self.fs.open(meta_path, "r") as f:
                    metadata = json.load(f)

                ttl = metadata.get("ttl")
                if ttl is not None:
                    timestamp = metadata["timestamp"]
                    if time.time() - timestamp > ttl:
                        # Expired - clean up
                        self.invalidate(hash_key)
                        return None
            except (json.JSONDecodeError, KeyError, OSError):  # pragma: no cover
                # Corrupted metadata - treat as miss
                return None

        # Load cached data using pickle
        try:
            with self.fs.open(data_path, "rb") as f:
                return pickle.load(f)  # type: ignore[arg-type]
        except (OSError, pickle.PickleError):  # pragma: no cover
            return None

    def put(self, hash_key: str, value: Any, ttl: int | None = None) -> None:
        """Store value in cache with optional expiration."""
        data_path, meta_path = self._get_paths(hash_key)
        shard_dir = "/".join(data_path.split("/")[:-1])
        self.fs.mkdirs(shard_dir, exist_ok=True)

        # Serialize data using pickle
        with self.fs.open(data_path, "wb") as f:
            pickle.dump(value, f)  # type: ignore[arg-type]

        # Write metadata
        metadata = {"timestamp": time.time(), "ttl": ttl}
        with self.fs.open(meta_path, "w") as f:
            json.dump(metadata, f)

    def invalidate(self, hash_key: str) -> None:
        """Remove specific cache entry."""
        data_path, meta_path = self._get_paths(hash_key)

        try:
            self.fs.rm(data_path)
        except FileNotFoundError:  # pragma: no cover
            pass

        try:
            self.fs.rm(meta_path)
        except FileNotFoundError:  # pragma: no cover
            pass

        # Try to remove empty shard directory
        shard_dir = "/".join(data_path.split("/")[:-1])
        try:
            if self.fs.exists(shard_dir):
                files = self.fs.ls(shard_dir, detail=False)
                if not files:  # pragma: no cover
                    self.fs.rmdir(shard_dir)
        except (FileNotFoundError, OSError):  # pragma: no cover
            pass

    def clear(self) -> None:
        """Remove all cached entries."""
        if self.fs.exists(self.base_path):  # pragma: no branch
            self.fs.rm(self.base_path, recursive=True)
            self.fs.mkdirs(self.base_path, exist_ok=True)

    def __getstate__(self) -> dict[str, Any]:
        """
        Serialize FileCacheStore for pickling (needed for distributed backends).

        Returns only the base_path - filesystem will be reconstructed on unpickling.
        """
        return {"base_path": self.base_path}

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Reconstruct FileCacheStore from pickled state.

        Recreates the filesystem instance from the base_path.
        """
        from fsspec import filesystem
        from fsspec.utils import get_protocol

        self.base_path = state["base_path"]
        self.fs = filesystem(get_protocol(self.base_path))
