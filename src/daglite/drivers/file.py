"""File-based driver using fsspec."""

from __future__ import annotations

import posixpath
from typing import TYPE_CHECKING, Any

from typing_extensions import override

from daglite.drivers.base import Driver

if TYPE_CHECKING:
    from fsspec import AbstractFileSystem


class FileDriver(Driver):
    """
    File-based implementation of the Driver ABC using fsspec.

    Supports both local and remote filesystems (s3://, gcs://, etc.) via fsspec.

    This is a thin wrapper - it saves exactly what you give it to exactly where
    you specify. When you save "output.csv", you get "output.csv" on disk.

    Args:
        base_path: Base directory/prefix for storage (e.g., "/tmp/storage" or
            "s3://my-bucket/storage").
        fs: Optional fsspec AbstractFileSystem instance to use. If not provided, it will be created
            based on the protocol in base_path.

    Examples:
        >>> # Local filesystem
        >>> driver = FileDriver("/tmp/storage")
        >>>
        >>> # S3
        >>> driver = FileDriver("s3://my-bucket/storage")  # doctest: +SKIP
        >>>
        >>> # Google Cloud Storage
        >>> driver = FileDriver("gcs://my-bucket/storage")  # doctest: +SKIP
    """

    def __init__(self, base_path: str, fs: AbstractFileSystem | None = None) -> None:
        from fsspec import filesystem
        from fsspec.utils import get_protocol

        self.base_path = base_path.rstrip("/")

        if fs is None:
            self.fs = filesystem(get_protocol(base_path))
        else:
            self.fs = fs

        self.fs.mkdirs(self.base_path, exist_ok=True)

    def _full_path(self, key: str) -> str:
        """Get the full path for a key."""
        return f"{self.base_path}/{key}"

    @override
    def save(self, key: str, data: bytes) -> str:
        path = self._full_path(key)

        # Create parent directories
        parent = posixpath.dirname(path)
        if parent:  # pragma: no branch - defensive check for root-level keys
            self.fs.mkdirs(parent, exist_ok=True)

        # Write data
        with self.fs.open(path, "wb") as f:
            f.write(data)  # type: ignore

        return path

    @override
    def load(self, key: str) -> bytes:
        path = self._full_path(key)

        if not self.fs.exists(path):
            raise KeyError(f"Key '{key}' not found")

        with self.fs.open(path, "rb") as f:
            return f.read()  # type: ignore[return-value]

    @override
    def exists(self, key: str) -> bool:
        return self.fs.exists(self._full_path(key))

    @override
    def delete(self, key: str) -> None:
        path = self._full_path(key)
        if self.fs.exists(path):
            self.fs.rm(path)

    @override
    def list_keys(self) -> list[str]:
        all_files = self.fs.glob(f"{self.base_path}/**", detail=False)

        # Normalize base_path to forward slashes for comparison; on Windows
        # self.base_path may use backslashes while fsspec glob returns
        # forward-slash paths, causing startswith() to fail.
        normalized_base = self.base_path.replace("\\", "/")
        prefix_len = len(normalized_base) + 1  # +1 for the trailing slash

        keys = []
        for f in all_files:
            if self.fs.isdir(f):
                continue  # Skip directories

            if isinstance(f, str):  # pragma: no branch - defensive check
                normalized_f = f.replace("\\", "/")
                if normalized_f.startswith(normalized_base):  # pragma: no branch - defensive check
                    key = normalized_f[prefix_len:]
                    if key:  # pragma: no branch - defensive check
                        keys.append(key)

        return sorted(keys)

    @override
    def get_format_hint(self, key: str) -> str | None:
        # Normalize to forward slashes in case of Windows paths
        normalized_key = key.replace("\\", "/")
        ext = posixpath.splitext(normalized_key)[1]
        if ext:
            return ext[1:].lower()  # Remove leading dot
        return None

    def __getstate__(self) -> dict[str, Any]:
        return {"base_path": self.base_path}

    def __setstate__(self, state: dict[str, Any]) -> None:
        from fsspec import filesystem
        from fsspec.utils import get_protocol

        self.base_path = state["base_path"]
        self.fs = filesystem(get_protocol(self.base_path))
