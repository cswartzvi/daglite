"""File-based driver using fsspec."""

from __future__ import annotations

import posixpath
import re
from pathlib import PureWindowsPath
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
        self._protocol = get_protocol(base_path)

        if fs is None:
            self.fs = filesystem(self._protocol)
        else:
            self.fs = fs

        self.fs.mkdirs(self.base_path, exist_ok=True)

    @property
    @override
    def is_local(self) -> bool:
        """
        Return True only for local file paths.

        Plain paths (no protocol) and the ``file`` protocol are considered
        local.  All other protocols (e.g. ``s3``, ``gcs``) are remote.
        """
        return self._protocol in ("", "file")

    def _full_path(self, key: str) -> str:
        """
        Gets the full path for the specified key.

        Relative keys are resolved under `base_path`. Absolute local paths (POSIX, Windows drive,
        and UNC) bypass `base_path`. URI keys are only accepted when they use the same protocol as
        `base_path` (or`file://` for local stores); mixed protocols raise `ValueError``.
        """
        # key = re.sub(r"\\", "/", key)  # replace backward slashes with forward slashes

        key_protocol = self._get_uri_protocol(key)

        if key_protocol is not None:
            if self._is_local_protocol(key_protocol) and self.is_local:
                return key

            if key_protocol == self._protocol:
                return key

            raise ValueError(
                f"Key protocol '{key_protocol}' does not match driver protocol "
                f"'{self._protocol or 'file'}'"
            )

        if self._is_absolute_local_path(key):
            return key

        return f"{self.base_path}/{key}"

    @staticmethod
    def _is_local_protocol(protocol: str) -> bool:
        return protocol in ("", "file")

    @staticmethod
    def _get_uri_protocol(key: str) -> str | None:
        match = re.match(r"^([a-zA-Z][a-zA-Z0-9+.-]*)://", key)
        if match:
            return match.group(1)
        return None

    @staticmethod
    def _is_absolute_local_path(key: str) -> bool:
        return key.startswith("/") or PureWindowsPath(key).is_absolute()

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
