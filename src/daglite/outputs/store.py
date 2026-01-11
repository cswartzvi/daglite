"""Storage protocols and implementations for task outputs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from fsspec import AbstractFileSystem

    from daglite.serialization import SerializationRegistry
else:
    AbstractFileSystem = object
    SerializationRegistry = object


T = TypeVar("T")


class FileOutputStore:
    """
    File-based implementation of OutputStore using fsspec.

    Supports both local and remote filesystems (s3://, gcs://, etc.) via fsspec.

    Storage layout:
        base_path/
            {key}.{ext}  # Serialized data

    Examples:
        >>> # Local filesystem
        >>> store = FileOutputStore("file:///tmp/outputs")
        >>> store = FileOutputStore("/tmp/outputs")  # Equivalent
        >>>
        >>> # S3
        >>> store = FileOutputStore("s3://my-bucket/outputs")  # doctest: +SKIP
        >>>
        >>> # Google Cloud Storage
        >>> store = FileOutputStore("gcs://my-bucket/outputs")  # doctest: +SKIP
    """

    def __init__(
        self,
        base_path: str,
        registry: SerializationRegistry | None = None,
        fs: AbstractFileSystem | None = None,
    ) -> None:
        """
        Initialize file-based output store.

        Args:
            base_path: Root directory/prefix for outputs (can be fsspec URL like s3://bucket/path)
            registry: Serialization registry (uses default if None)
            fs: fsspec filesystem instance (auto-detected from base_path if None)
        """
        from fsspec import filesystem
        from fsspec.utils import get_protocol

        from daglite.serialization import default_registry

        self.base_path = base_path.rstrip("/")

        if registry is None:
            self.registry = default_registry
        else:
            self.registry = registry

        if fs is None:
            self.fs = filesystem(get_protocol(base_path))
        else:
            self.fs = fs

        self.fs.mkdirs(self.base_path, exist_ok=True)

    def save(
        self,
        key: str,
        value: Any,
        format: str | None = None,
    ) -> None:
        """
        Save output to file.

        Args:
            key: Storage key/path. Extension controls filename, not serialization format.
            value: Value to serialize and save.
            format: Serialization format (independent of filename).
        """
        data, ext = self.registry.serialize(value, format=format)

        if "." in key.split("/")[-1]:  # Has extension
            output_path = f"{self.base_path}/{key}"
        else:
            output_path = f"{self.base_path}/{key}.{ext}"

        parent = "/".join(output_path.rsplit("/", 1)[:-1])
        if parent:  # pragma: no branch
            self.fs.mkdirs(parent, exist_ok=True)

        with self.fs.open(output_path, "wb") as f:
            f.write(data)  # type: ignore

    def load(self, key: str, return_type: type[T] | None = None) -> T:
        """Load output from file."""
        if "/" in key:
            parent_dir = "/".join([self.base_path] + key.split("/")[:-1])
            filename_prefix = key.split("/")[-1]
        else:
            parent_dir = self.base_path
            filename_prefix = key

        try:
            all_files = self.fs.ls(parent_dir, detail=False)
        except FileNotFoundError:
            all_files = []

        matching_files = [
            f for f in all_files if f.split("/")[-1].startswith(f"{filename_prefix}.")
        ]

        if not matching_files:
            raise KeyError(f"Output '{key}' not found")

        if len(matching_files) > 1:
            raise ValueError(f"Multiple files found for key '{key}': {matching_files}")

        output_path = matching_files[0]
        with self.fs.open(output_path, "rb") as f:
            data = f.read()

        ext = output_path.rsplit(".", 1)[-1]

        if return_type is None:
            # Check if this extension is used by pickle format (works without type hints)
            formats = self.registry.get_formats_for_extension(ext)

            if "pickle" in formats:
                import pickle

                return pickle.loads(data)  # type: ignore
            else:
                raise ValueError(
                    f"return_type is required for extension '{ext}'. "
                    "Only pickle format supports loading without type hints."
                )

        # With type hint, look up format from extension
        format_name = self.registry.get_format_from_extension(return_type, ext)
        if format_name is None:  # pragma: no cover
            format_name = ext

        return self.registry.deserialize(data, return_type, format=format_name)  # type: ignore

    def exists(self, key: str) -> bool:
        """Check if an output exists."""
        if "/" in key:
            parent_dir = "/".join([self.base_path] + key.split("/")[:-1])
            filename_prefix = key.split("/")[-1]
        else:
            parent_dir = self.base_path
            filename_prefix = key

        try:
            all_files = self.fs.ls(parent_dir, detail=False)
        except FileNotFoundError:
            return False

        matching_files = [
            f for f in all_files if f.split("/")[-1].startswith(f"{filename_prefix}.")
        ]
        return len(matching_files) > 0

    def delete(self, key: str) -> None:
        """Delete an output."""
        if "/" in key:
            parent_dir = "/".join([self.base_path] + key.split("/")[:-1])
            filename_prefix = key.split("/")[-1]
        else:
            parent_dir = self.base_path
            filename_prefix = key

        try:
            all_files = self.fs.ls(parent_dir, detail=False)
        except FileNotFoundError:
            return

        matching_files = [
            f for f in all_files if f.split("/")[-1].startswith(f"{filename_prefix}.")
        ]

        for f in matching_files:
            self.fs.rm(f)

    def list_keys(self) -> list[str]:
        """List all stored output keys."""
        try:
            all_files = self.fs.ls(self.base_path, detail=False)
        except FileNotFoundError:
            return []

        keys = set()
        for f in all_files:
            filename = f.split("/")[-1]
            # Remove extension
            key = filename.rsplit(".", 1)[0] if "." in filename else filename
            keys.add(key)

        return sorted(keys)

    def __getstate__(self) -> dict[str, Any]:
        """
        Serialize FileOutputStore for pickling (needed for distributed backends).

        Returns only the base_path - filesystem will be reconstructed on unpickling.
        """
        return {"base_path": self.base_path}

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Reconstruct FileOutputStore from pickled state.

        Recreates the filesystem instance and registry from the base_path.
        """
        from fsspec import filesystem
        from fsspec.utils import get_protocol

        from daglite.serialization import default_registry

        self.base_path = state["base_path"]
        self.registry = default_registry
        self.fs = filesystem(get_protocol(self.base_path))
