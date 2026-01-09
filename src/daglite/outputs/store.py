"""Storage protocols and implementations for task outputs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from daglite.serialization import SerializationRegistry
else:
    SerializationRegistry = object


T = TypeVar("T")


class FileOutputStore:
    """
    File-based implementation of OutputStore.

    Storage layout:
        base_path/
            {key}.{ext}  # Serialized data
    """

    def __init__(
        self,
        base_path: Path | str | None = None,
        registry: SerializationRegistry | None = None,
    ) -> None:
        """
        Initialize file-based output store.

        Args:
            base_path: Root directory for outputs
            registry: Serialization registry (uses default if None)
        """
        from daglite.serialization import default_registry

        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.base_path.mkdir(parents=True, exist_ok=True)

        if registry is None:
            self.registry = default_registry
        else:
            self.registry = registry

    def save(
        self,
        key: str,
        value: Any,
        format: str | None = None,
    ) -> None:
        """Save output to file."""
        data, ext = self.registry.serialize(value, format=format)
        output_file = self.base_path / f"{key}.{ext}"
        output_file.write_bytes(data)

    def load(self, key: str, return_type: type[T] | None = None) -> T:
        """Load output from file."""
        # Find the file with this key (any extension)
        matching_files = list(self.base_path.glob(f"{key}.*"))

        if not matching_files:
            raise KeyError(f"Output '{key}' not found")

        if len(matching_files) > 1:
            raise ValueError(f"Multiple files found for key '{key}': {matching_files}")

        output_file = matching_files[0]
        data = output_file.read_bytes()
        ext = output_file.suffix[1:]  # Remove leading dot

        if return_type is None:
            # Check if this extension is used by pickle format (works without type hints)
            registrations = self.registry._extension_to_format.get(ext, [])
            formats = {fmt for _, fmt in registrations}

            if "pickle" in formats:
                import pickle

                return pickle.loads(data)
            else:
                raise ValueError(
                    f"return_type is required for extension '{ext}'. "
                    "Only pickle format supports loading without type hints."
                )

        # With type hint, look up format from extension
        format_name = self.registry.get_format_from_extension(return_type, ext)
        if format_name is None:
            format_name = ext

        return self.registry.deserialize(data, return_type, format=format_name)

    def exists(self, key: str) -> bool:
        """Check if an output exists."""
        matching_files = list(self.base_path.glob(f"{key}.*"))
        return len(matching_files) > 0

    def delete(self, key: str) -> None:
        """Delete an output."""
        for f in self.base_path.glob(f"{key}.*"):
            f.unlink()

    def list_keys(self) -> list[str]:
        """List all stored output keys."""
        keys = set()
        for f in self.base_path.iterdir():
            if f.is_file():
                keys.add(f.stem)
        return sorted(keys)
