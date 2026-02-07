"""
Abstract base class for datasets with automatic registration.

Datasets handle serialization/deserialization of specific types to/from bytes.
They auto-register via __init_subclass__, enabling lazy plugin discovery.
"""

from __future__ import annotations

import importlib.metadata
from abc import ABC
from abc import abstractmethod
from typing import Any, ClassVar


class AbstractDataset(ABC):
    """
    Abstract base class for type-based serialization.

    Subclasses auto-register via __init_subclass__. Supports lazy plugin discovery
    via entry points when a type/format combination is not found.

    Examples:
        Create a custom dataset using class parameters:

        >>> class JsonDataset(AbstractDataset, format="json", types=dict, extensions="json"):
        ...     def serialize(self, value: dict) -> bytes:
        ...         import json
        ...
        ...         return json.dumps(value).encode("utf-8")
        ...
        ...     def deserialize(self, data: bytes) -> dict:
        ...         import json
        ...
        ...         return json.loads(data.decode("utf-8"))

        Or using class variables:

        >>> class YamlDataset(AbstractDataset):
        ...     format = "yaml"
        ...     types = (dict,)
        ...     extensions = ("yaml", "yml")
        ...
        ...     def serialize(self, value: dict) -> bytes: ...
        ...     def deserialize(self, data: bytes) -> dict: ...

        The dataset is now registered and can be retrieved:

        >>> dataset = AbstractDataset.get(dict, "json")
        >>> dataset.serialize({"key": "value"})
        b'{"key": "value"}'
    """

    # Class-level registry: (type, format) -> Dataset class
    _registry: ClassVar[dict[tuple[type, str], type[AbstractDataset]]] = {}

    # Reverse lookup: extension -> list of (type, format) for Driver hints
    _extension_hints: ClassVar[dict[str, list[tuple[type, str]]]] = {}

    # Track which entry point modules we've already tried to load
    _discovered: ClassVar[set[str]] = set()

    # Whether to auto-discover plugins on lookup failure
    _auto_discover: ClassVar[bool] = True

    # Defined by subclasses via __init_subclass__ or as class variables
    format: ClassVar[str]
    types: ClassVar[tuple[type, ...]]
    extensions: ClassVar[tuple[str, ...]]

    def __init_subclass__(
        cls,
        *,
        format: str | None = None,
        types: type | tuple[type, ...] | None = None,
        extensions: str | tuple[str, ...] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Auto-register subclasses in the dataset registry.

        Parameters can be provided as class parameters or as class variables.
        Class parameters take precedence over class variables.

        Args:
            format: Format identifier (e.g., 'pickle', 'parquet'). If None, checks for class
                variable. If neither exists, this is an abstract intermediate class and won't be
                registered.
            types: Python type(s) this dataset can serialize. Can be a single
                type or a tuple of types.
            extensions: Optional file extension(s) for format inference (without dots). Can be a
                single string or a tuple of strings.
            kwargs: Additional keyword arguments.

        Examples:
            Using class parameters:

            >>> class JsonDataset(AbstractDataset, format="json", types=dict): ...

            Using class variables:

            >>> class CsvDataset(AbstractDataset):
            ...     format = "csv"
            ...     supported_types = (dict,)
            ...     file_extensions = ("csv",)
            ...     ...
        """
        super().__init_subclass__(**kwargs)

        # Resolve format: parameter > class variable > None (abstract class)
        resolved_format = format
        if resolved_format is None:
            resolved_format = getattr(cls, "format", None)

        if resolved_format is None:
            return

        # Resolve types: parameter > class variable > empty
        resolved_types: tuple[type, ...]
        if types is not None:
            resolved_types = (types,) if isinstance(types, type) else types
        else:
            resolved_types = getattr(cls, "types", ())

        # Resolve extensions: parameter > class variable > empty
        resolved_extensions: tuple[str, ...]
        if extensions is not None:
            resolved_extensions = (extensions,) if isinstance(extensions, str) else extensions
        else:
            resolved_extensions = getattr(cls, "extensions", ())

        cls.format = resolved_format
        cls.supported_types = resolved_types
        cls.file_extensions = resolved_extensions

        # Register for each supported type
        for t in resolved_types:
            AbstractDataset._registry[(t, resolved_format)] = cls

        # Register extension hints for Driver
        for ext in resolved_extensions:
            if ext not in AbstractDataset._extension_hints:
                AbstractDataset._extension_hints[ext] = []
            for t in resolved_types:
                AbstractDataset._extension_hints[ext].append((t, resolved_format))

    @abstractmethod
    def serialize(self, value: Any, **options: Any) -> bytes:
        """
        Convert a value to bytes.

        Args:
            value: The value to serialize.
            **options: Optional settings that may be used for serialization.

        Returns:
            Serialized bytes representation.
        """
        ...

    @abstractmethod
    def deserialize(self, data: bytes, **options: Any) -> Any:
        """
        Convert bytes back to a value.

        Args:
            data: Serialized bytes to deserialize.
            **options: Optional settings that may be used for deserialization.

        Returns:
            The deserialized value.
        """
        ...

    @classmethod
    def get(cls, type_: type, format: str) -> AbstractDataset:
        """
        Look up and instantiate the Dataset for a type/format combination.

        Supports lazy plugin discovery - if the combination isn't found,
        attempts to load relevant entry points before failing.

        Args:
            type_: The Python type to serialize/deserialize.
            format: The format identifier (e.g., 'pickle', 'parquet').

        Returns:
            An instantiated Dataset that can handle the type/format.

        Raises:
            ValueError: If no dataset is registered for the type/format.

        Examples:
            >>> dataset = AbstractDataset.get(str, "text")  # doctest: +SKIP
            >>> dataset.serialize("hello")  # doctest: +SKIP
            b'hello'
        """
        key = (type_, format)

        # Fast path: exact match in registry
        if key in cls._registry:
            return cls._registry[key]()

        # Check for subclass matches (e.g., custom dict subclass → dict handler)
        for (reg_type, reg_format), dataset_cls in cls._registry.items():
            if reg_format == format:
                try:
                    if issubclass(type_, reg_type):
                        return dataset_cls()
                except TypeError:
                    continue

        # Slow path: try discovering plugins for this type
        if cls._auto_discover:
            cls._discover_for_type(type_)

            if key in cls._registry:
                return cls._registry[key]()

            for (reg_type, reg_format), dataset_cls in cls._registry.items():
                if reg_format == format:
                    try:
                        if issubclass(type_, reg_type):
                            return dataset_cls()
                    except TypeError:
                        continue

        module = type_.__module__.split(".")[0]
        available_formats = cls.get_formats_for_type(type_)

        if available_formats:
            raise ValueError(
                f"No dataset registered for {type_.__name__} with format '{format}'.\n"
                f"Available formats for this type: {', '.join(sorted(available_formats))}"
            )
        else:
            raise ValueError(
                f"No dataset registered for {type_.__name__} with format '{format}'.\n"
                f"Try: pip install daglite-datasets[{module}]"
            )

    @classmethod
    def infer_format(cls, type_: type, extension: str | None = None) -> str:
        """
        Infer the format for a type, optionally using extension as a hint.

        Args:
            type_: The Python type.
            extension: Optional file extension (without dot) to help infer format.

        Returns:
            The inferred format identifier.

        Raises:
            ValueError: If no format can be inferred for the type.

        Examples:
            >>> AbstractDataset.infer_format(str, "txt")  # doctest: +SKIP
            'text'
            >>> AbstractDataset.infer_format(dict, "pkl")  # doctest: +SKIP
            'pickle'
        """
        # Try extension hint first
        if extension:
            hints = cls._extension_hints.get(extension, [])
            for hint_type, hint_format in hints:
                if hint_type == type_:
                    return hint_format
            # Check subclasses
            for hint_type, hint_format in hints:
                try:
                    if issubclass(type_, hint_type):
                        return hint_format
                except TypeError:
                    continue

        # Fall back to first registered format for this type
        for (reg_type, reg_format), _ in cls._registry.items():
            if reg_type == type_:
                return reg_format

        # Check subclass matches
        for (reg_type, reg_format), _ in cls._registry.items():
            try:
                if issubclass(type_, reg_type):
                    return reg_format
            except TypeError:
                continue

        # Try auto-discovery
        if cls._auto_discover:
            cls._discover_for_type(type_)

            for (reg_type, reg_format), _ in cls._registry.items():
                if reg_type == type_:
                    return reg_format

        raise ValueError(f"No default format registered for {type_.__name__}")

    @classmethod
    def get_formats_for_type(cls, type_: type) -> set[str]:
        """
        Get all registered formats for a type.

        Args:
            type_: The Python type.

        Returns:
            Set of format identifiers registered for this type.

        Examples:
            >>> "text" in AbstractDataset.get_formats_for_type(str)  # doctest: +SKIP
            True
        """
        formats: set[str] = set()
        for (reg_type, reg_format), _ in cls._registry.items():
            if reg_type == type_:
                formats.add(reg_format)
            else:
                try:
                    if issubclass(type_, reg_type):
                        formats.add(reg_format)
                except TypeError:
                    continue
        return formats

    @classmethod
    def get_extension(cls, type_: type, format: str) -> str | None:
        """
        Get the preferred file extension for a type/format combination.

        Args:
            type_: The Python type.
            format: The format identifier.

        Returns:
            The preferred file extension (without dot), or None if not found.

        Examples:
            >>> AbstractDataset.get_extension(str, "text")  # doctest: +SKIP
            'txt'
        """
        key = (type_, format)
        if key in cls._registry:
            dataset_cls = cls._registry[key]
            if dataset_cls.file_extensions:
                return dataset_cls.file_extensions[0]
        return None

    @classmethod
    def load_plugins(cls, *names: str, auto_discover: bool | None = None) -> None:
        """
        Explicitly load dataset plugins.

        Args:
            *names: Plugin names to load (e.g., "pandas", "numpy").
                If empty, loads all installed plugins.
            auto_discover: If provided, sets whether lazy discovery is enabled.
                Set to False to disable automatic plugin loading on lookup failure.

        Examples:
            Load specific plugins:

            >>> AbstractDataset.load_plugins("pandas", "numpy")  # doctest: +SKIP

            Load all installed plugins:

            >>> AbstractDataset.load_plugins()  # doctest: +SKIP

            Disable auto-discovery (strict mode):

            >>> AbstractDataset.load_plugins("pandas", auto_discover=False)  # doctest: +SKIP
        """
        if auto_discover is not None:
            cls._auto_discover = auto_discover

        try:
            eps = importlib.metadata.entry_points(group="daglite.datasets")
        except TypeError:  # pragma: no cover
            # Python < 3.10 compatibility
            all_eps = importlib.metadata.entry_points()
            eps = all_eps.get("daglite.datasets", [])  # type: ignore[assignment]

        for ep in eps:
            if not names or ep.name in names:
                try:
                    ep.load()  # Imports module, triggers __init_subclass__
                    cls._discovered.add(ep.name)
                except Exception:  # pragma: no cover
                    pass  # Plugin failed to load, continue with others

    @classmethod
    def _discover_for_type(cls, type_: type) -> None:
        """
        Lazily discover and load plugins that might handle this type.

        Uses the type's module name to find relevant entry points.
        For example, pandas.DataFrame → looks for "pandas" entry point.
        """
        module_name = type_.__module__.split(".")[0]

        if module_name in cls._discovered:
            return

        cls._discovered.add(module_name)

        try:
            eps = importlib.metadata.entry_points(group="daglite.datasets")
        except TypeError:  # pragma: no cover
            # Python < 3.10 compatibility
            all_eps = importlib.metadata.entry_points()
            eps = all_eps.get("daglite.datasets", [])  # type: ignore[assignment]

        for ep in eps:
            if ep.name == module_name:
                try:
                    ep.load()
                except Exception:  # pragma: no cover
                    pass  # Plugin failed to load
                break


__all__ = ["AbstractDataset"]
