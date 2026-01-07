"""
Type-based serialization and hashing.

This module provides a central registry for type-based serialization, deserialization, and hashing
of built-in and custom Python types. Adding support for external types (e.g., numpy arrays, pandas
dataframes, PIL images, PyTorch tensors) can be done via separate `daglite-serialization` package.
"""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from typing import Any, Callable, Type, TypeVar

T = TypeVar("T")


class SerializationRegistry:
    """
    Central registry for type-based serialization and hashing.

    Hash strategies support recursive hashing, so collections containing registered types (like
    dict[str, np.ndarray]) automatically work.

    Examples:
        >>> from daglite.serialization import SerializationRegistry
        >>> registry = SerializationRegistry()

        Register the custom type for serialization
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class MyModel:
        ...     def to_bytes(self) -> bytes:
        ...         return b"model_bytes"
        ...
        ...     @staticmethod
        ...     def from_bytes(b: bytes) -> "MyModel":
        ...         return MyModel()
        ...
        ...     def get_version_hash(self) -> str:
        ...         return "unique_model_hash"

        Register serialization handlers for the custom type
        >>> registry.register(
        ...     MyModel,
        ...     lambda m: m.to_bytes(),
        ...     lambda b: MyModel.from_bytes(b),
        ...     format="default",
        ...     file_extensions="model",
        ... )

        Register a hash strategy for the custom type
        >>> registry.register_hash_strategy(
        ...     MyModel, lambda m: m.get_version_hash(), "Hash model version and config"
        ... )

        Serialize and hash an instance of the custom type
        >>> my_model = MyModel()
        >>> registry.serialize(my_model)
        (b'model_bytes', 'model')
        >>> registry.hash_value(my_model)
        'unique_model_hash'
    """

    def __init__(self) -> None:
        """Initialize registry with built-in types."""
        self._handlers: dict[tuple[Type, str], _SerializationHandler] = {}
        self._default_formats: dict[Type, str] = {}
        self._extension_to_format: dict[str, list[tuple[Type, str]]] = {}
        self._hash_strategies: dict[Type, _HashStrategy] = {}
        self._register_builtin_types()

    def register(
        self,
        type_: Type[T],
        serializer: Callable[[T], bytes],
        deserializer: Callable[[bytes], T],
        format: str = "default",
        file_extensions: list[str] | str = "bin",
        make_default: bool = False,
    ) -> None:
        """
        Register a serialization handler for a type.

        Args:
            type_: Python type to register the handler for.
            serializer: Function to convert object to bytes.
            deserializer: Function to convert bytes back to object.
            format: Format identifier (e.g., 'pickle', 'csv', 'parquet'), defaults to 'default'.
            file_extensions: File extension(s) for this format. Can be a single string or list.
                First extension is preferred (used when saving). Defaults to 'bin'.
            make_default: Whether to make this the default format for the type.
        """
        # Normalize to list
        if isinstance(file_extensions, str):
            file_extensions = [file_extensions]

        handler = _SerializationHandler(
            type_=type_,
            format=format,
            file_extensions=file_extensions,
            serializer=serializer,
            deserializer=deserializer,
            is_default=make_default,
        )

        key = (type_, format)
        self._handlers[key] = handler

        # Register all extensions for reverse lookup
        # Multiple types can share the same extension (e.g., dict, list, tuple all use .pkl)
        for ext in file_extensions:
            if ext not in self._extension_to_format:
                self._extension_to_format[ext] = []
            self._extension_to_format[ext].append((type_, format))

        # Set as default if requested or if it's the first format for this type
        if make_default or type_ not in self._default_formats:
            self._default_formats[type_] = format

    def register_hash_strategy(
        self,
        type_: Type,
        hasher: Callable[[Any], str],
        description: str = "",
    ) -> None:
        """
        Register a hash strategy for a type.

        Args:
            type_: Python type to register the strategy for.
            hasher: Function to compute hash string from object.
            description: Human-readable description of the strategy.
        """
        strategy = _HashStrategy(
            type_=type_,
            hasher=hasher,
            description=description,
        )
        self._hash_strategies[type_] = strategy

    def serialize(
        self,
        obj: Any,
        format: str | None = None,
    ) -> tuple[bytes, str]:
        """
        Serialize an object using registered handler.

        Args:
            obj: Object to serialize.
            format: Optional format specifier (uses default if None)

        Returns:
            Tuple of (serialized_bytes, file_extension)

        Raises:
            ValueError: If no handler is registered for the type/format

        Examples:
            >>> from daglite.serialization import SerializationRegistry
            >>> registry = SerializationRegistry()
            >>> registry.serialize("hello", format="text")
            (b'hello', 'txt')
        """
        obj_type = type(obj)

        # Determine format
        if format is None:
            format = self._default_formats.get(obj_type, "pickle")

        # Find handler (exact match or check subclasses)
        handler = self._find_handler(obj_type, format)
        if handler is None:
            raise ValueError(
                f"No serialization handler registered for type {obj_type.__name__} "
                f"with format '{format}'. Register using registry.register()."
            )

        # Serialize
        data = handler.serializer(obj)
        return data, handler.file_extensions[0]  # Return preferred extension

    def deserialize(
        self,
        data: bytes,
        type_: Type[T],
        format: str | None = None,
    ) -> T:
        """
        Deserialize bytes back to an object.

        Args:
            data: Serialized data as bytes.
            type_: Expected Python type of the object.
            format: Optional format specifier (uses default if None)

        Returns:
            The deserialized object

        Raises:
            ValueError: If no handler is registered for the type/format

        Examples:
            >>> from daglite.serialization import SerializationRegistry
            >>> registry = SerializationRegistry()
            >>> registry.deserialize(b"hello", str, format="text")
            'hello'
        """
        # Determine format
        if format is None:
            format = self._default_formats.get(type_, "pickle")

        # Find handler
        handler = self._find_handler(type_, format)
        if handler is None:
            raise ValueError(
                f"No deserialization handler registered for type {type_.__name__} "
                f"with format '{format}'. Register using registry.register()."
            )

        # Deserialize
        return handler.deserializer(data)

    def hash_value(self, obj: Any) -> str:
        """
        Hash an object using registered strategy.

        Supports recursive hashing - collections containing registered types
        will automatically use the appropriate hash strategies for nested values.

        Args:
            obj: Object to be hashed.

        Returns:
            SHA256 hex digest string

        Raises:
            TypeError: If no hash strategy is registered for this type.

        Example:
            >>> from daglite.serialization import SerializationRegistry
            >>> registry = SerializationRegistry()
            >>> registry.hash_value([1, 2, 3])
            '37b739850c210ed5bf75eb0803169d39e1b8e827a692f14b598872598406536b'
        """
        obj_type = type(obj)

        # Find strategy (exact match or check subclasses)
        strategy = self._find_hash_strategy(obj_type)
        if strategy is None:
            # No strategy found - raise helpful error
            module = obj_type.__module__
            type_name = obj_type.__name__

            # Infer plugin from module name
            plugin_suggestion = None
            if module.startswith("numpy"):
                plugin_suggestion = "daglite_serialization[numpy]"
            elif module.startswith("pandas"):
                plugin_suggestion = "daglite_serialization[pandas]"
            elif module.startswith("PIL") or module.startswith("pillow"):
                plugin_suggestion = "daglite_serialization[pillow]"
            elif module.startswith("torch"):
                plugin_suggestion = "daglite_serialization[torch]"

            if plugin_suggestion:
                raise TypeError(
                    f"No hash strategy registered for {type_name} from {module}.\n"
                    f"\n"
                    f"To fix:\n"
                    f" 1. Install: pip install {plugin_suggestion}\n"
                    f" 2. Register: from daglite_serialization import register_all; register_all()"
                    f"\n\n"
                    f"Or register a custom hash strategy:\n"
                    f" from daglite.serialization import default_registry\n"
                    f" default_registry.register_hash_strategy({type_name}, my_hasher)\n"
                )
            else:
                # Generic error for unknown types
                raise TypeError(
                    f"No hash strategy registered for {type_name}.\n"
                    f"\n"
                    f"To use this type with caching, register a hash strategy:\n"
                    f"  from daglite.serialization import default_registry\n"
                    f"  import hashlib\n"
                    f"  import pickle\n"
                    f"  default_registry.register_hash_strategy(\n"
                    f"      {type_name},\n"
                    f"      lambda obj: hashlib.sha256(pickle.dumps(obj)).hexdigest()\n"
                    f"  )\n"
                )

        return strategy.hasher(obj)

    def set_default_format(self, type_: Type, format: str) -> None:
        """
        Set the default format for a type.

        Args:
            type_: The Python type
            format: The format identifier

        Raises:
            ValueError: If the format is not registered for the type
        """
        if (type_, format) not in self._handlers:
            raise ValueError(
                f"Format '{format}' is not registered for type {type_.__name__}. "
                f"Register it first using registry.register()."
            )
        self._default_formats[type_] = format

    def get_format_from_extension(
        self,
        type_: Type,
        extension: str,
    ) -> str | None:
        """
        Get the format identifier for a given type and file extension.

        Args:
            type_: The Python type
            extension: File extension (without leading dot)

        Returns:
            Format identifier if found, None otherwise

        Examples:
            >>> from daglite.serialization import default_registry
            >>> default_registry.get_format_from_extension(dict, "pkl")
            'pickle'
            >>> default_registry.get_format_from_extension(dict, "pickle")
            'pickle'
        """
        registrations = self._extension_to_format.get(extension, [])

        # Try exact type match first
        for registered_type, format in registrations:
            if registered_type == type_:
                return format

        # Check if any registered type is a parent class
        for registered_type, format in registrations:
            try:
                if issubclass(type_, registered_type):
                    return format
            except TypeError:  # pragma: no cover
                continue

        return None

    def get_extension(
        self,
        type_: Type,
        format: str | None = None,
    ) -> str:
        """
        Get the file extension for a type/format combination.

        Args:
            type_: The Python type
            format: Optional format specifier (uses default if None)

        Returns:
            The file extension (without leading dot)

        Raises:
            ValueError: If no handler is registered for the type/format
        """
        if format is None:
            format = self._default_formats.get(type_, "pickle")

        handler = self._find_handler(type_, format)
        if handler is None:
            raise ValueError(
                f"No handler registered for type {type_.__name__} with format '{format}'."
            )

        return handler.file_extensions[0]  # Return preferred extension

    def _find_handler(
        self,
        type_: Type,
        format: str,
    ) -> _SerializationHandler | None:
        """Find handler for type/format, checking subclasses if needed."""
        # Try exact match first
        key = (type_, format)
        if key in self._handlers:
            return self._handlers[key]

        # Check if any registered type is a parent class
        for (registered_type, registered_format), handler in self._handlers.items():
            if registered_format == format:
                try:
                    if issubclass(type_, registered_type):
                        return handler
                except TypeError:  # pragma: no cover
                    # issubclass raises TypeError for non-class types (e.g., generics, unions)
                    # This is defensive - in practice, Type annotations ensure we have classes
                    continue

        return None

    def _find_hash_strategy(self, type_: Type) -> _HashStrategy | None:
        """Find hash strategy for type, checking subclasses if needed."""
        # Try exact match first
        if type_ in self._hash_strategies:
            return self._hash_strategies[type_]

        # Check if any registered type is a parent class
        for registered_type, strategy in self._hash_strategies.items():
            try:
                if issubclass(type_, registered_type):
                    return strategy
            except TypeError:  # pragma: no cover
                # issubclass raises TypeError for non-class types (e.g., generics, unions)
                # This is defensive - in practice, Type annotations ensure we have classes
                continue

        return None

    def _register_builtin_types(self) -> None:
        """
        Register serialization and hash strategies for built-in Python types.

        Hash strategies use closures to enable recursive hashing - collections
        can contain any registered type and hashing will work automatically.
        """

        # Helper: hash simple immutable values via repr
        def hash_simple(obj: Any) -> str:
            """Hash simple values (str, int, float, bool, None) using repr."""
            return hashlib.sha256(repr(obj).encode()).hexdigest()

        # bytes - direct hash
        self.register(
            bytes,
            lambda b: b,
            lambda b: b,
            format="raw",
            file_extensions="bin",
        )
        self.register_hash_strategy(
            bytes,
            lambda data: hashlib.sha256(data).hexdigest(),
            "Direct SHA256 hash of bytes",
        )

        # str
        self.register(
            str,
            lambda s: s.encode("utf-8"),
            lambda b: b.decode("utf-8"),
            format="text",
            file_extensions="txt",
        )
        self.register_hash_strategy(str, hash_simple, "Hash str via repr")

        # int
        self.register(
            int,
            lambda x: str(x).encode(),
            lambda b: int(b.decode()),
            format="text",
            file_extensions="txt",
        )
        self.register_hash_strategy(int, hash_simple, "Hash int via repr")

        # float
        self.register(
            float,
            lambda x: str(x).encode(),
            lambda b: float(b.decode()),
            format="text",
            file_extensions="txt",
        )
        self.register_hash_strategy(float, hash_simple, "Hash float via repr")

        # bool
        self.register(
            bool,
            lambda x: str(x).encode(),
            lambda b: b.decode() == "True",
            format="text",
            file_extensions="txt",
        )
        self.register_hash_strategy(bool, hash_simple, "Hash bool via repr")

        # NoneType
        self.register(
            type(None),
            lambda _: b"None",
            lambda _: None,
            format="text",
            file_extensions="txt",
        )
        self.register_hash_strategy(type(None), hash_simple, "Hash NoneType via repr")

        # dict - recursive hashing via closure
        def hash_dict(d: dict) -> str:
            """Hash dict recursively using registry for values."""
            h = hashlib.sha256()
            for key in sorted(d.keys()):
                h.update(str(key).encode())
                # Recursively hash value using registry
                h.update(self.hash_value(d[key]).encode())
            return h.hexdigest()

        self.register(
            dict,
            pickle.dumps,
            pickle.loads,
            format="pickle",
            file_extensions=["pkl", "pickle"],
        )
        self.register_hash_strategy(dict, hash_dict, "Recursive hash of dict values")

        # list, tuple - recursive hashing via closure
        def hash_sequence(items) -> str:
            """Hash sequence recursively using registry for items."""
            h = hashlib.sha256()
            for item in items:
                # Recursively hash item using registry
                h.update(self.hash_value(item).encode())
            return h.hexdigest()

        for type_ in [list, tuple]:
            self.register(
                type_,
                pickle.dumps,
                pickle.loads,
                format="pickle",
                file_extensions=["pkl", "pickle"],
            )
            self.register_hash_strategy(
                type_, hash_sequence, f"Recursive hash of {type_.__name__} items"
            )

        # set, frozenset - sort then recursive hash
        def hash_unordered(items) -> str:
            """Hash unordered collection recursively using registry for items."""
            h = hashlib.sha256()
            # Sort by hash to get deterministic order
            item_hashes = sorted(self.hash_value(item) for item in items)
            for item_hash in item_hashes:
                h.update(item_hash.encode())
            return h.hexdigest()

        for type_ in [set, frozenset]:
            self.register(
                type_,
                pickle.dumps,
                pickle.loads,
                format="pickle",
                file_extensions=["pkl", "pickle"],
            )
            self.register_hash_strategy(
                type_, hash_unordered, f"Recursive hash of {type_.__name__} items"
            )


@dataclass
class _SerializationHandler:
    """Handler for serializing/deserializing a specific type in a specific format."""

    type_: Type
    """Python type this handler applies to"""

    format: str
    """Format identifier (e.g., 'pickle', 'csv', 'parquet')"""

    file_extensions: list[str]
    """File extensions for this format (first is preferred, e.g., ['pkl', 'pickle'])"""

    serializer: Callable[[Any], bytes]
    """Function to convert object to bytes"""

    deserializer: Callable[[bytes], Any]
    """Function to convert bytes back to object"""

    is_default: bool = False
    """Whether this is the default format for the type"""


@dataclass
class _HashStrategy:
    """
    Strategy for hashing a specific type.

    Attributes:
        type_: The Python type this strategy applies to
        hasher: Function to compute hash string from object
        description: Human-readable description of the strategy
    """

    type_: Type
    """Python type this strategy applies to"""

    hasher: Callable[[Any], str]
    """Function to compute hash string from object"""

    description: str = ""
    """Human-readable description of the strategy"""


# Global default registry instance
default_registry = SerializationRegistry()


__all__ = [
    "SerializationRegistry",
    "_SerializationHandler",
    "_HashStrategy",
    "default_registry",
]
