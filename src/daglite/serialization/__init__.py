"""Type-based serialization and hashing.

This module provides a central registry for type-based serialization,
deserialization, and hashing. It's designed to support efficient caching
and checkpointing with smart hash strategies for large objects.

Key components:
- SerializationRegistry: Main registry for types
- SerializationHandler: Per-format handler
- HashStrategy: Per-type hash function
- default_registry: Global registry instance

Example:
    >>> from daglite.serialization import default_registry
    >>>
    >>> # Register a custom type
    >>> default_registry.register(
    ...     MyModel,
    ...     lambda m: m.to_bytes(),
    ...     lambda b: MyModel.from_bytes(b),
    ...     format='default',
    ...     file_extension='model'
    ... )
    >>>
    >>> # Register hash strategy
    >>> default_registry.register_hash_strategy(
    ...     MyModel,
    ...     lambda m: m.get_version_hash(),
    ...     "Hash model version and config"
    ... )
    >>>
    >>> # Use it
    >>> data, ext = default_registry.serialize(my_model)
    >>> hash_key = default_registry.hash_value(my_model)
"""

from .registry import (
    HashStrategy,
    SerializationHandler,
    SerializationRegistry,
    default_registry,
)

__all__ = [
    'SerializationRegistry',
    'SerializationHandler',
    'HashStrategy',
    'default_registry',
]
