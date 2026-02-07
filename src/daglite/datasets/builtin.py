"""
Built-in dataset implementations for common Python types.

These datasets are automatically registered when the module is imported.
"""

from __future__ import annotations

import pickle
from typing import Any

from typing_extensions import override

from daglite.datasets.base import AbstractDataset


class PickleDataset(
    AbstractDataset,
    format="pickle",
    types=(object,),
    extensions=("pkl", "pickle"),
):
    """
    General-purpose pickle serialization.

    This is the fallback dataset for types without a specific handler.
    Works with most Python objects but produces binary, non-human-readable output.

    Examples:
        >>> dataset = PickleDataset()
        >>> data = dataset.serialize({"key": [1, 2, 3]})
        >>> dataset.deserialize(data)
        {'key': [1, 2, 3]}
    """

    @override
    def serialize(self, value: Any, **options) -> bytes:
        return pickle.dumps(value)

    @override
    def deserialize(self, data: bytes, **options) -> Any:
        return pickle.loads(data)


class TextDataset(AbstractDataset, format="text", types=str, extensions="txt"):
    """
    UTF-8 text serialization for strings.

    Produces human-readable .txt files.

    Examples:
        >>> dataset = TextDataset()
        >>> dataset.serialize("hello world")
        b'hello world'
        >>> dataset.deserialize(b"hello world")
        'hello world'
    """

    @override
    def serialize(self, value: str, **options) -> bytes:
        return value.encode("utf-8")

    @override
    def deserialize(self, data: bytes, **options) -> str:
        return data.decode("utf-8")


class BytesDataset(AbstractDataset, format="raw", types=bytes, extensions="bin"):
    """
    Pass-through dataset for raw bytes.

    Simply returns bytes unchanged - useful for already-serialized data.

    Examples:
        >>> dataset = BytesDataset()
        >>> dataset.serialize(b"raw data")
        b'raw data'
        >>> dataset.deserialize(b"raw data")
        b'raw data'
    """

    @override
    def serialize(self, value: bytes, **options) -> bytes:
        return value

    @override
    def deserialize(self, data: bytes, **options) -> bytes:
        return data


# Register pickle format for common collection types
# These get their own registrations so they're found before the generic `object` fallback


class DictPickleDataset(
    AbstractDataset,
    format="pickle",
    types=dict,
    extensions=("pkl", "pickle"),
):
    """Pickle serialization for dictionaries."""

    @override
    def serialize(self, value: dict, **options) -> bytes:
        return pickle.dumps(value)

    @override
    def deserialize(self, data: bytes, **options) -> dict:
        return pickle.loads(data)


class ListPickleDataset(
    AbstractDataset,
    format="pickle",
    types=list,
    extensions=("pkl", "pickle"),
):
    """Pickle serialization for lists."""

    @override
    def serialize(self, value: list, **options) -> bytes:
        return pickle.dumps(value)

    @override
    def deserialize(self, data: bytes, **options) -> list:
        return pickle.loads(data)


class TuplePickleDataset(
    AbstractDataset,
    format="pickle",
    types=tuple,
    extensions=("pkl", "pickle"),
):
    """Pickle serialization for tuples."""

    @override
    def serialize(self, value: tuple, **options) -> bytes:
        return pickle.dumps(value)

    @override
    def deserialize(self, data: bytes, **options) -> tuple:
        return pickle.loads(data)


class SetPickleDataset(
    AbstractDataset,
    format="pickle",
    types=set,
    extensions=("pkl", "pickle"),
):
    """Pickle serialization for sets."""

    @override
    def serialize(self, value: set, **options) -> bytes:
        return pickle.dumps(value)

    @override
    def deserialize(self, data: bytes, **options) -> set:
        return pickle.loads(data)


class FrozenSetPickleDataset(
    AbstractDataset,
    format="pickle",
    types=frozenset,
    extensions=("pkl", "pickle"),
):
    """Pickle serialization for frozensets."""

    @override
    def serialize(self, value: frozenset, **options) -> bytes:
        return pickle.dumps(value)

    @override
    def deserialize(self, data: bytes, **options) -> frozenset:
        return pickle.loads(data)
