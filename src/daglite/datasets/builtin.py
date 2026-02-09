"""
Built-in dataset implementations for common Python types.

These datasets are automatically registered when the module is imported.
"""

from __future__ import annotations

import pickle
from typing import Any

from typing_extensions import override

from daglite.datasets.base import AbstractDataset


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
    def serialize(self, obj: str) -> bytes:
        return obj.encode("utf-8")

    @override
    def deserialize(self, data: bytes) -> str:
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
    def serialize(self, obj: bytes) -> bytes:
        return obj

    @override
    def deserialize(self, data: bytes) -> bytes:
        return data


class PickleDataset(AbstractDataset, format="pickle", types=object, extensions=("pkl", "pickle")):
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

    def __init__(self, protocol: int | None = None, encoding: str = "ASCII") -> None:
        self.protocol = protocol
        self.encoding = encoding

    @override
    def serialize(self, obj: Any) -> bytes:
        return pickle.dumps(obj, protocol=self.protocol)

    @override
    def deserialize(self, data: bytes) -> Any:
        return pickle.loads(data, encoding=self.encoding)


class DictPickleDataset(PickleDataset, types=dict):
    """Pickle serialization for dictionaries."""


class ListPickleDataset(PickleDataset, types=list):
    """Pickle serialization for lists."""


class TuplePickleDataset(PickleDataset, types=tuple):
    """Pickle serialization for tuples."""


class SetPickleDataset(PickleDataset, types=set):
    """Pickle serialization for sets."""


class FrozenSetPickleDataset(PickleDataset, types=frozenset):
    """Pickle serialization for frozensets."""
