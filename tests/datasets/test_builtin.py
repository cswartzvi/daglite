"""Unit tests for built-in dataset implementations."""

import pickle

import pytest

from daglite.datasets.base import AbstractDataset
from daglite.datasets.builtin import BytesDataset
from daglite.datasets.builtin import DictPickleDataset
from daglite.datasets.builtin import FrozenSetPickleDataset
from daglite.datasets.builtin import ListPickleDataset
from daglite.datasets.builtin import PickleDataset
from daglite.datasets.builtin import SetPickleDataset
from daglite.datasets.builtin import TextDataset
from daglite.datasets.builtin import TuplePickleDataset


class TestPickleDataset:
    """Tests for the general-purpose PickleDataset."""

    def test_serialize_dict(self):
        ds = PickleDataset()
        data = ds.serialize({"key": [1, 2, 3]})
        assert pickle.loads(data) == {"key": [1, 2, 3]}

    def test_deserialize_dict(self):
        ds = PickleDataset()
        data = pickle.dumps({"key": [1, 2, 3]})
        assert ds.deserialize(data) == {"key": [1, 2, 3]}

    def test_roundtrip_complex_object(self):
        ds = PickleDataset()
        obj = {"nested": {"data": [1, 2.5, None, True]}, "tuple": (1, 2)}
        assert ds.deserialize(ds.serialize(obj)) == obj

    def test_registered_for_object(self):
        """PickleDataset is registered for (object, 'pickle')."""
        assert (object, "pickle") in AbstractDataset._registry

    def test_extensions(self):
        """PickleDataset has pkl and pickle extensions."""
        ds_cls = AbstractDataset._registry[(object, "pickle")]
        assert "pkl" in ds_cls.file_extensions
        assert "pickle" in ds_cls.file_extensions


class TestTextDataset:
    """Tests for TextDataset (UTF-8 string serialization)."""

    def test_serialize(self):
        ds = TextDataset()
        assert ds.serialize("hello world") == b"hello world"

    def test_deserialize(self):
        ds = TextDataset()
        assert ds.deserialize(b"hello world") == "hello world"

    def test_roundtrip(self):
        ds = TextDataset()
        text = "Unicode: café ☕ 日本語"
        assert ds.deserialize(ds.serialize(text)) == text

    def test_empty_string(self):
        ds = TextDataset()
        assert ds.deserialize(ds.serialize("")) == ""

    def test_registered_for_str(self):
        assert (str, "text") in AbstractDataset._registry

    def test_extension_txt(self):
        ds_cls = AbstractDataset._registry[(str, "text")]
        assert "txt" in ds_cls.file_extensions


class TestBytesDataset:
    """Tests for BytesDataset (pass-through raw bytes)."""

    def test_serialize_passthrough(self):
        ds = BytesDataset()
        raw = b"\x00\x01\xff"
        assert ds.serialize(raw) == raw

    def test_deserialize_passthrough(self):
        ds = BytesDataset()
        raw = b"\x00\x01\xff"
        assert ds.deserialize(raw) == raw

    def test_roundtrip(self):
        ds = BytesDataset()
        raw = b"raw data content"
        assert ds.deserialize(ds.serialize(raw)) == raw

    def test_registered_for_bytes(self):
        assert (bytes, "raw") in AbstractDataset._registry

    def test_extension_bin(self):
        ds_cls = AbstractDataset._registry[(bytes, "raw")]
        assert "bin" in ds_cls.file_extensions


_TYPED_PICKLE_CASES = [
    pytest.param(DictPickleDataset, dict, {"a": 1, "b": [2, 3]}, id="dict"),
    pytest.param(ListPickleDataset, list, [1, "two", 3.0], id="list"),
    pytest.param(TuplePickleDataset, tuple, (1, 2, 3), id="tuple"),
    pytest.param(SetPickleDataset, set, {1, 2, 3}, id="set"),
    pytest.param(FrozenSetPickleDataset, frozenset, frozenset([1, 2, 3]), id="frozenset"),
]


class TestTypedPickleDatasets:
    """Parameterized tests for typed pickle dataset round-trips and registration."""

    @pytest.mark.parametrize("ds_cls,python_type,value", _TYPED_PICKLE_CASES)
    def test_roundtrip(self, ds_cls, python_type, value):
        ds = ds_cls()
        assert ds.deserialize(ds.serialize(value)) == value

    @pytest.mark.parametrize("ds_cls,python_type,value", _TYPED_PICKLE_CASES)
    def test_registered(self, ds_cls, python_type, value):
        assert (python_type, "pickle") in AbstractDataset._registry


_GET_BUILTIN_CASES = [
    pytest.param(str, "text", "hi", b"hi", id="str-text"),
    pytest.param(bytes, "raw", b"raw", b"raw", id="bytes-raw"),
    pytest.param(dict, "pickle", {"a": 1}, None, id="dict-pickle"),
    pytest.param(list, "pickle", [1, 2], None, id="list-pickle"),
    pytest.param(object, "pickle", 42, None, id="object-pickle"),
]


class TestGetViaBuiltins:
    """Tests for AbstractDataset.get() with built-in types."""

    @pytest.mark.parametrize("python_type,fmt,value,expected_bytes", _GET_BUILTIN_CASES)
    def test_get_roundtrip(self, python_type, fmt, value, expected_bytes):
        """get() returns a working dataset for each built-in type/format pair."""
        ds = AbstractDataset.get(python_type, fmt)()
        data = ds.serialize(value)
        if expected_bytes is not None:
            assert data == expected_bytes
        assert ds.deserialize(data) == value
