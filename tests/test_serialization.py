"""Tests for serialization registry and hash strategies."""

import pickle
import time
from dataclasses import dataclass

import pytest

from daglite.serialization import (
    HashStrategy,
    SerializationHandler,
    SerializationRegistry,
    default_registry,
)
from daglite.serialization import hash_strategies


# Module-level test classes (needed for pickle support)
@dataclass
class TestData:
    """Test data class for pickle serialization tests."""

    value: int


class TestSerializationRegistry:
    """Tests for SerializationRegistry."""

    def test_builtin_types_registered(self):
        """Test that built-in types are registered by default."""
        registry = SerializationRegistry()

        # Test serialization
        assert registry.serialize("hello") == (b"hello", "txt")
        assert registry.serialize(42) == (b"42", "txt")
        assert registry.serialize(3.14) == (b"3.14", "txt")
        assert registry.serialize(True) == (b"True", "txt")
        assert registry.serialize(b"bytes") == (b"bytes", "bin")

    def test_builtin_types_deserialization(self):
        """Test that built-in types can be deserialized."""
        registry = SerializationRegistry()

        assert registry.deserialize(b"hello", str) == "hello"
        assert registry.deserialize(b"42", int) == 42
        assert registry.deserialize(b"3.14", float) == 3.14
        assert registry.deserialize(b"True", bool) is True
        assert registry.deserialize(b"False", bool) is False

    def test_builtin_types_hash(self):
        """Test that built-in types can be hashed."""
        registry = SerializationRegistry()

        # Hash should be deterministic
        hash1 = registry.hash_value("hello")
        hash2 = registry.hash_value("hello")
        assert hash1 == hash2

        # Different values should have different hashes
        hash3 = registry.hash_value("world")
        assert hash1 != hash3

    def test_register_custom_type(self):
        """Test registering a custom type."""
        registry = SerializationRegistry()

        @dataclass
        class Point:
            x: int
            y: int

            def to_bytes(self) -> bytes:
                return f"{self.x},{self.y}".encode()

            @classmethod
            def from_bytes(cls, data: bytes) -> "Point":
                x, y = data.decode().split(",")
                return cls(int(x), int(y))

        # Register
        registry.register(
            Point,
            lambda p: p.to_bytes(),
            lambda b: Point.from_bytes(b),
            format="csv",
            file_extension="csv",
        )

        # Test serialization
        point = Point(10, 20)
        data, ext = registry.serialize(point)
        assert data == b"10,20"
        assert ext == "csv"

        # Test deserialization
        restored = registry.deserialize(data, Point)
        assert restored.x == 10
        assert restored.y == 20

    def test_multiple_formats(self):
        """Test registering multiple formats for the same type."""
        registry = SerializationRegistry()

        # Register CSV format
        registry.register(
            TestData,
            lambda d: str(d.value).encode(),
            lambda b: TestData(int(b.decode())),
            format="csv",
            file_extension="csv",
            make_default=True,
        )

        # Register pickle format
        registry.register(
            TestData,
            pickle.dumps,
            pickle.loads,
            format="pickle",
            file_extension="pkl",
        )

        data = TestData(42)

        # Default should be CSV
        serialized, ext = registry.serialize(data)
        assert ext == "csv"
        assert serialized == b"42"

        # Can explicitly request pickle
        serialized, ext = registry.serialize(data, format="pickle")
        assert ext == "pkl"

    def test_set_default_format(self):
        """Test changing the default format."""
        registry = SerializationRegistry()

        # Register both formats
        registry.register(
            TestData,
            lambda d: str(d.value).encode(),
            lambda b: TestData(int(b.decode())),
            format="csv",
            file_extension="csv",
        )

        registry.register(
            TestData,
            pickle.dumps,
            pickle.loads,
            format="pickle",
            file_extension="pkl",
        )

        data = TestData(42)

        # Default should be CSV (first registered)
        _, ext = registry.serialize(data)
        assert ext == "csv"

        # Change default to pickle
        registry.set_default_format(TestData, "pickle")

        # Now pickle should be default
        _, ext = registry.serialize(data)
        assert ext == "pkl"

    def test_register_hash_strategy(self):
        """Test registering a custom hash strategy."""
        registry = SerializationRegistry()

        @dataclass
        class Data:
            value: int
            metadata: str  # We don't want to include this in hash

        # Register serialization
        registry.register(
            Data,
            pickle.dumps,
            pickle.loads,
        )

        # Register hash strategy (only hash value, not metadata)
        registry.register_hash_strategy(
            Data,
            lambda d: hash_strategies.hash_int(d.value),
            "Hash only the value field",
        )

        # Same value should produce same hash, even with different metadata
        data1 = Data(42, "foo")
        data2 = Data(42, "bar")
        assert registry.hash_value(data1) == registry.hash_value(data2)

        # Different value should produce different hash
        data3 = Data(43, "foo")
        assert registry.hash_value(data1) != registry.hash_value(data3)

    def test_get_extension(self):
        """Test getting file extension for type/format."""
        registry = SerializationRegistry()

        assert registry.get_extension(str) == "txt"
        assert registry.get_extension(int) == "txt"
        assert registry.get_extension(dict) == "pkl"

    def test_unregistered_type_raises(self):
        """Test that unregistered types raise ValueError."""
        registry = SerializationRegistry()

        class CustomType:
            pass

        with pytest.raises(ValueError, match="No serialization handler registered"):
            registry.serialize(CustomType())

    def test_unregistered_format_raises(self):
        """Test that unregistered formats raise ValueError."""
        registry = SerializationRegistry()

        with pytest.raises(ValueError, match="No serialization handler registered"):
            registry.serialize("hello", format="invalid_format")

    def test_hash_raises_for_unregistered_type(self):
        """Test that unregistered types raise TypeError."""
        registry = SerializationRegistry()

        class CustomType:
            def __init__(self, value):
                self.value = value

            def __repr__(self):
                return f"CustomType({self.value})"

        obj = CustomType(42)

        # Should raise TypeError with helpful message
        with pytest.raises(TypeError, match="No hash strategy registered"):
            registry.hash_value(obj)


class TestHashStrategies:
    """Tests for hash strategies."""

    def test_hash_string(self):
        """Test string hashing."""
        hash1 = hash_strategies.hash_string("hello")
        hash2 = hash_strategies.hash_string("hello")
        hash3 = hash_strategies.hash_string("world")

        assert hash1 == hash2
        assert hash1 != hash3

    def test_hash_int(self):
        """Test integer hashing."""
        assert hash_strategies.hash_int(42) == hash_strategies.hash_int(42)
        assert hash_strategies.hash_int(42) != hash_strategies.hash_int(43)

    def test_hash_float(self):
        """Test float hashing."""
        assert hash_strategies.hash_float(3.14) == hash_strategies.hash_float(3.14)
        assert hash_strategies.hash_float(3.14) != hash_strategies.hash_float(3.15)

    def test_hash_bool(self):
        """Test boolean hashing."""
        assert hash_strategies.hash_bool(True) == hash_strategies.hash_bool(True)
        assert hash_strategies.hash_bool(True) != hash_strategies.hash_bool(False)

    def test_hash_none(self):
        """Test None hashing."""
        assert hash_strategies.hash_none(None) == hash_strategies.hash_none(None)

    def test_hash_dict(self):
        """Test dictionary hashing."""
        dict1 = {"a": 1, "b": 2}
        dict2 = {"b": 2, "a": 1}  # Different order, same content
        dict3 = {"a": 1, "b": 3}

        # Same content should have same hash (order-independent)
        assert hash_strategies.hash_dict(dict1) == hash_strategies.hash_dict(dict2)

        # Different content should have different hash
        assert hash_strategies.hash_dict(dict1) != hash_strategies.hash_dict(dict3)

    def test_hash_list(self):
        """Test list hashing."""
        list1 = [1, 2, 3]
        list2 = [1, 2, 3]
        list3 = [3, 2, 1]

        assert hash_strategies.hash_list(list1) == hash_strategies.hash_list(list2)
        assert hash_strategies.hash_list(list1) != hash_strategies.hash_list(list3)

    def test_hash_tuple(self):
        """Test tuple hashing."""
        tuple1 = (1, 2, 3)
        tuple2 = (1, 2, 3)
        tuple3 = (3, 2, 1)

        assert hash_strategies.hash_tuple(tuple1) == hash_strategies.hash_tuple(tuple2)
        assert hash_strategies.hash_tuple(tuple1) != hash_strategies.hash_tuple(tuple3)

    def test_hash_set(self):
        """Test set hashing."""
        set1 = {1, 2, 3}
        set2 = {3, 1, 2}  # Different order, same content
        set3 = {1, 2, 4}

        # Same content should have same hash (order-independent)
        assert hash_strategies.hash_set(set1) == hash_strategies.hash_set(set2)

        # Different content should have different hash
        assert hash_strategies.hash_set(set1) != hash_strategies.hash_set(set3)



class TestDefaultRegistry:
    """Tests for the global default_registry instance."""

    def test_default_registry_exists(self):
        """Test that default_registry is available."""
        assert default_registry is not None
        assert isinstance(default_registry, SerializationRegistry)

    def test_default_registry_has_builtins(self):
        """Test that default_registry has built-in types registered."""
        # Should be able to serialize/deserialize basic types
        assert default_registry.serialize("hello") == (b"hello", "txt")
        assert default_registry.deserialize(b"hello", str) == "hello"

        # Should be able to hash basic types
        hash_value = default_registry.hash_value(42)
        assert isinstance(hash_value, str)
        assert len(hash_value) == 64  # SHA256 hex digest


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_dict(self):
        """Test hashing empty dictionary."""
        assert hash_strategies.hash_dict({}) == hash_strategies.hash_dict({})

    def test_empty_list(self):
        """Test hashing empty list."""
        assert hash_strategies.hash_list([]) == hash_strategies.hash_list([])

    def test_empty_string(self):
        """Test hashing empty string."""
        assert hash_strategies.hash_string("") == hash_strategies.hash_string("")

    def test_nested_dict(self):
        """Test hashing nested dictionary."""
        dict1 = {"a": {"b": 1, "c": 2}, "d": 3}
        dict2 = {"a": {"b": 1, "c": 2}, "d": 3}
        dict3 = {"a": {"b": 1, "c": 3}, "d": 3}

        assert hash_strategies.hash_dict(dict1) == hash_strategies.hash_dict(dict2)
        assert hash_strategies.hash_dict(dict1) != hash_strategies.hash_dict(dict3)

    def test_nested_list(self):
        """Test hashing nested list."""
        list1 = [[1, 2], [3, 4]]
        list2 = [[1, 2], [3, 4]]
        list3 = [[1, 2], [3, 5]]

        assert hash_strategies.hash_list(list1) == hash_strategies.hash_list(list2)
        assert hash_strategies.hash_list(list1) != hash_strategies.hash_list(list3)

    def test_mixed_types_in_dict(self):
        """Test hashing dictionary with mixed value types."""
        dict1 = {"a": 1, "b": "hello", "c": [1, 2, 3]}
        dict2 = {"a": 1, "b": "hello", "c": [1, 2, 3]}

        assert hash_strategies.hash_dict(dict1) == hash_strategies.hash_dict(dict2)

    def test_unicode_string(self):
        """Test hashing Unicode strings."""
        str1 = "Hello 世界 🌍"
        str2 = "Hello 世界 🌍"
        str3 = "Hello World"

        assert hash_strategies.hash_string(str1) == hash_strategies.hash_string(str2)
        assert hash_strategies.hash_string(str1) != hash_strategies.hash_string(str3)
