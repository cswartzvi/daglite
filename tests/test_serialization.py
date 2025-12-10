"""Tests for serialization registry and hash strategies."""

import pickle
from dataclasses import dataclass

import pytest

from daglite.serialization import SerializationRegistry
from daglite.serialization import default_registry


# Module-level test classes (needed for pickle support)
@dataclass
class SampleData:
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
            SampleData,
            lambda d: str(d.value).encode(),
            lambda b: SampleData(int(b.decode())),
            format="csv",
            file_extension="csv",
            make_default=True,
        )

        # Register pickle format
        registry.register(
            SampleData,
            pickle.dumps,
            pickle.loads,
            format="pickle",
            file_extension="pkl",
        )

        data = SampleData(42)

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
            SampleData,
            lambda d: str(d.value).encode(),
            lambda b: SampleData(int(b.decode())),
            format="csv",
            file_extension="csv",
        )

        registry.register(
            SampleData,
            pickle.dumps,
            pickle.loads,
            format="pickle",
            file_extension="pkl",
        )

        data = SampleData(42)

        # Default should be CSV (first registered)
        _, ext = registry.serialize(data)
        assert ext == "csv"

        # Change default to pickle
        registry.set_default_format(SampleData, "pickle")

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
            lambda d: registry.hash_value(d.value),
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

        # Test with default format (None)
        assert registry.get_extension(str) == "txt"
        assert registry.get_extension(int) == "txt"
        assert registry.get_extension(dict) == "pkl"

        # Test with explicit format
        assert registry.get_extension(str, format="text") == "txt"
        assert registry.get_extension(dict, format="pickle") == "pkl"

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
        hash1 = default_registry.hash_value("hello")
        hash2 = default_registry.hash_value("hello")
        hash3 = default_registry.hash_value("world")

        assert hash1 == hash2
        assert hash1 != hash3

    def test_hash_int(self):
        """Test integer hashing."""
        assert default_registry.hash_value(42) == default_registry.hash_value(42)
        assert default_registry.hash_value(42) != default_registry.hash_value(43)

    def test_hash_float(self):
        """Test float hashing."""
        assert default_registry.hash_value(3.14) == default_registry.hash_value(3.14)
        assert default_registry.hash_value(3.14) != default_registry.hash_value(3.15)

    def test_hash_bool(self):
        """Test boolean hashing."""
        assert default_registry.hash_value(True) == default_registry.hash_value(True)
        assert default_registry.hash_value(True) != default_registry.hash_value(False)

    def test_hash_none(self):
        """Test None hashing."""
        assert default_registry.hash_value(None) == default_registry.hash_value(None)

    def test_hash_dict(self):
        """Test dictionary hashing."""
        dict1 = {"a": 1, "b": 2}
        dict2 = {"b": 2, "a": 1}  # Different order, same content
        dict3 = {"a": 1, "b": 3}

        # Same content should have same hash (order-independent)
        assert default_registry.hash_value(dict1) == default_registry.hash_value(dict2)

        # Different content should have different hash
        assert default_registry.hash_value(dict1) != default_registry.hash_value(dict3)

    def test_hash_list(self):
        """Test list hashing."""
        list1 = [1, 2, 3]
        list2 = [1, 2, 3]
        list3 = [3, 2, 1]

        assert default_registry.hash_value(list1) == default_registry.hash_value(list2)
        assert default_registry.hash_value(list1) != default_registry.hash_value(list3)

    def test_hash_tuple(self):
        """Test tuple hashing."""
        tuple1 = (1, 2, 3)
        tuple2 = (1, 2, 3)
        tuple3 = (3, 2, 1)

        assert default_registry.hash_value(tuple1) == default_registry.hash_value(tuple2)
        assert default_registry.hash_value(tuple1) != default_registry.hash_value(tuple3)

    def test_hash_set(self):
        """Test set hashing."""
        set1 = {1, 2, 3}
        set2 = {3, 1, 2}  # Different order, same content
        set3 = {1, 2, 4}

        # Same content should have same hash (order-independent)
        assert default_registry.hash_value(set1) == default_registry.hash_value(set2)

        # Different content should have different hash
        assert default_registry.hash_value(set1) != default_registry.hash_value(set3)


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


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_deserialize_unregistered_format(self):
        """Test that deserialize raises ValueError for unregistered format."""
        registry = SerializationRegistry()

        with pytest.raises(ValueError, match="No deserialization handler registered"):
            registry.deserialize(b"test", str, format="invalid_format")

    def test_set_default_format_unregistered(self):
        """Test that set_default_format raises ValueError for unregistered format."""
        registry = SerializationRegistry()

        with pytest.raises(ValueError, match="Format 'invalid' is not registered"):
            registry.set_default_format(str, "invalid")

    def test_hash_numpy_type_suggests_plugin(self):
        """Test that numpy types suggest the numpy plugin."""
        registry = SerializationRegistry()

        # Create a mock numpy-like type
        class FakeNumpyArray:
            __module__ = "numpy"

        with pytest.raises(TypeError, match="daglite_serialization\\[numpy\\]"):
            registry.hash_value(FakeNumpyArray())

    def test_hash_pandas_type_suggests_plugin(self):
        """Test that pandas types suggest the pandas plugin."""
        registry = SerializationRegistry()

        # Create a mock pandas-like type
        class FakePandasDataFrame:
            __module__ = "pandas"

        with pytest.raises(TypeError, match="daglite_serialization\\[pandas\\]"):
            registry.hash_value(FakePandasDataFrame())

    def test_hash_pillow_type_suggests_plugin(self):
        """Test that PIL types suggest the pillow plugin."""
        registry = SerializationRegistry()

        # Create a mock PIL-like type
        class FakePILImage:
            __module__ = "PIL.Image"

        with pytest.raises(TypeError, match="daglite_serialization\\[pillow\\]"):
            registry.hash_value(FakePILImage())

    def test_hash_torch_type_suggests_plugin(self):
        """Test that torch types suggest the torch plugin."""
        registry = SerializationRegistry()

        # Create a mock torch-like type
        class FakeTorchTensor:
            __module__ = "torch"

        with pytest.raises(TypeError, match="daglite_serialization\\[torch\\]"):
            registry.hash_value(FakeTorchTensor())

    def test_get_extension_unregistered_type(self):
        """Test get_extension with unregistered type."""
        registry = SerializationRegistry()

        class CustomType:
            pass

        with pytest.raises(ValueError, match="No handler registered"):
            registry.get_extension(CustomType)

    def test_find_handler_with_subclass(self):
        """Test that _find_handler works with subclasses."""
        registry = SerializationRegistry()

        class BaseClass:
            pass

        class DerivedClass(BaseClass):
            pass

        # Register for base class
        registry.register(
            BaseClass,
            lambda obj: b"base",
            lambda data: BaseClass(),
            format="test",
            file_extension="txt",
        )

        # Should work for derived class too
        derived = DerivedClass()
        data, ext = registry.serialize(derived, format="test")
        assert data == b"base"
        assert ext == "txt"

    def test_find_hash_strategy_with_subclass(self):
        """Test that _find_hash_strategy works with subclasses."""
        registry = SerializationRegistry()

        class BaseClass:
            pass

        class DerivedClass(BaseClass):
            pass

        # Register hash strategy for base class
        registry.register_hash_strategy(
            BaseClass,
            lambda obj: "base_hash",
            "Hash for base class",
        )

        # Should work for derived class too
        derived = DerivedClass()
        hash_value = registry.hash_value(derived)
        assert hash_value == "base_hash"


class TestEdgeCases:
    """Tests for edge cases with built-in types."""

    def test_empty_dict(self):
        """Test hashing empty dictionary."""
        assert default_registry.hash_value({}) == default_registry.hash_value({})

    def test_empty_list(self):
        """Test hashing empty list."""
        assert default_registry.hash_value([]) == default_registry.hash_value([])

    def test_empty_string(self):
        """Test hashing empty string."""
        assert default_registry.hash_value("") == default_registry.hash_value("")

    def test_nested_dict(self):
        """Test hashing nested dictionary."""
        dict1 = {"a": {"b": 1, "c": 2}, "d": 3}
        dict2 = {"a": {"b": 1, "c": 2}, "d": 3}
        dict3 = {"a": {"b": 1, "c": 3}, "d": 3}

        assert default_registry.hash_value(dict1) == default_registry.hash_value(dict2)
        assert default_registry.hash_value(dict1) != default_registry.hash_value(dict3)

    def test_nested_list(self):
        """Test hashing nested list."""
        list1 = [[1, 2], [3, 4]]
        list2 = [[1, 2], [3, 4]]
        list3 = [[1, 2], [3, 5]]

        assert default_registry.hash_value(list1) == default_registry.hash_value(list2)
        assert default_registry.hash_value(list1) != default_registry.hash_value(list3)

    def test_mixed_types_in_dict(self):
        """Test hashing dictionary with mixed value types."""
        dict1 = {"a": 1, "b": "hello", "c": [1, 2, 3]}
        dict2 = {"a": 1, "b": "hello", "c": [1, 2, 3]}

        assert default_registry.hash_value(dict1) == default_registry.hash_value(dict2)

    def test_unicode_string(self):
        """Test hashing Unicode strings."""
        str1 = "Hello 世界 🌍"
        str2 = "Hello 世界 🌍"
        str3 = "Hello World"

        assert default_registry.hash_value(str1) == default_registry.hash_value(str2)
        assert default_registry.hash_value(str1) != default_registry.hash_value(str3)

    def test_recursive_hashing(self):
        """Test that hashing works recursively through collections.

        This is important for Phase 2 caching - we want to be able to hash
        complex nested structures like dict[str, np.ndarray] automatically.
        """

        # Register a custom type
        @dataclass
        class Point:
            x: int
            y: int

        registry = SerializationRegistry()
        registry.register_hash_strategy(
            Point, lambda p: registry.hash_value((p.x, p.y)), "Hash Point as tuple"
        )

        # Test nested structures
        point1 = Point(1, 2)
        point2 = Point(1, 2)
        point3 = Point(3, 4)

        # Direct hash
        assert registry.hash_value(point1) == registry.hash_value(point2)
        assert registry.hash_value(point1) != registry.hash_value(point3)

        # In a list (recursive!)
        list1 = [point1, "hello", 42]
        list2 = [point2, "hello", 42]
        list3 = [point3, "hello", 42]

        assert registry.hash_value(list1) == registry.hash_value(list2)
        assert registry.hash_value(list1) != registry.hash_value(list3)

        # In a dict (recursive!)
        dict1 = {"point": point1, "name": "A"}
        dict2 = {"point": point2, "name": "A"}
        dict3 = {"point": point3, "name": "A"}

        assert registry.hash_value(dict1) == registry.hash_value(dict2)
        assert registry.hash_value(dict1) != registry.hash_value(dict3)
