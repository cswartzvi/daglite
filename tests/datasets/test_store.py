"""Unit tests for DatasetStore."""

import pickle
import tempfile

import pytest

from daglite.datasets.store import DatasetStore
from daglite.drivers import FileDriver


class TestDatasetStoreInit:
    """Tests for DatasetStore initialization."""

    def test_init_with_string_creates_file_driver(self):
        """String path creates a FileDriver internally."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)
            assert isinstance(store._driver, FileDriver)

    def test_init_with_driver_instance(self):
        """Accepts a Driver instance directly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            driver = FileDriver(tmpdir)
            store = DatasetStore(driver)
            assert store._driver is driver

    def test_base_path_from_file_driver(self):
        """base_path returns the underlying driver's base_path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)
            assert store.base_path == tmpdir

    def test_base_path_from_custom_driver(self):
        """base_path returns empty string when driver has no base_path."""
        from daglite.drivers.base import Driver

        class InMemoryDriver(Driver):
            def __init__(self):
                self._data: dict[str, bytes] = {}

            def save(self, key, data):
                self._data[key] = data
                return key

            def load(self, key):
                return self._data[key]

            def exists(self, key):
                return key in self._data  # pragma: no cover

            def delete(self, key):
                self._data.pop(key, None)  # pragma: no cover

            def list_keys(self):
                return list(self._data.keys())  # pragma: no cover

        store = DatasetStore(InMemoryDriver())
        assert store.base_path == ""

    def test_is_local_default_true(self):
        """is_local is True for FileDriver."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)
            assert store.is_local is True


class TestDatasetStoreSave:
    """Tests for DatasetStore.save()."""

    def test_save_string_infers_text(self):
        """Saving a string to a .txt key uses TextDataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)
            store.save("hello.txt", "hello world")
            data = store._driver.load("hello.txt")
            assert data == b"hello world"

    def test_save_dict_infers_pickle(self):
        """Saving a dict to a .pkl key uses DictPickleDataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)
            store.save("data.pkl", {"a": 1})
            data = store._driver.load("data.pkl")
            assert pickle.loads(data) == {"a": 1}

    def test_save_explicit_format(self):
        """Explicit format overrides extension inference."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)
            store.save("data.bin", "text content", format="text")
            data = store._driver.load("data.bin")
            assert data == b"text content"

    def test_save_returns_path(self):
        """save() returns the actual path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)
            path = store.save("output.txt", "value")
            assert path.endswith("output.txt")

    def test_save_raw_bytes(self):
        """Saving bytes to a .bin key uses BytesDataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)
            store.save("raw.bin", b"\x00\x01\xff")
            data = store._driver.load("raw.bin")
            assert data == b"\x00\x01\xff"

    def test_save_with_options(self):
        """Options are passed through to the dataset serialize method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)
            # Options are passed but PickleDataset ignores them - just verify no crash
            store.save("data.pkl", {"a": 1}, options={"protocol": 2})

    def test_save_nested_key(self):
        """Save with a nested path creates subdirectories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)
            store.save("sub/dir/data.txt", "nested")
            assert store._driver.load("sub/dir/data.txt") == b"nested"


class TestDatasetStoreLoad:
    """Tests for DatasetStore.load()."""

    def test_load_with_return_type(self):
        """Load with return_type uses the correct dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)
            store.save("hello.txt", "hello world")
            result = store.load("hello.txt", return_type=str)
            assert result == "hello world"

    def test_load_without_return_type_uses_pickle(self):
        """Load without return_type falls back to pickle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)
            # Save as pickle
            store._driver.save("data.pkl", pickle.dumps({"x": 42}))
            result = store.load("data.pkl")
            assert result == {"x": 42}

    def test_load_dict(self):
        """Load dict roundtrips correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)
            store.save("data.pkl", {"a": 1, "b": 2})
            result = store.load("data.pkl", return_type=dict)
            assert result == {"a": 1, "b": 2}

    def test_load_nonexistent_raises(self):
        """Loading a non-existent key raises KeyError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)
            with pytest.raises(KeyError):
                store.load("nonexistent.pkl")

    def test_load_with_options(self):
        """Options are passed through to deserialize."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)
            store.save("data.pkl", [1, 2, 3])
            result = store.load("data.pkl", return_type=list, options={"extra": True})
            assert result == [1, 2, 3]


class TestDatasetStoreOperations:
    """Tests for exists, delete, list_keys."""

    def test_exists_true(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)
            store.save("key.txt", "value")
            assert store.exists("key.txt")

    def test_exists_false(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)
            assert not store.exists("no_such_key.txt")

    def test_delete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)
            store.save("key.txt", "value")
            assert store.exists("key.txt")
            store.delete("key.txt")
            assert not store.exists("key.txt")

    def test_list_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)
            store.save("a.txt", "alpha")
            store.save("b.txt", "beta")
            keys = store.list_keys()
            assert sorted(keys) == ["a.txt", "b.txt"]


class TestDatasetStorePickle:
    """Tests for DatasetStore pickling support."""

    def test_pickle_roundtrip(self):
        """DatasetStore can be pickled and unpickled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)
            store.save("data.txt", "hello")

            # Pickle and restore
            pickled = pickle.dumps(store)
            restored = pickle.loads(pickled)

            assert restored.load("data.txt", return_type=str) == "hello"

    def test_getstate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)
            state = store.__getstate__()
            assert "driver" in state

    def test_setstate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            driver = FileDriver(tmpdir)
            store = DatasetStore.__new__(DatasetStore)
            store.__setstate__({"driver": driver})
            assert store._driver is driver
