"""Unit tests for FileOutputStore.

Tests in this file should NOT use evaluate(). Evaluation tests are in tests/evaluation/.
"""

import pickle
import tempfile
from pathlib import Path

import pytest

from daglite.outputs.store import FileOutputStore


class TestFileOutputStore:
    """Tests for FileOutputStore."""

    def test_init_with_string_path(self):
        """Test initialization with string path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileOutputStore(tmpdir)
            assert store.base_path == tmpdir
            assert store.fs is not None
            assert store.registry is not None

    def test_init_creates_directory(self):
        """Test that initialization creates the base directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = f"{tmpdir}/nested/path"
            _ = FileOutputStore(nested_path)
            assert Path(nested_path).exists()

    def test_save_and_load_basic_types(self):
        """Test saving and loading basic Python types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileOutputStore(tmpdir)

            # Save various types
            store.save("int_value", 42)
            store.save("str_value", "hello")
            store.save("float_value", 3.14)
            store.save("list_value", [1, 2, 3])

            # Load and verify
            assert store.load("int_value", int) == 42
            assert store.load("str_value", str) == "hello"
            assert store.load("float_value", float) == 3.14
            assert store.load("list_value", list) == [1, 2, 3]

    def test_save_with_extension_in_key(self):
        """Test that keys with extensions are preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileOutputStore(tmpdir)

            store.save("data.txt", "content")
            assert store.load("data", str) == "content"  # Load without extension

            # Check file exists with correct name
            assert (Path(tmpdir) / "data.txt").exists()

    def test_save_without_extension_adds_extension(self):
        """Test that keys without extensions get automatic extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileOutputStore(tmpdir)

            store.save("mydata", "content")
            assert store.load("mydata", str) == "content"

            # Check that file has extension
            files = list(Path(tmpdir).glob("mydata.*"))
            assert len(files) == 1
            assert files[0].suffix in [".txt", ".pkl"]

    def test_save_creates_nested_directories(self):
        """Test that save creates nested directories for keys with paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileOutputStore(tmpdir)

            store.save("subdir/nested/file", "content")
            assert store.load("subdir/nested/file", str) == "content"
            # Check that nested directory structure exists
            assert (Path(tmpdir) / "subdir" / "nested").exists()

    def test_exists(self):
        """Test exists method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileOutputStore(tmpdir)

            assert not store.exists("nonexistent")

            store.save("existing", "value")
            assert store.exists("existing")

    def test_load_nonexistent_raises_keyerror(self):
        """Test that loading nonexistent key raises KeyError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileOutputStore(tmpdir)

            with pytest.raises(KeyError, match="not found"):
                store.load("nonexistent", str)

    def test_load_multiple_files_raises_valueerror(self):
        """Test that multiple files with same key raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileOutputStore(tmpdir)

            # Manually create multiple files with same prefix
            (Path(tmpdir) / "data.txt").write_text("text")
            (Path(tmpdir) / "data.pkl").write_bytes(b"pickle")

            with pytest.raises(ValueError, match="Multiple files found"):
                store.load("data", str)

    def test_serialization_for_distributed_backends(self):
        """Test that FileOutputStore can be pickled (for Ray, Dask, etc.)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileOutputStore(tmpdir)

            # Pickle and unpickle
            pickled = pickle.dumps(store)
            restored = pickle.loads(pickled)

            # Verify it still works
            assert restored.base_path == tmpdir
            restored.save("test", "value")
            assert restored.load("test", str) == "value"

    def test_format_parameter(self):
        """Test explicit format parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileOutputStore(tmpdir)

            # Save with explicit format
            store.save("data", [1, 2, 3], format="pickle")
            assert store.load("data", list) == [1, 2, 3]

    def test_delete(self):
        """Test delete method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileOutputStore(tmpdir)

            store.save("temp", "value")
            assert store.exists("temp")

            store.delete("temp")
            assert not store.exists("temp")

    def test_delete_nested_key(self):
        """Test delete with nested paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileOutputStore(tmpdir)

            store.save("dir/subdir/file", "value")
            assert store.exists("dir/subdir/file")

            store.delete("dir/subdir/file")
            assert not store.exists("dir/subdir/file")

    def test_delete_nonexistent_is_safe(self):
        """Test that deleting nonexistent key doesn't raise error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileOutputStore(tmpdir)

            # Should not raise
            store.delete("nonexistent")

    def test_list_keys(self):
        """Test list_keys method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileOutputStore(tmpdir)

            # Empty store
            assert store.list_keys() == []

            # Add some keys
            store.save("key1", "value1")
            store.save("key2", 123)
            store.save("key3", [1, 2, 3])

            keys = store.list_keys()
            assert sorted(keys) == ["key1", "key2", "key3"]

    def test_list_keys_empty_directory(self):
        """Test list_keys on nonexistent directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create store but delete the directory
            store = FileOutputStore(f"{tmpdir}/nonexistent")
            import shutil

            shutil.rmtree(f"{tmpdir}/nonexistent")

            # Should return empty list, not raise
            assert store.list_keys() == []

    def test_init_with_custom_registry_and_fs(self):
        """Test initialization with custom registry and filesystem."""
        from fsspec import filesystem

        from daglite.serialization import default_registry

        with tempfile.TemporaryDirectory() as tmpdir:
            custom_fs = filesystem("file")
            store = FileOutputStore(tmpdir, registry=default_registry, fs=custom_fs)

            # Verify it works
            store.save("test", "value")
            assert store.load("test", str) == "value"

    def test_load_without_type_hint_for_pickle(self):
        """Test loading pickle files without return_type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileOutputStore(tmpdir)

            # Save as pickle
            store.save("data", [1, 2, 3], format="pickle")

            # Load without type hint should work for pickle
            result = store.load("data")
            assert result == [1, 2, 3]

    def test_list_keys_with_files_without_extension(self):
        """Test list_keys with files that have no extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileOutputStore(tmpdir)

            # Manually create a file without extension
            (Path(tmpdir) / "noext").write_text("data")

            # Should still list it
            keys = store.list_keys()
            assert "noext" in keys

    def test_load_with_unregistered_extension_fallback(self):
        """Test load works correctly even with various extensions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileOutputStore(tmpdir)

            # Save data with explicit format
            store.save("data", [1, 2, 3], format="pickle")

            # Load should work fine
            result = store.load("data", list)
            assert result == [1, 2, 3]

    def test_load_without_type_hint_non_pickle_raises(self):
        """Test that loading non-pickle format without type hint raises clear error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileOutputStore(tmpdir)

            # Save as text format (int)
            store.save("number", 42)

            # Load without type hint should fail for non-pickle
            with pytest.raises(ValueError, match="return_type is required"):
                store.load("number")

    def test_delete_with_non_nested_key(self):
        """Test delete with simple (non-nested) key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileOutputStore(tmpdir)

            # Save and delete simple key
            store.save("simple", "value")
            assert store.exists("simple")

            store.delete("simple")
            assert not store.exists("simple")

    def test_save_without_subdirectory(self):
        """Test saving a file at root level (no parent directory creation)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileOutputStore(tmpdir)
            # Save at root - triggers parent='' case
            store.save("rootfile.txt", "content", format="text")
            assert store.load("rootfile", str) == "content"

    def test_load_from_nonexistent_subdirectory(self):
        """Test loading from non-existent subdirectory path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileOutputStore(tmpdir)
            # Try loading from path with non-existent parent directory
            with pytest.raises(KeyError):
                store.load("missing_dir/file", str)

    def test_delete_from_nonexistent_subdirectory(self):
        """Test deleting from non-existent subdirectory doesn't raise error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileOutputStore(tmpdir)
            # Should not raise even if parent directory doesn't exist
            store.delete("missing_dir/file")

    def test_exists_in_nonexistent_subdirectory(self):
        """Test checking existence in non-existent subdirectory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileOutputStore(tmpdir)
            # Should return False without raising error
            assert not store.exists("missing_dir/file")


class TestFileOutputStoreProtocols:
    """Tests for fsspec protocol support."""

    def test_file_protocol_explicit(self):
        """Test explicit file:// protocol."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileOutputStore(f"file://{tmpdir}")
            store.save("test", "value")
            assert store.load("test", str) == "value"

    @pytest.mark.parametrize(
        "path_format",
        [
            "{tmpdir}",
            "file://{tmpdir}",
        ],
    )
    def test_local_filesystem_paths(self, path_format):
        """Test various local filesystem path formats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = path_format.format(tmpdir=tmpdir)
            store = FileOutputStore(path)

            store.save("test_value", 123)
            assert store.load("test_value", int) == 123
