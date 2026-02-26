"""Unit tests for FileDriver.

Tests in this file should NOT use evaluate(). Evaluation behavior tests are in tests/behavior/ and
cross-subsystem scenarios are in tests/integration/.
"""

import pickle
import tempfile
from pathlib import Path

import pytest

from daglite.drivers import FileDriver


class TestFileDriverOperations:
    """Tests for FileDriver (raw byte storage)."""

    def test_init_with_string_path(self):
        """Test initialization with string path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            driver = FileDriver(tmpdir)
            assert driver.base_path == tmpdir
            assert driver.fs is not None

    def test_init_creates_directory(self):
        """Test that initialization creates the base directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = f"{tmpdir}/nested/path"
            _ = FileDriver(nested_path)
            assert Path(nested_path).exists()

    def test_save_and_load_bytes(self):
        """Test saving and loading raw bytes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            driver = FileDriver(tmpdir)

            # Save raw bytes
            driver.save("key1.bin", b"hello world")
            driver.save("key2.bin", b"\x00\x01\x02\xff")

            # Load and verify
            data1 = driver.load("key1.bin")
            assert data1 == b"hello world"

            data2 = driver.load("key2.bin")
            assert data2 == b"\x00\x01\x02\xff"

    def test_save_creates_exact_filename(self):
        """Test that save creates the exact filename given."""
        with tempfile.TemporaryDirectory() as tmpdir:
            driver = FileDriver(tmpdir)

            driver.save("output.csv", b"a,b,c")
            assert (Path(tmpdir) / "output.csv").exists()
            assert (Path(tmpdir) / "output.csv").read_bytes() == b"a,b,c"

    def test_save_creates_nested_directories(self):
        """Test that save creates nested directories for keys with paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            driver = FileDriver(tmpdir)

            driver.save("subdir/nested/file.txt", b"content")
            data = driver.load("subdir/nested/file.txt")
            assert data == b"content"
            # Check that nested directory structure exists
            assert (Path(tmpdir) / "subdir" / "nested" / "file.txt").exists()

    def test_save_returns_path(self):
        """Test that save returns the path where data was stored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            driver = FileDriver(tmpdir)

            path = driver.save("mykey.bin", b"content")
            assert path == f"{tmpdir}/mykey.bin"

    def test_exists(self):
        """Test exists method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            driver = FileDriver(tmpdir)

            assert not driver.exists("nonexistent")

            driver.save("existing.bin", b"value")
            assert driver.exists("existing.bin")

    def test_load_nonexistent_raises_keyerror(self):
        """Test that loading nonexistent key raises KeyError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            driver = FileDriver(tmpdir)

            with pytest.raises(KeyError, match="not found"):
                driver.load("nonexistent")

    def test_serialization_for_distributed_backends(self):
        """Test that FileDriver can be pickled (for Ray, Dask, etc.)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            driver = FileDriver(tmpdir)

            # Pickle and unpickle
            pickled = pickle.dumps(driver)
            restored = pickle.loads(pickled)

            # Verify it still works
            assert restored.base_path == tmpdir
            restored.save("test.bin", b"value")
            data = restored.load("test.bin")
            assert data == b"value"

    def test_delete(self):
        """Test delete method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            driver = FileDriver(tmpdir)

            driver.save("temp.bin", b"value")
            assert driver.exists("temp.bin")
            assert (Path(tmpdir) / "temp.bin").exists()

            driver.delete("temp.bin")
            assert not driver.exists("temp.bin")
            assert not (Path(tmpdir) / "temp.bin").exists()

    def test_delete_nested_key(self):
        """Test delete with nested paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            driver = FileDriver(tmpdir)

            driver.save("dir/subdir/file.txt", b"value")
            assert driver.exists("dir/subdir/file.txt")

            driver.delete("dir/subdir/file.txt")
            assert not driver.exists("dir/subdir/file.txt")

    def test_delete_nonexistent_is_safe(self):
        """Test that deleting nonexistent key doesn't raise error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            driver = FileDriver(tmpdir)

            # Should not raise
            driver.delete("nonexistent")

    def test_list_keys(self):
        """Test list_keys method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            driver = FileDriver(tmpdir)

            # Empty driver
            assert driver.list_keys() == []

            # Add some keys
            driver.save("key1.bin", b"value1")
            driver.save("key2.bin", b"value2")
            driver.save("key3.bin", b"value3")

            keys = driver.list_keys()
            assert sorted(keys) == ["key1.bin", "key2.bin", "key3.bin"]

    def test_list_keys_raises_no_error_on_nonexistent_directory(self):
        """Test that list_keys does not raise error if base directory doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create driver but delete the directory
            driver = FileDriver(f"{tmpdir}/nonexistent")
            import shutil

            shutil.rmtree(f"{tmpdir}/nonexistent")

            # Should return empty list, not raise
            assert driver.list_keys() == []

    def test_list_keys_empty_directory(self):
        """Test list_keys on nonexistent directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create driver but delete the directory
            driver = FileDriver(f"{tmpdir}/nonexistent")
            import shutil

            shutil.rmtree(f"{tmpdir}/nonexistent")

            # Should return empty list, not raise
            assert driver.list_keys() == []

    def test_get_format_hint(self):
        """Test get_format_hint method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            driver = FileDriver(tmpdir)

            # Returns extension for files with extensions
            assert driver.get_format_hint("data.bin") == "bin"
            assert driver.get_format_hint("notes.txt") == "txt"
            assert driver.get_format_hint("output.csv") == "csv"
            assert driver.get_format_hint("archive.tar.gz") == "gz"

            # Returns None for files without extensions
            assert driver.get_format_hint("README") is None
            assert driver.get_format_hint("Makefile") is None

            # Handles nested paths correctly
            assert driver.get_format_hint("path/to/data.json") == "json"
            assert driver.get_format_hint("path/to/file") is None

            # Handles Windows-style paths (normalized internally)
            assert driver.get_format_hint("path\\to\\data.parquet") == "parquet"


class TestFileDriverPathResolution:
    """Tests for absolute vs relative key path resolution."""

    def test_relative_key_uses_base_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            driver = FileDriver(tmpdir)

            saved_path = driver.save("relative/data.bin", b"value")

            assert saved_path == f"{tmpdir}/relative/data.bin"
            assert driver.exists("relative/data.bin")
            assert driver.load("relative/data.bin") == b"value"

            driver.delete("relative/data.bin")
            assert not driver.exists("relative/data.bin")

    def test_absolute_posix_key_bypasses_base_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with tempfile.TemporaryDirectory() as other:
                driver = FileDriver(tmpdir)
                absolute_path = f"{other}/absolute.bin"

                saved_path = driver.save(absolute_path, b"value")

                assert saved_path == absolute_path
                assert Path(absolute_path).exists()
                assert Path(absolute_path).read_bytes() == b"value"

                assert driver.exists(absolute_path)
                assert driver.load(absolute_path) == b"value"

                driver.delete(absolute_path)
                assert not driver.exists(absolute_path)
                assert not Path(absolute_path).exists()

    def test_windows_absolute_key_bypasses_base_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            driver = FileDriver(tmpdir)

            windows_key = "C:/tmp/windows-absolute.bin"
            assert driver._full_path(windows_key) == windows_key

    def test_local_uri_key_bypasses_base_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with tempfile.TemporaryDirectory() as other:
                driver = FileDriver(tmpdir)
                uri_key = f"file://{other}/uri.bin"

                saved_path = driver.save(uri_key, b"value")

                assert saved_path == uri_key
                assert Path(other, "uri.bin").read_bytes() == b"value"
                assert driver.exists(uri_key)
                assert driver.load(uri_key) == b"value"

                driver.delete(uri_key)
                assert not driver.exists(uri_key)

    def test_remote_uri_protocol_mismatch_raises(self):
        from unittest.mock import MagicMock

        mock_fs = MagicMock()
        mock_fs.exists.return_value = False

        driver = FileDriver("/tmp/base", fs=mock_fs)

        with pytest.raises(ValueError, match="does not match"):
            driver.save("s3://bucket/key.bin", b"value")

    def test_remote_driver_accepts_matching_uri_keys(self):
        from unittest.mock import MagicMock

        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.__exit__.return_value = None

        mock_fs = MagicMock()
        mock_fs.open.return_value = mock_file
        mock_fs.exists.return_value = True
        mock_fs.protocol = "s3"

        driver = FileDriver("s3://bucket/base", fs=mock_fs)

        key = "s3://bucket/explicit-target.bin"
        driver.save(key, b"value")
        driver.load(key)
        assert driver.exists(key)
        driver.delete(key)

        mock_fs.open.assert_any_call(key, "wb")
        mock_fs.open.assert_any_call(key, "rb")
        mock_fs.rm.assert_called_once_with(key)

    def test_remote_driver_rejects_local_uri_keys(self):
        from unittest.mock import MagicMock

        mock_fs = MagicMock()
        mock_fs.exists.return_value = False

        driver = FileDriver("s3://bucket/base", fs=mock_fs)

        with pytest.raises(ValueError, match="does not match"):
            driver.save("file:///tmp/data.bin", b"value")


class TestFileDriverFsspecSupport:
    """Tests for fsspec protocol support."""

    def test_init_with_custom_fs(self):
        """Test initialization with custom filesystem."""
        from fsspec import filesystem

        with tempfile.TemporaryDirectory() as tmpdir:
            custom_fs = filesystem("file")
            driver = FileDriver(tmpdir, fs=custom_fs)

            # Verify it works
            driver.save("test.bin", b"value")
            data = driver.load("test.bin")
            assert data == b"value"

    def test_file_protocol_explicit(self):
        """Test explicit file:// protocol."""
        with tempfile.TemporaryDirectory() as tmpdir:
            driver = FileDriver(f"file://{tmpdir}")
            driver.save("test.bin", b"value")
            data = driver.load("test.bin")
            assert data == b"value"

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
            driver = FileDriver(path)

            driver.save("test_value.bin", b"\x01\x02\x03")
            data = driver.load("test_value.bin")
            assert data == b"\x01\x02\x03"


class TestFileDriverIsLocal:
    """Tests for FileDriver.is_local property."""

    def test_local_path_is_local(self):
        """Plain local path reports is_local=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            driver = FileDriver(tmpdir)
            assert driver.is_local is True

    def test_file_protocol_is_local(self):
        """Explicit file:// protocol reports is_local=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            driver = FileDriver(f"file://{tmpdir}")
            assert driver.is_local is True

    def test_remote_protocol_is_not_local(self):
        """Remote protocol (e.g. s3://) reports is_local=False."""
        from unittest.mock import MagicMock

        mock_fs = MagicMock()
        driver = FileDriver("s3://my-bucket/prefix", fs=mock_fs)
        assert driver.is_local is False
