"""Tests for backend discovery and registration."""

import pytest

from daglite.backends import find_backend
from daglite.backends.base import Backend
from daglite.backends.local import SequentialBackend
from daglite.backends.local import ThreadBackend
from daglite.exceptions import BackendError


class TestFindBackend:
    """Test backend discovery via find_backend()."""

    def test_find_sequential_by_name(self) -> None:
        """find_backend() returns SequentialBackend for 'sequential'."""
        backend = find_backend("sequential")
        assert isinstance(backend, SequentialBackend)

    def test_find_synchronous_alias(self) -> None:
        """find_backend() accepts 'synchronous' as alias for sequential."""
        backend = find_backend("synchronous")
        assert isinstance(backend, SequentialBackend)

    def test_find_threading_by_name(self) -> None:
        """find_backend() returns ThreadBackend for 'threading'."""
        backend = find_backend("threading")
        assert isinstance(backend, ThreadBackend)

    def test_find_backend_none_defaults_to_sequential(self) -> None:
        """find_backend(None) returns default SequentialBackend."""
        backend = find_backend(None)
        assert isinstance(backend, SequentialBackend)

    def test_find_backend_with_instance_returns_instance(self) -> None:
        """find_backend() returns backend instance unchanged."""
        original = ThreadBackend()
        backend = find_backend(original)
        assert backend is original

    def test_find_backend_unknown_raises_error(self) -> None:
        """find_backend() raises BackendError for unknown backend name."""
        with pytest.raises(BackendError, match="Unknown backend 'nonexistent'"):
            find_backend("nonexistent")

    def test_find_backend_error_lists_available(self) -> None:
        """BackendError message lists available backends."""
        with pytest.raises(BackendError, match="available:"):
            find_backend("invalid")

    @pytest.mark.parametrize(
        "backend_name,expected_type",
        [
            ("sequential", SequentialBackend),
            ("synchronous", SequentialBackend),
            ("threading", ThreadBackend),
        ],
    )
    def test_find_backend_types(self, backend_name: str, expected_type: type[Backend]) -> None:
        """Parameterized test for all backend types."""
        backend = find_backend(backend_name)
        assert isinstance(backend, expected_type)

    def test_find_backend_returns_new_instances(self) -> None:
        """find_backend() returns new instances each time."""
        backend1 = find_backend("sequential")
        backend2 = find_backend("sequential")
        assert backend1 is not backend2
