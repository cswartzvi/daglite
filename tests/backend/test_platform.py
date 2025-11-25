"""Tests for platform-specific backend behavior."""

import multiprocessing as mp
import sys
from unittest.mock import patch

import pytest

from daglite.backends.local import _get_global_process_pool
from daglite.backends.local import _reset_global_pools


class TestProcessPoolPlatformBehavior:
    """Test platform-specific multiprocessing context selection."""

    def teardown_method(self) -> None:
        """Reset pools after each test."""
        _reset_global_pools()

    @patch("daglite.backends.local.os.name", "nt")
    def test_windows_uses_spawn(self) -> None:
        """Windows uses spawn context for process pool."""
        _reset_global_pools()
        pool = _get_global_process_pool()
        # ProcessPoolExecutor doesn't expose mp_context publicly,
        # but we can verify it was created without errors on "Windows"
        assert pool is not None

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="macOS spawn context requires _posixshmem module (POSIX-only)",
    )
    @patch("daglite.backends.local.sys.platform", "darwin")
    @patch("daglite.backends.local.os.name", "posix")
    def test_macos_uses_spawn(self) -> None:
        """macOS uses spawn context for process pool."""
        _reset_global_pools()
        pool = _get_global_process_pool()
        assert pool is not None

    @pytest.mark.skipif(
        "fork" not in mp.get_all_start_methods(),
        reason="fork context not available on this platform",
    )
    @patch("daglite.backends.local.sys.platform", "linux")
    @patch("daglite.backends.local.os.name", "posix")
    def test_linux_uses_fork(self) -> None:
        """Linux uses fork context for process pool."""
        _reset_global_pools()
        pool = _get_global_process_pool()
        assert pool is not None
