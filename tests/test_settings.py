"""Tests for daglite settings configuration."""

from daglite.settings import DagliteSettings
from daglite.settings import get_global_settings
from daglite.settings import set_global_settings


class TestDagliteSettings:
    """Test DagliteSettings dataclass."""

    def test_default_settings_has_values(self) -> None:
        """Default settings have non-None computed values."""
        settings = DagliteSettings()
        assert settings.max_backend_threads > 0
        assert settings.max_parallel_processes > 0

    def test_custom_thread_settings(self) -> None:
        """Custom thread pool size is respected."""
        settings = DagliteSettings(max_backend_threads=10)
        assert settings.max_backend_threads == 10

    def test_custom_process_settings(self) -> None:
        """Custom process pool size is respected."""
        settings = DagliteSettings(max_parallel_processes=4)
        assert settings.max_parallel_processes == 4

    def test_settings_immutable(self) -> None:
        """Settings are frozen (immutable)."""
        import pytest

        settings = DagliteSettings()
        with pytest.raises(Exception):  # FrozenInstanceError in dataclasses
            settings.max_backend_threads = 100  # type: ignore


class TestGlobalSettings:
    """Test global settings management."""

    def test_get_global_settings_returns_default(self) -> None:
        """get_global_settings returns default instance if not set."""
        # Note: This test assumes no prior set_global_settings call
        # In practice, settings persist across tests due to global state
        settings = get_global_settings()
        assert isinstance(settings, DagliteSettings)

    def test_set_and_get_global_settings(self) -> None:
        """set_global_settings stores and retrieves custom settings."""
        from daglite.backends.local import _reset_global_pools

        custom_settings = DagliteSettings(
            max_backend_threads=16,
            max_parallel_processes=8,
        )
        set_global_settings(custom_settings)
        _reset_global_pools()  # Reset pools so they pick up new settings

        retrieved = get_global_settings()
        assert retrieved.max_backend_threads == 16
        assert retrieved.max_parallel_processes == 8

        # Cleanup: reset to defaults for other tests
        set_global_settings(DagliteSettings())
        _reset_global_pools()

    def test_settings_persist_across_calls(self) -> None:
        """Global settings persist across multiple get_global_settings calls."""
        from daglite.backends.local import _reset_global_pools

        custom_settings = DagliteSettings(max_backend_threads=24)
        set_global_settings(custom_settings)
        _reset_global_pools()

        settings1 = get_global_settings()
        settings2 = get_global_settings()

        assert settings1.max_backend_threads == 24
        assert settings2.max_backend_threads == 24
        assert settings1 is settings2  # Same instance

        # Cleanup
        set_global_settings(DagliteSettings())
        _reset_global_pools()
