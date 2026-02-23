"""
Unit tests for daglite settings configuration.

Tests in this file should NOT focus on evaluation. Evaluation behavior tests are in tests/behavior/ and cross-subsystem scenarios are in tests/integration/.
"""

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
        """set_global_settings persists settings that can be retrieved."""
        custom = DagliteSettings(max_backend_threads=42, enable_plugin_tracing=True)
        set_global_settings(custom)
        retrieved = get_global_settings()
        assert retrieved.max_backend_threads == 42
        assert retrieved.enable_plugin_tracing is True

        # Clean up
        set_global_settings(DagliteSettings())

    def test_settings_enable_plugin_tracing(self) -> None:
        """Test enable_plugin_tracing setting."""
        settings = DagliteSettings(enable_plugin_tracing=True)
        assert settings.enable_plugin_tracing is True

        settings = DagliteSettings(enable_plugin_tracing=False)
        assert settings.enable_plugin_tracing is False


class TestSettingsEnvironmentVariables:
    """Test settings from environment variables."""

    def test_env_get_bool_true_values(self, monkeypatch) -> None:
        """Test _env_get_bool with various true values."""
        from daglite.settings import _env_get_bool

        monkeypatch.setenv("TEST_VAR", "1")
        assert _env_get_bool("TEST_VAR") is True

        monkeypatch.setenv("TEST_VAR", "true")
        assert _env_get_bool("TEST_VAR") is True

        monkeypatch.setenv("TEST_VAR", "TRUE")
        assert _env_get_bool("TEST_VAR") is True

        monkeypatch.setenv("TEST_VAR", "yes")
        assert _env_get_bool("TEST_VAR") is True

        monkeypatch.setenv("TEST_VAR", "YES")
        assert _env_get_bool("TEST_VAR") is True

    def test_env_get_bool_false_values(self, monkeypatch) -> None:
        """Test _env_get_bool with false values."""
        from daglite.settings import _env_get_bool

        monkeypatch.setenv("TEST_VAR", "0")
        assert _env_get_bool("TEST_VAR") is False

        monkeypatch.setenv("TEST_VAR", "false")
        assert _env_get_bool("TEST_VAR") is False

        monkeypatch.setenv("TEST_VAR", "no")
        assert _env_get_bool("TEST_VAR") is False

    def test_env_get_bool_default(self) -> None:
        """Test _env_get_bool returns default when not set."""
        from daglite.settings import _env_get_bool

        # Use a unique name unlikely to be set
        assert _env_get_bool("DAGLITE_NONEXISTENT_VAR_123", default=False) is False
        assert _env_get_bool("DAGLITE_NONEXISTENT_VAR_123", default=True) is True

        custom_settings = DagliteSettings(
            max_backend_threads=16,
            max_parallel_processes=8,
        )
        set_global_settings(custom_settings)

        retrieved = get_global_settings()
        assert retrieved.max_backend_threads == 16
        assert retrieved.max_parallel_processes == 8

        # Cleanup: reset to defaults for other tests
        set_global_settings(DagliteSettings())

    def test_settings_persist_across_calls(self) -> None:
        """Global settings persist across multiple get_global_settings calls."""

        custom_settings = DagliteSettings(max_backend_threads=24)
        set_global_settings(custom_settings)

        settings1 = get_global_settings()
        settings2 = get_global_settings()

        assert settings1.max_backend_threads == 24
        assert settings2.max_backend_threads == 24
        assert settings1 is settings2  # Same instance

        # Cleanup
        set_global_settings(DagliteSettings())
