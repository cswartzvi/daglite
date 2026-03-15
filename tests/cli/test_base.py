"""Tests for CLI base functionality."""

from __future__ import annotations

import pytest

from daglite.cli.core import main


class TestCLIBase:
    """Tests for basic CLI functionality."""

    def test_cli_help(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "Daglite" in out
        assert "run" in out

    def test_cli_version(self, capsys):
        import daglite

        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert daglite.__version__ in out

    def test_cli_no_command(self, capsys):
        with pytest.raises(SystemExit):
            main([])
        out = capsys.readouterr().out
        assert "Usage:" in out

    def test_cli_invalid_command(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["invalid"])
        assert exc_info.value.code != 0
