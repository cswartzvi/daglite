"""Tests for CLI base functionality."""

from __future__ import annotations

from click.testing import CliRunner

from daglite.cli import cli


class TestCLIBase:
    """Tests for basic CLI functionality."""

    def test_cli_help(self):
        """Test that --help works."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Daglite" in result.output
        assert "run" in result.output

    def test_cli_version(self):
        """Test that --version works."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.0.1" in result.output

    def test_cli_no_command(self):
        """Test CLI with no command shows help."""
        runner = CliRunner()
        result = runner.invoke(cli, [])
        # Click shows usage info which exits with code 0
        # Note: actually exits with 2 in click when no command provided
        assert "Usage:" in result.output

    def test_cli_invalid_command(self):
        """Test CLI with invalid command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["invalid"])
        assert result.exit_code != 0
        assert "No such command" in result.output
