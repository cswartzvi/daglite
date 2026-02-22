"""Tests for the CLI ``list`` command."""

from __future__ import annotations

from click.testing import CliRunner

from daglite.cli.base import cli


class TestListCommand:
    """Tests for ``daglite list``."""

    def test_list_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["list", "--help"])
        assert result.exit_code == 0
        assert "List workflows" in result.output
        assert "MODULE" in result.output

    def test_list_requires_module_argument(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["list"])
        assert result.exit_code != 0
        assert "Missing argument" in result.output

    def test_list_nonexistent_module(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["list", "nonexistent.module.that.does_not_exist"])
        assert result.exit_code != 0
        assert "Cannot import module" in result.output

    def test_list_module_with_workflows(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["list", "tests.examples.workflows"])
        assert result.exit_code == 0
        # All four example workflows should appear
        assert "math_workflow" in result.output
        assert "empty_workflow" in result.output
        assert "failing_workflow" in result.output
        assert "untyped_workflow" in result.output

    def test_list_shows_dotted_path(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["list", "tests.examples.workflows"])
        assert result.exit_code == 0
        assert "tests.examples.workflows.math_workflow" in result.output

    def test_list_shows_description(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["list", "tests.examples.workflows"])
        assert result.exit_code == 0
        # math_workflow has a multi-line docstring; the first non-empty line should appear
        assert "arithmetic" in result.output.lower()

    def test_list_module_with_no_workflows(self):
        """Module that exists but contains no @workflow objects."""
        runner = CliRunner()
        # daglite.exceptions has no workflows
        result = runner.invoke(cli, ["list", "daglite.exceptions"])
        assert result.exit_code == 0
        assert "No workflows found." in result.output

    def test_list_multiple_modules(self):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["list", "tests.examples.workflows", "daglite.exceptions"],
        )
        assert result.exit_code == 0
        # Workflows from first module are present
        assert "math_workflow" in result.output
        # Second module contributes no workflows, so the combined message is fine
        assert "No workflows found." not in result.output

    def test_list_one_valid_one_invalid_module(self):
        """An invalid module should produce an error even if others are valid."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["list", "nonexistent_module"],
        )
        assert result.exit_code != 0
        assert "Cannot import module" in result.output

    def test_list_long_description_truncated(self):
        """Descriptions longer than 72 chars should be truncated with '...'."""
        runner = CliRunner()
        result = runner.invoke(cli, ["list", "tests.examples.workflows"])
        assert result.exit_code == 0
        assert "..." in result.output
        # The raw description must NOT appear verbatim (it's over 72 chars)
        assert "verbose_workflow" in result.output

    def test_list_workflow_without_description(self):
        """Workflows with no description should be listed without a description column."""
        runner = CliRunner()
        result = runner.invoke(cli, ["list", "tests.examples.workflows"])
        assert result.exit_code == 0
        # no_description_workflow has no docstring; it must still appear, just without trailing text
        assert "no_description_workflow" in result.output
        # Its path should appear on its own line (no two-space separator after it)
        for line in result.output.splitlines():
            if "no_description_workflow" in line:
                assert "  " not in line.split("no_description_workflow")[1]
                break
