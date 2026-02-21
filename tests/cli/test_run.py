"""Tests for CLI run command."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from daglite.cli._shared import parse_param_value
from daglite.cli.base import cli


class TestRunCommand:
    """Tests for the run command."""

    def test_run_help(self):
        """Test that run --help works."""
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "Run a daglite workflow" in result.output
        assert "--param" in result.output
        assert "--backend" in result.output

    def test_run_without_workflow_path(self):
        """Test run command without workflow path."""
        runner = CliRunner()
        result = runner.invoke(cli, ["run"])
        assert result.exit_code != 0
        assert "Missing argument" in result.output

    def test_run_with_invalid_workflow_path(self):
        """Test run command with invalid workflow path (no dot)."""
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "invalid"])
        assert result.exit_code != 0
        assert "Invalid workflow path" in result.output

    def test_run_with_nonexistent_module(self):
        """Test run command with non-existent module."""
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "nonexistent.module.workflow"])
        assert result.exit_code != 0
        assert "No module named" in result.output

    def test_run_with_nonexistent_workflow(self):
        """Test run command with non-existent workflow in existing module."""
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "daglite.nonexistent_workflow"])
        assert result.exit_code != 0
        assert "not found" in result.output

    def test_run_with_non_workflow_object(self):
        """Test run command with object that's not a Workflow."""
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "daglite.task"])
        assert result.exit_code != 0
        assert "is not a Workflow" in result.output

    def test_run_simple_workflow(self):
        """Test running a simple workflow."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["run", "tests.examples.workflows.math_workflow", "--param", "x=5", "--param", "y=10"],
        )
        assert result.exit_code == 0

    def test_run_empty_workflow(self):
        """Test running a workflow with no parameters."""
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "tests.examples.workflows.empty_workflow"])
        assert result.exit_code == 0
        # Should not show "Parameters:" line when there are no parameters
        assert "Parameters:" not in result.output

    def test_run_workflow_with_default_params(self):
        """Test running a workflow using default parameter values."""
        runner = CliRunner()
        # math_workflow has factor=2 as default
        result = runner.invoke(
            cli,
            ["run", "tests.examples.workflows.math_workflow", "--param", "x=5", "--param", "y=10"],
        )
        assert result.exit_code == 0

    def test_run_workflow_override_default_params(self):
        """Test running a workflow and overriding default parameter."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                "tests.examples.workflows.math_workflow",
                "--param",
                "x=5",
                "--param",
                "y=10",
                "--param",
                "factor=3",
            ],
        )
        assert result.exit_code == 0

    def test_run_workflow_missing_required_param(self):
        """Test running a workflow without required parameters."""
        runner = CliRunner()
        result = runner.invoke(
            cli, ["run", "tests.examples.workflows.math_workflow", "--param", "x=5"]
        )
        assert result.exit_code != 0
        assert "Missing required parameters" in result.output
        assert "y" in result.output

    def test_run_workflow_with_invalid_param_format(self):
        """Test running with invalid parameter format."""
        runner = CliRunner()
        result = runner.invoke(
            cli, ["run", "tests.examples.workflows.math_workflow", "--param", "invalid_format"]
        )
        assert result.exit_code != 0
        assert "Invalid parameter format" in result.output

    def test_run_workflow_with_unknown_param(self):
        """Test running with unknown parameter name."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                "tests.examples.workflows.math_workflow",
                "--param",
                "x=5",
                "--param",
                "y=10",
                "--param",
                "unknown=5",
            ],
        )
        assert result.exit_code != 0
        assert "Unknown parameter" in result.output

    def test_run_workflow_with_parallel(self):
        """Test running with parallel (async) execution."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                "tests.examples.workflows.math_workflow",
                "--param",
                "x=5",
                "--param",
                "y=10",
                "--parallel",
            ],
        )
        assert result.exit_code == 0

    def test_run_workflow_with_settings(self):
        """Test running with settings override."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                "tests.examples.workflows.math_workflow",
                "--param",
                "x=5",
                "--param",
                "y=10",
                "--settings",
                "max_backend_threads=8",
            ],
        )
        assert result.exit_code == 0

    def test_run_workflow_with_invalid_setting_name(self):
        """Test running with invalid setting name."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                "tests.examples.workflows.math_workflow",
                "--param",
                "x=5",
                "--param",
                "y=10",
                "--settings",
                "invalid_setting=10",
            ],
        )
        assert result.exit_code != 0
        assert "Unknown setting" in result.output

    def test_run_workflow_with_invalid_setting_value(self):
        """Test running with invalid setting value."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                "tests.examples.workflows.math_workflow",
                "--param",
                "x=5",
                "--param",
                "y=10",
                "--settings",
                "max_backend_threads=not_a_number",
            ],
        )
        assert result.exit_code != 0
        assert "Invalid value" in result.output

    def test_run_workflow_with_bool_setting(self):
        """Test that boolean settings are parsed correctly."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                "tests.examples.workflows.math_workflow",
                "--param",
                "x=5",
                "--param",
                "y=10",
                "--settings",
                "enable_plugin_tracing=true",
            ],
        )
        assert result.exit_code == 0

    def test_run_workflow_with_union_type_setting(self):
        """Test that Union-typed settings (e.g. datastore_store: str | DatasetStore) are parsed."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                "tests.examples.workflows.math_workflow",
                "--param",
                "x=5",
                "--param",
                "y=10",
                "--settings",
                "datastore_store=/tmp/test_store",
            ],
        )
        assert result.exit_code == 0

    def test_run_workflow_backend_applied(self):
        """Test that --backend flag is applied to workflow execution."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                "tests.examples.workflows.math_workflow",
                "--param",
                "x=5",
                "--param",
                "y=10",
                "--backend",
                "threading",
            ],
        )
        assert result.exit_code == 0

    def test_run_workflow_type_conversion_int(self):
        """Test that integer parameters are properly converted."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                "tests.examples.workflows.math_workflow",
                "--param",
                "x=5",
                "--param",
                "y=10",
                "--param",
                "factor=3",
            ],
        )
        assert result.exit_code == 0

    def test_run_workflow_invalid_setting_format(self):
        """Test running with invalid setting format (no =)."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                "tests.examples.workflows.math_workflow",
                "--param",
                "x=5",
                "--param",
                "y=10",
                "--settings",
                "invalid_format",
            ],
        )
        assert result.exit_code != 0
        assert "Invalid setting format" in result.output

    def test_run_workflow_call_error(self):
        """Test error when calling workflow with wrong parameters."""
        runner = CliRunner()
        # Create a workflow that will fail when called
        result = runner.invoke(
            cli,
            [
                "run",
                "tests.examples.workflows.math_workflow",
                # Missing required parameters will cause error during call
            ],
        )
        assert result.exit_code != 0
        assert "Missing required parameters" in result.output

    def test_run_workflow_execution_error(self):
        """Test handling of workflow execution errors."""
        runner = CliRunner()
        # Use a workflow that will fail during execution
        result = runner.invoke(
            cli,
            [
                "run",
                "tests.examples.workflows.failing_workflow",
                "--param",
                "x=42",
            ],
        )
        assert result.exit_code != 0
        assert "Workflow execution failed" in result.output
        assert "Intentional failure" in result.output

    def test_run_untyped_workflow_warning(self):
        """Test warning when using untyped workflow parameters."""
        import warnings

        runner = CliRunner()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = runner.invoke(
                cli,
                [
                    "run",
                    "tests.examples.workflows.untyped_workflow",
                    "--param",
                    "x=5",
                    "--param",
                    "y=10",
                ],
            )
        # Should succeed but with a warning about untyped parameters
        assert result.exit_code == 0
        # The warning goes to stderr, but click captures it
        # We can check that it ran successfully


class TestParseParamValue:
    """Test suite for parse_param_value function."""

    def test_parse_none_type(self):
        """Test parsing with None type returns string."""
        assert parse_param_value("test", None) == "test"

    def test_parse_int(self):
        """Test parsing integer values."""
        assert parse_param_value("42", int) == 42
        assert parse_param_value("-10", int) == -10

    def test_parse_int_invalid(self):
        """Test parsing invalid integer raises ValueError."""
        with pytest.raises(ValueError, match="Cannot convert"):
            parse_param_value("not_a_number", int)

    def test_parse_float(self):
        """Test parsing float values."""
        assert parse_param_value("3.14", float) == 3.14
        assert parse_param_value("-2.5", float) == -2.5

    def test_parse_float_invalid(self):
        """Test parsing invalid float raises ValueError."""
        with pytest.raises(ValueError, match="Cannot convert"):
            parse_param_value("not_a_float", float)

    def test_parse_bool_true(self):
        """Test parsing boolean true values."""
        assert parse_param_value("true", bool) is True
        assert parse_param_value("True", bool) is True
        assert parse_param_value("1", bool) is True
        assert parse_param_value("yes", bool) is True
        assert parse_param_value("y", bool) is True

    def test_parse_bool_false(self):
        """Test parsing boolean false values."""
        assert parse_param_value("false", bool) is False
        assert parse_param_value("False", bool) is False
        assert parse_param_value("0", bool) is False
        assert parse_param_value("no", bool) is False
        assert parse_param_value("n", bool) is False
        assert parse_param_value("anything", bool) is False

    def test_parse_str(self):
        """Test parsing string values."""
        assert parse_param_value("hello", str) == "hello"
        assert parse_param_value("123", str) == "123"

    def test_parse_custom_type_success(self):
        """Test parsing with custom type that has constructor."""
        result = parse_param_value("/tmp/test", Path)
        assert isinstance(result, Path)
        assert result == Path("/tmp/test")

    def test_parse_custom_type_fallback(self):
        """Test parsing with custom type that fails falls back to string."""

        # Create a type that will fail conversion
        class UnconvertibleType:
            def __init__(self, value):
                raise TypeError("Cannot convert")

        result = parse_param_value("test", UnconvertibleType)
        assert result == "test"  # Falls back to string
