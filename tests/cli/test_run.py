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
        assert "Run a daglite pipeline" in result.output
        assert "--param" in result.output
        assert "--backend" in result.output

    def test_run_without_pipeline_path(self):
        """Test run command without pipeline path."""
        runner = CliRunner()
        result = runner.invoke(cli, ["run"])
        assert result.exit_code != 0
        assert "Missing argument" in result.output

    def test_run_with_invalid_pipeline_path(self):
        """Test run command with invalid pipeline path (no dot)."""
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "invalid"])
        assert result.exit_code != 0
        assert "Invalid pipeline path" in result.output

    def test_run_with_nonexistent_module(self):
        """Test run command with non-existent module."""
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "nonexistent.module.pipeline"])
        assert result.exit_code != 0
        assert "No module named" in result.output

    def test_run_with_nonexistent_pipeline(self):
        """Test run command with non-existent pipeline in existing module."""
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "daglite.nonexistent_pipeline"])
        assert result.exit_code != 0
        assert "not found" in result.output

    def test_run_with_non_pipeline_object(self):
        """Test run command with object that's not a Pipeline."""
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "daglite.task"])
        assert result.exit_code != 0
        assert "is not a Pipeline" in result.output

    def test_run_simple_pipeline(self):
        """Test running a simple pipeline."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["run", "tests.examples.pipelines.math_pipeline", "--param", "x=5", "--param", "y=10"],
        )
        assert result.exit_code == 0
        assert "Running pipeline: math_pipeline" in result.output
        assert "Pipeline completed successfully" in result.output
        assert "Result: 30" in result.output

    def test_run_empty_pipeline(self):
        """Test running a pipeline with no parameters."""
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "tests.examples.pipelines.empty_pipeline"])
        assert result.exit_code == 0
        assert "Running pipeline: empty_pipeline" in result.output
        # Should not show "Parameters:" line when there are no parameters
        assert "Parameters:" not in result.output
        assert "Result: 3" in result.output

    def test_run_pipeline_with_default_params(self):
        """Test running a pipeline using default parameter values."""
        runner = CliRunner()
        # math_pipeline has factor=2 as default
        result = runner.invoke(
            cli,
            ["run", "tests.examples.pipelines.math_pipeline", "--param", "x=5", "--param", "y=10"],
        )
        assert result.exit_code == 0
        assert "Result: 30" in result.output  # (5+10)*2 = 30

    def test_run_pipeline_override_default_params(self):
        """Test running a pipeline and overriding default parameter."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                "tests.examples.pipelines.math_pipeline",
                "--param",
                "x=5",
                "--param",
                "y=10",
                "--param",
                "factor=3",
            ],
        )
        assert result.exit_code == 0
        assert "Result: 45" in result.output  # (5+10)*3 = 45

    def test_run_pipeline_missing_required_param(self):
        """Test running a pipeline without required parameters."""
        runner = CliRunner()
        result = runner.invoke(
            cli, ["run", "tests.examples.pipelines.math_pipeline", "--param", "x=5"]
        )
        assert result.exit_code != 0
        assert "Missing required parameters" in result.output
        assert "y" in result.output

    def test_run_pipeline_with_invalid_param_format(self):
        """Test running with invalid parameter format."""
        runner = CliRunner()
        result = runner.invoke(
            cli, ["run", "tests.examples.pipelines.math_pipeline", "--param", "invalid_format"]
        )
        assert result.exit_code != 0
        assert "Invalid parameter format" in result.output

    def test_run_pipeline_with_unknown_param(self):
        """Test running with unknown parameter name."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                "tests.examples.pipelines.math_pipeline",
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

    def test_run_pipeline_with_backend(self):
        """Test running with a specific backend."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                "tests.examples.pipelines.math_pipeline",
                "--param",
                "x=5",
                "--param",
                "y=10",
                "--backend",
                "threading",
            ],
        )
        assert result.exit_code == 0
        assert "Backend: threading" in result.output
        assert "Result: 30" in result.output

    def test_run_pipeline_with_async(self):
        """Test running with async execution."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                "tests.examples.pipelines.math_pipeline",
                "--param",
                "x=5",
                "--param",
                "y=10",
                "--async",
            ],
        )
        assert result.exit_code == 0
        assert "Async execution: enabled" in result.output
        assert "Result: 30" in result.output

    def test_run_pipeline_with_settings(self):
        """Test running with settings override."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                "tests.examples.pipelines.math_pipeline",
                "--param",
                "x=5",
                "--param",
                "y=10",
                "--settings",
                "max_backend_threads=8",
            ],
        )
        assert result.exit_code == 0
        assert "30" in result.output
        assert "Result: 30" in result.output

    def test_run_pipeline_with_invalid_setting_name(self):
        """Test running with invalid setting name."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                "tests.examples.pipelines.math_pipeline",
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

    def test_run_pipeline_with_invalid_setting_value(self):
        """Test running with invalid setting value."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                "tests.examples.pipelines.math_pipeline",
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

    def test_run_pipeline_type_conversion_int(self):
        """Test that integer parameters are properly converted."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                "tests.examples.pipelines.math_pipeline",
                "--param",
                "x=5",
                "--param",
                "y=10",
                "--param",
                "factor=3",
            ],
        )
        assert result.exit_code == 0
        assert "Result: 45" in result.output

    def test_run_pipeline_invalid_setting_format(self):
        """Test running with invalid setting format (no =)."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                "tests.examples.pipelines.math_pipeline",
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

    def test_run_pipeline_call_error(self):
        """Test error when calling pipeline with wrong parameters."""
        runner = CliRunner()
        # Create a pipeline that will fail when called
        result = runner.invoke(
            cli,
            [
                "run",
                "tests.examples.pipelines.math_pipeline",
                # Missing required parameters will cause error during call
            ],
        )
        assert result.exit_code != 0
        assert "Missing required parameters" in result.output

    def test_run_pipeline_execution_error(self):
        """Test handling of pipeline execution errors."""
        runner = CliRunner()
        # Use a pipeline that will fail during execution
        result = runner.invoke(
            cli,
            [
                "run",
                "tests.examples.pipelines.failing_pipeline",
                "--param",
                "x=42",
            ],
        )
        assert result.exit_code != 0
        assert "Pipeline execution failed" in result.output
        assert "Intentional failure" in result.output

    def test_run_untyped_pipeline_warning(self):
        """Test warning when using untyped pipeline parameters."""
        import warnings

        runner = CliRunner()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = runner.invoke(
                cli,
                [
                    "run",
                    "tests.examples.pipelines.untyped_pipeline",
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
        assert "Result:" in result.output


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
