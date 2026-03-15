"""Tests for CLI run command."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from daglite.cli._shared import _parse_param_value
from daglite.cli._shared import normalize_tokens
from daglite.cli._shared import parse_settings_overrides
from daglite.cli.core import main


class TestRunCommand:
    """Tests for the run command."""

    def test_run_help(self, capsys):
        """Test that ``daglite run --help`` works."""
        with pytest.raises(SystemExit) as exc_info:
            main(["run", "--help"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "Run a workflow target" in out

    def test_run_workflow_help(self, capsys):
        """Test that ``daglite run <target> --help`` shows workflow parameters."""
        with pytest.raises(SystemExit) as exc_info:
            main(["run", "tests.examples.workflows:math_workflow", "--help"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "--x" in out
        assert "--y" in out
        assert "--factor" in out

    def test_run_without_workflow_path(self, capsys):
        """Test run command without workflow path shows help."""
        with pytest.raises(SystemExit) as exc_info:
            main(["run"])
        assert exc_info.value.code == 0  # redirects to help

    def test_run_with_nonexistent_module(self, capsys):
        """Test run command with non-existent module."""
        with pytest.raises(SystemExit) as exc_info:
            main(["run", "nonexistent.module:workflow"])
        assert exc_info.value.code == 2

    def test_run_with_nonexistent_workflow(self, capsys):
        """Test run command with non-existent workflow in existing module."""
        with pytest.raises(SystemExit) as exc_info:
            main(["run", "tests.examples.workflows:does_not_exist", "--x", "1"])
        assert exc_info.value.code == 2

    def test_run_with_non_workflow_object(self, capsys):
        """Test run command with object that's not a Workflow."""
        with pytest.raises(SystemExit):
            main(["run", "tests.examples.workflows:add"])

    def test_run_simple_workflow(self, capsys):
        """Test running a simple workflow."""
        with pytest.raises(SystemExit) as exc_info:
            main(["run", "tests.examples.workflows:math_workflow", "--x", "5", "--y", "10"])
        assert exc_info.value.code == 0

    def test_run_empty_workflow(self, capsys):
        """Test running a workflow with no parameters."""
        with pytest.raises(SystemExit) as exc_info:
            main(["run", "tests.examples.workflows:empty_workflow"])
        assert exc_info.value.code == 0

    def test_run_workflow_with_default_params(self, capsys):
        """Test running a workflow using default parameter values."""
        with pytest.raises(SystemExit) as exc_info:
            main(["run", "tests.examples.workflows:math_workflow", "--x", "5", "--y", "10"])
        assert exc_info.value.code == 0

    def test_run_workflow_override_default_params(self, capsys):
        """Test running a workflow and overriding default parameter."""
        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "run",
                    "tests.examples.workflows:math_workflow",
                    "--x",
                    "5",
                    "--y",
                    "10",
                    "--factor",
                    "3",
                ]
            )
        assert exc_info.value.code == 0

    def test_run_workflow_missing_required_param(self, capsys):
        """Test running a workflow without required parameters."""
        with pytest.raises(SystemExit) as exc_info:
            main(["run", "tests.examples.workflows:math_workflow", "--x", "5"])
        assert exc_info.value.code != 0

    def test_run_workflow_with_invalid_param_value(self, capsys):
        """Test running with a value that can't be converted to the parameter's type."""
        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "run",
                    "tests.examples.workflows:math_workflow",
                    "--x",
                    "notanumber",
                    "--y",
                    "10",
                ]
            )
        assert exc_info.value.code != 0

    def test_run_async_workflow(self, capsys):
        """Test running an async workflow through the CLI (exercises asyncio.run path)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["run", "tests.examples.workflows:async_workflow", "--x", "3", "--y", "4"])
        assert exc_info.value.code == 0

    def test_run_with_dotted_target(self, capsys):
        """Test running with dotted target (module.workflow) syntax."""
        with pytest.raises(SystemExit) as exc_info:
            main(["run", "tests.examples.workflows.math_workflow", "--x", "5", "--y", "10"])
        assert exc_info.value.code == 0

    def test_run_with_filepath_target(self, capsys):
        """Test running with filepath:workflow syntax."""
        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "run",
                    "tests/examples/workflows.py:math_workflow",
                    "--x",
                    "5",
                    "--y",
                    "10",
                ]
            )
        assert exc_info.value.code == 0

    def test_run_version_flag(self, capsys):
        """Test that ``daglite run --version`` shows version instead of treating it as target."""
        import daglite

        with pytest.raises(SystemExit) as exc_info:
            main(["run", "--version"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert daglite.__version__ in out

    def test_run_short_help_flag(self, capsys):
        """Test that ``daglite run -h`` shows help."""
        with pytest.raises(SystemExit) as exc_info:
            main(["run", "-h"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "Run a workflow target" in out


class TestDescribeCommand:
    """Tests for the describe command."""

    def test_describe_workflow(self, capsys):
        """Test describing a workflow shows its parameters."""
        with pytest.raises(SystemExit) as exc_info:
            main(["describe", "tests.examples.workflows:math_workflow"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "--x" in out
        assert "--y" in out
        assert "--factor" in out

    def test_describe_help(self, capsys):
        """Test that ``daglite describe --help`` shows full docstring."""
        with pytest.raises(SystemExit) as exc_info:
            main(["describe", "--help"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "Describe a workflow target" in out
        assert "path/to/module.py" in out

    def test_describe_nonexistent_target(self, capsys):
        """Test describing a nonexistent workflow."""
        with pytest.raises(SystemExit) as exc_info:
            main(["describe", "tests.examples.workflows:does_not_exist"])
        assert exc_info.value.code == 2

    def test_describe_with_filepath_target(self, capsys):
        """Test describing with filepath:workflow syntax."""
        with pytest.raises(SystemExit) as exc_info:
            main(["describe", "tests/examples/workflows.py:math_workflow"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "--x" in out

    def test_describe_with_arg_validation(self, capsys):
        """Test describe with argument validation via ``--`` separator."""
        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "describe",
                    "tests.examples.workflows:math_workflow",
                    "--",
                    "--x",
                    "5",
                    "--y",
                    "10",
                ]
            )
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "Parsed values" in out
        assert "x = 5" in out
        assert "y = 10" in out


class TestSetupCliPlugins:
    """Tests for setup_cli_plugins() in _shared.py."""

    def test_skips_registration_when_plugin_already_present(self):
        """Calling setup_cli_plugins twice must not re-register the plugin."""
        from daglite.cli._shared import setup_cli_plugins
        from daglite.logging.plugin import LifecycleLoggingPlugin
        from daglite.plugins.manager import has_plugin
        from daglite.plugins.manager import register_plugins

        register_plugins(LifecycleLoggingPlugin())
        assert has_plugin(LifecycleLoggingPlugin)

        setup_cli_plugins()
        assert has_plugin(LifecycleLoggingPlugin)


class TestParseParamValue:
    """Test suite for _parse_param_value function."""

    def test_parse_none_type(self):
        assert _parse_param_value("test", None) == "test"

    def test_parse_int(self):
        assert _parse_param_value("42", int) == 42
        assert _parse_param_value("-10", int) == -10

    def test_parse_int_invalid(self):
        with pytest.raises(ValueError, match="Cannot convert"):
            _parse_param_value("not_a_number", int)

    def test_parse_float(self):
        assert _parse_param_value("3.14", float) == 3.14
        assert _parse_param_value("-2.5", float) == -2.5

    def test_parse_float_invalid(self):
        with pytest.raises(ValueError, match="Cannot convert"):
            _parse_param_value("not_a_float", float)

    def test_parse_bool_true(self):
        assert _parse_param_value("true", bool) is True
        assert _parse_param_value("True", bool) is True
        assert _parse_param_value("1", bool) is True
        assert _parse_param_value("yes", bool) is True
        assert _parse_param_value("y", bool) is True

    def test_parse_bool_false(self):
        assert _parse_param_value("false", bool) is False
        assert _parse_param_value("False", bool) is False
        assert _parse_param_value("0", bool) is False
        assert _parse_param_value("no", bool) is False
        assert _parse_param_value("n", bool) is False
        assert _parse_param_value("anything", bool) is False

    def test_parse_str(self):
        assert _parse_param_value("hello", str) == "hello"
        assert _parse_param_value("123", str) == "123"

    def test_parse_custom_type_success(self):
        result = _parse_param_value("/tmp/test", Path)
        assert isinstance(result, Path)
        assert result == Path("/tmp/test")

    def test_parse_custom_type_fallback(self):
        class UnconvertibleType:
            def __init__(self, value):
                raise TypeError("Cannot convert")

        result = _parse_param_value("test", UnconvertibleType)
        assert result == "test"


class TestFilepathToModule:
    """Tests for _filepath_to_module in _shared.py."""

    def test_passthrough_dotted_path(self):
        from daglite.cli._shared import _filepath_to_module

        assert _filepath_to_module("tests.examples.workflows") == "tests.examples.workflows"

    def test_passthrough_colon_target(self):
        from daglite.cli._shared import _filepath_to_module

        assert _filepath_to_module("mod.sub:func") == "mod.sub:func"

    def test_filepath_with_slashes(self):
        from daglite.cli._shared import _filepath_to_module

        assert _filepath_to_module("tests/examples/workflows.py") == "tests.examples.workflows"

    def test_filepath_with_colon_target(self):
        from daglite.cli._shared import _filepath_to_module

        assert (
            _filepath_to_module("tests/examples/workflows.py:math_workflow")
            == "tests.examples.workflows:math_workflow"
        )

    def test_filepath_relative_prefix(self):
        from daglite.cli._shared import _filepath_to_module

        assert _filepath_to_module("./tests/examples/workflows.py") == "tests.examples.workflows"

    def test_directory_trailing_slash(self):
        from daglite.cli._shared import _filepath_to_module

        assert _filepath_to_module("tests/examples/") == "tests.examples"

    def test_backslash_filepath(self):
        from daglite.cli._shared import _filepath_to_module

        assert _filepath_to_module("tests\\examples\\workflows.py") == "tests.examples.workflows"

    def test_backslash_relative_prefix(self):
        from daglite.cli._shared import _filepath_to_module

        assert _filepath_to_module(".\\tests\\examples\\workflows.py") == "tests.examples.workflows"

    def test_windows_drive_letter_no_attr(self):
        """Drive-letter colon (C:\\...) should not be split as a target separator."""
        from daglite.cli._shared import _filepath_to_module

        assert _filepath_to_module("C:\\proj\\workflows.py") == "proj.workflows"

    def test_windows_drive_letter_with_attr(self):
        """Drive-letter path with :attr suffix should be split correctly."""
        from daglite.cli._shared import _filepath_to_module

        assert _filepath_to_module("C:\\proj\\workflows.py:my_wf") == "proj.workflows:my_wf"


class TestNormalizeTokens:
    """Tests for normalize_tokens."""

    def test_none_returns_empty_list(self):
        assert normalize_tokens(None) == []

    def test_list_passthrough(self):
        assert normalize_tokens(["a", "b"]) == ["a", "b"]

    def test_tuple_converted(self):
        assert normalize_tokens(("a", "b")) == ["a", "b"]


class TestParseSettingsOverrides:
    """Tests for parse_settings_overrides."""

    def test_valid_int_setting(self):
        result = parse_settings_overrides(("max_backend_threads=8",))
        assert result == {"max_backend_threads": 8}

    def test_missing_equals(self):
        with pytest.raises(ValueError, match="Invalid setting format"):
            parse_settings_overrides(("no_equals_here",))

    def test_unknown_setting(self):
        with pytest.raises(ValueError, match="Unknown setting"):
            parse_settings_overrides(("nonexistent_setting=10",))

    def test_invalid_value(self):
        with pytest.raises(ValueError, match="Invalid value"):
            parse_settings_overrides(("max_backend_threads=notanumber",))

    def test_union_type_setting(self):
        result = parse_settings_overrides(("dataset_store=/tmp/ds",))
        assert result["dataset_store"] == "/tmp/ds"

    def test_empty_tuple(self):
        assert parse_settings_overrides(()) == {}


class TestSplitTarget:
    """Tests for _split_target."""

    def test_no_separator_raises(self):
        from daglite.cli._shared import _split_target

        with pytest.raises(ValueError, match="Expected"):
            _split_target("nodotshere")

    def test_empty_parts_raises(self):
        from daglite.cli._shared import _split_target

        with pytest.raises(ValueError, match="Expected"):
            _split_target(":attr")

    def test_colon_split(self):
        from daglite.cli._shared import _split_target

        assert _split_target("mod.sub:func") == ("mod.sub", "func")

    def test_dot_split(self):
        from daglite.cli._shared import _split_target

        assert _split_target("mod.sub.func") == ("mod.sub", "func")


class TestResolvedSignature:
    """Tests for _resolved_signature."""

    def test_returns_signature_with_type_hints(self):
        from daglite.cli._shared import _resolved_signature
        from tests.examples.workflows import math_workflow

        sig = _resolved_signature(math_workflow)
        assert "x" in sig.parameters
        assert sig.parameters["x"].annotation is int

    def test_returns_return_annotation(self):
        from daglite.cli._shared import _resolved_signature
        from tests.examples.workflows import sync_workflow

        sig = _resolved_signature(sync_workflow)
        assert sig.return_annotation is int


class TestIsAsyncWorkflow:
    """Tests for _is_async_workflow."""

    def test_sync_workflow(self):
        from daglite.cli._shared import _is_async_workflow
        from tests.examples.workflows import math_workflow

        assert _is_async_workflow(math_workflow) is False

    def test_async_workflow(self):
        from daglite.cli._shared import _is_async_workflow
        from tests.examples.workflows import async_workflow

        assert _is_async_workflow(async_workflow) is True


class TestValidateWorkflowArgs:
    """Tests for validate_workflow_args."""

    def test_unrecognized_args_raises(self, mocker):
        from daglite.cli._shared import validate_workflow_args

        # The `ignored` branch is defensive — cyclopts raises before returning
        # ignored tokens.  Mock parse_args to simulate returned ignored tokens.
        mocker.patch(
            "daglite.cli._shared.build_workflow_app",
            return_value=mocker.Mock(
                parse_args=mocker.Mock(return_value=(None, None, ["--bogus"])),
            ),
        )
        with pytest.raises(ValueError, match="Unrecognized arguments"):
            validate_workflow_args(
                "tests.examples.workflows:math_workflow",
                ["--x", "5"],
            )


class TestDescribeEdgeCases:
    """Edge-case tests for the describe command."""

    def test_describe_bare_name_shows_tip(self, capsys):
        """Target without ``:`` or ``.`` shows a helpful tip."""
        with pytest.raises(SystemExit) as exc_info:
            main(["describe", "math_workflow"])
        assert exc_info.value.code == 2
        out = capsys.readouterr().out
        assert "daglite list" in out

    def test_describe_bad_attr_shows_error(self, capsys):
        """Target with valid module but bad attribute shows error."""
        with pytest.raises(SystemExit) as exc_info:
            main(["describe", "tests.examples.workflows:nonexistent"])
        assert exc_info.value.code == 2
        out = capsys.readouterr().out
        assert "Error:" in out

    def test_describe_arg_validation_failure(self, capsys):
        """Invalid args after ``--`` produce a validation error."""
        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "describe",
                    "tests.examples.workflows:math_workflow",
                    "--",
                    "--x",
                    "notanint",
                    "--y",
                    "10",
                ]
            )
        assert exc_info.value.code == 2
        out = capsys.readouterr().out
        assert "Argument validation failed" in out

    def test_describe_suggests_workflows(self, capsys):
        """Describing a package (not a workflow) suggests workflows in that module."""
        with pytest.raises(SystemExit) as exc_info:
            main(["describe", "tests.examples:bogus"])
        assert exc_info.value.code == 2
        out = capsys.readouterr().out
        assert "Did you mean" in out

    def test_describe_bad_dotted_attr_shows_suggestions(self, capsys):
        """Dotted target with bad attr hits the elif branch of _print_workflow_suggestions."""
        with pytest.raises(SystemExit) as exc_info:
            main(["describe", "tests.examples.workflows.nonexistent"])
        assert exc_info.value.code == 2
        out = capsys.readouterr().out
        assert "Did you mean" in out

    def test_describe_nonexistent_module_no_suggestions(self, capsys):
        """Nonexistent module triggers ModuleNotFoundError in suggestions."""
        with pytest.raises(SystemExit) as exc_info:
            main(["describe", "no_such_module:func"])
        assert exc_info.value.code == 2

    def test_describe_module_without_workflows_no_suggestions(self, capsys):
        """Module that exists but has no workflows hits the empty-list return."""
        with pytest.raises(SystemExit) as exc_info:
            main(["describe", "tests:conftest"])
        assert exc_info.value.code == 2


class TestCoreRunEdgeCases:
    """Edge-case tests for core.py run handling."""

    def test_run_no_args_shows_help(self, capsys):
        """``daglite run`` with no target shows run help."""
        with pytest.raises(SystemExit) as exc_info:
            main(["run"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "Run a workflow target" in out

    def test_run_error_prints_tip(self, capsys):
        """A run-time error shows the error message and a describe tip."""
        with pytest.raises(SystemExit) as exc_info:
            main(["run", "tests.examples.workflows:failing_workflow", "--x", "42"])
        assert exc_info.value.code == 2
        out = capsys.readouterr().out
        assert "Error:" in out
        assert "daglite describe" in out


class TestCliName:
    """Tests for _cli_name."""

    def test_normal_argv0(self, monkeypatch):
        from daglite.cli._shared import _cli_name

        monkeypatch.setattr("sys.argv", ["daglite", "list"])
        assert _cli_name() == "daglite"

    def test_dunder_main(self, monkeypatch):
        from daglite.cli._shared import _cli_name

        monkeypatch.setattr("sys.argv", ["/some/path/__main__.py"])
        assert _cli_name() == "py -m daglite.cli"


class TestEnsureCwdOnPath:
    """Tests for _ensure_cwd_on_path."""

    def test_inserts_cwd_when_missing(self, monkeypatch):
        from daglite.cli._shared import _ensure_cwd_on_path

        cwd = str(Path.cwd())
        # Remove CWD from sys.path temporarily
        orig = sys.path[:]
        monkeypatch.setattr("sys.path", [p for p in orig if p != cwd])
        _ensure_cwd_on_path()
        assert cwd in sys.path


class TestResolvedSignatureException:
    """Test _resolved_signature when get_type_hints fails."""

    def test_falls_back_on_type_hint_error(self, monkeypatch):
        import typing

        from daglite.cli._shared import _resolved_signature
        from tests.examples.workflows import math_workflow

        monkeypatch.setattr(
            typing, "get_type_hints", lambda *a, **kw: (_ for _ in ()).throw(TypeError("boom"))
        )
        sig = _resolved_signature(math_workflow)
        # Should still return a signature, just without resolved hints
        assert "x" in sig.parameters


class TestDescribeArgPreview:
    """Tests for the argument preview in describe."""

    def test_describe_with_valid_args(self, capsys):
        """Workflow args are validated and displayed."""
        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "describe",
                    "tests.examples.workflows:math_workflow",
                    "--",
                    "--x",
                    "5",
                    "--y",
                    "10",
                ]
            )
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "x = 5" in out
        assert "y = 10" in out


class TestIterModulesUnder:
    """Tests for _iter_modules_under."""

    def test_walks_package(self):
        from daglite.cli._shared import _iter_modules_under

        modules = list(_iter_modules_under("tests.examples"))
        assert "tests.examples" in modules
        assert any("workflows" in m for m in modules)
