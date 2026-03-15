"""Tests for the CLI ``list`` command."""

from __future__ import annotations

import pytest

from daglite.cli.core import main


class TestListCommand:
    """Tests for ``daglite list``."""

    def test_list_help(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["list", "--help"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "List" in out

    def test_list_requires_module_argument(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["list"])
        assert exc_info.value.code != 0

    def test_list_nonexistent_module(self, capsys):
        with pytest.raises(SystemExit):
            main(["list", "nonexistent.module.that.does_not_exist"])
        out = capsys.readouterr().out
        assert "Error" in out or "No module named" in out

    def test_list_module_with_workflows(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["list", "tests.examples.workflows"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "math_workflow" in out
        assert "empty_workflow" in out
        assert "failing_workflow" in out
        assert "untyped_workflow" in out

    def test_list_shows_colon_path(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["list", "tests.examples.workflows"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "tests.examples.workflows:math_workflow" in out

    def test_list_module_with_no_workflows(self, capsys):
        """Module that exists but contains no @workflow objects."""
        with pytest.raises(SystemExit) as exc_info:
            main(["list", "daglite.settings"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "No workflows found" in out

    def test_list_one_valid_one_invalid_module(self, capsys):
        """An invalid module should produce an error."""
        with pytest.raises(SystemExit):
            main(["list", "nonexistent_module"])
        out = capsys.readouterr().out
        assert "Error" in out or "No module named" in out

    def test_list_workflow_without_description(self, capsys):
        """Workflows with no description should still be listed."""
        with pytest.raises(SystemExit) as exc_info:
            main(["list", "tests.examples.workflows"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "no_description_workflow" in out

    def test_list_with_filepath(self, capsys):
        """Filesystem paths should be accepted in addition to dotted module paths."""
        with pytest.raises(SystemExit) as exc_info:
            main(["list", "tests/examples/workflows.py"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "math_workflow" in out
        assert "tests.examples.workflows:" in out
