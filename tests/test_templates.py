"""Unit tests for key template parsing and resolution."""

import pytest

from daglite._templates import parse_template
from daglite._templates import resolve_template


class TestParseTemplate:
    """Tests for parse_template() — syntax validation and placeholder extraction."""

    def test_empty_string(self):
        assert parse_template("") == frozenset()

    def test_escaped_braces(self):
        assert parse_template("{{literal}}") == frozenset()

    def test_multiple_placeholders(self):
        assert parse_template("output_{data_id}_{version}") == frozenset({"data_id", "version"})

    def test_nested_path(self):
        assert parse_template("outputs/{data_id}/result.pkl") == frozenset({"data_id"})

    def test_empty_placeholder_raises(self):
        """Empty {} placeholders are rejected."""
        with pytest.raises(ValueError, match="empty placeholder"):
            parse_template("output_{}")

    def test_malformed_template_unclosed_brace(self):
        """Unclosed brace raises ValueError."""
        with pytest.raises(ValueError, match="Invalid key template"):
            parse_template("output_{unclosed")


class TestResolveTemplate:
    """Tests for resolve_template()."""

    def test_basic_substitution(self):
        assert resolve_template("output_{split}.csv", {"split": "train"}) == "output_train.csv"

    def test_multiple_placeholders(self):
        result = resolve_template(
            "{model}_{split}_{epoch}", {"model": "bert", "split": "val", "epoch": 3}
        )
        assert result == "bert_val_3"

    def test_no_placeholders_passthrough(self):
        assert resolve_template("plain.txt", {"x": 1}) == "plain.txt"

    def test_missing_placeholder_survives(self):
        """Unresolvable placeholders are left as ``{name}``."""
        assert resolve_template("{a}_{b}", {"a": "ok"}) == "ok_{b}"
