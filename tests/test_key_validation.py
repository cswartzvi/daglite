"""Unit tests for key template validation."""

import pytest

from daglite._templates import parse_template
from daglite._templates import resolve_template


class TestParseTemplate:
    """Tests for parse_template() — syntax validation and placeholder extraction."""

    # -- plain strings --

    def test_plain_string_returns_empty(self):
        """Plain strings have no placeholders."""
        assert parse_template("simple_key") == frozenset()

    def test_empty_string(self):
        assert parse_template("") == frozenset()

    def test_escaped_braces(self):
        assert parse_template("{{literal}}") == frozenset()

    # -- valid templates --

    def test_single_placeholder(self):
        assert parse_template("{x}") == frozenset({"x"})

    def test_multiple_placeholders(self):
        assert parse_template("output_{data_id}_{version}") == frozenset({"data_id", "version"})

    def test_nested_path(self):
        assert parse_template("outputs/{data_id}/result.pkl") == frozenset({"data_id"})

    def test_three_placeholders(self):
        assert parse_template("{a}_{b}_{c}") == frozenset({"a", "b", "c"})

    def test_mixed_literal_and_placeholder(self):
        assert parse_template("prefix_{name}_suffix") == frozenset({"name"})

    def test_single_placeholder_only(self):
        assert parse_template("{key}") == frozenset({"key"})

    # -- invalid templates --

    def test_empty_placeholder_raises(self):
        """Empty {} placeholders are rejected."""
        with pytest.raises(ValueError, match="empty placeholder"):
            parse_template("output_{}")

    def test_malformed_template_unclosed_brace(self):
        """Unclosed brace raises ValueError."""
        with pytest.raises(ValueError, match="Invalid key template"):
            parse_template("output_{unclosed")


class TestParseTemplatePlaceholderChecks:
    """Tests verifying placeholder sets can be compared against allowed names."""

    def test_all_placeholders_available(self):
        placeholders = parse_template("output_{a}_{b}")
        unknown = placeholders - {"a", "b", "c"}
        assert not unknown

    def test_missing_placeholder_detected(self):
        placeholders = parse_template("output_{missing}")
        unknown = placeholders - {"a", "b"}
        assert unknown == {"missing"}

    def test_partial_match_detected(self):
        placeholders = parse_template("output_{a}_{missing}")
        unknown = placeholders - {"a", "b"}
        assert unknown == {"missing"}

    def test_no_placeholders_always_passes(self):
        placeholders = parse_template("literal_key")
        assert not placeholders - set()

    def test_empty_available_with_placeholder(self):
        placeholders = parse_template("{x}")
        assert placeholders - set() == {"x"}


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

    def test_all_missing(self):
        assert resolve_template("{x}_{y}", {}) == "{x}_{y}"

    def test_empty_args(self):
        assert resolve_template("no_args.txt", {}) == "no_args.txt"

    def test_numeric_value(self):
        assert resolve_template("epoch_{n}", {"n": 42}) == "epoch_42"
