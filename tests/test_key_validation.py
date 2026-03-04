"""Unit tests for key template validation."""

import pytest

from daglite._validation import check_key_placeholders
from daglite._validation import check_key_template
from daglite._validation import has_placeholders
from daglite._validation import resolve_template


class TestValidateKeyTemplate:
    """Tests for check_key_template() syntax validation."""

    def test_valid_template_no_placeholders(self):
        """Plain strings pass validation."""
        check_key_template("simple_key")

    def test_valid_template_with_placeholders(self):
        """Templates with named placeholders pass."""
        check_key_template("output_{data_id}_{version}")

    def test_valid_template_nested_path(self):
        """Templates with path separators pass."""
        check_key_template("outputs/{data_id}/result.pkl")

    def test_empty_placeholder_raises(self):
        """Empty {} placeholders are rejected."""
        with pytest.raises(ValueError, match="empty placeholder"):
            check_key_template("output_{}")

    def test_malformed_template_unclosed_brace(self):
        """Unclosed brace raises ValueError."""
        with pytest.raises(ValueError, match="Invalid key template"):
            check_key_template("output_{unclosed")

    def test_multiple_valid_placeholders(self):
        """Multiple named placeholders all pass."""
        check_key_template("{a}_{b}_{c}")

    def test_mixed_literal_and_placeholder(self):
        """Mix of literal text and placeholders passes."""
        check_key_template("prefix_{name}_suffix")

    def test_single_placeholder_only(self):
        """A key that is just a single placeholder passes."""
        check_key_template("{key}")


class TestValidateKeyPlaceholders:
    """Tests for check_key_placeholders() placeholder-name validation."""

    def test_all_placeholders_available(self):
        """No error when all placeholders match available names."""
        check_key_placeholders("output_{a}_{b}", {"a", "b", "c"})

    def test_missing_placeholder_raises(self):
        """Missing placeholder raises ValueError."""
        with pytest.raises(ValueError, match="won't be available"):
            check_key_placeholders("output_{missing}", {"a", "b"})

    def test_partial_match_raises(self):
        """One matching and one missing still raises."""
        with pytest.raises(ValueError, match="won't be available"):
            check_key_placeholders("output_{a}_{missing}", {"a", "b"})

    def test_no_placeholders_always_passes(self):
        """A literal key with no placeholders always passes."""
        check_key_placeholders("literal_key", set())

    def test_empty_available_with_placeholder_raises(self):
        """Placeholders with empty available set raises."""
        with pytest.raises(ValueError, match="won't be available"):
            check_key_placeholders("{x}", set())

    def test_error_message_lists_available(self):
        """Error message includes the sorted available variables."""
        with pytest.raises(ValueError, match="Available placeholders: \\['alpha', 'beta'\\]"):
            check_key_placeholders("{gamma}", {"alpha", "beta"})


class TestHasPlaceholders:
    """Tests for has_placeholders()."""

    def test_plain_string(self):
        assert has_placeholders("hello") is False

    def test_single_placeholder(self):
        assert has_placeholders("{x}") is True

    def test_multiple_placeholders(self):
        assert has_placeholders("{a}_{b}") is True

    def test_escaped_braces(self):
        assert has_placeholders("{{literal}}") is False

    def test_empty_string(self):
        assert has_placeholders("") is False


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
