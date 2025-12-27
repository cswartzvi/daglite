"""Conftest for all pytest configuration - fixtures, hooks, and doctest setup."""

import doctest

import pytest

# Doctest Configuration


def pytest_configure(config):
    """Configure pytest with custom doctest options."""
    doctest.ELLIPSIS_MARKER = "..."


def pytest_collection_modifyitems(items):
    """Automatically mark doctest items with the 'doctest' marker."""
    for item in items:
        if isinstance(item, pytest.DoctestItem):
            item.add_marker(pytest.mark.doctest)
