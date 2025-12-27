"""Conftest for all pytest configuration - fixtures, hooks, and doctest setup."""

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest settings before tests are run."""
    # Placeholder for future global pytest configuration.


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Automatically mark doctest items with the 'doctest' marker."""
    # Doctest marker addition
    for item in items:
        if isinstance(item, pytest.DoctestItem):
            item.add_marker(pytest.mark.doctest)
