"""Conftest for all pytest configuration - fixtures, hooks, and doctest setup."""

from pathlib import Path

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest settings before tests are run."""
    # Placeholder for future global pytest configuration.


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Apply suite markers based on test location and mark doctest items."""
    for item in items:
        path = Path(str(item.fspath)).as_posix()
        normed = f"/{path}"

        if isinstance(item, pytest.DoctestItem):
            item.add_marker(pytest.mark.doctest)

        if "/tests/behavior/" in normed:
            item.add_marker(pytest.mark.behavior)
        elif "/tests/integration/" in normed or "/tests/cli/" in normed:
            item.add_marker(pytest.mark.integration)
        elif "/tests/contracts/" in normed:
            item.add_marker(pytest.mark.contracts)

            if "/tests/contracts/typing/" in normed:
                item.add_marker(pytest.mark.typing_contract)
        elif "/tests/" in normed and "/tests/examples/" not in normed:
            item.add_marker(pytest.mark.unit)
