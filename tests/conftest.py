"""Conftest for all pytest configuration - fixtures, hooks, and doctest setup."""

from pathlib import Path

import pytest

# Source modules whose doctests use the old futures/graph execution model.
# These doctests are skipped until Phase 5 removes the legacy code.
_LEGACY_DOCTEST_MODULES = frozenset(
    {
        "daglite.engine",
        "daglite.tasks",
        "daglite.futures.base",
        "daglite.futures.task_future",
        "daglite.futures.map_future",
        "daglite.futures.load_future",
    }
)


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest settings before tests are run."""
    # Placeholder for future global pytest configuration.


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Apply suite markers based on test location and mark doctest items."""
    skip_legacy = pytest.mark.skip(reason="Legacy futures/graph execution model")

    for item in items:
        path = Path(str(item.fspath)).as_posix()
        normed = f"/{path}"

        if isinstance(item, pytest.DoctestItem):
            item.add_marker(pytest.mark.doctest)

            # Skip doctests in legacy source modules.
            module_name = getattr(item, "name", "")
            for prefix in _LEGACY_DOCTEST_MODULES:
                if module_name.startswith(prefix):
                    item.add_marker(skip_legacy)
                    break

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
