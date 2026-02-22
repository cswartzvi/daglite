"""Shared fixtures for CLI tests."""

from __future__ import annotations

import logging

import pytest

from daglite.plugins.manager import reset_global_plugin_manager


@pytest.fixture(autouse=True)
def reset_plugins_and_logging():
    """
    Reset plugin manager and logging state around each CLI test.

    This fixture saves and restores both the plugin manager state and the
    propagation/handler state of affected loggers so that test cases are fully
    isolated from one another.
    """
    # Save state of loggers that might be affected by plugin registration and dictConfig calls.
    _watched = ["daglite", "daglite.lifecycle", "daglite.tasks"]
    _saved: dict[str, tuple[bool, int, list[logging.Handler]]] = {}
    for name in _watched:
        lg = logging.getLogger(name)
        _saved[name] = (lg.propagate, lg.level, lg.handlers[:])

    root_level = logging.root.level
    root_handlers = logging.root.handlers[:]

    reset_global_plugin_manager()

    yield

    reset_global_plugin_manager()

    # Restore state of watched loggers and root logger to ensure test isolation.
    for name, (propagate, level, handlers) in _saved.items():
        lg = logging.getLogger(name)
        lg.propagate = propagate
        lg.setLevel(level)
        lg.handlers = handlers

    logging.root.handlers = root_handlers
    logging.root.setLevel(root_level)
