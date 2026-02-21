"""Shared fixtures for CLI tests."""

from __future__ import annotations

import logging

import pytest

from daglite.plugins.manager import reset_global_plugin_manager


@pytest.fixture(autouse=True)
def reset_plugins_and_logging():
    """Reset plugin manager and logging state around each CLI test.

    CLI tests invoke ``daglite run`` via Click's CliRunner.  The run command:

    1. Calls ``_setup_cli_plugins``, which registers ``LifecycleLoggingPlugin``
       (or rich equivalents) into the *global* plugin manager.
    2. ``LifecycleLoggingPlugin.__init__`` calls ``logging.config.dictConfig``,
       which sets the ``daglite`` logger to ``propagate=False``.  This breaks
       pytest's ``caplog`` fixture for any subsequent test that expects log
       records from ``daglite.*`` loggers to reach the root handler.

    This fixture saves and restores both the plugin manager state and the
    propagation/handler state of affected loggers so that test cases are fully
    isolated from one another.
    """
    # --- save state ---------------------------------------------------------
    # Named loggers that dictConfig touches
    _watched = ["daglite", "daglite.lifecycle", "daglite.tasks"]
    _saved: dict[str, tuple[bool, int, list[logging.Handler]]] = {}
    for name in _watched:
        lg = logging.getLogger(name)
        _saved[name] = (lg.propagate, lg.level, lg.handlers[:])

    root_level = logging.root.level
    root_handlers = logging.root.handlers[:]

    reset_global_plugin_manager()

    yield  # ----------------------------------------------------------------

    reset_global_plugin_manager()

    # --- restore state ------------------------------------------------------
    for name, (propagate, level, handlers) in _saved.items():
        lg = logging.getLogger(name)
        lg.propagate = propagate
        lg.setLevel(level)
        lg.handlers = handlers

    logging.root.handlers = root_handlers
    logging.root.setLevel(root_level)
