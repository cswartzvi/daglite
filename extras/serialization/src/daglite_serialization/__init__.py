"""Serialization plugin for daglite with support for numpy, pandas, and PIL."""

import importlib
import pkgutil

__version__ = "0.1.0"


def register_all():
    """
    Register all available serialization handlers.

    This function auto-discovers and registers handlers for all available
    serialization modules in this package. Each module should provide a
    `register_handlers()` function.

    Modules that can't be imported (missing dependencies) are silently skipped.

    Example:
        >>> from daglite_serialization import register_all
        >>> register_all()  # Auto-discover and register all available handlers
    """
    # Auto-discover all modules in this package
    for finder, module_name, ispkg in pkgutil.iter_modules(__path__, __name__ + "."):
        # Skip __init__ and private modules
        if module_name.endswith("__init__") or module_name.split(".")[-1].startswith(
            "_"
        ):  # pragma: no cover
            # Defensive check - we don't create such modules in this package
            continue

        try:
            # Import the module
            module = importlib.import_module(module_name)

            # If it has register_handlers(), call it
            if hasattr(module, "register_handlers"):  # pragma: no branch
                # All our plugin modules have register_handlers()
                module.register_handlers()
        except ImportError:  # pragma: no cover
            # Module dependencies not installed - skip silently
            # (tested manually but hard to automate without uninstalling deps)
            pass


__all__ = ["register_all", "__version__"]
