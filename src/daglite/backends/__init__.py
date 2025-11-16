from typing import TYPE_CHECKING

from daglite.backends.local import LocalBackend
from daglite.backends.threading import ThreadBackend

if TYPE_CHECKING:
    from daglite.engine import Backend
else:
    Backend = object  # type: ignore[misc]


__all__ = ["LocalBackend", "ThreadBackend"]


def find_backend(name: str | None = None) -> Backend:
    """
    Find a backend class by name.

    Args:
        name (str):
            Name of the backend to find.

    Returns:
        Backend class.
    """
    name = name if name is not None else "local"

    backends = {
        "local": LocalBackend,
        "synchronous": LocalBackend,
        "threading": ThreadBackend,
    }

    # TODO : dynamic discovery of backends from entry points

    if name not in backends:
        raise ValueError(f"Unknown backend '{name}'; available: {list(backends.keys())}")
    return backends[name]()
