from daglite.backends.local import LocalBackend
from daglite.backends.threading import ThreadBackend

from .base import Backend


def find_backend(backend: str | Backend | None = None) -> Backend:
    """
    Find a backend class by name.

    Args:
        backend (daglite.engine.Backend | str, optional):
            Name or instance of the backend to find. If an instance is given, it is
            returned directly. If None, defaults to "local".

    Returns:
        Backend class.
    """

    if isinstance(backend, Backend):
        return backend

    backend = backend if backend is not None else "local"

    backends = {
        "local": LocalBackend,
        "synchronous": LocalBackend,
        "threading": ThreadBackend,
    }

    # TODO : dynamic discovery of backends from entry points

    if backend not in backends:
        raise ValueError(f"Unknown backend '{backend}'; available: {list(backends.keys())}")
    return backends[backend]()
