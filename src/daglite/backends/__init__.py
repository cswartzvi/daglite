from daglite.backends.local import ProcessBackend
from daglite.backends.local import SequentialBackend
from daglite.backends.local import ThreadBackend
from daglite.exceptions import BackendError

from .base import Backend


def find_backend(backend: str | Backend | None = None) -> Backend:
    """
    Find a backend class by name.

    Args:
        backend (daglite.engine.Backend | str, optional):
            Name or instance of the backend to find. If an instance is given, it is
            returned directly. If None, defaults to "sequential".

    Returns:
        An instance of the requested backend class (or the default).
    """

    if isinstance(backend, Backend):
        return backend

    backend = backend if backend is not None else "sequential"

    backends = {
        "sequential": SequentialBackend,
        "synchronous": SequentialBackend,
        "threading": ThreadBackend,
        "threads": ThreadBackend,
        "multiprocessing": ProcessBackend,
        "processes": ProcessBackend,
    }

    # TODO : dynamic discovery of backends from entry points

    if backend not in backends:
        raise BackendError(f"Unknown backend '{backend}'; available: {list(backends.keys())}")
    return backends[backend]()
