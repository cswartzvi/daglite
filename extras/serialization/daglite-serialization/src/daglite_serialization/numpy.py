"""NumPy serialization and hashing handlers."""

import hashlib

import numpy as np

from daglite.serialization import default_registry


def hash_numpy_array(arr: np.ndarray) -> str:
    """Fast hash for numpy arrays using metadata + sample.

    Strategy:
    - Hash shape, dtype (always fast)
    - Sample small chunks from beginning, middle, and end
    - Full hash for small arrays

    Performance: <100ms for 800MB array (vs ~2000ms for full hash)

    Warning:
        This is a sample-based hash that may miss changes in the middle of
        very large arrays. For critical applications where you need to detect
        all changes, consider using full hashing:

        >>> def hash_full(arr):
        ...     return hashlib.sha256(arr.tobytes()).hexdigest()
        >>> default_registry.register_hash_strategy(np.ndarray, hash_full)

    Args:
        arr: numpy ndarray

    Returns:
        SHA256 hex digest
    """
    h = hashlib.sha256()

    # Hash metadata
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())

    # Sample data
    if arr.size > 10000:
        # For very large arrays, sample from start, middle, and end
        if arr.ndim == 1:
            # 1D array - sample from three locations
            h.update(arr[:1000].tobytes())
            mid = arr.size // 2
            h.update(arr[mid : mid + 1000].tobytes())
            h.update(arr[-1000:].tobytes())
        elif arr.ndim == 2:
            # 2D array - sample first, middle, and last rows
            h.update(arr[:10].tobytes())
            mid = arr.shape[0] // 2
            h.update(arr[mid : mid + 10].tobytes())
            h.update(arr[-10:].tobytes())
        else:
            # Multi-dimensional - flatten just the portions we need
            flat = arr.ravel()
            h.update(flat[:1000].tobytes())
            mid = flat.size // 2
            h.update(flat[mid : mid + 1000].tobytes())
            h.update(flat[-1000:].tobytes())
    else:
        # Small enough to hash completely
        h.update(arr.tobytes())

    return h.hexdigest()


def register_numpy_handlers():
    """Register numpy array handlers with the default registry.

    This registers:
    - Hash strategy for np.ndarray (sample-based for performance)

    Example:
        >>> from daglite_serialization.numpy import register_numpy_handlers
        >>> register_numpy_handlers()
    """
    # Register hash strategy
    default_registry.register_hash_strategy(
        np.ndarray, hash_numpy_array, "Sample-based hash for numpy arrays"
    )

    # Could also register serialization formats here if needed
    # For example: .npy format, compressed formats, etc.
