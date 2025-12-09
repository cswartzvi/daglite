"""Serialization plugin for daglite with support for numpy, pandas, and PIL."""

__version__ = "0.1.0"


def register_all():
    """Register all available serialization handlers.

    This function attempts to register handlers for:
    - NumPy arrays (if numpy is installed)
    - Pandas DataFrames and Series (if pandas is installed)
    - PIL Images (if pillow is installed)

    Handlers that can't be imported are silently skipped.

    Example:
        >>> from daglite_serialization import register_all
        >>> register_all()  # Register all available handlers
    """
    # Try to register numpy handlers
    try:
        from . import numpy

        numpy.register_numpy_handlers()
    except ImportError:
        pass

    # Try to register pandas handlers
    try:
        from . import pandas

        pandas.register_pandas_handlers()
    except ImportError:
        pass

    # Try to register pillow handlers
    try:
        from . import pillow

        pillow.register_pillow_handlers()
    except ImportError:
        pass


__all__ = ["register_all", "__version__"]
