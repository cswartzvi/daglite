"""Pandas serialization and hashing handlers."""

import hashlib

import pandas as pd

from daglite.serialization import default_registry


def hash_pandas_dataframe(df: pd.DataFrame) -> str:
    """
    Fast hash for pandas DataFrames using schema + sample rows.

    Strategy:
    - Hash shape, column names, dtypes (always fast)
    - Sample first/last 500 rows for large DataFrames
    - Full hash for small DataFrames

    Performance: ~10ms for 1M rows (vs ~5000ms for full hash)

    Warning:
        This is a sample-based hash that may miss changes in the middle of
        very large DataFrames. For critical applications, consider full hashing.

    Args:
        df: pandas DataFrame

    Returns:
        SHA256 hex digest
    """
    h = hashlib.sha256()

    # Hash schema
    h.update(str(df.shape).encode())
    h.update(str(df.dtypes.to_dict()).encode())
    h.update(str(df.columns.tolist()).encode())

    # Sample rows
    if len(df) > 1000:
        # Sample from start, middle, and end
        mid = len(df) // 2
        sample = pd.concat([df.head(333), df.iloc[mid : mid + 334], df.tail(333)])
        h.update(pd.util.hash_pandas_object(sample).values.tobytes())  # type: ignore
    else:
        h.update(pd.util.hash_pandas_object(df).values.tobytes())  # type: ignore

    return h.hexdigest()


def hash_pandas_series(series: pd.Series) -> str:
    """
    Fast hash for pandas Series using dtype + sample values.

    Args:
        series: pandas Series

    Returns:
        SHA256 hex digest
    """
    h = hashlib.sha256()

    # Hash metadata
    h.update(str(len(series)).encode())
    h.update(str(series.dtype).encode())
    h.update(str(series.name).encode())

    # Sample values
    if len(series) > 1000:
        mid = len(series) // 2
        sample = pd.concat([series.head(333), series.iloc[mid : mid + 334], series.tail(333)])
        h.update(pd.util.hash_pandas_object(sample).values.tobytes())  # type: ignore
    else:
        h.update(pd.util.hash_pandas_object(series).values.tobytes())  # type: ignore

    return h.hexdigest()


def register_handlers():
    """
    Register pandas handlers with the default registry.

    This registers:
    - Hash strategy for pd.DataFrame (schema + sample-based)
    - Hash strategy for pd.Series (dtype + sample-based)

    Example:
        >>> from daglite_serialization.pandas import register_handlers
        >>> register_handlers()
    """
    # Register hash strategies
    default_registry.register_hash_strategy(
        pd.DataFrame, hash_pandas_dataframe, "Sample-based hash for pandas DataFrames"
    )

    default_registry.register_hash_strategy(
        pd.Series, hash_pandas_series, "Sample-based hash for pandas Series"
    )

    # Could also register serialization formats here if needed
    # For example: CSV, Parquet, Feather, etc.
