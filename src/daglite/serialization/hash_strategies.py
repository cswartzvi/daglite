"""
Hash strategies for built-in Python types.

This module provides hash functions for Python's built-in types only.
For specialized types (numpy, pandas, PIL, etc.), see the daglite_serialization plugin.

All hash functions return SHA256 hex digests for consistency.
"""

import hashlib


def hash_bytes(data: bytes) -> str:
    """Hash bytes directly using SHA256."""
    return hashlib.sha256(data).hexdigest()


def hash_string(s: str) -> str:
    """Hash string using UTF-8 encoding."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def hash_int(n: int) -> str:
    """Hash integer."""
    return hashlib.sha256(str(n).encode()).hexdigest()


def hash_float(f: float) -> str:
    """Hash float."""
    return hashlib.sha256(str(f).encode()).hexdigest()


def hash_bool(b: bool) -> str:
    """Hash boolean."""
    return hashlib.sha256(str(b).encode()).hexdigest()


def hash_none(_: None) -> str:
    """Hash None."""
    return hashlib.sha256(b"None").hexdigest()


def hash_dict(d: dict) -> str:
    """
    Hash dictionary by sorting keys and hashing key-value pairs.

    Note: This uses repr() for values, so it's only suitable for dicts
    containing built-in types. For complex types, register them separately.
    """
    h = hashlib.sha256()
    for key in sorted(d.keys()):
        h.update(str(key).encode())
        h.update(repr(d[key]).encode())
    return h.hexdigest()


def hash_list(lst: list) -> str:
    """
    Hash list by hashing each element in order.

    Note: This uses repr() for values, so it's only suitable for lists
    containing built-in types. For complex types, register them separately.
    """
    h = hashlib.sha256()
    for item in lst:
        h.update(repr(item).encode())
    return h.hexdigest()


def hash_tuple(tup: tuple) -> str:
    """Hash tuple by hashing each element in order."""
    h = hashlib.sha256()
    for item in tup:
        h.update(repr(item).encode())
    return h.hexdigest()


def hash_set(s: set) -> str:
    """Hash set by sorting and hashing elements."""
    h = hashlib.sha256()
    for item in sorted(s, key=repr):
        h.update(repr(item).encode())
    return h.hexdigest()


def hash_frozenset(fs: frozenset) -> str:
    """Hash frozenset by sorting and hashing elements."""
    h = hashlib.sha256()
    for item in sorted(fs, key=repr):
        h.update(repr(item).encode())
    return h.hexdigest()
