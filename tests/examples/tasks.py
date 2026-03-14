"""
Reusable task definitions for daglite tests.

This module provides canonical `@task`-decorated functions used across unit, integration, and
type-assertion tests.  Tasks here must be **pure** (no mutable state) so they are safe to share
between test files without isolation concerns.

Tasks that require specific decorator parameters (`cache=True`, `retries=3`, `dataset=...`, etc.)
should be defined inline in the test that exercises them.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from collections.abc import Iterator

from daglite import task

# region Unary tasks


@task
def double(x: int) -> int:
    """Unary operation — doubles the input."""
    return x * 2


@task
async def async_double(x: int) -> int:
    """Async unary operation — doubles the input."""
    return x * 2


@task
def to_string(x: int) -> str:
    """Type transformation int → str."""
    return str(x)


@task
async def async_to_string(x: int) -> str:
    """Async type transformation int → str."""
    return str(x)


@task
def maybe_none(x: int) -> int | None:
    """Task returning an optional type."""
    return x if x > 0 else None


@task
async def async_maybe_none(x: int) -> int | None:
    """Async task returning an optional type."""
    return x if x > 0 else None


# region Binary tasks


@task
def add(x: int, y: int) -> int:
    """Binary operation — adds two integers."""
    return x + y


@task
async def async_add(x: int, y: int) -> int:
    """Async binary operation — adds two integers."""
    return x + y


@task
def multiply(x: int, factor: int) -> int:
    """Multiply a number by a factor."""
    return x * factor


# region Named tasks

# NOTE: These tasks are used to test the use of the `@task` decorator with explicit parameters.


@task(name="named_add")
def named_add(x: int, y: int) -> int:
    """Binary add with an explicit ``name`` — exercises the decorator-with-kwargs path."""
    return x + y


@task(name="async_named_add")
async def async_named_add(x: int, y: int) -> int:
    """Async binary add with an explicit ``name``."""
    return x + y


# region Erroring tasks


@task
def broken(x: int) -> int:
    """A task that always raises RuntimeError."""
    raise RuntimeError("boom")


@task
async def async_broken(x: int) -> int:
    """An async task that always raises RuntimeError."""
    raise RuntimeError("async boom")


@task
def fail_on_three(x: int) -> int:
    """A task that raises ValueError when input is 3."""
    if x == 3:
        raise ValueError("three is bad")
    return x


@task
def failing_task(x: int) -> int:
    """A task that always raises with a descriptive message."""
    raise RuntimeError(f"Intentional failure with input: {x}")


# region Streaming tasks


@task
def count_up(n: int) -> Iterator[int]:
    """Sync generator — yields 0..n-1."""
    for i in range(n):
        yield i


@task
async def async_count_up(n: int) -> AsyncIterator[int]:
    """Async generator — yields 0..n-1."""
    for i in range(n):
        yield i


@task
def broken_stream(n: int) -> Iterator[int]:
    """Sync generator that raises after yielding one item."""
    yield 0
    raise RuntimeError("stream boom")


@task
async def async_broken_stream(n: int) -> AsyncIterator[int]:
    """Async generator that raises after yielding one item."""
    yield 0
    raise RuntimeError("async stream boom")
