"""
Type assertions for the daglite eager execution API.

These tests do not run any actual computations; they only verify that type
checkers infer correct types for task bindings, workflow decorators, and
composition primitives.
"""

# pyright: reportUnusedCoroutine=false
# mypy: disable-error-code="unused-coroutine"

from collections.abc import AsyncIterator
from collections.abc import Iterator

from typing_extensions import assert_type

from daglite.composers import gather_tasks
from daglite.composers import map_tasks

from ..examples.tasks import add
from ..examples.tasks import async_add
from ..examples.tasks import async_count_up
from ..examples.tasks import async_double
from ..examples.tasks import async_maybe_none
from ..examples.tasks import async_named_add
from ..examples.tasks import async_to_string
from ..examples.tasks import count_up
from ..examples.tasks import double
from ..examples.tasks import maybe_none
from ..examples.tasks import named_add
from ..examples.tasks import to_string
from ..examples.workflows import async_workflow
from ..examples.workflows import named_async_workflow
from ..examples.workflows import named_sync_workflow
from ..examples.workflows import sync_workflow

# region Task assertions


def test_sync_task_return_types() -> None:
    """Verify that `@task` produces the correct return types."""
    # unary tasks
    assert_type(double(x=5), int)
    assert_type(to_string(x=5), str)
    assert_type(maybe_none(x=5), int | None)

    # binary tasks
    assert_type(add(x=1, y=2), int)
    assert_type(named_add(x=1, y=2), int)


async def test_async_task_awaited_return_types() -> None:
    """Verify that awaiting `@task` produces the correct return types."""
    # unary tasks
    assert_type(await async_double(x=5), int)
    assert_type(await async_to_string(x=5), str)
    assert_type(await async_maybe_none(x=5), int | None)

    # binary tasks
    assert_type(await async_add(x=1, y=2), int)
    assert_type(await async_named_add(x=1, y=2), int)


def test_map_tasks_return_types() -> None:
    """Verify that `map_tasks` produces the correct return types."""

    # NOTE: ty currently does not support type narrowing for `list` return types (return
    # `list[Unknown]` instead of `list[int]`), therefore, we must use a ty-specific ignore.

    # unary tasks
    assert_type(map_tasks(double, [1, 2, 3]), list[int])  # ty: ignore
    assert_type(map_tasks(to_string, [1, 2, 3]), list[str])  # ty: ignore
    assert_type(map_tasks(maybe_none, [1, 2, 3]), list[int | None])  # ty: ignore

    # binary tasks
    assert_type(map_tasks(add, [1, 2, 3], [10, 20, 30]), list[int])  # ty: ignore
    assert_type(map_tasks(named_add, [1, 2], [10, 20]), list[int])  # ty: ignore


async def test_async_map_tasks_return_types() -> None:
    """Verify that `async_map_tasks` produces the correct return types."""

    # NOTE: ty currently does not support type narrowing for `list` return types (return
    # `list[Unknown]` instead of `list[int]`), therefore, we must use a ty-specific ignore.

    # unary tasks
    assert_type(await gather_tasks(async_double, [1, 2, 3]), list[int])  # ty: ignore
    assert_type(await gather_tasks(async_to_string, [1, 2, 3]), list[str])  # ty: ignore
    assert_type(await gather_tasks(async_maybe_none, [1, 2, 3]), list[int | None])  # ty: ignore

    # binary tasks
    assert_type(await gather_tasks(async_add, [1, 2, 3], [10, 20, 30]), list[int])  # ty: ignore
    assert_type(await gather_tasks(async_named_add, [1, 2], [10, 20]), list[int])  # ty: ignore


# region Streaming task assertions


def test_sync_stream_return_types() -> None:
    """Verify that `@task` on a sync generator produces an Iterator return type."""
    assert_type(count_up(n=5), Iterator[int])


def test_async_stream_return_types() -> None:
    """Verify that `@task` on an async generator produces an AsyncIterator return type."""
    assert_type(async_count_up(n=5), AsyncIterator[int])


# region Workflow assertions


def test_sync_workflow_return_types() -> None:
    """Verify that `@workflow` produces the correct return types for sync workflows."""
    assert_type(sync_workflow(x=1, y=2), int)
    assert_type(named_sync_workflow(x=1, y=2), int)


async def test_async_workflow_return_types() -> None:
    """Verify that `@workflow` produces the correct return types for async workflows."""
    assert_type(await async_workflow(x=1, y=2), int)
    assert_type(await named_async_workflow(x=1, y=2), int)
