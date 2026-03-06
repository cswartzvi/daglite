"""
Type assertions for the daglite eager execution API.

These tests do not run any actual computations; they only verify that type
checkers infer correct types for task bindings, workflow decorators, and
composition primitives.
"""

# pyright: reportUnusedCoroutine=false
# mypy: disable-error-code="unused-coroutine"

from collections.abc import Coroutine
from typing import Any

from typing_extensions import assert_type

from daglite import task
from daglite import workflow
from daglite.mapping import gather_tasks
from daglite.mapping import map_task
from daglite.tasks import AsyncTask
from daglite.tasks import SyncTask
from daglite.workflows import AsyncWorkflow
from daglite.workflows import SyncWorkflow

# region Helper tasks


@task
def add(x: int, y: int) -> int:
    """Binary operation for basic tests."""
    return x + y


@task
def double(x: int) -> int:
    """Unary operation for map tests."""
    return x * 2


@task
def to_string(x: int) -> str:
    """Type transformation int -> str."""
    return str(x)


@task
def maybe_none(x: int) -> int | None:
    """Task returning optional type."""
    return x if x > 0 else None


@task
async def async_add(x: int, y: int) -> int:
    """Async binary operation."""
    return x + y


@task
async def async_double(x: int) -> int:
    """Async unary operation."""
    return x * 2


# region Bare @task — decorator type


def test_sync_task_bare_decorator_type() -> None:
    """``@task`` on a sync function produces a ``SyncTask``."""
    assert isinstance(add, SyncTask)


def test_async_task_bare_decorator_type() -> None:
    """``@task`` on an async function produces an ``AsyncTask``."""
    assert isinstance(async_add, AsyncTask)


# region Bare @task — call return type


def test_sync_task_call_returns_value() -> None:
    """Calling a ``SyncTask`` returns the raw value type ``R``."""
    result = add(x=1, y=2)
    assert_type(result, int)


def test_sync_task_preserves_optional() -> None:
    """Optional return types are preserved."""
    result = maybe_none(x=5)
    assert_type(result, int | None)


def test_sync_task_str_return() -> None:
    """Type transformation is preserved."""
    result = to_string(x=5)
    assert_type(result, str)


def test_async_task_call_returns_coroutine() -> None:
    """Calling an ``AsyncTask`` returns ``Coroutine[Any, Any, R]``."""
    result = async_add(x=1, y=2)
    assert_type(result, Coroutine[Any, Any, int])


async def test_async_task_awaited_returns_value() -> None:
    """Awaiting an ``AsyncTask`` call returns the raw value type ``R``."""
    result = await async_add(x=1, y=2)
    assert_type(result, int)


# region @task(keyword) — decorator type


def test_sync_task_keyword_decorator_type() -> None:
    """``@task(cache=True)`` on a sync function produces a ``SyncTask``."""

    @task(cache=True)
    def cached_add(x: int, y: int) -> int:
        return x + y

    assert isinstance(cached_add, SyncTask)


def test_async_task_keyword_decorator_type() -> None:
    """``@task(cache=True)`` on an async function produces an ``AsyncTask``."""

    @task(cache=True)
    async def cached_async(x: int, y: int) -> int:
        return x + y

    assert isinstance(cached_async, AsyncTask)


# region @task(keyword) — call return type


def test_sync_task_keyword_call_returns_value() -> None:
    """Keyword-decorated sync task call returns ``R``."""

    @task(cache=True)
    def cached_add(x: int, y: int) -> int:
        return x + y

    result = cached_add(x=1, y=2)
    assert_type(result, int)


def test_async_task_keyword_call_returns_coroutine() -> None:
    """Keyword-decorated async task call returns ``Coroutine[Any, Any, R]``."""

    @task(cache=True)
    async def cached_async(x: int, y: int) -> int:
        return x + y

    result = cached_async(x=1, y=2)
    assert_type(result, Coroutine[Any, Any, int])


async def test_async_task_keyword_awaited_returns_value() -> None:
    """Awaiting a keyword-decorated async task returns ``R``."""

    @task(cache=True)
    async def cached_async(x: int, y: int) -> int:
        return x + y

    result = await cached_async(x=1, y=2)
    assert_type(result, int)


# region Bare @workflow — decorator type


def test_sync_workflow_bare_decorator_type() -> None:
    """``@workflow`` on a sync function produces a ``SyncWorkflow``."""

    @workflow
    def wf(x: int, y: int) -> int:
        return add(x=x, y=y)

    assert isinstance(wf, SyncWorkflow)


def test_async_workflow_bare_decorator_type() -> None:
    """``@workflow`` on an async function produces an ``AsyncWorkflow``."""

    @workflow
    async def wf(x: int, y: int) -> int:
        return await async_add(x=x, y=y)

    assert isinstance(wf, AsyncWorkflow)


# region @workflow(keyword) — decorator type


def test_sync_workflow_keyword_decorator_type() -> None:
    """``@workflow(name=...)`` on a sync function produces a ``SyncWorkflow``."""

    @workflow(name="custom")
    def wf(x: int) -> int:
        return double(x=x)

    assert isinstance(wf, SyncWorkflow)


def test_async_workflow_keyword_decorator_type() -> None:
    """``@workflow(name=...)`` on an async function produces an ``AsyncWorkflow``."""

    @workflow(name="custom")
    async def wf(x: int) -> int:
        return await async_double(x=x)

    assert isinstance(wf, AsyncWorkflow)


# region Workflow call return types


def test_sync_workflow_call_returns_value() -> None:
    """Calling a ``SyncWorkflow`` returns the raw value type ``R``."""

    @workflow
    def wf(x: int, y: int) -> int:
        return add(x=x, y=y)

    result = wf(1, 2)
    assert_type(result, int)


def test_async_workflow_call_returns_coroutine() -> None:
    """Calling an ``AsyncWorkflow`` returns ``Coroutine[Any, Any, R]``."""

    @workflow
    async def wf(x: int, y: int) -> int:
        return await async_add(x=x, y=y)

    result = wf(1, 2)
    assert_type(result, Coroutine[Any, Any, int])


async def test_async_workflow_awaited_returns_value() -> None:
    """Awaiting an ``AsyncWorkflow`` call returns ``R``."""

    @workflow
    async def wf(x: int, y: int) -> int:
        return await async_add(x=x, y=y)

    result = await wf(1, 2)
    assert_type(result, int)


# region task_map / async_task_map


def test_task_map_returns_list() -> None:
    """``task_map`` returns ``list[R]``."""
    result = map_task(double, [1, 2, 3])
    assert_type(result, list[int])  # ty: ignore[type-assertion-failure]  # ty infers list[Unknown]


async def test_async_task_map_returns_list() -> None:
    """``async_task_map`` returns ``list[R]``."""
    result = await gather_tasks(async_double, [1, 2, 3])
    assert_type(result, list[int])  # ty: ignore[type-assertion-failure]  # ty infers list[Unknown]
