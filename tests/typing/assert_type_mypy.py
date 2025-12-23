# pyright: ignore

"""Type assertions specific to mypy's coroutine/async handling."""

from collections.abc import AsyncGenerator
from collections.abc import Coroutine
from typing import Any

from typing_extensions import assert_type

from daglite import task
from daglite.engine import evaluate_async
from daglite.futures import MapTaskFuture
from daglite.futures import TaskFuture


@task
async def async_add(x: int, y: int) -> int:
    """Async binary operation."""
    return x + y


# -- Async tasks (mypy-specific assertions) --


def test_async_task_basic() -> None:
    """Async tasks return TaskFuture[Coroutine[...]]."""
    result = async_add(x=5, y=10)
    assert_type(result, TaskFuture[Coroutine[Any, Any, int]])


def test_async_task_with_product() -> None:
    """Async tasks work with product()."""
    result = async_add.product(x=[1, 2, 3], y=[10, 20, 30])
    assert_type(result, MapTaskFuture[Coroutine[Any, Any, int]])


async def test_async_task_evaluation() -> None:
    """evaluate() unwraps coroutines from async tasks."""
    future = async_add(x=5, y=10)
    result = await evaluate_async(future)
    assert_type(result, int)


# -- Async generators (mypy-specific assertions) --


def test_async_generator_types() -> None:
    """Async generators wrapped in Coroutine."""

    @task
    async def async_generate(n: int) -> AsyncGenerator[int, None]:
        async def _gen() -> AsyncGenerator[int, None]:
            for i in range(n):
                yield i

        return _gen()

    future = async_generate(n=5)
    assert_type(future, TaskFuture[Coroutine[Any, Any, AsyncGenerator[int, None]]])
