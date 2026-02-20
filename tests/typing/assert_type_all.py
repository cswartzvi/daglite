"""Tests for type assertions using `assert_type` from `typing_extensions`."""

# NOTE: These tests do not run any actual computations; they only verify that the types of task
# bindings and compositions are as expected.

from typing import Any

from typing_extensions import assert_type

from daglite import Dag
from daglite import pipeline
from daglite import task
from daglite.futures import MapTaskFuture
from daglite.futures import TaskFuture
from daglite.futures.base import BaseTaskFuture
from daglite.workflow_result import WorkflowResult

# -- Helper tasks for type tests --


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
def sum_list(xs: list[int]) -> int:
    """Aggregation for join tests."""
    return sum(xs)


@task
def join_strings(xs: list[str]) -> str:
    """Aggregation for string join tests."""
    return ",".join(xs)


@task
def get_dict() -> dict[str, Any]:
    """Task returning dict type."""
    return {"key": "value"}


@task
def maybe_none(x: int) -> int | None:
    """Task returning optional type."""
    return x if x > 0 else None


@task
async def async_add(x: int, y: int) -> int:
    """Async binary operation."""
    return x + y


# -- Core API: __call__() --


def test_bind_basic() -> None:
    """Test basic task binding with scalar parameters."""
    result = add(x=1, y=2)
    assert_type(result, TaskFuture[int])
    assert_type(result.run(), int)


def test_bind_with_dependencies() -> None:
    """Test binding with TaskFuture dependencies."""
    dep = double(x=5)
    result = add(x=dep, y=10)
    assert_type(result, TaskFuture[int])
    assert_type(result.run(), int)


def test_bind_return_types() -> None:
    """Test that bind() preserves various return types."""
    int_result = add(x=1, y=2)
    assert_type(int_result, TaskFuture[int])
    assert_type(int_result.run(), int)

    str_result = to_string(x=5)
    assert_type(str_result, TaskFuture[str])
    assert_type(str_result.run(), str)

    dict_result = get_dict()
    assert_type(dict_result, TaskFuture[dict[str, Any]])
    assert_type(dict_result.run(), dict[str, Any])

    optional_result = maybe_none(x=5)
    assert_type(optional_result, TaskFuture[int | None])
    assert_type(optional_result.run(), int | None)


# -- Core API: partial() --


def test_partial_binding() -> None:
    """Test partial() for partial parameter binding."""
    partial_add = add.partial(y=10)
    result = partial_add(x=5)
    assert_type(result, TaskFuture[int])
    assert_type(result.run(), int)


def test_partial_with_options() -> None:
    """Test that partial() and with_options() work together."""
    partial_with_opts = add.partial(y=10).with_options(backend_name="threading")(x=5)
    assert_type(partial_with_opts, TaskFuture[int])
    assert_type(partial_with_opts.run(), int)


# -- Fan-out API: product() and zip() --


def test_product_basic() -> None:
    """Test product() creates MapTaskFuture."""
    result = double.map(x=[1, 2, 3])
    assert_type(result, MapTaskFuture[int])
    assert_type(result.run(), list[int])


def test_zip_basic() -> None:
    """Test zip() creates MapTaskFuture."""
    result = add.map(x=[1, 2, 3], y=[10, 20, 30])
    assert_type(result, MapTaskFuture[int])
    assert_type(result.run(), list[int])


def test_product_vs_zip_semantics() -> None:
    """Test Cartesian product vs pairwise zip."""
    cartesian = add.map(x=[1, 2], y=[10, 20], map_mode="product")
    assert_type(cartesian, MapTaskFuture[int])
    assert_type(cartesian.run(), list[int])

    pairwise = add.map(x=[1, 2], y=[10, 20])
    assert_type(pairwise, MapTaskFuture[int])
    assert_type(pairwise.run(), list[int])


def test_product_with_partial() -> None:
    """Test product() with partially fixed parameters."""
    result = add.partial(y=10).map(x=[1, 2, 3])
    assert_type(result, MapTaskFuture[int])
    assert_type(result.run(), list[int])


def test_product_nested() -> None:
    """Test nested product operations."""
    level1 = double.map(x=[1, 2, 3])
    level2 = add.map(x=level1, y=[100, 200])
    assert_type(level2, MapTaskFuture[int])
    assert_type(level2.run(), list[int])


# -- Map API: then() --


def test_then_basic() -> None:
    """Test then() for mapping over MapTaskFuture."""
    mapped = double.map(x=[1, 2, 3]).then(double)
    assert_type(mapped, MapTaskFuture[int])
    assert_type(mapped.run(), list[int])


def test_then_type_transformation() -> None:
    """Test then() with type transformation."""
    result = double.map(x=[1, 2, 3]).then(to_string)
    assert_type(result, MapTaskFuture[str])
    assert_type(result.run(), list[str])


def test_then_chaining() -> None:
    """Test chaining multiple then() calls."""
    result = double.map(x=[1, 2, 3]).then(double).then(to_string)
    assert_type(result, MapTaskFuture[str])
    assert_type(result.run(), list[str])


# -- Join API: join() --


def test_join_from_product() -> None:
    """Test join() aggregates MapTaskFuture to TaskFuture."""
    result = double.map(x=[1, 2, 3]).join(sum_list)
    assert_type(result, TaskFuture[int])
    assert_type(result.run(), int)


def test_join_with_type_change() -> None:
    """Test join() with type transformation."""
    result = double.map(x=[1, 2, 3]).then(to_string).join(join_strings)
    assert_type(result, TaskFuture[str])
    assert_type(result.run(), str)


def test_join_after_then_chain() -> None:
    """Test join() after chained then() operations."""
    result = double.map(x=[1, 2, 3]).then(double).then(add.partial(y=10)).join(sum_list)
    assert_type(result, TaskFuture[int])
    assert_type(result.run(), int)


# -- Fluent API: then_map() --


def test_then_map_product_basic() -> None:
    """Test then_map() in product mode on TaskFuture."""
    prep = double(x=5)
    result = prep.then_map(add, y=[10, 20, 30], map_mode="product")
    assert_type(result, MapTaskFuture[int])
    assert_type(result.run(), list[int])


def test_then_map_product_chaining() -> None:
    """Test then_map() in product mode  with then() and join()."""
    prep = double(x=5)
    chained = prep.then_map(add, y=[10, 20], map_mode="product").then(double).join(sum_list)
    assert_type(chained, TaskFuture[int])
    assert_type(chained.run(), int)


def test_then_map_zip_basic() -> None:
    """Test then_map() in zip mode on TaskFuture."""
    prep = double(x=5)
    result = prep.then_map(add, y=[10, 20, 30])
    assert_type(result, MapTaskFuture[int])
    assert_type(result.run(), list[int])


def test_then_map_zip_chaining() -> None:
    """Test then_map() in zip mode with then() and join()."""
    prep = double(x=5)
    chained = prep.then_map(add, y=[10, 20]).then(to_string).join(join_strings)
    assert_type(chained, TaskFuture[str])
    assert_type(chained.run(), str)


# -- Pipeline decorator --


def test_pipeline_decorator() -> None:
    """Test @pipeline decorator preserves return type."""

    @pipeline
    def compute(x: int, y: int) -> TaskFuture[int]:
        return add(x=x, y=y)

    result = compute(5, 10)
    assert_type(result, Any)


def test_pipeline_run() -> None:
    """Test Pipeline.run() returns WorkflowResult (pipeline is now an alias for workflow)."""

    @pipeline
    def compute(x: int, y: int) -> TaskFuture[int]:
        return add(x=x, y=y)

    result = compute.run(5, 10)
    assert_type(result, WorkflowResult)


def test_pipeline_run_map() -> None:
    """Test Pipeline.run() returns WorkflowResult for MapTaskFuture pipelines."""

    @pipeline
    def sweep(values: list[int]) -> MapTaskFuture[int]:
        return double.map(x=values)

    result = sweep.run([1, 2, 3])
    assert_type(result, WorkflowResult)


def test_pipeline_dag_annotation() -> None:
    """Test Dag[T] annotation works as workflow return type."""

    @pipeline
    def compute(x: int, y: int) -> Dag[int]:
        return add(x=x, y=y)

    result = compute(5, 10)
    assert_type(result, Any)


def test_pipeline_dag_annotation_map() -> None:
    """Test Dag[T] annotation works for mapped workflows."""

    @pipeline
    def sweep(values: list[int]) -> Dag[int]:
        return double.map(x=values)

    result = sweep(values=[1, 2, 3])
    assert_type(result, Any)


# -- Async tasks --


def test_async_task_basic() -> None:
    """Async tasks return TaskFuture[Coroutine[...]].."""
    # See type checker specific assertions in separate files


def test_async_task_with_product() -> None:
    """Async tasks work with product()."""
    # See type checker specific assertions in separate files


async def test_async_task_evaluation() -> None:
    """evaluate_async() unwraps coroutines."""
    # See assert_type_tests_mypy.py and assert_type_tests_pyright.py for type assertions


def test_mixed_sync_async() -> None:
    """Sync tasks can depend on async task results."""
    async_result = async_add(x=5, y=10)
    sync_result = double(x=async_result)
    assert_type(sync_result, TaskFuture[int])


# -- Generators --


def test_sync_generator_types() -> None:
    """Sync generators preserve static type, run() returns list."""
    from collections.abc import Generator

    @task
    def generate(n: int) -> Generator[int, None, None]:
        for i in range(n):
            yield i

    future = generate(n=5)
    assert_type(future, TaskFuture[Generator[int, None, None]])
    assert_type(future.run(), list[int])


def test_async_generator_types() -> None:
    """Async generators wrapped in Coroutine."""
    # See assert_type_tests_mypy.py and assert_type_tests_pyright.py for type assertions
