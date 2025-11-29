"""Tests for type assertions using `assert_type` from `typing_extensions`."""

# NOTE: These tests do not run any actual computations; they only verify that the types of task
# bindings and compositions are as expected.

# NOTE: `assert_type` checks do not handle type `ParamSpec` correctly. Therefore asserting the
# types of `Task` or `FixedParamTask` cannot be done here. We will have to rely on the overall
# type checking of the daglite package for those.

from typing import TYPE_CHECKING, Any

from typing_extensions import assert_type

# Type checkers disagree on coroutine types:
# - pyright infers CoroutineType (implementation from types)
# - mypy infers Coroutine (protocol from collections.abc)
# We use Coroutine and add pyright: ignore comments where needed
if TYPE_CHECKING:
    from collections.abc import Coroutine

from daglite import pipeline
from daglite import task
from daglite.futures import MapTaskFuture
from daglite.futures import TaskFuture


@task
def prepare(n: int) -> int:
    return n


@task
def double(x: int) -> int:
    return 2 * x


@task
def sum_list(xs: list[int]) -> int:
    return sum(xs)


@task
def score(x: int, y: int) -> int:
    return x + 2 * y


@task
def get_constant() -> str:
    return "constant"


@task
def fetch_data() -> dict[str, Any]:
    return {"key": "value"}


@task
def get_items() -> list[str]:
    return ["a", "b", "c"]


@task
def three_args(x: int, y: int, z: int) -> int:
    return x + y + z


@task
def maybe_value(x: int) -> int | None:
    return x if x > 0 else None


@task
def split(x: int) -> tuple[int, int]:
    return (x, x * 2)


@task
def to_string(x: int) -> str:
    return str(x)


@task
def join_strings(xs: list[str]) -> str:
    return ",".join(xs)


@task
async def async_fetch_data(url: str) -> dict[str, str]:
    """Async task that returns a dict."""
    return {"url": url, "data": "example"}


@task
async def async_double(x: int) -> int:
    """Async task that doubles a value."""
    return x * 2


@task
async def async_process_list(xs: list[int]) -> int:
    """Async task that processes a list."""
    return sum(xs)


# -- Basic task operations --
def test_task_basic_operations() -> None:
    """Test basic task binding, dependencies, and with_options."""
    # Simple scalar parameters
    simple_score = score.bind(x=1, y=2)
    assert_type(simple_score, TaskFuture[int])

    # Reference parameters (dependencies)
    prepared = prepare.bind(n=3)
    mixed_score = score.bind(x=prepared, y=2)
    assert_type(mixed_score, TaskFuture[int])
    combined = double.bind(x=prepared)
    assert_type(combined, TaskFuture[int])

    # with_options works on tasks and fixed tasks
    with_options = score.with_options(backend="threading").bind(x=1, y=2)
    assert_type(with_options, TaskFuture[int])
    fixed_with_options = score.fix(y=10).with_options(backend="threading").bind(x=5)
    assert_type(fixed_with_options, TaskFuture[int])


# -- Fan-out operations (product and zip) --
def test_task_fanout_operations() -> None:
    """Test product/zip with map and join operations."""
    # product: fan-out -> map -> join
    product_result = prepare.product(n=[1, 2, 3])
    assert_type(product_result, MapTaskFuture[int])
    product_mapped = product_result.map(double)
    assert_type(product_mapped, MapTaskFuture[int])
    product_joined = product_mapped.join(sum_list)
    assert_type(product_joined, TaskFuture[int])

    # zip: fan-out -> map -> join
    zip_result = prepare.zip(n=[1, 2, 3])
    assert_type(zip_result, MapTaskFuture[int])
    zip_mapped = zip_result.map(double)
    assert_type(zip_mapped, MapTaskFuture[int])
    zip_joined = zip_mapped.join(sum_list)
    assert_type(zip_joined, TaskFuture[int])

    # product/zip with fixed parameters
    product_fixed = score.fix(y=10).product(x=[1, 2, 3])
    assert_type(product_fixed, MapTaskFuture[int])
    zip_fixed = score.fix(y=20).zip(x=[1, 2, 3])
    assert_type(zip_fixed, MapTaskFuture[int])

    # Empty iterables
    empty_product = prepare.product(n=[])
    assert_type(empty_product, MapTaskFuture[int])
    empty_zip = prepare.zip(n=[])
    assert_type(empty_zip, MapTaskFuture[int])


# -- Return type preservation --
def test_task_return_types() -> None:
    """Test that various return types are preserved correctly."""
    # No parameters
    constant = get_constant.bind()
    assert_type(constant, TaskFuture[str])

    # Different types
    data = fetch_data.bind()
    assert_type(data, TaskFuture[dict[str, Any]])
    items = get_items.bind()
    assert_type(items, TaskFuture[list[str]])
    tuple_result = split.bind(x=10)
    assert_type(tuple_result, TaskFuture[tuple[int, int]])

    # Optional types
    maybe = maybe_value.bind(x=5)
    assert_type(maybe, TaskFuture[int | None])


# -- Complex graph patterns --
def test_task_complex_graphs() -> None:
    """Test complex dependency graphs and chaining."""
    # Multiple dependencies
    dep_a = prepare.bind(n=1)
    dep_b = prepare.bind(n=2)
    dep_c = three_args.bind(x=dep_a, y=dep_b, z=10)
    combined = score.bind(x=dep_a, y=dep_c)
    assert_type(combined, TaskFuture[int])

    # Nested map chains
    nested_maps = prepare.product(n=[1, 2, 3]).map(double).map(double).join(sum_list)
    assert_type(nested_maps, TaskFuture[int])

    # MapTaskFuture without join
    mapped_only = prepare.product(n=[1, 2]).map(double)
    assert_type(mapped_only, MapTaskFuture[int])


# -- Partial application with fix --
def test_task_fix_operations() -> None:
    """Test fix() for partial parameter binding."""
    # Fix one parameter
    fixed_once = three_args.fix(z=100)
    partially_bound = fixed_once.bind(x=1, y=10)
    assert_type(partially_bound, TaskFuture[int])

    # Fix multiple parameters
    fixed_two_params = three_args.fix(y=10, z=100)
    final_bound = fixed_two_params.bind(x=1)
    assert_type(final_bound, TaskFuture[int])

    # product vs zip with multiple parameters (Cartesian vs pairwise)
    cartesian_scores = score.product(x=[1, 2], y=[10, 20])
    assert_type(cartesian_scores, MapTaskFuture[int])
    zipped_scores = score.zip(x=[1, 2], y=[10, 20])
    assert_type(zipped_scores, MapTaskFuture[int])


# -- Nested fan-out operations --
def test_task_nested_fanout() -> None:
    """Test nested product/zip operations (fan-out over fan-out results)."""
    # Nested product
    level1_product = double.product(x=[1, 2, 3])
    assert_type(level1_product, MapTaskFuture[int])
    level2_product = score.product(x=level1_product, y=[100, 200])
    assert_type(level2_product, MapTaskFuture[int])
    nested_product_result = level2_product.join(sum_list)
    assert_type(nested_product_result, TaskFuture[int])

    # Nested zip
    level1_zip = double.zip(x=[1, 2, 3])
    assert_type(level1_zip, MapTaskFuture[int])
    level2_zip = score.zip(x=level1_zip, y=[100, 200, 300])
    assert_type(level2_zip, MapTaskFuture[int])
    nested_zip_result = level2_zip.join(sum_list)
    assert_type(nested_zip_result, TaskFuture[int])


# -- Type transformations in map chains --
def test_task_type_transformations() -> None:
    """Test that type changes propagate correctly through map chains."""
    # Type changes: int -> int -> str
    type_changing_chain = prepare.product(n=[1, 2, 3]).map(double).map(to_string)
    assert_type(type_changing_chain, MapTaskFuture[str])
    type_change_joined = type_changing_chain.join(join_strings)
    assert_type(type_change_joined, TaskFuture[str])

    # Mixing concrete values and TaskFutures
    prep_future = prepare.bind(n=10)
    mixed_product = score.product(x=[1, 2], y=prep_future)
    assert_type(mixed_product, MapTaskFuture[int])
    mixed_zip = score.zip(x=[1, 2], y=prep_future)
    assert_type(mixed_zip, MapTaskFuture[int])


# -- Pipeline decorator --
def test_pipeline_types() -> None:
    """Test that @pipeline decorator preserves types correctly."""

    @task
    def add(x: int, y: int) -> int:
        return x + y

    @pipeline
    def simple_pipeline(x: int, y: int) -> TaskFuture[int]:
        return add.bind(x=x, y=y)

    pipeline_result = simple_pipeline(5, 10)
    assert_type(pipeline_result, TaskFuture[int])


# -- Async tasks (honest coroutine types) --
# NOTE: Type checkers disagree: mypy sees Coroutine, pyright sees CoroutineType.
# Both are correct - CoroutineType is the implementation, Coroutine is the protocol.
# We use Coroutine (the protocol) since it's more general and works at runtime.
def test_async_task_types() -> None:
    """Async tasks return TaskFuture[Coroutine[...]] - the honest type."""
    # Simple async task
    result = async_fetch_data.bind(url="example.com")
    assert_type(result, TaskFuture[Coroutine[Any, Any, dict[str, str]]])  # pyright: ignore[reportAssertTypeFailure]

    # Async task with dependencies
    prep = prepare.bind(n=5)
    doubled = async_double.bind(x=prep)
    assert_type(doubled, TaskFuture[Coroutine[Any, Any, int]])  # pyright: ignore[reportAssertTypeFailure]

    # Async task with map operations
    values = async_double.product(x=[1, 2, 3])
    assert_type(values, MapTaskFuture[Coroutine[Any, Any, int]])  # pyright: ignore[reportAssertTypeFailure]

    # Mixed sync/async composition
    async_result = async_double.bind(x=10)
    sync_result = double.bind(x=async_result)
    assert_type(sync_result, TaskFuture[int])
    sync_first = prepare.bind(n=5)
    async_after = async_double.bind(x=sync_first)
    assert_type(async_after, TaskFuture[Coroutine[Any, Any, int]])  # pyright: ignore[reportAssertTypeFailure]

    # evaluate() unwraps coroutines
    from daglite import evaluate

    result_future = async_double.bind(x=5)
    evaluated = evaluate(result_future)
    assert_type(evaluated, int)  # pyright: ignore[reportAssertTypeFailure,reportUnusedCoroutine]


# -- Sync generator/iterator materialization --
def test_sync_generator_types() -> None:
    """Sync generators preserve static type but materialize to list on evaluation."""
    from collections.abc import Generator
    from collections.abc import Iterator

    from daglite import evaluate

    @task
    def generate_numbers(n: int) -> Generator[int, None, None]:
        for i in range(n):
            yield i

    @task
    def generate_range(n: int) -> Iterator[int]:
        return (i * 2 for i in range(n))

    # Static types preserved
    gen_result = generate_numbers.bind(n=5)
    assert_type(gen_result, TaskFuture[Generator[int, None, None]])
    iter_result = generate_range.bind(n=5)
    assert_type(iter_result, TaskFuture[Iterator[int]])

    # evaluate() returns list[T] for generators/iterators
    gen_evaluated = evaluate(gen_result)
    assert_type(gen_evaluated, list[int])
    iter_evaluated = evaluate(iter_result)
    assert_type(iter_evaluated, list[int])

    # Generators work in map operations
    ranges = generate_range.product(n=[3, 4, 5])
    assert_type(ranges, MapTaskFuture[Iterator[int]])
    evaluated_ranges = evaluate(ranges)
    assert_type(evaluated_ranges, list[Iterator[int]])


def test_async_generator_types() -> None:
    """Async generators wrapped in coroutines preserve type information."""
    from collections.abc import AsyncGenerator
    from collections.abc import AsyncIterator
    from collections.abc import Generator

    @task
    async def async_generate_numbers(n: int) -> AsyncGenerator[int, None]:
        async def _gen():
            for i in range(n):
                yield i

        return _gen()

    @task
    async def async_generate_range(n: int) -> AsyncIterator[str]:
        async def _gen():
            for i in range(n):
                yield str(i)

        return _gen()

    # Async generators are wrapped in Coroutine
    async_gen_result = async_generate_numbers.bind(n=5)
    assert_type(async_gen_result, TaskFuture[Coroutine[Any, Any, AsyncGenerator[int, None]]])  # pyright: ignore[reportAssertTypeFailure]
    async_iter_result = async_generate_range.bind(n=5)
    assert_type(async_iter_result, TaskFuture[Coroutine[Any, Any, AsyncIterator[str]]])  # pyright: ignore[reportAssertTypeFailure]

    # Async generators work in map operations
    ranges = async_generate_numbers.product(n=[3, 4, 5])
    assert_type(ranges, MapTaskFuture[Coroutine[Any, Any, AsyncGenerator[int, None]]])  # pyright: ignore[reportAssertTypeFailure]

    # Mixed sync and async generators
    @task
    def sync_generate(n: int) -> Generator[int, None, None]:
        for i in range(n):
            yield i

    sync_gen = sync_generate.bind(n=3)
    assert_type(sync_gen, TaskFuture[Generator[int, None, None]])
    async_gen = async_generate_numbers.bind(n=3)
    assert_type(async_gen, TaskFuture[Coroutine[Any, Any, AsyncGenerator[int, None]]])  # pyright: ignore[reportAssertTypeFailure]
