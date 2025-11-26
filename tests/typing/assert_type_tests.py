"""Tests for type assertions using `assert_type` from `typing_extensions`."""

# NOTE: These tests do not run any actual computations; they only verify that the types of task
# bindings and compositions are as expected.

# NOTE: `assert_type` checks do not handle type `ParamSpec` correctly. Therefore asserting the
# types of `Task` or `FixedParamTask` cannot be done here. We will have to rely on the overall
# type checking of the daglite package for those.

from typing import Any

from typing_extensions import assert_type

from daglite import pipeline
from daglite import task
from daglite.tasks import MapTaskFuture
from daglite.tasks import TaskFuture


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


# -- Scalar parameters --
def test_task_simple_parameters() -> None:
    simple_score = score.bind(x=1, y=2)
    assert_type(simple_score, TaskFuture[int])


# -- Scalar parameters with options --
def test_task_parameters_with_options() -> None:
    simple_score_options = score.with_options(backend="threading").bind(x=1, y=2)
    assert_type(simple_score_options, TaskFuture[int])


# -- Reference parameters --
def test_task_reference_parameters() -> None:
    prepared = prepare.bind(n=3)

    mixed_score = score.bind(x=prepared, y=2)
    assert_type(mixed_score, TaskFuture[int])

    combined = double.bind(x=prepared)
    assert_type(combined, TaskFuture[int])


# -- Reference parameters with options --
def test_task_reference_parameters_with_options() -> None:
    prepared = prepare.with_options(backend="threading").bind(n=3)
    mixed_score = score.with_options(backend="threading").bind(x=prepared, y=2)
    assert_type(mixed_score, TaskFuture[int])


# -- Fan-out with extend map and join --
def test_task_extend_map_join() -> None:
    prepared = prepare.extend(n=[1, 2, 3])
    assert_type(prepared, MapTaskFuture[int])

    doubled = prepared.map(double)
    assert_type(doubled, MapTaskFuture[int])

    joined = doubled.join(sum_list)
    assert_type(joined, TaskFuture[int])


# -- Fan-out with extend and mixed parameters --
def test_task_extend_with_fix() -> None:
    extend_score = score.fix(y=10).extend(x=[1, 2, 3])
    assert_type(extend_score, MapTaskFuture[int])


# -- Fan-out with zip, map, and join --
def test_task_zip_map_join() -> None:
    zip_prepared = prepare.zip(n=[1, 2, 3])
    assert_type(zip_prepared, MapTaskFuture[int])

    zip_doubled = zip_prepared.map(double)
    assert_type(zip_doubled, MapTaskFuture[int])

    zip_joined = zip_doubled.join(sum_list)
    assert_type(zip_joined, TaskFuture[int])


# -- Fan-out with zip and mixed parameters --
def test_task_zip_with_fix() -> None:
    zip_score = score.fix(y=20).zip(x=[1, 2, 3])
    assert_type(zip_score, MapTaskFuture[int])


# -- Tasks with no parameters --
def test_task_no_parameters() -> None:
    constant = get_constant.bind()
    assert_type(constant, TaskFuture[str])


# -- Different return types --
def test_task_different_return_types() -> None:
    data = fetch_data.bind()
    assert_type(data, TaskFuture[dict[str, Any]])

    items = get_items.bind()
    assert_type(items, TaskFuture[list[str]])

    tuple_result = split.bind(x=10)
    assert_type(tuple_result, TaskFuture[tuple[int, int]])


# -- Optional return types --
def test_task_optional_return_types() -> None:
    maybe = maybe_value.bind(x=5)
    assert_type(maybe, TaskFuture[int | None])

    maybe_negative = maybe_value.bind(x=-5)
    assert_type(maybe_negative, TaskFuture[int | None])


# -- Multiple dependencies (complex graph) --
def test_task_multiple_dependencies() -> None:
    dep_a = prepare.bind(n=1)
    dep_b = prepare.bind(n=2)
    dep_c = three_args.bind(x=dep_a, y=dep_b, z=10)

    combined = score.bind(x=dep_a, y=dep_c)
    assert_type(combined, TaskFuture[int])


# -- Nested map chains --
def test_task_nested_map_chains() -> None:
    nested_maps = prepare.extend(n=[1, 2, 3]).map(double).map(double).join(sum_list)
    assert_type(nested_maps, TaskFuture[int])


# -- MapTaskFuture without join --
def test_task_map_without_join() -> None:
    mapped_only = prepare.extend(n=[1, 2]).map(double)
    assert_type(mapped_only, MapTaskFuture[int])


# -- Using fix with partial application --
def test_task_fix_partial_application() -> None:
    fixed_once = three_args.fix(z=100)
    partially_bound = fixed_once.bind(x=1, y=10)
    assert_type(partially_bound, TaskFuture[int])

    fixed_two_params = three_args.fix(y=10, z=100)
    final_bound = fixed_two_params.bind(x=1)
    assert_type(final_bound, TaskFuture[int])


# -- extend vs zip with multiple parameters --
def test_task_extend_vs_zip_multiple_params() -> None:
    cartesian_scores = score.extend(x=[1, 2], y=[10, 20])
    assert_type(cartesian_scores, MapTaskFuture[int])

    zipped_scores = score.zip(x=[1, 2], y=[10, 20])
    assert_type(zipped_scores, MapTaskFuture[int])


# -- Nested extend (fan-out over fan-out results) --
def test_task_nested_extend() -> None:
    level1_extend = double.extend(x=[1, 2, 3])
    assert_type(level1_extend, MapTaskFuture[int])

    level2_extend = score.extend(x=level1_extend, y=[100, 200])
    assert_type(level2_extend, MapTaskFuture[int])

    nested_extend_result = level2_extend.join(sum_list)
    assert_type(nested_extend_result, TaskFuture[int])


# -- Nested zip (fan-out over fan-out results) --
def test_task_nested_zip() -> None:
    level1_zip = double.zip(x=[1, 2, 3])
    assert_type(level1_zip, MapTaskFuture[int])

    level2_zip = score.zip(x=level1_zip, y=[100, 200, 300])
    assert_type(level2_zip, MapTaskFuture[int])

    nested_zip_result = level2_zip.join(sum_list)
    assert_type(nested_zip_result, TaskFuture[int])


# -- Map chains with different return types --
def test_task_map_chain_type_changes() -> None:
    type_changing_chain = prepare.extend(n=[1, 2, 3]).map(double).map(to_string)
    assert_type(type_changing_chain, MapTaskFuture[str])

    type_change_joined = type_changing_chain.join(join_strings)
    assert_type(type_change_joined, TaskFuture[str])


# -- Mixing concrete values and TaskFutures in extend/zip --
def test_task_mixed_concrete_and_futures() -> None:
    prep_future = prepare.bind(n=10)
    mixed_extend = score.extend(x=[1, 2], y=prep_future)
    assert_type(mixed_extend, MapTaskFuture[int])

    mixed_zip = score.zip(x=[1, 2], y=prep_future)
    assert_type(mixed_zip, MapTaskFuture[int])


# -- with_options on FixedParamTask --
def test_task_with_options_on_fixed_param_task() -> None:
    fixed_with_options = score.fix(y=10).with_options(backend="threading")
    options_result = fixed_with_options.bind(x=5)
    assert_type(options_result, TaskFuture[int])


# -- Empty iterables in extend/zip --
def test_task_empty_iterables() -> None:
    empty_extend = prepare.extend(n=[])
    assert_type(empty_extend, MapTaskFuture[int])

    empty_zip = prepare.zip(n=[])
    assert_type(empty_zip, MapTaskFuture[int])


# -- Scalar pipeline --
def test_pipeline_with_scalar_parameters() -> None:
    @task
    def add(x: int, y: int) -> int:
        return x + y

    @pipeline
    def simple_pipeline(x: int, y: int) -> TaskFuture[int]:
        return add.bind(x=x, y=y)

    pipeline_result = simple_pipeline(5, 10)
    assert_type(pipeline_result, TaskFuture[int])


# -- Pipeline with extend --
def test_pipeline_with_extend() -> None:
    @task
    def increment(x: int) -> int:
        return x + 1

    @pipeline
    def extend_pipeline(values: list[int]) -> MapTaskFuture[int]:
        return increment.extend(x=values)

    pipeline_result = extend_pipeline([1, 2, 3])
    assert_type(pipeline_result, MapTaskFuture[int])


# -- Pipeline with zip --
def test_pipeline_with_zip() -> None:
    @task
    def square(x: int) -> int:
        return x * x

    @pipeline
    def zip_pipeline(values: list[int]) -> MapTaskFuture[int]:
        return square.zip(x=values)

    pipeline_result = zip_pipeline([1, 2, 3])
    assert_type(pipeline_result, MapTaskFuture[int])
