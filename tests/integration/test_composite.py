"""
Integration tests for composite nodes and the graph optimizer.

These tests verify end-to-end execution of pipelines that get folded into
composite nodes by the optimizer.  They exercise both the optimised path
(default) and the un-optimised fallback (via `enable_graph_optimization=False`).
"""

import pytest

from daglite import task
from daglite.exceptions import ParameterError
from daglite.settings import DagliteSettings
from daglite.settings import set_global_settings


@pytest.fixture(autouse=True)
def _reset_settings():
    """Ensure default settings are restored after each test."""
    yield
    set_global_settings(DagliteSettings())


def _with_optimization(enabled: bool) -> None:
    set_global_settings(DagliteSettings(enable_graph_optimization=enabled))


class TestCompositeTaskExecution:
    """End-to-end tests for task chains folded into CompositeTaskNode."""

    def test_two_node_chain(self) -> None:
        """a.then(b) produces the correct result when folded."""

        @task
        def double(x: int) -> int:
            return x * 2

        @task
        def add_ten(x: int) -> int:
            return x + 10

        result = double(x=5).then(add_ten).run()
        assert result == 20  # 5*2 + 10

    def test_three_node_chain(self) -> None:
        """a.then(b).then(c) produces the correct result."""

        @task
        def a(x: int) -> int:
            return x + 1

        @task
        def b(x: int) -> int:
            return x * 3

        @task
        def c(x: int) -> int:
            return x - 2

        result = a(x=4).then(b).then(c).run()
        assert result == 13  # (4+1)*3 - 2

    def test_chain_with_extra_kwargs(self) -> None:
        """then() with extra kwargs resolves correctly inside composite."""

        @task
        def start(x: int) -> int:
            return x

        @task
        def compute(x: int, factor: int, offset: int) -> int:
            return x * factor + offset

        result = start(x=5).then(compute, factor=3, offset=7).run()
        assert result == 22  # 5*3 + 7

    def test_chain_with_partial(self) -> None:
        """then() with a partial task works correctly."""

        @task
        def start(x: int) -> int:
            return x * 2

        @task
        def scale(x: int, factor: int, offset: int) -> int:
            return x * factor + offset

        result = start(x=3).then(scale.partial(offset=100), factor=5).run()
        assert result == 130  # (3*2)*5 + 100

    def test_optimised_matches_unoptimised(self) -> None:
        """Optimised and un-optimised paths produce the same result."""

        @task
        def a(x: int) -> int:
            return x + 1

        @task
        def b(x: int) -> int:
            return x * 2

        @task
        def c(x: int) -> int:
            return x - 3

        future_factory = lambda: a(x=10).then(b).then(c)  # noqa

        _with_optimization(True)
        optimised_result = future_factory().run()

        _with_optimization(False)
        unoptimised_result = future_factory().run()

        assert optimised_result == unoptimised_result == 19  # (10+1)*2 - 3

    def test_long_chain(self) -> None:
        """A 5-node chain folds and runs correctly."""

        @task
        def inc(x: int) -> int:
            return x + 1

        result = inc(x=0).then(inc).then(inc).then(inc).then(inc).run()
        assert result == 5

    def test_chain_returning_complex_type(self) -> None:
        """Composite chain correctly passes complex types between links."""

        @task
        def make_dict(key: str) -> dict:
            return {"key": key, "values": [1, 2, 3]}

        @task
        def add_field(data: dict) -> dict:
            return {**data, "total": sum(data["values"])}

        @task
        def format_result(data: dict) -> str:
            return f"{data['key']}: {data['total']}"

        result = make_dict(key="test").then(add_field).then(format_result).run()
        assert result == "test: 6"


class TestCompositeMapExecution:
    """End-to-end tests for map chains folded into CompositeMapTaskNode."""

    def test_map_then(self) -> None:
        """map().then() produces correct results when folded."""

        @task
        def double(x: int) -> int:
            return x * 2

        @task
        def add_one(x: int) -> int:
            return x + 1

        result = double.map(x=[1, 2, 3]).then(add_one).run()
        assert result == [3, 5, 7]

    def test_map_then_then(self) -> None:
        """map().then().then() produces correct results."""

        @task
        def identity(x: int) -> int:
            return x

        @task
        def add(x: int, y: int) -> int:
            return x + y

        @task
        def negate(x: int) -> int:
            return -x

        result = identity.map(x=[1, 2, 3]).then(add, y=10).then(negate).run()
        assert result == [-11, -12, -13]

    def test_map_then_with_kwargs(self) -> None:
        """map().then() with extra kwargs resolves correctly."""

        @task
        def scale(x: int, factor: int) -> int:
            return x * factor

        @task
        def add(x: int, offset: int) -> int:
            return x + offset

        result = scale.map(x=[1, 2, 3], factor=[2, 2, 2]).then(add, offset=100).run()
        assert result == [102, 104, 106]

    def test_map_join(self) -> None:
        """map().join() produces the correct aggregated result."""

        @task
        def square(x: int) -> int:
            return x**2

        @task
        def total(values: list[int]) -> int:
            return sum(values)

        result = square.map(x=[1, 2, 3, 4]).join(total).run()
        assert result == 30  # 1 + 4 + 9 + 16

    def test_map_then_join(self) -> None:
        """map().then().join() produces the correct result."""

        @task
        def double(x: int) -> int:
            return x * 2

        @task
        def add(x: int, y: int) -> int:
            return x + y

        @task
        def total(values: list[int], offset: int) -> int:
            return sum(values) + offset

        result = double.map(x=[1, 2, 3]).then(add, y=5).join(total, offset=10).run()
        assert result == 37  # (2+5) + (4+5) + (6+5) + 10 = 7+9+11+10

    def test_map_product_then(self) -> None:
        """map() with product mode followed by .then() works correctly."""

        @task
        def scale(x: int, factor: int) -> int:
            return x * factor

        @task
        def inc(x: int) -> int:
            return x + 1

        result = scale.map(x=[1, 2], factor=[10, 100], map_mode="product").then(inc).run()
        # product: (1,10)=10, (1,100)=100, (2,10)=20, (2,100)=200
        # +1: 11, 101, 21, 201
        assert result == [11, 101, 21, 201]

    def test_optimised_matches_unoptimised_map(self) -> None:
        """Optimised and un-optimised map paths produce the same result."""

        @task
        def double(x: int) -> int:
            return x * 2

        @task
        def add(x: int, y: int) -> int:
            return x + y

        @task
        def total(values: list[int]) -> int:
            return sum(values)

        def make_pipeline():
            return double.map(x=[1, 2, 3]).then(add, y=10).join(total)

        _with_optimization(True)
        optimised = make_pipeline().run()

        _with_optimization(False)
        unoptimised = make_pipeline().run()

        assert optimised == unoptimised

    def test_map_then_empty_list(self) -> None:
        """map() over an empty list followed by .then() returns empty list."""

        @task
        def double(x: int) -> int:  # pragma: no cover
            return x * 2

        @task
        def add_one(x: int) -> int:  # pragma: no cover
            return x + 1

        result = double.map(x=[]).then(add_one).run()
        assert result == []


class TestReduceExecution:
    """End-to-end tests for the .reduce() streaming fold API."""

    def test_reduce_sum(self) -> None:
        """Simple reduce computes a sum correctly."""

        @task
        def double(x: int) -> int:
            return x * 2

        @task
        def accumulate(acc: int, item: int) -> int:
            return acc + item

        result = double.map(x=[1, 2, 3]).reduce(accumulate, initial=0).run()
        assert result == 12  # (2 + 4 + 6)

    def test_reduce_with_then_before(self) -> None:
        """map().then().reduce() works correctly."""

        @task
        def double(x: int) -> int:
            return x * 2

        @task
        def add(x: int, y: int) -> int:
            return x + y

        @task
        def accumulate(acc: int, item: int) -> int:
            return acc + item

        result = double.map(x=[1, 2, 3]).then(add, y=10).reduce(accumulate, initial=0).run()
        assert result == 42  # (2+10) + (4+10) + (6+10) = 12 + 14 + 16

    def test_reduce_ordered(self) -> None:
        """Ordered reduce preserves iteration order."""

        @task
        def identity(x: int) -> int:
            return x

        @task
        def append_to_str(acc: str, item: int) -> str:
            return f"{acc},{item}" if acc else str(item)

        result = identity.map(x=[1, 2, 3]).reduce(append_to_str, initial="", ordered=True).run()
        assert result == "1,2,3"

    def test_reduce_unordered(self) -> None:
        """Unordered reduce still produces correct result for commutative ops."""

        @task
        def double(x: int) -> int:
            return x * 2

        @task
        def accumulate(acc: int, item: int) -> int:
            return acc + item

        result = double.map(x=[1, 2, 3]).reduce(accumulate, initial=0, ordered=False).run()
        assert result == 12  # 2 + 4 + 6 (order doesn't matter for sum)

    def test_reduce_validates_params(self) -> None:
        """reduce() rejects tasks with wrong number of parameters."""

        @task
        def double(x: int) -> int:  # pragma: no cover
            return x * 2

        @task
        def bad_reducer(a: int, b: int, c: int) -> int:  # pragma: no cover
            return a + b + c

        with pytest.raises(ParameterError, match="must have exactly 2 unbound parameter"):
            double.map(x=[1, 2]).reduce(bad_reducer, initial=0)

    def test_reduce_optimised_matches_unoptimised(self) -> None:
        """Optimised streaming reduce matches unoptimised functools.reduce fallback."""

        @task
        def square(x: int) -> int:
            return x**2

        @task
        def accumulate(acc: int, item: int) -> int:
            return acc + item

        def make_pipeline():
            return square.map(x=[1, 2, 3, 4]).reduce(accumulate, initial=0)

        _with_optimization(True)
        optimised = make_pipeline().run()

        _with_optimization(False)
        unoptimised = make_pipeline().run()

        assert optimised == unoptimised == 30  # 1 + 4 + 9 + 16

    def test_reduce_with_complex_accumulator(self) -> None:
        """Reduce with a dict accumulator works correctly."""

        @task
        def fetch(key: str) -> dict:
            return {"key": key, "value": len(key)}

        @task
        def merge(acc: dict, item: dict) -> dict:
            result = dict(acc)
            result[item["key"]] = item["value"]
            return result

        result = fetch.map(key=["a", "bb", "ccc"]).reduce(merge, initial={}).run()
        assert result == {"a": 1, "bb": 2, "ccc": 3}


class TestCompositeAsyncTasks:
    """Test that async task functions work inside composites."""

    def test_async_task_chain(self) -> None:
        """Async tasks in a .then() chain execute correctly."""

        @task
        async def async_double(x: int) -> int:
            return x * 2

        @task
        async def async_add(x: int, y: int) -> int:
            return x + y

        result = async_double(x=5).then(async_add, y=3).run()
        assert result == 13  # 5*2 + 3

    def test_async_map_then(self) -> None:
        """Async tasks in map().then() execute correctly."""

        @task
        async def async_double(x: int) -> int:
            return x * 2

        @task
        async def async_inc(x: int) -> int:
            return x + 1

        result = async_double.map(x=[1, 2, 3]).then(async_inc).run()
        assert result == [3, 5, 7]

    def test_mixed_sync_async_chain(self) -> None:
        """Mixed sync/async tasks in a chain work correctly."""

        @task
        def sync_start(x: int) -> int:
            return x * 2

        @task
        async def async_middle(x: int) -> int:
            return x + 10

        @task
        def sync_end(x: int) -> int:
            return x * 3

        result = sync_start(x=5).then(async_middle).then(sync_end).run()
        assert result == 60  # (5*2 + 10) * 3

    def test_async_reduce(self) -> None:
        """Async reduce function works correctly."""

        @task
        def double(x: int) -> int:
            return x * 2

        @task
        async def async_accumulate(acc: int, item: int) -> int:
            return acc + item

        result = double.map(x=[1, 2, 3]).reduce(async_accumulate, initial=0).run()
        assert result == 12


class TestOptimizationToggle:
    """Test that optimization can be disabled and the pipeline still works."""

    def test_disabled_optimization_task_chain(self) -> None:
        """Task chain runs correctly without optimization."""
        _with_optimization(False)

        @task
        def a(x: int) -> int:
            return x + 1

        @task
        def b(x: int) -> int:
            return x * 2

        result = a(x=10).then(b).run()
        assert result == 22  # (10+1) * 2

    def test_disabled_optimization_map_chain(self) -> None:
        """Map chain runs correctly without optimization."""
        _with_optimization(False)

        @task
        def double(x: int) -> int:
            return x * 2

        @task
        def inc(x: int) -> int:
            return x + 1

        result = double.map(x=[1, 2, 3]).then(inc).run()
        assert result == [3, 5, 7]

    def test_disabled_optimization_reduce_fallback(self) -> None:
        """Reduce falls back to functools.reduce when optimization is off."""
        _with_optimization(False)

        @task
        def double(x: int) -> int:
            return x * 2

        @task
        def accumulate(acc: int, item: int) -> int:
            return acc + item

        result = double.map(x=[1, 2, 3]).reduce(accumulate, initial=0).run()
        assert result == 12  # 2 + 4 + 6


class TestCompositeEdgeCases:
    """Edge cases and corner scenarios for composite execution."""

    def test_chain_with_none_result(self) -> None:
        """Chain correctly passes None between links."""

        @task
        def return_none(x: int) -> None:
            return None

        @task
        def handle_none(x: None) -> str:
            return f"got: {x}"

        result = return_none(x=42).then(handle_none).run()
        assert result == "got: None"

    def test_map_single_element(self) -> None:
        """map() with a single element still optimises correctly."""

        @task
        def double(x: int) -> int:
            return x * 2

        @task
        def add(x: int, y: int) -> int:
            return x + y

        result = double.map(x=[5]).then(add, y=3).run()
        assert result == [13]  # 5*2 + 3

    def test_task_then_followed_by_standalone(self) -> None:
        """A folded chain feeding into a non-chain task works correctly."""

        @task
        def a(x: int) -> int:
            return x + 1

        @task
        def b(x: int) -> int:
            return x * 2

        @task
        def c(x: int, y: int) -> int:
            return x + y

        # a → b folds into composite, c depends on composite AND a literal
        chain_result = a(x=4).then(b)
        result = c(x=chain_result, y=100).run()
        assert result == 110  # (4+1)*2 + 100

    def test_reduce_initial_zero(self) -> None:
        """Reduce with initial=0 works for multiplication."""

        @task
        def identity(x: int) -> int:
            return x

        @task
        def multiply(acc: int, item: int) -> int:
            return acc * item

        result = identity.map(x=[2, 3, 4]).reduce(multiply, initial=1).run()
        assert result == 24  # 2 * 3 * 4

    def test_map_join_with_extra_kwargs(self) -> None:
        """map().join() with extra kwargs on join works correctly."""

        @task
        def double(x: int) -> int:
            return x * 2

        @task
        def weighted_sum(values: list[int], weight: float) -> float:
            return sum(values) * weight

        result = double.map(x=[1, 2, 3]).join(weighted_sum, weight=0.5).run()
        assert result == 6.0  # (2 + 4 + 6) * 0.5


class TestReduceRetriesExecution:
    """Reduce retries should actually retry on failure during execution."""

    def test_reduce_retries_on_flaky_reducer(self) -> None:
        """A reducer that fails once should succeed with retries=1."""
        call_count = 0

        @task(retries=1)
        def flaky_add(acc: int, item: int) -> int:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient error")
            return acc + item

        @task
        def identity(x: int) -> int:
            return x

        _with_optimization(False)
        result = identity.map(x=[10, 20]).reduce(flaky_add, initial=0).run()
        assert result == 30  # 0 + 10 + 20


class TestCompositeErrorPaths:
    """Cover the error-raising paths in composite execution."""

    def test_composite_task_error_raises(self) -> None:
        """CompositeTaskNode propagates errors from chain execution."""

        @task
        def good(x: int) -> int:
            return x + 1

        @task
        def bad(x: int) -> int:
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            good(x=1).then(bad).run()

    def test_composite_map_error_raises(self) -> None:
        """CompositeMapTaskNode propagates errors from iteration execution."""

        @task
        def exploder(x: int) -> int:
            if x == 2:
                raise RuntimeError("iteration failed")
            return x * 2

        @task
        def identity(x: int) -> int:
            return x

        with pytest.raises(RuntimeError, match="iteration failed"):
            exploder.map(x=[1, 2, 3]).then(identity).run()


class TestReduceNodeFallbackCoverage:
    """Cover ReduceNode.execute() async path and sync retry continue."""

    def test_reduce_node_async_fallback(self) -> None:
        """ReduceNode.execute() should handle async reduce functions."""

        @task
        def identity(x: int) -> int:
            return x

        @task
        async def async_add(acc: int, item: int) -> int:
            return acc + item

        _with_optimization(False)
        result = identity.map(x=[10, 20, 30]).reduce(async_add, initial=0).run()
        assert result == 60

    def test_reduce_node_sync_retry_continue(self) -> None:
        """ReduceNode sync fallback retries on transient failure."""
        call_count = 0

        @task(retries=2)
        def flaky_sum(acc: int, item: int) -> int:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient")
            return acc + item

        @task
        def identity(x: int) -> int:
            return x

        _with_optimization(False)
        result = identity.map(x=[5, 10]).reduce(flaky_sum, initial=0).run()
        assert result == 15

    def test_reduce_node_async_retry_continue(self) -> None:
        """ReduceNode async fallback retries on transient failure."""
        call_count = 0

        @task(retries=2)
        async def flaky_async_sum(acc: int, item: int) -> int:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient")
            return acc + item

        @task
        def identity(x: int) -> int:
            return x

        _with_optimization(False)
        result = identity.map(x=[5, 10]).reduce(flaky_async_sum, initial=0).run()
        assert result == 15


class TestStreamingReduceRetryCoverage:
    """Cover the retry continue inside _execute_reduce._apply_reduce."""

    def test_streaming_reduce_retries_transient_failure(self) -> None:
        """Streaming reduce should retry on transient reduce failures."""
        call_count = 0

        @task(retries=2)
        def flaky_sum(acc: int, item: int) -> int:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient")
            return acc + item

        @task
        def identity(x: int) -> int:
            return x

        result = identity.map(x=[5, 10]).reduce(flaky_sum, initial=0).run()
        assert result == 15


class TestReduceFutureExecution:
    """Cover ReduceFuture.run_async() and initial value as a future."""

    def test_reduce_run_async(self) -> None:
        """ReduceFuture.run_async() should work."""
        import asyncio

        @task
        def double(x: int) -> int:
            return x * 2

        @task
        def add(acc: int, item: int) -> int:
            return acc + item

        result = asyncio.run(double.map(x=[1, 2, 3]).reduce(add, initial=0).run_async())
        assert result == 12  # 2 + 4 + 6

    def test_reduce_initial_as_future(self) -> None:
        """ReduceFuture should support another future as the initial value."""

        @task
        def make_initial(x: int) -> int:
            return x

        @task
        def double(x: int) -> int:
            return x * 2

        @task
        def add(acc: int, item: int) -> int:
            return acc + item

        result = double.map(x=[1, 2, 3]).reduce(add, initial=make_initial(x=100)).run()
        assert result == 112  # 100 + 2 + 4 + 6

    def test_map_then_task_node_executes(self) -> None:
        """map().then(task) execution produces correct results."""

        @task
        def double(x: int) -> int:
            return x * 2

        @task
        def add_one(x: int) -> int:
            return x + 1

        result = double.map(x=[1, 2, 3]).then(add_one).run()
        assert result == [3, 5, 7]  # (1*2+1, 2*2+1, 3*2+1)
