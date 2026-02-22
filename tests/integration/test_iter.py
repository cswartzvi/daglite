"""
Integration tests for lazy iterator (.iter()) support.

These tests verify end-to-end execution of pipelines using `.iter()` to lazily
iterate a generator on the coordinator while dispatching map work to backend
workers. Tests exercise both the optimised path (default) and the un-optimised
fallback (via `enable_graph_optimization=False`).
"""

import tempfile
import threading
from typing import Iterator

import pytest

from daglite import task
from daglite.datasets.store import DatasetStore
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


# -- Shared task definitions --------------------------------------------------


@task
def generate_numbers(n: int) -> Iterator[int]:
    """Generate numbers from 0 to n-1."""
    for i in range(n):
        yield i


@task
def double(x: int) -> int:
    return x * 2


@task
def add_ten(x: int) -> int:
    return x + 10


@task
def reduce_sum(acc: int, item: int) -> int:
    return acc + item


@task
def reduce_str(acc: str, item: int) -> str:
    return f"{acc},{item}" if acc else str(item)


@task
def sum_values(values: list[int]) -> int:
    return sum(values)


# -- Tests --------------------------------------------------------------------


class TestIterMapReduce:
    """Tests for iter + map + reduce pipelines."""

    def test_iter_map_reduce_optimized(self) -> None:
        """iter + map + reduce produces correct result with optimization enabled."""
        result = double.map(x=generate_numbers.iter(n=10)).reduce(reduce_sum, initial=0).run()
        assert result == 90  # sum(0,2,4,...,18)

    def test_iter_map_reduce_unoptimized(self) -> None:
        """iter + map + reduce produces correct result with optimization disabled."""
        _with_optimization(False)
        result = double.map(x=generate_numbers.iter(n=10)).reduce(reduce_sum, initial=0).run()
        assert result == 90

    def test_iter_map_reduce_matches_materialized(self) -> None:
        """iter() and materialized call() produce identical results."""
        iter_result = double.map(x=generate_numbers.iter(n=10)).reduce(reduce_sum, initial=0).run()
        mat_result = double.map(x=generate_numbers(n=10)).reduce(reduce_sum, initial=0).run()
        assert iter_result == mat_result

    def test_iter_map_reduce_ordered(self) -> None:
        """Ordered reduce preserves iteration order."""
        result = (
            double.map(x=generate_numbers.iter(n=5))
            .reduce(reduce_str, initial="", ordered=True)
            .run()
        )
        assert result == "0,2,4,6,8"

    def test_iter_map_reduce_unordered(self) -> None:
        """Unordered reduce still produces correct result for commutative ops."""
        result = (
            double.map(x=generate_numbers.iter(n=10))
            .reduce(reduce_sum, initial=0, ordered=False)
            .run()
        )
        assert result == 90

    def test_iter_map_reduce_empty(self) -> None:
        """iter + map + reduce with empty generator returns initial value."""
        result = double.map(x=generate_numbers.iter(n=0)).reduce(reduce_sum, initial=42).run()
        assert result == 42


class TestIterMapCollect:
    """Tests for iter + map + collect (default terminal) pipelines."""

    def test_iter_map_collect_optimized(self) -> None:
        """iter + map collects results into a list (optimized)."""
        result = double.map(x=generate_numbers.iter(n=5)).run()
        assert result == [0, 2, 4, 6, 8]

    def test_iter_map_collect_unoptimized(self) -> None:
        """iter + map collects results into a list (unoptimized)."""
        _with_optimization(False)
        result = double.map(x=generate_numbers.iter(n=5)).run()
        assert result == [0, 2, 4, 6, 8]

    def test_iter_map_collect_matches_materialized(self) -> None:
        """iter() and materialized call() produce the same list."""
        iter_result = double.map(x=generate_numbers.iter(n=5)).run()
        mat_result = double.map(x=generate_numbers(n=5)).run()
        assert iter_result == mat_result


class TestIterMapThenReduce:
    """Tests for iter + map + then + reduce pipelines."""

    def test_iter_map_then_reduce(self) -> None:
        """iter + map + then + reduce chains work correctly."""
        result = (
            double.map(x=generate_numbers.iter(n=5))
            .then(add_ten)
            .reduce(reduce_sum, initial=0)
            .run()
        )
        # double: [0,2,4,6,8], add_ten: [10,12,14,16,18], sum = 70
        assert result == 70

    def test_iter_map_then_reduce_unoptimized(self) -> None:
        """iter + map + then + reduce works with optimization disabled."""
        _with_optimization(False)
        result = (
            double.map(x=generate_numbers.iter(n=5))
            .then(add_ten)
            .reduce(reduce_sum, initial=0)
            .run()
        )
        assert result == 70


class TestIterMapJoin:
    """Tests for iter + map + join pipelines."""

    def test_iter_map_join(self) -> None:
        """iter + map + join produces correct result."""
        result = double.map(x=generate_numbers.iter(n=5)).join(sum_values).run()
        # double: [0,2,4,6,8], sum = 20
        assert result == 20

    def test_iter_map_join_unoptimized(self) -> None:
        """iter + map + join works with optimization disabled."""
        _with_optimization(False)
        result = double.map(x=generate_numbers.iter(n=5)).join(sum_values).run()
        assert result == 20


class TestIterWithPartial:
    """Tests for .iter() with partial tasks."""

    def test_iter_with_partial_task(self) -> None:
        """PartialTask.iter() works as expected."""

        @task
        def generate_range(start: int, n: int) -> Iterator[int]:
            for i in range(start, start + n):
                yield i

        partial = generate_range.partial(start=5)
        result = double.map(x=partial.iter(n=3)).reduce(reduce_sum, initial=0).run()
        # generate: [5,6,7], double: [10,12,14], sum = 36
        assert result == 36


class TestIterValidation:
    """Tests for iter-related validation errors."""

    def test_mixed_iter_and_list_in_map_raises(self) -> None:
        """Mixing .iter() with other mapped arguments raises ParameterError."""

        @task
        def combine(x: int, y: int) -> int:
            return x + y

        with pytest.raises(ParameterError, match="iter.*must be the only mapped argument"):
            combine.map(x=generate_numbers.iter(n=5), y=[1, 2, 3])


# -- Fluent API tests (.then/.join/.reduce on IterTaskFuture) -----------------


class TestIterFluentThen:
    """Tests for IterTaskFuture.then() end-to-end."""

    def test_iter_then_run_optimized(self) -> None:
        """gen.iter().then(double).run() produces correct list."""
        result = generate_numbers.iter(n=5).then(double).run()
        assert result == [0, 2, 4, 6, 8]

    def test_iter_then_run_unoptimized(self) -> None:
        """gen.iter().then(double).run() works without optimization."""
        _with_optimization(False)
        result = generate_numbers.iter(n=5).then(double).run()
        assert result == [0, 2, 4, 6, 8]

    def test_iter_then_matches_map_api(self) -> None:
        """.iter().then(f) and f.map(x=.iter()) produce identical results."""
        fluent = generate_numbers.iter(n=8).then(double).run()
        map_api = double.map(x=generate_numbers.iter(n=8)).run()
        assert fluent == map_api

    def test_iter_then_chain(self) -> None:
        """gen.iter().then(double).then(add_ten) chains work."""
        result = generate_numbers.iter(n=4).then(double).then(add_ten).run()
        # [0,1,2,3] → double → [0,2,4,6] → add_ten → [10,12,14,16]
        assert result == [10, 12, 14, 16]

    def test_iter_then_with_partial(self) -> None:
        """gen.iter().then(partial_task) works as expected."""

        @task
        def add(x: int, offset: int) -> int:
            return x + offset

        result = generate_numbers.iter(n=3).then(add.partial(offset=100)).run()
        assert result == [100, 101, 102]


class TestIterFluentThenReduce:
    """Tests for IterTaskFuture.then().reduce() end-to-end."""

    def test_iter_then_reduce_optimized(self) -> None:
        """gen.iter().then(double).reduce(sum, 0) works optimized."""
        result = generate_numbers.iter(n=10).then(double).reduce(reduce_sum, initial=0).run()
        assert result == 90

    def test_iter_then_reduce_unoptimized(self) -> None:
        """gen.iter().then(double).reduce(sum, 0) works unoptimized."""
        _with_optimization(False)
        result = generate_numbers.iter(n=10).then(double).reduce(reduce_sum, initial=0).run()
        assert result == 90

    def test_iter_then_reduce_matches_map_api(self) -> None:
        """.iter().then(f).reduce() matches f.map(x=.iter()).reduce()."""
        fluent = generate_numbers.iter(n=10).then(double).reduce(reduce_sum, initial=0).run()
        map_api = double.map(x=generate_numbers.iter(n=10)).reduce(reduce_sum, initial=0).run()
        assert fluent == map_api


class TestIterFluentJoin:
    """Tests for IterTaskFuture.then().join() and .join() end-to-end."""

    def test_iter_then_join_optimized(self) -> None:
        """gen.iter().then(double).join(sum_values) works."""
        result = generate_numbers.iter(n=5).then(double).join(sum_values).run()
        assert result == 20

    def test_iter_then_join_unoptimized(self) -> None:
        """gen.iter().then(double).join(sum_values) works unoptimized."""
        _with_optimization(False)
        result = generate_numbers.iter(n=5).then(double).join(sum_values).run()
        assert result == 20

    def test_iter_join_direct(self) -> None:
        """gen.iter().join(sum_values) materializes and joins."""
        result = generate_numbers.iter(n=5).join(sum_values).run()
        # [0,1,2,3,4] → sum = 10
        assert result == 10

    def test_iter_join_direct_unoptimized(self) -> None:
        """gen.iter().join(sum_values) works unoptimized."""
        _with_optimization(False)
        result = generate_numbers.iter(n=5).join(sum_values).run()
        assert result == 10


class TestIterSave:
    """Tests for .save() on IterTaskFuture — each yielded item saved with iteration_index."""

    def test_iter_save_optimized(self) -> None:
        """iter().save() saves each yielded item with iteration_index (optimized)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)
            result = double.map(
                x=generate_numbers.iter(n=4).save("gen_{iteration_index}", save_store=store)
            ).run()
            assert result == [0, 2, 4, 6]
            keys = sorted(store.list_keys())
            assert keys == ["gen_0", "gen_1", "gen_2", "gen_3"]
            # Verify saved values are the raw yielded items (before double)
            assert store.load("gen_0") == 0
            assert store.load("gen_1") == 1
            assert store.load("gen_2") == 2
            assert store.load("gen_3") == 3

    def test_iter_save_unoptimized(self) -> None:
        """iter().save() saves per-item even with optimization disabled."""
        _with_optimization(False)
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)
            result = double.map(
                x=generate_numbers.iter(n=3).save("gen_{iteration_index}", save_store=store)
            ).run()
            assert result == [0, 2, 4]
            keys = sorted(store.list_keys())
            assert keys == ["gen_0", "gen_1", "gen_2"]
            assert store.load("gen_0") == 0
            assert store.load("gen_1") == 1
            assert store.load("gen_2") == 2

    def test_iter_save_with_reduce(self) -> None:
        """iter().save() + .reduce() saves each yielded item during streaming reduce."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)
            result = (
                double.map(
                    x=generate_numbers.iter(n=3).save("item_{iteration_index}", save_store=store)
                )
                .reduce(reduce_sum, initial=0)
                .run()
            )
            assert result == 6  # 0+2+4
            keys = sorted(store.list_keys())
            assert keys == ["item_0", "item_1", "item_2"]

    def test_iter_save_fluent_then(self) -> None:
        """gen.iter().save().then(double) saves yielded items."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)
            result = (
                generate_numbers.iter(n=3)
                .save("yielded_{iteration_index}", save_store=store)
                .then(double)
                .run()
            )
            assert result == [0, 2, 4]
            keys = sorted(store.list_keys())
            assert keys == ["yielded_0", "yielded_1", "yielded_2"]
            assert store.load("yielded_0") == 0
            assert store.load("yielded_1") == 1
            assert store.load("yielded_2") == 2

    def test_iter_save_empty_generator(self) -> None:
        """iter().save() with empty generator produces no saves."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)
            result = (
                double.map(
                    x=generate_numbers.iter(n=0).save("gen_{iteration_index}", save_store=store)
                )
                .reduce(reduce_sum, initial=42)
                .run()
            )
            assert result == 42
            assert store.list_keys() == []


# -- Streaming behavior tests ------------------------------------------------


class TestIterStreamingBehavior:
    """Tests that verify generation, processing, and reduction truly interleave."""

    def test_ordered_reduce_interleaves_generation_and_reduction(self) -> None:
        """Ordered reduce should not materialise all items before reducing.

        With a concurrency window of 8, a 20-item generator should show
        reduction events starting before all generation events.
        """
        log: list[tuple[str, int]] = []
        lock = threading.Lock()

        @task(backend_name="threading")
        def logging_gen(n: int) -> Iterator[int]:
            for i in range(n):
                with lock:
                    log.append(("gen", i))
                yield i

        @task(backend_name="threading")
        def logging_double(x: int) -> int:
            with lock:
                log.append(("map", x))
            return x * 2

        @task
        def logging_reduce(acc: int, item: int) -> int:
            with lock:
                log.append(("reduce", item))
            return acc + item

        result = (
            logging_gen.iter(n=20)
            .then(logging_double)
            .reduce(logging_reduce, initial=0, ordered=True)
            .run()
        )
        assert result == sum(i * 2 for i in range(20))

        # Find when the first reduce event occurs relative to gen events.
        gen_indices = [i for i, (kind, _) in enumerate(log) if kind == "gen"]
        reduce_indices = [i for i, (kind, _) in enumerate(log) if kind == "reduce"]

        assert len(gen_indices) == 20
        assert len(reduce_indices) == 20

        # The first reduce should happen BEFORE the last gen — i.e. interleaved.
        first_reduce_pos = reduce_indices[0]
        last_gen_pos = gen_indices[-1]
        assert first_reduce_pos < last_gen_pos, (
            f"First reduce at log position {first_reduce_pos} should be before "
            f"last gen at position {last_gen_pos}. Reduction did not interleave "
            f"with generation."
        )

    def test_unordered_reduce_interleaves_generation_and_reduction(self) -> None:
        """Unordered reduce should interleave generation with reduction."""
        log: list[tuple[str, int]] = []
        lock = threading.Lock()

        @task(backend_name="threading")
        def logging_gen(n: int) -> Iterator[int]:
            for i in range(n):
                with lock:
                    log.append(("gen", i))
                yield i

        @task(backend_name="threading")
        def logging_double(x: int) -> int:
            with lock:
                log.append(("map", x))
            return x * 2

        @task
        def logging_reduce(acc: int, item: int) -> int:
            with lock:
                log.append(("reduce", item))
            return acc + item

        result = (
            logging_gen.iter(n=20)
            .then(logging_double)
            .reduce(logging_reduce, initial=0, ordered=False)
            .run()
        )
        # Sum is the same regardless of order (commutative).
        assert result == sum(i * 2 for i in range(20))

        gen_indices = [i for i, (kind, _) in enumerate(log) if kind == "gen"]
        reduce_indices = [i for i, (kind, _) in enumerate(log) if kind == "reduce"]

        assert len(gen_indices) == 20
        assert len(reduce_indices) == 20

        first_reduce_pos = reduce_indices[0]
        last_gen_pos = gen_indices[-1]
        assert first_reduce_pos < last_gen_pos, (
            f"First reduce at log position {first_reduce_pos} should be before "
            f"last gen at position {last_gen_pos}. Reduction did not interleave "
            f"with generation."
        )

    def test_ordered_reduce_preserves_order(self) -> None:
        """Even with multiple workers, ordered reduce processes in generation order."""

        @task(backend_name="threading")
        def logging_gen(n: int) -> Iterator[int]:
            for i in range(n):
                yield i

        @task(backend_name="threading")
        def identity(x: int) -> int:
            return x

        @task
        def concat_reduce(acc: str, item: int) -> str:
            return f"{acc},{item}" if acc else str(item)

        result = (
            logging_gen.iter(n=15)
            .then(identity)
            .reduce(concat_reduce, initial="", ordered=True)
            .run()
        )
        assert result == ",".join(str(i) for i in range(15))


# -- Error propagation tests -------------------------------------------------


class TestIterErrorPropagation:
    """Tests that errors in generators and workers propagate correctly."""

    def test_generator_error_propagates_in_ordered_reduce(self) -> None:
        """If the generator raises mid-stream, the error surfaces to the caller."""

        @task(backend_name="threading")
        def failing_gen(n: int) -> Iterator[int]:
            for i in range(n):
                if i == 5:
                    raise ValueError("generator failed at 5")
                yield i

        @task(backend_name="threading")
        def identity(x: int) -> int:
            return x

        with pytest.raises(ValueError, match="generator failed at 5"):
            failing_gen.iter(n=10).then(identity).reduce(reduce_sum, initial=0, ordered=True).run()

    def test_generator_error_propagates_in_unordered_reduce(self) -> None:
        """If the generator raises mid-stream with unordered reduce, error surfaces."""

        @task(backend_name="threading")
        def failing_gen(n: int) -> Iterator[int]:
            for i in range(n):
                if i == 5:
                    raise ValueError("generator failed at 5")
                yield i

        @task(backend_name="threading")
        def identity(x: int) -> int:
            return x

        with pytest.raises(ValueError, match="generator failed at 5"):
            failing_gen.iter(n=10).then(identity).reduce(reduce_sum, initial=0, ordered=False).run()

    def test_worker_error_propagates_in_ordered_reduce(self) -> None:
        """If a worker raises, the error surfaces through ordered reduce."""

        @task(backend_name="threading")
        def failing_double(x: int) -> int:
            if x == 3:
                raise RuntimeError("worker failed on 3")
            return x * 2

        result_future = (
            generate_numbers.iter(n=10)
            .then(failing_double)
            .reduce(reduce_sum, initial=0, ordered=True)
        )
        with pytest.raises(RuntimeError, match="worker failed on 3"):
            result_future.run()

    def test_worker_error_propagates_in_unordered_reduce(self) -> None:
        """If a worker raises, the error surfaces through unordered reduce."""

        @task(backend_name="threading")
        def failing_double(x: int) -> int:
            if x == 3:
                raise RuntimeError("worker failed on 3")
            return x * 2

        result_future = (
            generate_numbers.iter(n=10)
            .then(failing_double)
            .reduce(reduce_sum, initial=0, ordered=False)
        )
        with pytest.raises(RuntimeError, match="worker failed on 3"):
            result_future.run()

    def test_generator_error_propagates_in_collect(self) -> None:
        """If the generator raises mid-stream in a collect pipeline, error surfaces."""

        @task(backend_name="threading")
        def failing_gen(n: int) -> Iterator[int]:
            for i in range(n):
                if i == 3:
                    raise ValueError("generator failed at 3")
                yield i

        @task(backend_name="threading")
        def identity(x: int) -> int:
            return x

        with pytest.raises(ValueError, match="generator failed at 3"):
            failing_gen.iter(n=10).then(identity).run()

    def test_worker_error_propagates_in_collect(self) -> None:
        """If a worker raises in a collect pipeline, error surfaces."""

        @task(backend_name="threading")
        def failing_double(x: int) -> int:
            if x == 3:
                raise RuntimeError("worker failed on 3")
            return x * 2

        with pytest.raises(RuntimeError, match="worker failed on 3"):
            generate_numbers.iter(n=10).then(failing_double).run()
