"""Behavior tests for task evaluation using .run() with both sync and async tasks."""

import asyncio
import os

import pytest

from daglite import task


# Module-level tasks for ProcessBackend (must be picklable)
@task(backend_name="processes")
def get_pid_process(value: int) -> tuple[int, int]:
    """Return (value, process_id). Module-level for pickling."""
    return (value, os.getpid())


@task
def combine_pids(a: tuple[int, int], b: tuple[int, int]) -> dict:
    """Combine results from process tasks."""
    return {"sum": a[0] + b[0], "pids": [a[1], b[1]]}


class TestRunSyncSinglePath:
    """Tests .run() evaluation of single path tasks."""

    def test_single_task_evaluation_empty(self) -> None:
        """Evaluation succeeds for tasks without parameters."""

        @task
        def dummy() -> None:
            return

        dummies = dummy()
        result = dummies.run()
        assert result is None

    def test_single_task_evaluation_without_params(self) -> None:
        """Evaluation succeeds for parameterless tasks."""

        @task
        def prepare() -> int:
            return 5

        prepared = prepare()
        result = prepared.run()
        assert result == 5

    def test_single_task_evaluation_with_params(self) -> None:
        """Evaluation succeeds when parameters are provided."""

        @task
        def multiply(x: int, y: int) -> int:
            return x * y

        multiplied = multiply(x=4, y=6)
        result = multiplied.run()
        assert result == 24

    def test_short_chain_evaluation_without_params(self) -> None:
        """Evaluation succeeds for a chain of dependent tasks."""

        @task
        def prepare() -> int:
            return 5

        @task
        def square(z: int) -> int:
            return z**2

        prepared = prepare()
        squared = square(z=prepared)
        result = squared.run()
        assert result == 25  # = 5 ** 2

    def test_long_chain_evaluation_with_params(self) -> None:
        """Evaluation succeeds for a longer chain of tasks with parameters."""

        @task
        def add(x: int, y: int) -> int:
            return x + y

        @task
        def multiply(z: int, factor: int) -> int:
            return z * factor

        @task
        def subtract(value: int, decrement: int) -> int:
            return value - decrement

        added = add(x=3, y=7)  # 10
        multiplied1 = multiply(z=added, factor=4)
        multiplied2 = multiply(z=multiplied1, factor=2)  # 80
        multiplied3 = multiply(z=multiplied2, factor=3)  # 240
        multiplied4 = multiply(z=multiplied3, factor=1)  # 240
        subtracted = subtract(value=multiplied4, decrement=40)  # 200
        result = subtracted.run()
        assert result == 200

    def test_partial_task_evaluation_without_params(self) -> None:
        """Evaluation succeeds for tasks with fixed parameters."""

        @task
        def power(base: int, exponent: int) -> int:
            return base**exponent

        powered = power.partial(exponent=3)(base=2)
        result = powered.run()
        assert result == 8  # = 2 ** 3

    def test_partial_task_evaluation_with_params(self) -> None:
        """Evaluation succeeds for tasks with some fixed and some provided parameters."""

        @task
        def compute_area(length: float, width: float) -> float:
            return length * width

        area_task = compute_area.partial(width=5.0)
        area = area_task(length=10.0)
        result = area.run()
        assert result == 50.0  # = 10.0 * 5.0

    def test_mixed_chain_evaluation_without_params(self) -> None:
        """Evaluation succeeds for a chain mixing fixed and provided parameters."""

        @task
        def add(x: int, y: int) -> int:
            return x + y

        @task
        def multiply(z: int, factor: int) -> int:
            return z * factor

        fixed_multiply = multiply.partial(factor=10)

        added = add(x=2, y=3)  # 5
        multiplied = fixed_multiply(z=added)  # 50
        result = multiplied.run()
        assert result == 50

    def test_mixed_chain_evaluation_with_params(self) -> None:
        """Evaluation succeeds for a chain with both fixed and provided parameters."""

        @task
        def add(x: int, y: int) -> int:
            return x + y

        @task
        def multiply(z: int, factor: int) -> int:
            return z * factor

        @task
        def subtract(value: int, decrement: int) -> int:
            return value - decrement

        fixed_multiply = multiply.partial(factor=3)
        fixed_subtract = subtract.partial(decrement=4)

        added = add(x=7, y=8)  # 15
        multiplied = fixed_multiply(z=added)  # 45
        subtracted = fixed_subtract(value=multiplied)  # 41
        result = subtracted.run()
        assert result == 41

    def test_deep_chain_with_task_reuse_and_independent_branches(self) -> None:
        """Evaluation succeeds for deep chains with reused tasks and independent branches."""

        @task
        def multiply(x: int, factor: int) -> int:
            return x * factor

        # Start with 2, multiply by 2 repeatedly
        chain_depth = 2000
        current = multiply(x=2, factor=2)
        for i in range(chain_depth - 1):
            current = multiply(x=current, factor=2)

        # Evaluate all three independently
        result_chain = current.run()

        assert result_chain == 2 * (2**chain_depth)


@pytest.mark.parametrize("mode", ["product", "zip"])
class TestRunSyncMappedOperations:
    """
    Tests for mapped task operations (product and zip).

    This class tests common patterns shared between product() and zip() operations
    using parametrization to reduce duplication. Operation-specific behaviors are
    tested in separate classes below.
    """

    def test_empty_sequence(self, mode: str) -> None:
        """Evaluation succeeds with empty sequences."""

        @task
        def double(x: int) -> int:
            return x * 2  # pragma: no cover

        doubled_seq = double.map(x=[], map_mode=mode)  # type: ignore
        result = doubled_seq.run()
        assert result == []

    def test_single_parameter(self, mode: str) -> None:
        """Evaluation succeeds for single parameter tasks."""

        @task
        def double(x: int) -> int:
            return x * 2

        doubled_seq = double.map(x=[1, 2, 3], map_mode=mode)  # type: ignore
        result = doubled_seq.run()
        assert result == [2, 4, 6]

    def test_single_element(self, mode: str) -> None:
        """Evaluation succeeds with single element sequences."""

        @task
        def triple(x: int) -> int:
            return x * 3

        tripled_seq = triple.map(x=[5], map_mode=mode)  # type: ignore
        result = tripled_seq.run()
        assert result == [15]

    def test_with_future_input(self, mode: str) -> None:
        """Evaluation succeeds with TaskFuture as input."""

        @task
        def generate() -> list[int]:
            return [1, 2, 3]

        @task
        def add(x: int, y: int) -> int:
            return x + y

        future = generate()
        seq = add.map(x=future, y=[10, 20, 30], map_mode=mode)  # type: ignore

        expected = [11, 21, 31, 12, 22, 32, 13, 23, 33] if mode == "product" else [11, 22, 33]
        result = seq.run()
        assert result == expected

    def test_with_fixed_parameters(self, mode: str) -> None:
        """Evaluation succeeds with some fixed parameters."""

        @task
        def multiply(x: int, factor: int) -> int:
            return x * factor

        fixed = multiply.partial(factor=10)
        seq = fixed.map(x=[1, 2, 3], map_mode=mode)  # type: ignore
        expected = [10, 20, 30]  # Same for both modes with single sequence

        result = seq.run()
        assert result == expected

    def test_with_fixed_future_parameters(self, mode: str) -> None:
        """Evaluation succeeds with fixed TaskFuture parameters."""

        @task
        def get_factor() -> int:
            return 4

        @task
        def multiply(x: int, factor: int) -> int:
            return x * factor

        future = get_factor()
        fixed = multiply.partial(factor=future)
        seq = fixed.map(x=[2, 3, 4], map_mode=mode)  # type: ignore
        expected = [8, 12, 16]  # Same for both modes with single sequence

        result = seq.run()
        assert result == expected

    def test_with_map(self, mode: str) -> None:
        """Evaluation succeeds when chaining with .then() for mapping."""

        @task
        def add(x: int, y: int) -> int:
            return x + y

        @task
        def square(z: int) -> int:
            return z**2

        seq = add.map(x=[1, 2, 3], y=[10, 20, 30], map_mode=mode).then(square)  # type: ignore
        expected = (
            [121, 441, 961, 144, 484, 1024, 169, 529, 1089]
            if mode == "product"
            else [121, 484, 1089]
        )

        result = seq.run()
        assert result == expected

    def test_with_join(self, mode: str) -> None:
        """Evaluation succeeds when chaining with .join() for reduction."""

        @task
        def add(x: int, y: int) -> int:
            return x + y

        @task
        def sum_all(values: list[int]) -> int:
            return sum(values)

        seq = add.map(x=[1, 2, 3], y=[10, 20, 30], map_mode=mode).join(sum_all)  # type: ignore
        expected = (
            198 if mode == "product" else 66
        )  # product: 9 combos sum to 198, zip: 11+22+33=66

        result = seq.run()
        assert result == expected

    def test_with_map_kwargs(self, mode: str) -> None:
        """Evaluation succeeds with .then() using kwargs."""

        @task
        def add(x: int, y: int) -> int:
            return x + y

        @task
        def multiply(z: int, factor: int) -> int:
            return z * factor

        seq = add.map(x=[1, 2, 3], y=[10, 20, 30], map_mode=mode).then(multiply, factor=2)  # type: ignore
        expected = [22, 42, 62, 24, 44, 64, 26, 46, 66] if mode == "product" else [22, 44, 66]

        result = seq.run()
        assert result == expected

    def test_with_join_kwargs(self, mode: str) -> None:
        """Evaluation succeeds with .join() using kwargs."""

        @task
        def add(x: int, y: int) -> int:
            return x + y

        @task
        def sum_with_bonus(values: list[int], bonus: int) -> int:
            return sum(values) + bonus

        seq = add.map(x=[1, 2, 3], y=[10, 20, 30], map_mode=mode).join(sum_with_bonus, bonus=5)  # type: ignore
        expected = 203 if mode == "product" else 71  # product: 198+5, zip: 66+5

        result = seq.run()
        assert result == expected


class TestRunSyncProductBehavior:
    """Tests .run() with product-specific behavior (Cartesian product semantics)."""

    def test_product_multiple_parameters(self) -> None:
        """Product creates Cartesian product of multiple parameter sequences."""

        @task
        def add(x: int, y: int) -> int:
            return x + y

        added_seq = add.map(x=[1, 2, 3], y=[10, 20, 30], map_mode="product")
        result = added_seq.run()
        # Cartesian product: all combinations
        assert result == [11, 21, 31, 12, 22, 32, 13, 23, 33]

    def test_product_with_nested_tasks(self) -> None:
        """Product with nested TaskFuture creates Cartesian product."""

        @task
        def add(x: int, y: int) -> int:
            return x + y

        @task
        def multiply(z: int, factor: int) -> int:
            return z * factor

        added_seq = add.map(x=[1, 2], y=[10, 20], map_mode="product")  # [11, 21, 12, 22]
        multiplied_seq = multiply.map(z=added_seq, factor=[2, 3], map_mode="product")
        result = multiplied_seq.run()
        # Cartesian product of results
        assert result == [22, 33, 42, 63, 24, 36, 44, 66]

    def test_product_with_complex_chain(self) -> None:
        """Product with fixed map and join (fan-out, transform, fan-in)."""

        @task
        def add(x: int, y: int) -> int:
            return x + y

        @task
        def triple(z: int) -> int:
            return z * 3

        @task
        def max_value(values: list[int]) -> int:
            return max(values)

        result = (
            add.map(x=[1, 2], y=[10, 20], map_mode="product")  # [11, 21, 12, 22]
            .then(triple)  # [33, 63, 36, 66]
            .join(max_value)
        ).run()
        assert result == 66


class TestRunSyncZipBehavior:
    """Tests .run() with zip-specific behavior (element-wise alignment semantics)."""

    def test_zip_multiple_parameters(self) -> None:
        """Zip aligns multiple parameter sequences element-wise."""

        @task
        def sum_three(x: int, y: int, z: int) -> int:
            return x + y + z

        summed_seq = sum_three.map(x=[1, 2], y=[10, 20], z=[100, 200])
        result = summed_seq.run()
        # Element-wise alignment
        assert result == [111, 222]  # (1+10+100), (2+20+200)

    def test_zip_with_nested_tasks(self) -> None:
        """Zip with nested TaskFuture maintains element-wise alignment."""

        @task
        def add(x: int, y: int) -> int:
            return x + y

        @task
        def multiply(z: int, factor: int) -> int:
            return z * factor

        added_seq = add.map(x=[1, 2], y=[10, 20])  # [11, 22]
        multiplied_seq = multiply.map(z=added_seq, factor=[2, 3])
        result = multiplied_seq.run()
        # Element-wise: (11*2), (22*3)
        assert result == [22, 66]

    def test_zip_with_complex_chain(self) -> None:
        """Zip with fixed map and join (fan-out, transform, fan-in)."""

        @task
        def add(x: int, y: int) -> int:
            return x + y

        @task
        def increment(z: int, increment_by: int) -> int:
            return z + increment_by

        @task
        def sum_all(values: list[int]) -> int:
            return sum(values)

        result = (
            add.map(x=[1, 2, 3], y=[10, 20, 30])  # [11, 22, 33]
            .then(increment.partial(increment_by=5))  # [16, 27, 38]
            .join(sum_all)
        ).run()
        assert result == 81


class TestRunSyncComplexPaths:
    """Tests .run() evaluation of complex task graphs."""

    def test_product_and_zip_with_join(self) -> None:
        """Evaluation succeeds combining product, zip, and join operations."""

        @task
        def multiply(x: int, y: int) -> int:
            return x * y

        @task
        def sum_all(values: list[int]) -> int:
            return sum(values)

        # Fan-out with product
        multiplied_seq = multiply.map(x=[1, 2], y=[10, 20], map_mode="product")  # [10, 20, 20, 40]
        # Fan-in with join
        total = multiplied_seq.join(sum_all)
        result = total.run()
        assert result == 90  # 10 + 20 + 20 + 40


class TestRunSyncWithAsyncTasks:
    """Tests that async tasks work through .run() via the async-first engine."""

    def test_async_task_works(self) -> None:
        """Async tasks are evaluated through the async-first engine."""

        @task
        async def async_add(x: int, y: int) -> int:
            await asyncio.sleep(0.001)
            return x + y

        result = async_add(x=5, y=10).run()  # type: ignore
        assert result == 15

    def test_async_map_task_works(self) -> None:
        """Async map tasks are evaluated through the async-first engine."""

        @task
        async def async_square(x: int) -> int:
            await asyncio.sleep(0.001)
            return x * x

        result = async_square.map(x=[1, 2, 3, 4]).run()
        assert result == [1, 4, 9, 16]

    def test_mixed_graph_with_async_works(self) -> None:
        """Graphs containing both sync and async tasks work through .run()."""

        @task
        def sync_task(x: int) -> int:
            return x + 1

        @task
        async def async_double(x: int) -> int:
            await asyncio.sleep(0.001)
            return x * 2

        @task
        def combine(a: int, b: int) -> int:
            return a + b

        start = sync_task(x=10)
        async_result = async_double(x=start)
        combined = combine(a=start, b=async_result)

        result = combined.run()
        assert result == 33  # sync_task(10)=11, async_double(11)=22, combine(11,22)=33


class TestRunSyncGeneratorMaterialization:
    """
    Tests that generators are automatically materialized to lists.

    Tests marked with @pytest.mark.parametrize run for both .run() and .run_async()
    to ensure consistent behavior across evaluation modes. Other tests are sync-only.
    """

    @pytest.mark.parametrize("evaluation_mode", ["sync", "async"])
    def test_generator_is_materialized(self, evaluation_mode: str) -> None:
        """Generators returned from tasks are materialized to lists."""
        from typing import Iterator

        @task
        def generate_numbers(n: int) -> Iterator[int]:
            for i in range(n):
                yield i * 2

        async def run_async():
            return await generate_numbers(n=5).run_async()

        if evaluation_mode == "sync":
            result = generate_numbers(n=5).run()
        else:
            result = asyncio.run(run_async())

        assert result == [0, 2, 4, 6, 8]
        assert isinstance(result, list)

    def test_generator_reusable_by_multiple_consumers(self) -> None:
        """Materialized generators can be consumed by multiple downstream tasks."""
        from typing import Iterator

        @task
        def generate_numbers(n: int) -> Iterator[int]:
            for i in range(n):
                yield i

        @task
        def sum_values(values: list[int]) -> int:
            return sum(values)

        @task
        def count_values(values: list[int]) -> int:
            return len(values)

        @task
        def combine(total: int, count: int) -> tuple[int, int]:
            return (total, count)

        numbers = generate_numbers(n=5)
        total = sum_values(values=numbers)
        count = count_values(values=numbers)
        result_future = combine(total=total, count=count)

        result = result_future.run()
        assert result == (10, 5)  # sum([0,1,2,3,4]) = 10, len = 5

    @pytest.mark.parametrize("evaluation_mode", ["sync", "async"])
    def test_generator_in_map_operation(self, evaluation_mode: str) -> None:
        """Generators in map operations are materialized."""
        from typing import Iterator

        @task
        def generate_range(n: int) -> Iterator[int]:
            for i in range(n):
                yield i

        @task
        def sum_values(values: list[int]) -> int:
            return sum(values)

        # Multiple generators from product operation
        ranges = generate_range.map(n=[3, 4, 5])
        totals = ranges.then(sum_values)

        async def run_async():
            return await totals.run_async()

        if evaluation_mode == "sync":
            result = totals.run()
        else:
            result = asyncio.run(run_async())

        # [0,1,2] -> 3, [0,1,2,3] -> 6, [0,1,2,3,4] -> 10
        assert result == [3, 6, 10]

    def test_non_generator_iterables_unchanged(self) -> None:
        """Non-generator iterables like lists are not affected."""

        @task
        def return_list(n: int) -> list[int]:
            return [i * 2 for i in range(n)]

        @task
        def return_tuple(n: int) -> tuple[int, ...]:
            return tuple(i * 2 for i in range(n))

        list_result = return_list(n=3).run()
        assert list_result == [0, 2, 4]
        assert isinstance(list_result, list)

        tuple_result = return_tuple(n=3).run()
        assert tuple_result == (0, 2, 4)
        assert isinstance(tuple_result, tuple)

    def test_strings_not_materialized(self) -> None:
        """Strings (which are iterable) are not converted to lists."""

        @task
        def return_string() -> str:
            return "hello"

        result = return_string().run()
        assert result == "hello"
        assert isinstance(result, str)

    def test_generator_with_then_chain(self) -> None:
        """Generator materialization works with .then() chains."""
        from typing import Iterator

        @task
        def generate_numbers(n: int) -> Iterator[int]:
            for i in range(n):
                yield i

        @task
        def double_all(values: list[int]) -> list[int]:
            return [v * 2 for v in values]

        result = generate_numbers(n=4).then(double_all).run()
        assert result == [0, 2, 4, 6]

    def test_generator_type_inference(self) -> None:
        """Type system correctly infers list[T] from Iterator[T] return."""
        from typing import Generator, Iterator

        @task
        def return_iterator(n: int) -> Iterator[int]:
            for i in range(n):
                yield i

        @task
        def return_generator(n: int) -> Generator[str, None, None]:
            for i in range(n):
                yield str(i)

        # Type checker should infer list[int] and list[str]
        iter_result = return_iterator(n=3).run()
        gen_result = return_generator(n=3).run()

        # These should work at runtime (list operations)
        assert isinstance(iter_result, list)
        assert isinstance(gen_result, list)
        assert iter_result == [0, 1, 2]
        assert gen_result == ["0", "1", "2"]

        # Verify list operations work
        iter_result.append(99)
        gen_result.append("99")
        assert len(iter_result) == 4
        assert len(gen_result) == 4

    def test_async_generator_materialization_sync_mode(self) -> None:
        """Async generators are materialized properly through .run()."""
        from collections.abc import AsyncGenerator

        @task
        async def async_generate_numbers(n: int) -> AsyncGenerator[int, None]:
            async def _gen():
                for i in range(n):
                    await asyncio.sleep(0.001)
                    yield i

            return _gen()

        @task
        def sum_values(values: list[int]) -> int:
            return sum(values)

        nums = async_generate_numbers(n=5)
        total = sum_values(values=nums)

        result = total.run()
        assert result == 10  # 0 + 1 + 2 + 3 + 4

    def test_async_generator_materialization_async_mode(self) -> None:
        """Async generators are materialized properly in async evaluation."""
        from collections.abc import AsyncGenerator

        @task
        async def async_generate_numbers(n: int) -> AsyncGenerator[int, None]:
            async def _gen():
                for i in range(n):
                    await asyncio.sleep(0.001)
                    yield i

            return _gen()

        @task
        def sum_values(values: list[int]) -> int:
            return sum(values)

        nums = async_generate_numbers(n=5)
        total = sum_values(values=nums)

        async def run_async():
            return await total.run_async()

        result = asyncio.run(run_async())
        assert result == 10  # 0 + 1 + 2 + 3 + 4

    def test_async_generator_map_materialization_sync_mode(self) -> None:
        """Map tasks with async generators are materialized properly through .run()."""
        from collections.abc import AsyncGenerator

        @task
        async def async_get_range(n: int) -> AsyncGenerator[int, None]:
            async def _gen():
                for i in range(n):
                    await asyncio.sleep(0.001)
                    yield i

            return _gen()

        @task
        def sum_all_ranges(ranges: list[list[int]]) -> int:
            return sum(sum(r) for r in ranges)

        ranges = async_get_range.map(n=[3, 4, 5])
        total = sum_all_ranges(ranges=ranges)

        result = total.run()
        assert result == 19  # (0+1+2) + (0+1+2+3) + (0+1+2+3+4) = 3+6+10

    def test_async_generator_map_materialization_async_mode(self) -> None:
        """Map tasks that return async generators are materialized properly in async mode."""
        from collections.abc import AsyncGenerator

        @task
        async def async_get_range(n: int) -> AsyncGenerator[int, None]:
            async def _gen():
                for i in range(n):
                    await asyncio.sleep(0.001)
                    yield i

            return _gen()

        @task
        def sum_all_ranges(ranges: list[list[int]]) -> int:
            return sum(sum(r) for r in ranges)

        # Create a map task where each result is an async generator
        ranges = async_get_range.map(n=[3, 4, 5])
        total = sum_all_ranges(ranges=ranges)

        async def run_async():
            return await total.run_async()

        result = asyncio.run(run_async())
        assert result == 19  # (0+1+2) + (0+1+2+3) + (0+1+2+3+4) = 3+6+10

    def test_async_generator_with_thread_backend_sync_mode(self) -> None:
        """Async generators with ThreadBackend work through .run()."""
        from collections.abc import AsyncGenerator

        @task(backend_name="threading")
        async def async_generate(n: int) -> AsyncGenerator[int, None]:
            async def _gen():
                for i in range(n):
                    await asyncio.sleep(0.001)
                    yield i * 2

            return _gen()

        @task
        def sum_values(values: list[int]) -> int:
            return sum(values)

        nums = async_generate(n=5)
        total = sum_values(values=nums)

        result = total.run()
        assert result == 20  # 0 + 2 + 4 + 6 + 8

    def test_async_generator_with_thread_backend_async_mode(self) -> None:
        """Async generators work with ThreadBackend in async mode."""
        from collections.abc import AsyncGenerator

        @task(backend_name="threading")
        async def async_generate(n: int) -> AsyncGenerator[int, None]:
            async def _gen():
                for i in range(n):
                    await asyncio.sleep(0.001)
                    yield i * 2

            return _gen()

        @task
        def sum_values(values: list[int]) -> int:
            return sum(values)

        nums = async_generate(n=5)
        total = sum_values(values=nums)

        async def run_async():
            return await total.run_async()

        result = asyncio.run(run_async())
        assert result == 20  # 0+2+4+6+8


def test_cycle_detection_raises_error() -> None:
    """Evaluation raises ExecutionError when a cycle is detected in the graph."""
    from uuid import uuid4

    from daglite.engine import _ExecutionState
    from daglite.exceptions import ExecutionError
    from daglite.graph.nodes import TaskNode
    from daglite.graph.nodes.base import NodeInput

    # Manually create a cyclic graph (can't do this with normal API)
    # Node A depends on Node B, Node B depends on Node A
    node_a_id = uuid4()
    node_b_id = uuid4()

    def dummy(x: int) -> int:
        return x

    # Create nodes with circular dependencies
    nodes = {
        node_a_id: TaskNode(
            id=node_a_id,
            name="node_a",
            description=None,
            backend_name=None,
            func=dummy,
            kwargs={"x": NodeInput(_kind="ref", value=None, reference=node_b_id)},
        ),
        node_b_id: TaskNode(
            id=node_b_id,
            name="node_b",
            description=None,
            backend_name=None,
            func=dummy,
            kwargs={"x": NodeInput(_kind="ref", value=None, reference=node_a_id)},
        ),
    }

    state = _ExecutionState.from_nodes(nodes)  # type: ignore

    with pytest.raises(ExecutionError, match="Cycle detected"):
        state.check_complete()


class TestRunSyncSiblingParallelism:
    """Tests for concurrent execution of sibling tasks using .run()."""

    def test_sync_siblings_with_threading_backend(self) -> None:
        """Sibling tasks with threading backend are submitted concurrently."""
        import threading
        import time

        execution_order: list[tuple[int, float]] = []
        lock = threading.Lock()

        @task(backend_name="threading")
        def track_execution(value: int) -> int:
            """Track execution order and add small delay."""
            start = time.perf_counter()
            time.sleep(0.01)  # Small delay to ensure overlap if parallel
            with lock:
                execution_order.append((value, start))
            return value

        @task
        def sum_values(a: int, b: int, c: int) -> int:
            return a + b + c

        # Create sibling tasks
        t1 = track_execution(value=10)
        t2 = track_execution(value=20)
        t3 = track_execution(value=30)
        total = sum_values(a=t1, b=t2, c=t3)

        result = total.run()

        # Verify results
        assert result == 60

        # Verify all tasks executed
        assert len(execution_order) == 3

        # If truly concurrent, start times should overlap
        # (all should start within a small window, not sequentially)
        start_times = [t[1] for t in execution_order]
        time_range = max(start_times) - min(start_times)

        # If running inline with 0.01s sleep each, would be ~0.02s between first and last
        # If parallel, should start within ~0.001s of each other
        assert time_range < 0.015, (
            f"Tasks appear to have run sequentially (not in parallel). "
            f"Time range between first and last start: {time_range:.4f}s"
        )

    def test_sync_siblings_with_processes_backend(self) -> None:
        """Sibling tasks with processes backend execute concurrently in sync evaluation."""

        # Create sibling tasks (using module-level tasks for pickling)
        t1 = get_pid_process(value=10)
        t2 = get_pid_process(value=20)
        result_future = combine_pids(a=t1, b=t2)

        result = result_future.run()

        # Verify results
        assert result["sum"] == 30

        # PIDs should be different if truly parallel (may be same if Inline)
        # Just verify we got valid PIDs
        assert all(pid > 0 for pid in result["pids"])


class TestRunSyncGuards:
    """Tests for .run() entry-point guards (loop detection, nested call prevention)."""

    def test_evaluate_rejects_running_loop(self) -> None:
        """.run() raises RuntimeError when called from async context."""

        async def inner():
            task(lambda: 1)().run()

        with pytest.raises(RuntimeError, match="Cannot call evaluate"):
            asyncio.run(inner())

    def test_evaluate_async_rejects_nested_call(self) -> None:
        """.run_async() raises RuntimeError when called from within a task."""

        @task
        async def outer() -> int:
            inner_task = task(lambda: 1)
            return await inner_task().run_async()

        with pytest.raises(RuntimeError, match="Cannot call evaluate.*from within another task"):
            asyncio.run(outer().run_async())

    def test_evaluate_rejects_nested_call(self) -> None:
        """.run() raises RuntimeError when called from within a task."""

        @task
        def outer() -> int:
            inner_task = task(lambda: 1)
            return inner_task().run()

        with pytest.raises(RuntimeError, match="Cannot call evaluate"):
            outer().run()
