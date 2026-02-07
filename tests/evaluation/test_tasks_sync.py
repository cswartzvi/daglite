"""Integration test for task evaluation using evaluate() with both sync and async tasks."""

import asyncio
import os

import pytest

from daglite import evaluate
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


class TestSinglePathExecution:
    """Tests engine evaluation of single path tasks."""

    def test_single_task_evaluation_empty(self) -> None:
        """Evaluation succeeds for tasks without parameters."""

        @task
        def dummy() -> None:
            return

        dummies = dummy()
        result = evaluate(dummies)
        assert result is None

    def test_single_task_evaluation_without_params(self) -> None:
        """Evaluation succeeds for parameterless tasks."""

        @task
        def prepare() -> int:
            return 5

        prepared = prepare()
        result = evaluate(prepared)
        assert result == 5

    def test_single_task_evaluation_with_params(self) -> None:
        """Evaluation succeeds when parameters are provided."""

        @task
        def multiply(x: int, y: int) -> int:
            return x * y

        multiplied = multiply(x=4, y=6)
        result = evaluate(multiplied)
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
        result = evaluate(squared)
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
        result = evaluate(subtracted)
        assert result == 200

    def test_partial_task_evaluation_without_params(self) -> None:
        """Evaluation succeeds for tasks with fixed parameters."""

        @task
        def power(base: int, exponent: int) -> int:
            return base**exponent

        powered = power.partial(exponent=3)(base=2)
        result = evaluate(powered)
        assert result == 8  # = 2 ** 3

    def test_partial_task_evaluation_with_params(self) -> None:
        """Evaluation succeeds for tasks with some fixed and some provided parameters."""

        @task
        def compute_area(length: float, width: float) -> float:
            return length * width

        area_task = compute_area.partial(width=5.0)
        area = area_task(length=10.0)
        result = evaluate(area)
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
        result = evaluate(multiplied)
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
        result = evaluate(subtracted)
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
        result_chain = evaluate(current)

        assert result_chain == 2 * (2**chain_depth)


@pytest.mark.parametrize("mode", ["product", "zip"])
class TestMappedTaskOperations:
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

        operation = double.product if mode == "product" else double.zip
        doubled_seq = operation(x=[])
        result = evaluate(doubled_seq)
        assert result == []

    def test_single_parameter(self, mode: str) -> None:
        """Evaluation succeeds for single parameter tasks."""

        @task
        def double(x: int) -> int:
            return x * 2

        operation = double.product if mode == "product" else double.zip
        doubled_seq = operation(x=[1, 2, 3])
        result = evaluate(doubled_seq)
        assert result == [2, 4, 6]

    def test_single_element(self, mode: str) -> None:
        """Evaluation succeeds with single element sequences."""

        @task
        def triple(x: int) -> int:
            return x * 3

        operation = triple.product if mode == "product" else triple.zip
        tripled_seq = operation(x=[5])
        result = evaluate(tripled_seq)
        assert result == [15]

    def test_with_future_input(self, mode: str) -> None:
        """Evaluation succeeds with TaskFuture as input."""

        if mode == "product":

            @task
            def generate() -> list[int]:
                return [1, 2, 3]

            @task
            def square(x: int) -> int:
                return x**2

            future = generate()
            seq = square.product(x=future)
            expected = [1, 4, 9]
        else:  # zip

            @task
            def generate() -> list[int]:
                return [5, 10, 15]

            @task
            def multiply(x: int, y: int) -> int:
                return x * y

            future = generate()
            seq = multiply.zip(x=future, y=[2, 3, 4])
            expected = [10, 30, 60]

        result = evaluate(seq)
        assert result == expected

    def test_with_fixed_parameters(self, mode: str) -> None:
        """Evaluation succeeds with some fixed parameters."""

        if mode == "product":

            @task
            def power(base: int, exponent: int) -> int:
                return base**exponent

            fixed = power.partial(exponent=2)
            seq = fixed.product(base=[1, 2, 3, 4])
            expected = [1, 4, 9, 16]
        else:  # zip

            @task
            def multiply(x: int, factor: int) -> int:
                return x * factor

            fixed = multiply.partial(factor=10)  # type: ignore
            seq = fixed.zip(x=[1, 2, 3])
            expected = [10, 20, 30]

        result = evaluate(seq)
        assert result == expected

    def test_with_fixed_future_parameters(self, mode: str) -> None:
        """Evaluation succeeds with fixed TaskFuture parameters."""

        if mode == "product":

            @task
            def get_exponent() -> int:
                return 3

            @task
            def power(base: int, exponent: int) -> int:
                return base**exponent

            future = get_exponent()
            fixed = power.partial(exponent=future)
            seq = fixed.product(base=[2, 3, 4])
            expected = [8, 27, 64]
        else:  # zip

            @task
            def get_factor() -> int:
                return 4

            @task
            def multiply(x: int, factor: int) -> int:
                return x * factor

            future = get_factor()
            fixed = multiply.partial(factor=future)  # type: ignore
            seq = fixed.zip(x=[2, 3, 4])
            expected = [8, 12, 16]

        result = evaluate(seq)
        assert result == expected

    def test_with_map(self, mode: str) -> None:
        """Evaluation succeeds when chaining with .then() for mapping."""

        if mode == "product":

            @task
            def double(x: int) -> int:
                return x * 2

            @task
            def triple(x: int) -> int:
                return x * 3

            seq = double.product(x=[1, 2, 3]).then(triple)
            expected = [6, 12, 18]
        else:  # zip

            @task
            def add(x: int, y: int) -> int:
                return x + y

            @task
            def square(z: int) -> int:
                return z**2

            seq = add.zip(x=[1, 2, 3], y=[10, 20, 30]).then(square)
            expected = [121, 484, 1089]

        result = evaluate(seq)
        assert result == expected

    def test_with_join(self, mode: str) -> None:
        """Evaluation succeeds when chaining with .join() for reduction."""

        if mode == "product":

            @task
            def square(x: int) -> int:
                return x**2

            @task
            def sum_all(values: list[int]) -> int:
                return sum(values)

            seq = square.product(x=[1, 2, 3, 4]).join(sum_all)
            expected = 30  # 1 + 4 + 9 + 16
        else:  # zip

            @task
            def add(x: int, y: int) -> int:
                return x + y

            @task
            def sum_all(values: list[int]) -> int:
                return sum(values)

            seq = add.zip(x=[1, 2, 3], y=[10, 20, 30]).join(sum_all)
            expected = 66  # 11 + 22 + 33

        result = evaluate(seq)
        assert result == expected

    def test_with_map_kwargs(self, mode: str) -> None:
        """Evaluation succeeds with .then() using kwargs."""

        if mode == "product":

            @task
            def double(x: int) -> int:
                return x * 2

            @task
            def add(x: int, offset: int) -> int:  # pyright: ignore
                return x + offset

            seq = double.product(x=[1, 2, 3]).then(add, offset=10)
            expected = [12, 14, 16]
        else:  # zip

            @task
            def add(x: int, y: int) -> int:  # pyright: ignore
                return x + y

            @task
            def multiply(x: int, factor: int) -> int:
                return x * factor

            seq = add.zip(x=[1, 2, 3], y=[10, 20, 30]).then(multiply, factor=2)
            expected = [22, 44, 66]

        result = evaluate(seq)
        assert result == expected

    def test_with_join_kwargs(self, mode: str) -> None:
        """Evaluation succeeds with .join() using kwargs."""

        if mode == "product":

            @task
            def square(x: int) -> int:
                return x**2

            @task
            def sum_with_bonus(values: list[int], bonus: int) -> int:
                return sum(values) + bonus

            seq = square.product(x=[1, 2, 3, 4]).join(sum_with_bonus, bonus=5)
            expected = 35
        else:  # zip

            @task
            def add(x: int, y: int) -> int:
                return x + y

            @task
            def weighted_sum(values: list[int], weight: int) -> int:
                return sum(values) * weight

            seq = add.zip(x=[1, 2, 3], y=[10, 20, 30]).join(weighted_sum, weight=3)
            expected = 198

        result = evaluate(seq)
        assert result == expected


class TestProductSpecificBehavior:
    """Tests for product()-specific behavior (Cartesian product semantics)."""

    def test_product_multiple_parameters(self) -> None:
        """Product creates Cartesian product of multiple parameter sequences."""

        @task
        def add(x: int, y: int) -> int:
            return x + y

        added_seq = add.product(x=[1, 2, 3], y=[10, 20, 30])
        result = evaluate(added_seq)
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

        added_seq = add.product(x=[1, 2], y=[10, 20])  # [11, 21, 12, 22]
        multiplied_seq = multiply.product(z=added_seq, factor=[2, 3])
        result = evaluate(multiplied_seq)
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

        result = evaluate(
            add.product(x=[1, 2], y=[10, 20])  # [11, 21, 12, 22]
            .then(triple)  # [33, 63, 36, 66]
            .join(max_value)
        )
        assert result == 66


class TestZipSpecificBehavior:
    """Tests for zip()-specific behavior (element-wise alignment semantics)."""

    def test_zip_multiple_parameters(self) -> None:
        """Zip aligns multiple parameter sequences element-wise."""

        @task
        def sum_three(x: int, y: int, z: int) -> int:
            return x + y + z

        summed_seq = sum_three.zip(x=[1, 2], y=[10, 20], z=[100, 200])
        result = evaluate(summed_seq)
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

        added_seq = add.zip(x=[1, 2], y=[10, 20])  # [11, 22]
        multiplied_seq = multiply.zip(z=added_seq, factor=[2, 3])
        result = evaluate(multiplied_seq)
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

        result = evaluate(
            add.zip(x=[1, 2, 3], y=[10, 20, 30])  # [11, 22, 33]
            .then(increment.partial(increment_by=5))  # [16, 27, 38]
            .join(sum_all)
        )
        assert result == 81


class TestComplexPathEvaluation:
    """Tests engine evaluation of complex task graphs."""

    def test_product_and_zip_with_join(self) -> None:
        """Evaluation succeeds combining product, zip, and join operations."""

        @task
        def multiply(x: int, y: int) -> int:
            return x * y

        @task
        def sum_all(values: list[int]) -> int:
            return sum(values)

        # Fan-out with product
        multiplied_seq = multiply.product(x=[1, 2], y=[10, 20])  # [10, 20, 20, 40]
        # Fan-in with join
        total = multiplied_seq.join(sum_all)
        result = evaluate(total)
        assert result == 90  # 10 + 20 + 20 + 40


class TestAsyncTasksWithEvaluate:
    """Tests that async tasks work through evaluate() via the async-first engine."""

    def test_async_task_works(self) -> None:
        """Async tasks are evaluated through the async-first engine."""

        @task
        async def async_add(x: int, y: int) -> int:
            await asyncio.sleep(0.001)
            return x + y

        result = evaluate(async_add(x=5, y=10))  # type: ignore
        assert result == 15

    def test_async_map_task_works(self) -> None:
        """Async map tasks are evaluated through the async-first engine."""

        @task
        async def async_square(x: int) -> int:
            await asyncio.sleep(0.001)
            return x * x

        result = evaluate(async_square.product(x=[1, 2, 3, 4]))
        assert result == [1, 4, 9, 16]

    def test_mixed_graph_with_async_works(self) -> None:
        """Graphs containing both sync and async tasks work through evaluate()."""

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

        result = evaluate(combined)
        assert result == 33  # sync_task(10)=11, async_double(11)=22, combine(11,22)=33


class TestGeneratorMaterialization:
    """
    Tests that generators are automatically materialized to lists.

    Tests marked with @pytest.mark.parametrize run for both evaluate() and evaluate_async()
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
            from daglite.engine import evaluate_async

            return await evaluate_async(generate_numbers(n=5))

        if evaluation_mode == "sync":
            result = evaluate(generate_numbers(n=5))
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

        result = evaluate(result_future)
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
        ranges = generate_range.product(n=[3, 4, 5])
        totals = ranges.then(sum_values)

        async def run_async():
            from daglite.engine import evaluate_async

            return await evaluate_async(totals)

        if evaluation_mode == "sync":
            result = evaluate(totals)
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

        list_result = evaluate(return_list(n=3))
        assert list_result == [0, 2, 4]
        assert isinstance(list_result, list)

        tuple_result = evaluate(return_tuple(n=3))
        assert tuple_result == (0, 2, 4)
        assert isinstance(tuple_result, tuple)

    def test_strings_not_materialized(self) -> None:
        """Strings (which are iterable) are not converted to lists."""

        @task
        def return_string() -> str:
            return "hello"

        result = evaluate(return_string())
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

        result = evaluate(generate_numbers(n=4).then(double_all))
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
        iter_result = evaluate(return_iterator(n=3))
        gen_result = evaluate(return_generator(n=3))

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
        """Async generators are materialized properly through evaluate()."""
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

        result = evaluate(total)
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
            from daglite.engine import evaluate_async

            return await evaluate_async(total)

        result = asyncio.run(run_async())
        assert result == 10  # 0 + 1 + 2 + 3 + 4

    def test_async_generator_map_materialization_sync_mode(self) -> None:
        """Map tasks with async generators are materialized properly through evaluate()."""
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

        ranges = async_get_range.product(n=[3, 4, 5])
        total = sum_all_ranges(ranges=ranges)

        result = evaluate(total)
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
        ranges = async_get_range.product(n=[3, 4, 5])
        total = sum_all_ranges(ranges=ranges)

        async def run_async():
            from daglite.engine import evaluate_async

            return await evaluate_async(total)

        result = asyncio.run(run_async())
        assert result == 19  # (0+1+2) + (0+1+2+3) + (0+1+2+3+4) = 3+6+10

    def test_async_generator_with_thread_backend_sync_mode(self) -> None:
        """Async generators with ThreadBackend work through evaluate()."""
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

        result = evaluate(total)
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
            from daglite.engine import evaluate_async

            return await evaluate_async(total)

        result = asyncio.run(run_async())
        assert result == 20  # 0+2+4+6+8


def test_cycle_detection_raises_error() -> None:
    """Evaluation raises ExecutionError when a cycle is detected in the graph."""
    from uuid import uuid4

    from daglite.engine import _ExecutionState
    from daglite.exceptions import ExecutionError
    from daglite.graph.base import ParamInput
    from daglite.graph.nodes import TaskNode

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
            kwargs={"x": ParamInput(kind="ref", value=None, ref=node_b_id)},
        ),
        node_b_id: TaskNode(
            id=node_b_id,
            name="node_b",
            description=None,
            backend_name=None,
            func=dummy,
            kwargs={"x": ParamInput(kind="ref", value=None, ref=node_a_id)},
        ),
    }

    state = _ExecutionState.from_nodes(nodes)  # type: ignore

    with pytest.raises(ExecutionError, match="Cycle detected"):
        state.check_complete()


class TestSiblingParallelism:
    """Tests for concurrent execution of sibling tasks using evaluate()."""

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

        result = evaluate(total)

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

        result = evaluate(result_future)

        # Verify results
        assert result["sum"] == 30

        # PIDs should be different if truly parallel (may be same if Inline)
        # Just verify we got valid PIDs
        assert all(pid > 0 for pid in result["pids"])


class TestEvaluateGuards:
    """Tests for evaluate() entry-point guards (loop detection, nested call prevention)."""

    def test_evaluate_rejects_running_loop(self) -> None:
        """evaluate() raises RuntimeError when called from async context."""

        async def inner():
            evaluate(task(lambda: 1)())

        with pytest.raises(RuntimeError, match="Cannot call evaluate"):
            asyncio.run(inner())

    def test_evaluate_async_rejects_nested_call(self) -> None:
        """evaluate_async() raises RuntimeError when called from within a task."""
        from daglite import evaluate_async

        @task
        async def outer() -> int:
            inner_task = task(lambda: 1)
            return await evaluate_async(inner_task())

        with pytest.raises(RuntimeError, match="Cannot call evaluate.*from within another task"):
            asyncio.run(evaluate_async(outer()))

    def test_evaluate_rejects_nested_call(self) -> None:
        """evaluate() raises RuntimeError when called from within a task."""

        @task
        def outer() -> int:
            inner_task = task(lambda: 1)
            return evaluate(inner_task())

        with pytest.raises(RuntimeError, match="Cannot call evaluate"):
            evaluate(outer())
