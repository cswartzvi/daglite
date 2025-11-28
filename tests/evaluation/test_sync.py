"""Tests for task evaluation using evaluate() with both sync and async tasks."""

import asyncio

from daglite import async_task
from daglite import evaluate
from daglite import task


class TestSinglePathExecution:
    """Tests engine evaluation of single path tasks."""

    def test_single_task_evaluation_empty(self) -> None:
        """Evaluation succeeds for tasks without parameters."""

        @task
        def dummy() -> None:
            return

        dummies = dummy.bind()
        result = evaluate(dummies)
        assert result is None

    def test_single_task_evaluation_without_params(self) -> None:
        """Evaluation succeeds for parameterless tasks."""

        @task
        def prepare() -> int:
            return 5

        prepared = prepare.bind()
        result = evaluate(prepared)
        assert result == 5

    def test_single_task_evaluation_with_params(self) -> None:
        """Evaluation succeeds when parameters are provided."""

        @task
        def multiply(x: int, y: int) -> int:
            return x * y

        multiplied = multiply.bind(x=4, y=6)
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

        prepared = prepare.bind()
        squared = square.bind(z=prepared)
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

        added = add.bind(x=3, y=7)  # 10
        multiplied1 = multiply.bind(z=added, factor=4)
        multiplied2 = multiply.bind(z=multiplied1, factor=2)  # 80
        multiplied3 = multiply.bind(z=multiplied2, factor=3)  # 240
        multiplied4 = multiply.bind(z=multiplied3, factor=1)  # 240
        subtracted = subtract.bind(value=multiplied4, decrement=40)  # 200
        result = evaluate(subtracted)
        assert result == 200

    def test_fixed_task_evaluation_without_params(self) -> None:
        """Evaluation succeeds for tasks with fixed parameters."""

        @task
        def power(base: int, exponent: int) -> int:
            return base**exponent

        powered = power.fix(exponent=3).bind(base=2)
        result = evaluate(powered)
        assert result == 8  # = 2 ** 3

    def test_fixed_task_evaluation_with_params(self) -> None:
        """Evaluation succeeds for tasks with some fixed and some provided parameters."""

        @task
        def compute_area(length: float, width: float) -> float:
            return length * width

        area_task = compute_area.fix(width=5.0)
        area = area_task.bind(length=10.0)
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

        fixed_multiply = multiply.fix(factor=10)

        added = add.bind(x=2, y=3)  # 5
        multiplied = fixed_multiply.bind(z=added)  # 50
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

        fixed_multiply = multiply.fix(factor=3)
        fixed_subtract = subtract.fix(decrement=4)

        added = add.bind(x=7, y=8)  # 15
        multiplied = fixed_multiply.bind(z=added)  # 45
        subtracted = fixed_subtract.bind(value=multiplied)  # 41
        result = evaluate(subtracted)
        assert result == 41

    def test_deep_chain_with_task_reuse_and_independent_branches(self) -> None:
        """Evaluation succeeds for deep chains with reused tasks and independent branches."""

        @task
        def multiply(x: int, factor: int) -> int:
            return x * factor

        # Start with 2, multiply by 2 repeatedly
        chain_depth = 2000
        current = multiply.bind(x=2, factor=2)
        for i in range(chain_depth - 1):
            current = multiply.bind(x=current, factor=2)

        # Evaluate all three independently
        result_chain = evaluate(current)

        assert result_chain == 2 * (2**chain_depth)


class TestProductEvaluation:
    """Tests engine evaluation of mapped tasks using extend."""

    def test_product_with_empty_sequence(self) -> None:
        """Evaluation succeeds for product with empty sequence."""

        @task
        def double(x: int) -> int:
            return x * 2  # pragma: no cover

        doubled_seq = double.product(x=[])
        result = evaluate(doubled_seq)
        assert result == []

    def test_product_single_parameter(self) -> None:
        """Evaluation succeeds for single parameter task with extend (fan-out) behavior."""

        @task
        def double(x: int) -> int:
            return x * 2

        multiplied_seq = double.product(x=[1, 2, 3])
        result = evaluate(multiplied_seq)
        assert result == [2, 4, 6]

    def test_product_with_single_element(self) -> None:
        """Evaluation succeeds for product with single element sequence."""

        @task
        def triple(x: int) -> int:
            return x * 3

        tripled_seq = triple.product(x=[5])
        result = evaluate(tripled_seq)
        assert result == [15]

    def test_product_with_future_input(self) -> None:
        """Evaluation succeeds for product with TaskFuture as input."""

        @task
        def generate_range() -> list[int]:
            return [1, 2, 3]

        @task
        def square(x: int) -> int:
            return x**2

        range_future = generate_range.bind()
        squared_seq = square.product(x=range_future)
        result = evaluate(squared_seq)
        assert result == [1, 4, 9]

    def test_product_multiple_parameters(self) -> None:
        """Evaluation succeeds for multiple parameter task with extend (fan-out) behavior."""

        @task
        def add(x: int, y: int) -> int:
            return x + y

        added_seq = add.product(x=[1, 2, 3], y=[10, 20, 30])
        result = evaluate(added_seq)
        assert result == [11, 21, 31, 12, 22, 32, 13, 23, 33]  # Cartesian product

    def test_product_with_fixed_parameters(self) -> None:
        """Evaluation succeeds for product tasks with some fixed parameters."""

        @task
        def power(base: int, exponent: int) -> int:
            return base**exponent

        fixed_power = power.fix(exponent=2)
        powered_seq = fixed_power.product(base=[1, 2, 3, 4])
        result = evaluate(powered_seq)
        assert result == [1, 4, 9, 16]  # Squares of 1, 2, 3, 4

    def test_product_with_fixed_future_parameters(self) -> None:
        """Evaluation succeeds for product tasks with fixed TaskFuture parameters."""

        @task
        def get_exponent() -> int:
            return 3

        @task
        def power(base: int, exponent: int) -> int:
            return base**exponent

        exponent_future = get_exponent.bind()
        fixed_power = power.fix(exponent=exponent_future)
        powered_seq = fixed_power.product(base=[2, 3, 4])
        result = evaluate(powered_seq)
        assert result == [8, 27, 64]  # Cubes of 2, 3, 4

    def test_product_with_map(self) -> None:
        """Evaluation succeeds for product tasks with mapping behavior."""

        @task
        def double(x: int) -> int:
            return x * 2

        @task
        def triple(x: int) -> int:
            return x * 3

        doubled = double.product(x=[1, 2, 3])
        tripled = doubled.map(triple)
        result = evaluate(tripled)
        assert result == [6, 12, 18]  # = [2*3, 4*3, 6*3]

    def test_product_with_join(self) -> None:
        """Evaluation succeeds for product followed by join (fan-out then fan-in)."""

        @task
        def square(x: int) -> int:
            return x**2

        @task
        def sum_all(values: list[int]) -> int:
            return sum(values)

        squared_seq = square.product(x=[1, 2, 3, 4])
        total = squared_seq.join(sum_all)
        result = evaluate(total)
        assert result == 30  # 1 + 4 + 9 + 16

    def test_product_with_fixed_map(self) -> None:
        """Evaluation succeeds for product tasks with both fixed parameters and mapping."""

        @task
        def add(x: int, y: int) -> int:
            return x + y

        @task
        def square(z: int) -> int:
            return z**2

        fixed_add = add.fix(y=5)
        added_seq = fixed_add.product(x=[1, 2, 3])
        squared_seq = added_seq.map(square)
        result = evaluate(squared_seq)
        assert result == [36, 49, 64]  # = [(1+5)^2, (2+5)^2, (3+5)^2]

    def test_product_with_fixed_join(self) -> None:
        """Evaluation succeeds for product followed by join with fixed parameters."""

        @task
        def multiply(x: int, y: int) -> int:
            return x * y

        @task
        def multiply_total(values: list[int], factor: int) -> int:
            return sum(values) * factor

        fixed_multiply_total = multiply_total.fix(factor=3)
        multiplied_seq = multiply.product(x=[1, 2, 3], y=[10, 20, 30])
        total = multiplied_seq.join(fixed_multiply_total)
        result = evaluate(total)
        assert result == 1080  # (10 + 20 + 30 + 20 + 40 + 60 + 30 + 60 + 90) * 3

    def test_product_with_fixed_map_and_join(self) -> None:
        """Evaluation succeeds for product with map then join (fan-out, transform, fan-in)."""

        @task
        def add(x: int, y: int) -> int:
            return x + y

        @task
        def triple(z: int) -> int:
            return z * 3

        @task
        def max_value(values: list[int]) -> int:
            return max(values)

        added_seq = add.product(x=[1, 2], y=[10, 20])  # [11, 21, 12, 22]
        tripled_seq = added_seq.map(triple)  # [33, 63, 36, 66]
        maximum = tripled_seq.join(max_value)
        result = evaluate(maximum)
        assert result == 66

    def test_product_with_nested_tasks(self) -> None:
        """Evaluation succeeds for nested extend tasks."""

        @task
        def add(x: int, y: int) -> int:
            return x + y

        @task
        def multiply(z: int, factor: int) -> int:
            return z * factor

        added_seq = add.product(x=[1, 2], y=[10, 20])  # [11, 21, 12, 22]
        multiplied_seq = multiply.product(z=added_seq, factor=[2, 3])  # Cartesian product
        result = evaluate(multiplied_seq)
        assert result == [22, 33, 42, 63, 24, 36, 44, 66]

    def test_product_with_map_kwargs(self) -> None:
        """Evaluation succeeds for product with .map() using kwargs."""

        @task
        def double(x: int) -> int:
            return x * 2

        @task
        def add(x: int, offset: int) -> int:
            return x + offset

        result = evaluate(double.product(x=[1, 2, 3]).map(add, offset=10))
        assert result == [12, 14, 16]  # [2, 4, 6] -> add 10 -> [12, 14, 16]

    def test_product_with_join_kwargs(self) -> None:
        """Evaluation succeeds for product with .join() using kwargs."""

        @task
        def square(x: int) -> int:
            return x**2

        @task
        def sum_with_bonus(values: list[int], bonus: int) -> int:
            return sum(values) + bonus

        result = evaluate(square.product(x=[1, 2, 3, 4]).join(sum_with_bonus, bonus=5))
        assert result == 35  # (1 + 4 + 9 + 16) + 5


class TestZipEvaluation:
    """Tests engine evaluation of mapped tasks using zip."""

    def test_zip_with_empty_sequence(self) -> None:
        """Evaluation succeeds for zip with empty sequence."""

        @task
        def add(x: int, y: int) -> int:
            return x + y  # pragma: no cover

        added_seq = add.zip(x=[], y=[])
        result = evaluate(added_seq)
        assert result == []

    def test_zip_with_single_parameter(self) -> None:
        """Evaluation succeeds for single parameter task with zip (aligned sequences)."""

        @task
        def double(x: int) -> int:
            return x * 2

        doubled_seq = double.zip(x=[1, 2, 3])
        result = evaluate(doubled_seq)
        assert result == [2, 4, 6]

    def test_zip_with_single_element(self) -> None:
        """Evaluation succeeds for zip with single element sequences."""

        @task
        def triple(x: int) -> int:
            return x * 3

        tripled_seq = triple.zip(x=[5])
        result = evaluate(tripled_seq)
        assert result == [15]

    def test_zip_with_future_input(self) -> None:
        """Evaluation succeeds for zip with TaskFuture as input."""

        @task
        def generate_values() -> list[int]:
            return [5, 10, 15]

        @task
        def multiply(x: int, y: int) -> int:
            return x * y

        values_future = generate_values.bind()
        multiplied_seq = multiply.zip(x=values_future, y=[2, 3, 4])
        result = evaluate(multiplied_seq)
        assert result == [10, 30, 60]  # (5*2), (10*3), (15*4)

    def test_task_zip_with_multiple_parameters(self) -> None:
        """Evaluation succeeds for zip with three aligned sequences."""

        @task
        def sum_three(x: int, y: int, z: int) -> int:
            return x + y + z

        summed_seq = sum_three.zip(x=[1, 2], y=[10, 20], z=[100, 200])
        result = evaluate(summed_seq)
        assert result == [111, 222]  # (1+10+100), (2+20+200)

    def test_zip_with_fixed_parameters(self) -> None:
        """Evaluation succeeds for zip tasks with some fixed parameters."""

        @task
        def multiply(x: int, factor: int) -> int:
            return x * factor

        fixed_multiply = multiply.fix(factor=10)
        multiplied_seq = fixed_multiply.zip(x=[1, 2, 3])
        result = evaluate(multiplied_seq)
        assert result == [10, 20, 30]

    def test_zip_with_fixed_future_parameters(self) -> None:
        """Evaluation succeeds for zip tasks with fixed TaskFuture parameters."""

        @task
        def get_factor() -> int:
            return 4

        @task
        def multiply(x: int, factor: int) -> int:
            return x * factor

        factor_future = get_factor.bind()
        fixed_multiply = multiply.fix(factor=factor_future)
        multiplied_seq = fixed_multiply.zip(x=[2, 3, 4])
        result = evaluate(multiplied_seq)
        assert result == [8, 12, 16]  # (2*4), (3*4), (4*4)

    def test_zip_with_map(self) -> None:
        """Evaluation succeeds for zip tasks with mapping behavior."""

        @task
        def add(x: int, y: int) -> int:
            return x + y

        @task
        def square(z: int) -> int:
            return z**2

        added_seq = add.zip(x=[1, 2, 3], y=[10, 20, 30])
        squared_seq = added_seq.map(square)
        result = evaluate(squared_seq)
        assert result == [121, 484, 1089]  # = [11^2, 22^2, 33^2]

    def test_zip_with_join(self) -> None:
        """Evaluation succeeds for zip followed by join (fan-out then fan-in)."""

        @task
        def add(x: int, y: int) -> int:
            return x + y

        @task
        def sum_all(values: list[int]) -> int:
            return sum(values)

        added_seq = add.zip(x=[1, 2, 3], y=[10, 20, 30])
        total = added_seq.join(sum_all)
        result = evaluate(total)
        assert result == 66  # 11 + 22 + 33

    def test_zip_with_fixed_map(self) -> None:
        """Evaluation succeeds for zip tasks with both fixed parameters and mapping."""

        @task
        def multiply(x: int, factor: int) -> int:
            return x * factor

        @task
        def double(z: int) -> int:
            return z * 2

        fixed_multiply = multiply.fix(factor=3)
        multiplied_seq = fixed_multiply.zip(x=[2, 4, 6])
        doubled_seq = multiplied_seq.map(double)
        result = evaluate(doubled_seq)
        assert result == [12, 24, 36]  # = [(2*3)*2, (4*3)*2, (6*3)*2]

    def test_zip_with_fixed_join(self) -> None:
        """Evaluation succeeds for zip followed by join with fixed parameters."""

        @task
        def add(x: int, y: int) -> int:
            return x + y

        @task
        def multiply_total(values: list[int], factor: int) -> int:
            return sum(values) * factor

        fixed_multiply_total = multiply_total.fix(factor=2)
        added_seq = add.zip(x=[1, 2, 3], y=[10, 20, 30])
        total = added_seq.join(fixed_multiply_total)
        result = evaluate(total)
        assert result == 132  # (11 + 22 + 33) * 2

    def test_zip_with_fix_map_and_join(self) -> None:
        """Evaluation succeeds for zip with fixed map then join (fan-out, transform, fan-in)."""

        @task
        def add(x: int, y: int) -> int:
            return x + y

        @task
        def increment(z: int, increment_by: int) -> int:
            return z + increment_by

        @task
        def sum_all(values: list[int]) -> int:
            return sum(values)

        fixed_increment = increment.fix(increment_by=5)
        added_seq = add.zip(x=[1, 2, 3], y=[10, 20, 30])  # [11, 22, 33]
        incremented_seq = added_seq.map(fixed_increment)  # [16, 27, 38]
        total = incremented_seq.join(sum_all)
        result = evaluate(total)
        assert result == 81  # 16 + 27 + 38

    def test_zip_with_nested_tasks(self) -> None:
        """Evaluation succeeds for nested zip tasks."""

        @task
        def add(x: int, y: int) -> int:
            return x + y

        @task
        def multiply(z: int, factor: int) -> int:
            return z * factor

        added_seq = add.zip(x=[1, 2], y=[10, 20])  # [11, 22]
        multiplied_seq = multiply.zip(z=added_seq, factor=[2, 3])  # [22, 66]
        result = evaluate(multiplied_seq)
        assert result == [22, 66]  # (11*2), (22*3)

    def test_zip_with_map_kwargs(self) -> None:
        """Evaluation succeeds for zip with .map() using kwargs."""

        @task
        def add(x: int, y: int) -> int:
            return x + y

        @task
        def multiply(x: int, factor: int) -> int:
            return x * factor

        result = evaluate(add.zip(x=[1, 2, 3], y=[10, 20, 30]).map(multiply, factor=2))
        assert result == [22, 44, 66]  # [11, 22, 33] -> multiply by 2

    def test_zip_with_join_kwargs(self) -> None:
        """Evaluation succeeds for zip with .join() using kwargs."""

        @task
        def add(x: int, y: int) -> int:
            return x + y

        @task
        def weighted_sum(values: list[int], weight: int) -> int:
            return sum(values) * weight

        result = evaluate(add.zip(x=[1, 2, 3], y=[10, 20, 30]).join(weighted_sum, weight=3))
        assert result == 198  # (11 + 22 + 33) * 3


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
    """Tests for async tasks evaluated with evaluate() (synchronous execution)."""

    def test_async_task_sync_evaluation(self) -> None:
        """Async tasks can be evaluated synchronously."""

        @async_task
        async def async_add(x: int, y: int) -> int:
            await asyncio.sleep(0.001)  # Simulate async work
            return x + y

        result = evaluate(async_add.bind(x=5, y=10))
        assert result == 15

    def test_async_task_with_dependencies(self) -> None:
        """Async tasks work with dependencies."""

        @task
        def sync_start(n: int) -> int:
            return n * 2

        @async_task
        async def async_process(x: int) -> int:
            await asyncio.sleep(0.001)
            return x + 10

        @task
        def sync_finish(x: int) -> int:
            return x * 3

        # Chain: sync -> async -> sync
        start = sync_start.bind(n=5)  # 10
        processed = async_process.bind(x=start)  # 20
        finished = sync_finish.bind(x=processed)  # 60

        result = evaluate(finished)
        assert result == 60

    def test_async_task_product(self) -> None:
        """Async tasks work with product operations."""

        @async_task
        async def async_square(x: int) -> int:
            await asyncio.sleep(0.001)
            return x * x

        @task
        def sum_all(values: list[int]) -> int:
            return sum(values)

        squared = async_square.product(x=[1, 2, 3, 4])
        total = squared.join(sum_all)

        result = evaluate(total)
        assert result == 30  # 1 + 4 + 9 + 16

    def test_async_task_zip(self) -> None:
        """Async tasks work with zip operations."""

        @async_task
        async def async_add(x: int, y: int) -> int:
            await asyncio.sleep(0.001)
            return x + y

        @task
        def product(values: list[int]) -> int:
            result = 1
            for v in values:
                result *= v
            return result

        added = async_add.zip(x=[1, 2, 3], y=[10, 20, 30])
        prod = added.join(product)

        result = evaluate(prod)
        assert result == 11 * 22 * 33

    def test_async_task_with_then_chain(self) -> None:
        """Async tasks work with .then() chains."""

        @async_task
        async def async_double(x: int) -> int:
            await asyncio.sleep(0.001)
            return x * 2

        @task
        def add_ten(x: int) -> int:
            return x + 10

        @async_task
        async def async_square(x: int) -> int:
            await asyncio.sleep(0.001)
            return x * x

        # Chain: async -> sync -> async
        result = evaluate(async_double.bind(x=5).then(add_ten).then(async_square))
        assert result == 400  # (5*2 + 10)^2 = 20^2 = 400

    def test_mixed_sync_async_complex_graph(self) -> None:
        """Complex graph with mixed sync and async tasks."""

        @task
        def sync_task(x: int) -> int:
            return x + 1

        @async_task
        async def async_double(x: int) -> int:
            await asyncio.sleep(0.001)
            return x * 2

        @task
        def combine(a: int, b: int, c: int) -> int:
            return a + b + c

        # Diamond pattern with mixed sync/async
        start = sync_task.bind(x=10)  # 11
        left = async_double.bind(x=start)  # 22
        right = sync_task.bind(x=start)  # 12
        result_future = combine.bind(a=start, b=left, c=right)

        result = evaluate(result_future)
        assert result == 45  # 11 + 22 + 12


class TestGeneratorMaterialization:
    """Tests that generators are automatically materialized to lists with evaluate()."""

    def test_generator_is_materialized_sync(self) -> None:
        """Generators returned from tasks are materialized to lists in sync execution."""
        from typing import Iterator

        @task
        def generate_numbers(n: int) -> Iterator[int]:
            for i in range(n):
                yield i * 2

        result = evaluate(generate_numbers.bind(n=5))
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

        numbers = generate_numbers.bind(n=5)
        total = sum_values.bind(values=numbers)
        count = count_values.bind(values=numbers)
        result_future = combine.bind(total=total, count=count)

        result = evaluate(result_future)
        assert result == (10, 5)  # sum([0,1,2,3,4]) = 10, len = 5

    def test_generator_in_map_operation(self) -> None:
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
        totals = ranges.map(sum_values)

        result = evaluate(totals)
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

        list_result = evaluate(return_list.bind(n=3))
        assert list_result == [0, 2, 4]
        assert isinstance(list_result, list)

        tuple_result = evaluate(return_tuple.bind(n=3))
        assert tuple_result == (0, 2, 4)
        assert isinstance(tuple_result, tuple)

    def test_strings_not_materialized(self) -> None:
        """Strings (which are iterable) are not converted to lists."""

        @task
        def return_string() -> str:
            return "hello"

        result = evaluate(return_string.bind())
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

        result = evaluate(generate_numbers.bind(n=4).then(double_all))
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
        iter_result = evaluate(return_iterator.bind(n=3))
        gen_result = evaluate(return_generator.bind(n=3))

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

    def test_async_generator_materialization_sync_evaluation(self) -> None:
        """Async generators returned from async tasks are materialized in sync evaluation."""
        from collections.abc import AsyncGenerator

        @async_task
        async def async_generate_numbers(n: int) -> AsyncGenerator[int, None]:
            async def _gen():
                for i in range(n):
                    await asyncio.sleep(0.001)
                    yield i

            return _gen()

        @task
        def sum_values(values: list[int]) -> int:
            return sum(values)

        nums = async_generate_numbers.bind(n=5)
        total = sum_values.bind(values=nums)
        result = evaluate(total)
        assert result == 10  # 0 + 1 + 2 + 3 + 4

    def test_async_generator_map_materialization_sync_evaluation(self) -> None:
        """Map tasks that return async generators are materialized properly in sync evaluation."""
        from collections.abc import AsyncGenerator

        @async_task
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
        total = sum_all_ranges.bind(ranges=ranges)
        result = evaluate(total)
        assert result == 19  # (0+1+2) + (0+1+2+3) + (0+1+2+3+4) = 3+6+10

    def test_async_generator_with_thread_backend_sync_evaluation(self) -> None:
        """Async generators work with ThreadBackend in sync evaluation."""
        from collections.abc import AsyncGenerator

        @async_task(backend="threading")
        async def async_generate(n: int) -> AsyncGenerator[int, None]:
            async def _gen():
                for i in range(n):
                    await asyncio.sleep(0.001)
                    yield i * 2

            return _gen()

        @task
        def sum_values(values: list[int]) -> int:
            return sum(values)

        nums = async_generate.bind(n=5)
        total = sum_values.bind(values=nums)
        result = evaluate(total)
        assert result == 20  # 0+2+4+6+8
