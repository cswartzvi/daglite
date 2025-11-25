import threading
import time

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


class TestExtendEvaluation:
    """Tests engine evaluation of mapped tasks using extend."""

    def test_extend_with_empty_sequence(self) -> None:
        """Evaluation succeeds for extend with empty sequence."""

        @task
        def double(x: int) -> int:
            return x * 2  # pragma: no cover

        doubled_seq = double.extend(x=[])
        result = evaluate(doubled_seq)
        assert result == []

    def test_extend_single_parameter(self) -> None:
        """Evaluation succeeds for single parameter task with extend (fan-out) behavior."""

        @task
        def double(x: int) -> int:
            return x * 2

        multiplied_seq = double.extend(x=[1, 2, 3])
        result = evaluate(multiplied_seq)
        assert result == [2, 4, 6]

    def test_extend_with_single_element(self) -> None:
        """Evaluation succeeds for extend with single element sequence."""

        @task
        def triple(x: int) -> int:
            return x * 3

        tripled_seq = triple.extend(x=[5])
        result = evaluate(tripled_seq)
        assert result == [15]

    def test_extend_with_future_input(self) -> None:
        """Evaluation succeeds for extend with TaskFuture as input."""

        @task
        def generate_range() -> list[int]:
            return [1, 2, 3]

        @task
        def square(x: int) -> int:
            return x**2

        range_future = generate_range.bind()
        squared_seq = square.extend(x=range_future)
        result = evaluate(squared_seq)
        assert result == [1, 4, 9]

    def test_extend_multiple_parameters(self) -> None:
        """Evaluation succeeds for multiple parameter task with extend (fan-out) behavior."""

        @task
        def add(x: int, y: int) -> int:
            return x + y

        added_seq = add.extend(x=[1, 2, 3], y=[10, 20, 30])
        result = evaluate(added_seq)
        assert result == [11, 21, 31, 12, 22, 32, 13, 23, 33]  # Cartesian product

    def test_extend_with_fixed_parameters(self) -> None:
        """Evaluation succeeds for extend tasks with some fixed parameters."""

        @task
        def power(base: int, exponent: int) -> int:
            return base**exponent

        fixed_power = power.fix(exponent=2)
        powered_seq = fixed_power.extend(base=[1, 2, 3, 4])
        result = evaluate(powered_seq)
        assert result == [1, 4, 9, 16]  # Squares of 1, 2, 3, 4

    def test_extend_with_fixed_future_parameters(self) -> None:
        """Evaluation succeeds for extend tasks with fixed TaskFuture parameters."""

        @task
        def get_exponent() -> int:
            return 3

        @task
        def power(base: int, exponent: int) -> int:
            return base**exponent

        exponent_future = get_exponent.bind()
        fixed_power = power.fix(exponent=exponent_future)
        powered_seq = fixed_power.extend(base=[2, 3, 4])
        result = evaluate(powered_seq)
        assert result == [8, 27, 64]  # Cubes of 2, 3, 4

    def test_extend_with_map(self) -> None:
        """Evaluation succeeds for extend tasks with mapping behavior."""

        @task
        def double(x: int) -> int:
            return x * 2

        @task
        def triple(x: int) -> int:
            return x * 3

        doubled = double.extend(x=[1, 2, 3])
        tripled = doubled.map(triple)
        result = evaluate(tripled)
        assert result == [6, 12, 18]  # = [2*3, 4*3, 6*3]

    def test_extend_with_join(self) -> None:
        """Evaluation succeeds for extend followed by join (fan-out then fan-in)."""

        @task
        def square(x: int) -> int:
            return x**2

        @task
        def sum_all(values: list[int]) -> int:
            return sum(values)

        squared_seq = square.extend(x=[1, 2, 3, 4])
        total = squared_seq.join(sum_all)
        result = evaluate(total)
        assert result == 30  # 1 + 4 + 9 + 16

    def test_extend_with_fixed_map(self) -> None:
        """Evaluation succeeds for extend tasks with both fixed parameters and mapping."""

        @task
        def add(x: int, y: int) -> int:
            return x + y

        @task
        def square(z: int) -> int:
            return z**2

        fixed_add = add.fix(y=5)
        added_seq = fixed_add.extend(x=[1, 2, 3])
        squared_seq = added_seq.map(square)
        result = evaluate(squared_seq)
        assert result == [36, 49, 64]  # = [(1+5)^2, (2+5)^2, (3+5)^2]

    def test_extend_with_fixed_join(self) -> None:
        """Evaluation succeeds for extend followed by join with fixed parameters."""

        @task
        def multiply(x: int, y: int) -> int:
            return x * y

        @task
        def multiply_total(values: list[int], factor: int) -> int:
            return sum(values) * factor

        fixed_multiply_total = multiply_total.fix(factor=3)
        multiplied_seq = multiply.extend(x=[1, 2, 3], y=[10, 20, 30])
        total = multiplied_seq.join(fixed_multiply_total)
        result = evaluate(total)
        assert result == 1080  # (10 + 20 + 30 + 20 + 40 + 60 + 30 + 60 + 90) * 3

    def test_extend_with_fixed_map_and_join(self) -> None:
        """Evaluation succeeds for extend with map then join (fan-out, transform, fan-in)."""

        @task
        def add(x: int, y: int) -> int:
            return x + y

        @task
        def triple(z: int) -> int:
            return z * 3

        @task
        def max_value(values: list[int]) -> int:
            return max(values)

        added_seq = add.extend(x=[1, 2], y=[10, 20])  # [11, 21, 12, 22]
        tripled_seq = added_seq.map(triple)  # [33, 63, 36, 66]
        maximum = tripled_seq.join(max_value)
        result = evaluate(maximum)
        assert result == 66

    def test_extend_with_nested_tasks(self) -> None:
        """Evaluation succeeds for nested extend tasks."""

        @task
        def add(x: int, y: int) -> int:
            return x + y

        @task
        def multiply(z: int, factor: int) -> int:
            return z * factor

        added_seq = add.extend(x=[1, 2], y=[10, 20])  # [11, 21, 12, 22]
        multiplied_seq = multiply.extend(z=added_seq, factor=[2, 3])  # Cartesian product
        result = evaluate(multiplied_seq)
        assert result == [22, 33, 42, 63, 24, 36, 44, 66]


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


class TestComplexPathEvaluation:
    """Tests engine evaluation of complex task graphs."""

    def test_extend_and_zip_with_join(self) -> None:
        """Evaluation succeeds combining extend, zip, and join operations."""

        @task
        def multiply(x: int, y: int) -> int:
            return x * y

        @task
        def sum_all(values: list[int]) -> int:
            return sum(values)

        # Fan-out with extend
        multiplied_seq = multiply.extend(x=[1, 2], y=[10, 20])  # [10, 20, 20, 40]
        # Fan-in with join
        total = multiplied_seq.join(sum_all)
        result = evaluate(total)
        assert result == 90  # 10 + 20 + 20 + 40


class TestAsyncExecution:
    """Tests engine evaluation with use_async=True."""

    def test_single_task_async(self) -> None:
        """Async evaluation succeeds for single task."""

        @task
        def add(x: int, y: int) -> int:
            return x + y

        result = evaluate(add.bind(x=10, y=20), use_async=True)
        assert result == 30

    def test_chain_async(self) -> None:
        """Async evaluation preserves dependency order in chains."""

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
        multiplied = multiply.bind(z=added, factor=4)  # 40
        subtracted = subtract.bind(value=multiplied, decrement=15)  # 25
        result = evaluate(subtracted, use_async=True)
        assert result == 25

    def test_sibling_tasks_async(self) -> None:
        """Async evaluation handles multiple sibling tasks correctly."""

        threads = set()

        @task(backend="threading")
        def left() -> int:
            time.sleep(0.1)
            threads.add(threading.get_ident())
            return 5

        @task(backend="threading")
        def right() -> int:
            threads.add(threading.get_ident())
            return 10

        @task
        def combine(a: int, b: int) -> int:
            return a + b

        left_future = left.bind()
        right_future = right.bind()
        combined = combine.bind(a=left_future, b=right_future)

        result = evaluate(combined, use_async=True)
        assert result == 15
        assert len(threads) == 2  # Both tasks ran in parallel threads

    def test_extend_async(self) -> None:
        """Async evaluation handles parallel extend operations."""

        @task
        def square(x: int) -> int:
            return x**2

        @task
        def sum_all(values: list[int]) -> int:
            return sum(values)

        squared_seq = square.extend(x=[1, 2, 3, 4])  # [1, 4, 9, 16]
        total = squared_seq.join(sum_all)
        result = evaluate(total, use_async=True)
        assert result == 30

    def test_zip_async(self) -> None:
        """Async evaluation handles zip operations."""

        @task
        def add(x: int, y: int) -> int:
            return x + y

        @task
        def product(values: list[int]) -> int:
            result = 1
            for v in values:
                result *= v
            return result

        added_seq = add.zip(x=[1, 2, 3], y=[10, 20, 30])  # [11, 22, 33]
        prod = added_seq.join(product)
        result = evaluate(prod, use_async=True)
        assert result == 11 * 22 * 33

    def test_error_propagation_async(self) -> None:
        """Async evaluation propagates exceptions correctly."""

        @task
        def divide(x: int, y: int) -> float:
            return x / y

        divided = divide.bind(x=10, y=0)

        try:
            evaluate(divided, use_async=True)
            assert False, "Should have raised ZeroDivisionError"  # pragma: no cover
        except ZeroDivisionError:
            pass  # Expected

    def test_complex_graph_async(self) -> None:
        """Async evaluation handles complex graphs with multiple paths."""

        @task
        def add(x: int, y: int) -> int:
            return x + y

        @task
        def multiply(x: int, y: int) -> int:
            return x * y

        # Diamond pattern: two paths that merge
        a = add.bind(x=1, y=2)  # 3
        b = multiply.bind(x=2, y=3)  # 6
        c = add.bind(x=a, y=b)  # 9
        result = evaluate(c, use_async=True)
        assert result == 9
