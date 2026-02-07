"""Integration tests for async task evaluation with evaluate_async()."""

import asyncio
import threading
import time

import pytest

from daglite import evaluate_async
from daglite import task
from daglite.tasks import MapTaskFuture
from daglite.tasks import TaskFuture


# Module-level async tasks for ProcessBackend (must be picklable)
@task(backend_name="processes")
async def async_square_process(x: int) -> int:
    """Module-level async task for process backend testing."""
    await asyncio.sleep(0.001)
    return x * x


@task(backend_name="processes")
async def async_double_process(x: int) -> int:
    """Module-level async task for process backend testing."""
    await asyncio.sleep(0.001)
    return x * 2


@task(backend_name="processes")
async def async_add_process(y: int, z: int) -> int:
    """Module-level async task for process backend testing."""
    await asyncio.sleep(0.001)
    return y + z


# Module-level sync tasks for ProcessBackend map testing (must be picklable)
@task(backend_name="processes")
def double_process(x: int) -> int:
    """Module-level sync task for process backend map testing."""
    return x * 2


@task(backend_name="processes")
def square_process(z: int) -> int:
    """Module-level sync task for process backend map testing."""
    return z**2


class TestSyncTasksWithEvaluateAsync:
    """Tests evaluate_async() with regular (sync) task functions."""

    def test_single_task_async(self) -> None:
        """Async evaluation succeeds for single task."""
        import asyncio

        @task
        def add(x: int, y: int) -> int:
            return x + y

        async def run():
            return await evaluate_async(add(x=10, y=20))

        result = asyncio.run(run())
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

        added = add(x=3, y=7)  # 10
        multiplied = multiply(z=added, factor=4)  # 40
        subtracted = subtract(value=multiplied, decrement=15)  # 25

        async def run():
            return await evaluate_async(subtracted)

        result = asyncio.run(run())
        assert result == 25

    def test_sibling_tasks_async(self) -> None:
        """Async evaluation handles multiple sibling tasks correctly."""

        threads = set()

        @task(backend_name="threading")
        def left() -> int:
            time.sleep(0.1)
            threads.add(threading.get_ident())
            return 5

        @task(backend_name="threading")
        def right() -> int:
            threads.add(threading.get_ident())
            return 10

        @task
        def combine(a: int, b: int) -> int:
            return a + b

        left_future: TaskFuture[int] = left()
        right_future: TaskFuture[int] = right()
        combined = combine(a=left_future, b=right_future)

        async def run():
            return await evaluate_async(combined)

        result = asyncio.run(run())
        assert result == 15
        assert len(threads) == 2  # Both tasks ran in parallel threads

    def test_product_async(self) -> None:
        """Async evaluation handles parallel product operations."""

        @task
        def square(x: int) -> int:
            return x**2

        @task
        def sum_all(values: list[int]) -> int:
            return sum(values)

        squared_seq = square.product(x=[1, 2, 3, 4])  # [1, 4, 9, 16]
        total = squared_seq.join(sum_all)

        async def run():
            return await evaluate_async(total)

        result = asyncio.run(run())
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

        async def run():
            return await evaluate_async(prod)

        result = asyncio.run(run())
        assert result == 11 * 22 * 33

    def test_error_propagation_async(self) -> None:
        """Async evaluation propagates exceptions correctly."""

        @task(backend_name="threading")
        def divide(x: int, y: int) -> float:
            return x / y

        divided = divide(x=10, y=0)

        async def run():
            return await evaluate_async(divided)

        with pytest.raises(ZeroDivisionError):
            asyncio.run(run())

    def test_complex_graph_async(self) -> None:
        """Async evaluation handles complex graphs with multiple paths."""

        @task
        def add(x: int, y: int) -> int:
            return x + y

        @task
        def multiply(x: int, y: int) -> int:
            return x * y

        # Diamond pattern: two paths that merge
        a = add(x=1, y=2)  # 3
        b = multiply(x=2, y=3)  # 6
        c = add(x=a, y=b)  # 9

        async def run():
            return await evaluate_async(c)

        result = asyncio.run(run())
        assert result == 9

    def test_Inline_backend_with_sync_tasks(self) -> None:
        """Inline backend with sync tasks works in evaluate_async()."""

        @task(backend_name="Inline")
        def add(x: int, y: int) -> int:
            return x + y

        result_future = add(x=10, y=20)

        async def run():
            return await evaluate_async(result_future)

        assert asyncio.run(run()) == 30

    def test_Inline_backend_with_sync_map_tasks(self) -> None:
        """Inline backend with sync map tasks works in evaluate_async()."""

        @task(backend_name="Inline")
        def double(x: int) -> int:
            return x * 2

        result_future = double.product(x=[1, 2, 3])

        async def run():
            return await evaluate_async(result_future)

        assert asyncio.run(run()) == [2, 4, 6]

    def test_map_async(self) -> None:
        """Async evaluation handles map operations."""

        @task
        def double(x: int) -> int:
            return x * 2

        @task
        def square(z: int) -> int:
            return z**2

        doubled = double.product(x=[1, 2, 3])
        squared = doubled.then(square)

        async def run():
            return await evaluate_async(squared)

        result = asyncio.run(run())
        assert result == [4, 16, 36]  # [2, 4, 6] squared

    def test_join_async(self) -> None:
        """Async evaluation handles join operations."""

        @task
        def triple(x: int) -> int:
            return x * 3

        @task
        def sum_all(values: list[int]) -> int:
            return sum(values)

        tripled = triple.product(x=[1, 2, 3, 4])
        total = tripled.join(sum_all)

        async def run():
            return await evaluate_async(total)

        result = asyncio.run(run())
        assert result == 30  # (3 + 6 + 9 + 12)

    def test_map_async_with_thread_backend(self) -> None:
        """Async evaluation handles map with ThreadBackend (tests asyncio.gather path)."""

        @task(backend_name="threading")
        def double(x: int) -> int:
            return x * 2

        @task(backend_name="threading")
        def square(z: int) -> int:
            return z**2

        doubled: MapTaskFuture[int] = double.product(x=[1, 2, 3])
        squared: MapTaskFuture[int] = doubled.then(square)

        async def run():
            return await evaluate_async(squared)

        result = asyncio.run(run())
        assert result == [4, 16, 36]  # [2, 4, 6] squared

    def test_map_async_with_process_backend(self) -> None:
        """Async evaluation handles map with ProcessBackend."""

        doubled: MapTaskFuture[int] = double_process.product(x=[1, 2, 3])
        squared: MapTaskFuture[int] = doubled.then(square_process)

        async def run():
            return await evaluate_async(squared)

        result = asyncio.run(run())
        assert result == [4, 16, 36]  # [2, 4, 6] squared


class TestAsyncTasksWithEvaluateAsync:
    """Tests evaluate_async() with async task functions (async def)."""

    def test_async_task_async_evaluation(self) -> None:
        """Async tasks can be evaluated asynchronously."""

        @task
        async def async_multiply(x: int, factor: int) -> int:
            await asyncio.sleep(0.001)
            return x * factor

        async def run():
            return await evaluate_async(async_multiply(x=7, factor=3))

        result = asyncio.run(run())
        assert result == 21

    def test_async_task_parallel_execution(self) -> None:
        """Multiple async tasks execute in parallel."""

        import time

        execution_times: list[float] = []

        @task
        async def slow_task(duration: float) -> float:
            start = time.time()
            await asyncio.sleep(duration)
            execution_times.append(time.time() - start)
            return duration

        # Create three tasks that should run in parallel
        t1 = slow_task(duration=0.1)
        t2 = slow_task(duration=0.1)
        t3 = slow_task(duration=0.1)

        @task
        def combine(a: float, b: float, c: float) -> float:
            return a + b + c

        combined = combine(a=t1, b=t2, c=t3)

        async def run():
            start = time.time()
            result = await evaluate_async(combined)
            total_time = time.time() - start
            return result, total_time

        result, total_time = asyncio.run(run())

        assert abs(result - 0.3) < 0.001  # Floating point tolerance
        # If they ran in parallel, total time should be ~0.1s not ~0.3s
        assert total_time < 0.2  # Allow some overhead

    def test_async_task_error_propagation(self) -> None:
        """Errors in async tasks are properly propagated."""

        @task
        async def failing_task(x: int) -> int:
            await asyncio.sleep(0.001)
            raise ValueError("Async task failed!")

        future = failing_task(x=10)

        async def run():
            return await evaluate_async(future)

        try:
            asyncio.run(run())  # pyright: ignore
            assert False, "Should have raised ValueError"  # pragma: no cover
        except ValueError as e:
            assert str(e) == "Async task failed!"

    def test_async_with_thread_backend_coroutine_result(self) -> None:
        """Async evaluation with ThreadBackend awaits coroutine results."""

        @task(backend_name="threading")
        async def async_compute(x: int) -> int:
            await asyncio.sleep(0.001)
            return x * 2

        async def run():
            result_future = async_compute(x=21)
            return await evaluate_async(result_future)

        result = asyncio.run(run())
        assert result == 42

    def test_async_with_process_backend(self) -> None:
        """Async tasks work with ProcessBackend using persistent event loop."""

        async def run():
            result_future = async_square_process(x=7)
            return await evaluate_async(result_future)

        result = asyncio.run(run())
        assert result == 49

    def test_async_chain_with_process_backend(self) -> None:
        """Chain of async tasks works with ProcessBackend."""

        doubled = async_double_process(x=5)  # 10
        result = async_add_process(y=doubled, z=3)  # 13

        async def run():
            return await evaluate_async(result)

        final = asyncio.run(run())
        assert final == 13


class TestMappedOperationsWithEvaluateAsync:
    """Tests product/zip operations with async tasks and evaluate_async()."""

    def test_product_empty_sequence(self) -> None:
        """Product evaluation succeeds with empty sequences."""

        @task
        async def double(x: int) -> int:
            await asyncio.sleep(0.001)
            return x * 2  # pragma: no cover

        doubled_seq = double.product(x=[])

        async def run():
            return await evaluate_async(doubled_seq)

        result = asyncio.run(run())
        assert result == []

    def test_zip_empty_sequence(self) -> None:
        """Zip evaluation succeeds with empty sequences."""

        @task
        async def add(x: int, y: int) -> int:
            await asyncio.sleep(0.001)
            return x + y  # pragma: no cover

        added_seq = add.zip(x=[], y=[])

        async def run():
            return await evaluate_async(added_seq)

        result = asyncio.run(run())
        assert result == []

    def test_product_with_future_input(self) -> None:
        """Product evaluation succeeds with TaskFuture as input."""

        @task
        async def generate() -> list[int]:
            await asyncio.sleep(0.001)
            return [1, 2, 3]

        @task
        async def square(x: int) -> int:
            await asyncio.sleep(0.001)
            return x**2

        future = generate()
        seq = square.product(x=future)

        async def run():
            return await evaluate_async(seq)

        result = asyncio.run(run())
        assert result == [1, 4, 9]

    def test_zip_with_future_input(self) -> None:
        """Zip evaluation succeeds with TaskFuture as input."""

        @task
        async def generate() -> list[int]:
            await asyncio.sleep(0.001)
            return [5, 10, 15]

        @task
        async def multiply(x: int, y: int) -> int:
            await asyncio.sleep(0.001)
            return x * y

        future = generate()
        seq = multiply.zip(x=future, y=[2, 3, 4])

        async def run():
            return await evaluate_async(seq)

        result = asyncio.run(run())
        assert result == [10, 30, 60]

    def test_product_multiple_parameters(self) -> None:
        """Product creates Cartesian product of multiple parameter sequences."""

        @task
        async def add(x: int, y: int) -> int:
            await asyncio.sleep(0.001)
            return x + y

        added_seq = add.product(x=[1, 2, 3], y=[10, 20, 30])

        async def run():
            return await evaluate_async(added_seq)

        result = asyncio.run(run())
        assert result == [11, 21, 31, 12, 22, 32, 13, 23, 33]

    def test_product_with_nested_tasks(self) -> None:
        """Product with nested TaskFuture creates Cartesian product."""

        @task
        async def add(x: int, y: int) -> int:
            await asyncio.sleep(0.001)
            return x + y

        @task
        async def multiply(z: int, factor: int) -> int:
            await asyncio.sleep(0.001)
            return z * factor

        added_seq = add.product(x=[1, 2], y=[10, 20])  # [11, 21, 12, 22]
        multiplied_seq = multiply.product(z=added_seq, factor=[2, 3])

        async def run():
            return await evaluate_async(multiplied_seq)

        result = asyncio.run(run())
        assert result == [22, 33, 42, 63, 24, 36, 44, 66]

    def test_zip_pairwise_combination(self) -> None:
        """Zip creates pairwise combinations."""

        @task
        async def add(x: int, y: int) -> int:
            await asyncio.sleep(0.001)
            return x + y

        added_seq = add.zip(x=[1, 2, 3], y=[10, 20, 30])

        async def run():
            return await evaluate_async(added_seq)

        result = asyncio.run(run())
        assert result == [11, 22, 33]

    def test_product_with_map_and_join(self) -> None:
        """Product with map and join (fan-out, transform, fan-in)."""

        @task
        async def add(x: int, y: int) -> int:
            await asyncio.sleep(0.001)
            return x + y

        @task
        async def triple(z: int) -> int:
            await asyncio.sleep(0.001)
            return z * 3

        @task
        async def max_value(values: list[int]) -> int:
            await asyncio.sleep(0.001)
            return max(values)

        result_future = (
            add.product(x=[1, 2], y=[10, 20])  # [11, 21, 12, 22]
            .then(triple)  # [33, 63, 36, 66]
            .join(max_value)
        )

        async def run():
            return await evaluate_async(result_future)

        result = asyncio.run(run())
        assert result == 66


class TestGeneratorMaterializationWithEvaluateAsync:
    """Tests async generator materialization with evaluate_async()."""

    def test_async_generator_is_materialized(self) -> None:
        """Async generators returned from tasks are materialized to lists."""
        from typing import AsyncIterator

        @task
        async def generate_numbers(n: int) -> AsyncIterator[int]:
            for i in range(n):
                await asyncio.sleep(0.001)
                yield i * 2

        async def run():
            return await evaluate_async(generate_numbers(n=5))

        result = asyncio.run(run())
        assert result == [0, 2, 4, 6, 8]
        assert isinstance(result, list)

    def test_async_generator_reusable_by_multiple_consumers(self) -> None:
        """Materialized async generators can be consumed by multiple downstream tasks."""
        from typing import AsyncIterator

        @task
        async def generate_numbers(n: int) -> AsyncIterator[int]:
            for i in range(n):
                await asyncio.sleep(0.001)
                yield i

        @task
        async def sum_values(values: list[int]) -> int:
            await asyncio.sleep(0.001)
            return sum(values)

        @task
        async def count_values(values: list[int]) -> int:
            await asyncio.sleep(0.001)
            return len(values)

        @task
        async def combine(total: int, count: int) -> tuple[int, int]:
            await asyncio.sleep(0.001)
            return (total, count)

        numbers = generate_numbers(n=5)
        total = sum_values(values=numbers)
        count = count_values(values=numbers)
        result_future = combine(total=total, count=count)

        async def run():
            return await evaluate_async(result_future)

        result = asyncio.run(run())
        assert result == (10, 5)  # sum([0,1,2,3,4]) = 10, len = 5

    def test_async_generator_in_map_operation(self) -> None:
        """Async generators in map operations are materialized."""
        from typing import AsyncIterator

        @task
        async def generate_range(n: int) -> AsyncIterator[int]:
            for i in range(n):
                await asyncio.sleep(0.001)
                yield i

        @task
        async def square(values: list[int]) -> int:
            await asyncio.sleep(0.001)
            return sum(v**2 for v in values)

        generated = generate_range.product(n=[2, 3, 4])
        squared = generated.then(square)

        async def run():
            return await evaluate_async(squared)

        result = asyncio.run(run())
        # [0,1] -> 1, [0,1,2] -> 5, [0,1,2,3] -> 14
        assert result == [1, 5, 14]


class TestConcurrentSiblingTaskExecution:
    """Tests for concurrent execution of sibling tasks without timing assertions."""

    def test_sync_tasks_with_threading_backend_in_async_context(self) -> None:
        """Sync tasks with threading backend execute concurrently in async evaluation."""
        thread_ids: set[int] = set()
        lock = threading.Lock()

        @task(backend_name="threading")
        def track_thread(value: int) -> int:
            """Track which thread executes this task."""
            with lock:
                thread_ids.add(threading.get_ident())

            return value

        @task
        def sum_values(a: int, b: int, c: int) -> int:
            return a + b + c

        # Create sibling tasks that should run in different threads
        t1 = track_thread(value=10)
        t2 = track_thread(value=20)
        t3 = track_thread(value=30)
        total = sum_values(a=t1, b=t2, c=t3)

        async def run():
            return await evaluate_async(total)

        result = asyncio.run(run())

        # Verify results
        assert result == 60

        # Verify multiple threads were used (proves concurrent execution).
        # Note: This assumes the threading backend's executor uses a pool with
        # multiple threads. It may fail on single-core systems or if the thread
        # pool size is 1.
        assert len(thread_ids) > 1, (
            "Threading backend should execute sibling tasks in parallel using "
            "multiple threads. This may fail on single-core systems or if the "
            f"thread pool size is 1 (observed {len(thread_ids)} thread(s))."
        )
