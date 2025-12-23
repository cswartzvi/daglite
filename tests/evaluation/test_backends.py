"""Integration tests for backend execution with @task decorated functions.

These tests ensure that decorated tasks can be properly serialized and executed
across different backends, including the multiprocessing backend which requires
pickle support.
"""

import asyncio
import threading

import pytest

from daglite import evaluate
from daglite import evaluate_async
from daglite import pipeline
from daglite import task

# Module-level tasks for processes backend pickle compatibility
# Tasks MUST be defined at module level to be picklable by multiprocessing


@task
def square(x: int) -> int:
    return x * x


@task
def double(x: int) -> int:
    return x * 2


@task
def add(x: int, y: int) -> int:
    return x + y


@task
def sum_values(values: list[int]) -> int:
    return sum(values)


@task
def fetch_data() -> list[int]:
    return [1, 2, 3, 4, 5]


@task
def generate_range(n: int) -> list[int]:
    return list(range(n))


@task
def merge_dicts(d1: dict, d2: dict) -> dict:
    return {**d1, **d2}


@task
def failing_task(x: int) -> int:
    if x < 0:
        raise ValueError("negative value not allowed")
    if x == 0:
        raise ZeroDivisionError("cannot divide by zero")
    return x * 2


@task
def nested_computation(x: int) -> int:
    # Nested logic is fine as long as the task itself is module-level
    def helper(n: int) -> int:
        return n * 2

    return helper(x) + helper(x)


@pipeline
def increment_all_pipeline(nums: list[int]):
    incremented = add.zip(x=nums, y=[1] * len(nums))
    return incremented


class TestBackendIntegration:
    """Tests for backend execution with actual @task decorated functions."""

    @pytest.mark.parametrize(
        "backend,x,y,expected",
        [
            ("sequential", 5, 10, 15),
            ("threading", 6, 7, 13),
            ("processes", 8, 9, 17),
        ],
    )
    def test_single_task_execution(self, backend: str, x: int, y: int, expected: int) -> None:
        """Different backends execute single tasks correctly."""
        task_with_backend = add.with_options(backend_name=backend)
        result = evaluate(task_with_backend(x=x, y=y))
        assert result == expected

    @pytest.mark.parametrize("backend", ["sequential", "threading", "processes"])
    def test_map_operation(self, backend: str) -> None:
        """Different backends handle map operations correctly."""
        task_with_backend = double.with_options(backend_name=backend)
        nums = [1, 2, 3, 4, 5]
        results = evaluate(task_with_backend.zip(x=nums))
        assert results == [2, 4, 6, 8, 10]

    def test_mixed_backends_in_pipeline(self) -> None:
        """Different backends can be mixed in same pipeline."""
        fetch = fetch_data.with_options(backend_name="sequential")
        process = square.with_options(backend_name="processes")
        sum_task = sum_values.with_options(backend_name="sequential")

        data = fetch()
        processed = process.zip(x=data)
        total = sum_task(values=processed)

        result = evaluate(total)
        assert result == 55  # 1+4+9+16+25

    def test_processes_backend_with_dependencies(self) -> None:
        """Processes backend handles tasks with dependencies."""
        gen = generate_range.with_options(backend_name="processes")
        sum_task = sum_values.with_options(backend_name="processes")

        nums = gen(n=10)
        total = sum_task(values=nums)

        result = evaluate(total)
        assert result == 45  # 0+1+2+...+9

    def test_processes_backend_with_complex_data(self) -> None:
        """Processes backend handles complex data types."""
        task_with_backend = merge_dicts.with_options(backend_name="processes")
        dict1 = {"a": 1, "b": 2}
        dict2 = {"c": 3, "d": 4}

        result = evaluate(task_with_backend(d1=dict1, d2=dict2))
        assert result == {"a": 1, "b": 2, "c": 3, "d": 4}

    def test_threading_backend_uses_multiple_threads_async(self) -> None:
        """Threading backend executes sibling tasks using multiple threads with async evaluation."""
        # Track which threads are used
        thread_ids: set[int] = set()
        lock = threading.Lock()

        @task(backend_name="threading")
        def track_thread(value: int) -> int:
            """Task that tracks which thread it runs in."""
            with lock:
                thread_ids.add(threading.get_ident())
            return value

        # Create sibling tasks (independent tasks that are inputs to a combiner)
        task1 = track_thread(value=1)
        task2 = track_thread(value=2)
        task3 = track_thread(value=3)

        # Combine the results
        combined = add(x=add(x=task1, y=task2), y=task3)

        async def run():
            return await evaluate_async(combined)

        result = asyncio.run(run())

        # Verify result is correct
        assert result == 6  # 1 + 2 + 3

        # Verify that multiple threads were used (proves parallel execution)
        # With threading backend + async evaluation, we should see more than one thread ID
        assert len(thread_ids) > 1, (
            f"Expected multiple threads, but only {len(thread_ids)} thread(s) used"
        )


class TestBackendWithPipelines:
    """Tests for backends used within @pipeline decorated functions."""

    def test_pipeline_with_processes_backend(self) -> None:
        """Pipeline can use processes backend for tasks."""
        square_proc = square.with_options(backend_name="processes")

        @pipeline
        def compute_pipeline(nums: list[int]):
            squared = square_proc.zip(x=nums)
            return sum_values(values=squared)

        result = evaluate(compute_pipeline(nums=[1, 2, 3, 4]))
        assert result == 30  # 1+4+9+16


class TestBackendErrorHandling:
    """Tests for error handling across different backends."""

    @pytest.mark.parametrize("backend", ["threading", "processes"])
    def test_exception_propagation(self, backend: str) -> None:
        """Exceptions are properly propagated across different backends."""
        task_with_backend = failing_task.with_options(backend_name=backend)

        # Should succeed
        result = evaluate(task_with_backend(x=5))
        assert result == 10

        # Should raise ValueError
        with pytest.raises(ValueError, match="negative value not allowed"):
            evaluate(task_with_backend(x=-5))

        # Should raise ZeroDivisionError
        with pytest.raises(ZeroDivisionError, match="cannot divide by zero"):
            evaluate(task_with_backend(x=0))


class TestBackendPickleRequirements:
    """Tests specifically for pickle support in multiprocessing backend."""

    def test_task_with_nested_function(self) -> None:
        """Tasks defined at module level can contain nested helper functions."""
        task_with_backend = nested_computation.with_options(backend_name="processes")
        result = evaluate(task_with_backend(x=5))
        assert result == 20  # (5*2) + (5*2)

    def test_task_with_lambda_function(self) -> None:
        """Lambda functions can be used with sequential/threading but may fail with processes."""
        # Lambda functions don't have __name__ in some Python implementations
        # This tests the hasattr branches in the pickle fix code
        lambda_task = task(lambda x: x * 3, name="triple")  # type: ignore

        # Should work with sequential backend
        result = evaluate(lambda_task.with_options(backend_name="sequential")(x=7))
        assert result == 21

        # Should work with threading backend
        result = evaluate(lambda_task.with_options(backend_name="threading")(x=8))
        assert result == 24

    def test_task_with_callable_object(self) -> None:
        """Callable objects without standard function attributes work with non-process backends."""

        class Multiplier:
            def __init__(self, factor: int):
                self.factor = factor

            def __call__(self, x: int) -> int:
                return x * self.factor

        # Callable objects may not have __module__ or __name__ set properly
        multiplier = Multiplier(5)
        mult_task = task(multiplier, name="multiplier")  # type: ignore

        # Should work with sequential backend (no pickling needed)
        result = evaluate(mult_task.with_options(backend_name="sequential")(x=4))
        assert result == 20

        # Should work with threading backend (pickle not strictly required)
        result = evaluate(mult_task.with_options(backend_name="threading")(x=6))
        assert result == 30
