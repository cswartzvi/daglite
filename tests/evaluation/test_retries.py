"""Integration tests for task retries and timeout functionality."""

import asyncio

import pytest

from daglite import evaluate
from daglite import evaluate_async
from daglite import task


class TestSyncRetries:
    """Tests for retry functionality with synchronous tasks."""

    def test_task_succeeds_without_retries(self) -> None:
        """Task that succeeds on first try doesn't need retries."""

        @task(retries=3)
        def add(x: int, y: int) -> int:
            return x + y

        result = evaluate(add(x=10, y=20))
        assert result == 30

    def test_task_retries_on_failure(self) -> None:
        """Task retries specified number of times on failure."""
        attempts = []

        @task(retries=2)
        def flaky_task(x: int) -> int:
            attempts.append(1)
            if len(attempts) < 3:
                raise ValueError(f"Attempt {len(attempts)} failed")
            return x * 2

        result = evaluate(flaky_task(x=5))
        assert result == 10
        assert len(attempts) == 3  # Initial attempt + 2 retries

    def test_task_fails_after_retries_exhausted(self) -> None:
        """Task raises error after all retries are exhausted."""
        attempts = []

        @task(retries=2)
        def always_fails(x: int) -> int:
            attempts.append(1)
            raise ValueError(f"Attempt {len(attempts)} failed")

        with pytest.raises(ValueError, match="Attempt 3 failed"):
            evaluate(always_fails(x=5))

        assert len(attempts) == 3  # Initial attempt + 2 retries

    def test_task_with_zero_retries(self) -> None:
        """Task with retries=0 fails on first error."""
        attempts = []

        @task(retries=0)
        def fails_once(x: int) -> int:
            attempts.append(1)
            raise ValueError("Failed")

        with pytest.raises(ValueError, match="Failed"):
            evaluate(fails_once(x=5))

        assert len(attempts) == 1  # Only initial attempt

    def test_retries_with_with_options(self) -> None:
        """Retries can be set via with_options."""
        attempts = []

        @task
        def flaky_task(x: int) -> int:
            attempts.append(1)
            if len(attempts) < 2:
                raise ValueError("First attempt failed")
            return x * 2

        task_with_retries = flaky_task.with_options(retries=1)
        result = evaluate(task_with_retries(x=5))
        assert result == 10
        assert len(attempts) == 2

    def test_retries_with_partial_task(self) -> None:
        """Retries work with partial tasks."""
        attempts = []

        @task(retries=1)
        def flaky_multiply(x: int, factor: int) -> int:
            attempts.append(1)
            if len(attempts) < 2:
                raise ValueError("First attempt failed")
            return x * factor

        partial = flaky_multiply.partial(factor=3)
        result = evaluate(partial(x=4))
        assert result == 12
        assert len(attempts) == 2


class TestAsyncRetries:
    """Tests for retry functionality with asynchronous tasks."""

    def test_async_task_succeeds_without_retries(self) -> None:
        """Async task that succeeds on first try doesn't need retries."""

        @task(retries=3)
        async def add(x: int, y: int) -> int:
            await asyncio.sleep(0.001)
            return x + y

        async def run():
            return await evaluate_async(add(x=10, y=20))

        result = asyncio.run(run())
        assert result == 30

    def test_async_task_retries_on_failure(self) -> None:
        """Async task retries specified number of times on failure."""
        attempts = []

        @task(retries=2)
        async def flaky_task(x: int) -> int:
            attempts.append(1)
            await asyncio.sleep(0.001)
            if len(attempts) < 3:
                raise ValueError(f"Attempt {len(attempts)} failed")
            return x * 2

        async def run():
            return await evaluate_async(flaky_task(x=5))

        result = asyncio.run(run())
        assert result == 10
        assert len(attempts) == 3  # Initial attempt + 2 retries

    def test_async_task_fails_after_retries_exhausted(self) -> None:
        """Async task raises error after all retries are exhausted."""
        attempts = []

        @task(retries=2)
        async def always_fails(x: int) -> int:
            attempts.append(1)
            await asyncio.sleep(0.001)
            raise ValueError(f"Attempt {len(attempts)} failed")

        async def run():
            return await evaluate_async(always_fails(x=5))

        with pytest.raises(ValueError, match="Attempt 3 failed"):
            asyncio.run(run())

        assert len(attempts) == 3  # Initial attempt + 2 retries


class TestMapTaskRetries:
    """Tests for retry functionality with map operations."""

    def test_product_task_retries_each_iteration(self) -> None:
        """Each iteration of a product task can retry independently."""
        attempt_counts = {}

        @task(retries=1)
        def flaky_square(x: int) -> int:
            if x not in attempt_counts:
                attempt_counts[x] = 0
            attempt_counts[x] += 1

            # Fail first attempt for even numbers
            if x % 2 == 0 and attempt_counts[x] == 1:
                raise ValueError(f"First attempt failed for {x}")

            return x * x

        result = evaluate(flaky_square.product(x=[1, 2, 3, 4]))
        assert result == [1, 4, 9, 16]

        # Odd numbers: 1 attempt each, Even numbers: 2 attempts each
        assert attempt_counts == {1: 1, 2: 2, 3: 1, 4: 2}

    def test_async_product_task_retries_each_iteration(self) -> None:
        """Each iteration of an async product task can retry independently."""
        attempt_counts = {}

        @task(retries=1)
        async def flaky_square(x: int) -> int:
            if x not in attempt_counts:
                attempt_counts[x] = 0
            attempt_counts[x] += 1

            await asyncio.sleep(0.001)

            # Fail first attempt for even numbers
            if x % 2 == 0 and attempt_counts[x] == 1:
                raise ValueError(f"First attempt failed for {x}")

            return x * x

        async def run():
            return await evaluate_async(flaky_square.product(x=[1, 2, 3, 4]))

        result = asyncio.run(run())
        assert result == [1, 4, 9, 16]

        # Odd numbers: 1 attempt each, Even numbers: 2 attempts each
        assert attempt_counts == {1: 1, 2: 2, 3: 1, 4: 2}
