"""Tests for backend-level timeout enforcement."""

import asyncio
import time

import pytest

from daglite import evaluate
from daglite import evaluate_async
from daglite import task


# Module-level tasks for ProcessBackend (must be picklable)
@task(timeout=0.05, backend_name="multiprocessing")
def slow_task_process(x: int) -> int:
    """Module-level slow task for process backend timeout testing."""
    time.sleep(0.2)  # 4x timeout value for CI stability
    return x * 2


@task(timeout=5.0, backend_name="multiprocessing")
def fast_task_process(x: int) -> int:
    """Module-level fast task for process backend timeout testing."""
    # No sleep - completes immediately, well within timeout
    return x * 2


@task(timeout=0.05, backend_name="multiprocessing")
async def slow_async_task_process(x: int) -> int:
    """Module-level async slow task for process backend timeout testing."""
    await asyncio.sleep(0.2)  # 4x timeout value for CI stability
    return x * 2


class TestThreadBackendTimeout:
    """Tests for ThreadBackend timeout handling."""

    def test_thread_backend_timeout_enforced(self) -> None:
        """ThreadBackend enforces timeout and raises TimeoutError."""

        @task(timeout=0.05, backend_name="threading")
        def slow_task(x: int) -> int:
            time.sleep(0.2)  # 4x timeout value for CI stability
            return x * 2

        with pytest.raises(TimeoutError, match="exceeded timeout"):
            evaluate(slow_task(x=5))

    def test_thread_backend_timeout_success(self) -> None:
        """ThreadBackend allows tasks that complete within timeout."""

        @task(timeout=5.0, backend_name="threading")
        def fast_task(x: int) -> int:
            # No sleep - completes immediately, well within timeout
            return x * 2

        result = evaluate(fast_task(x=5))
        assert result == 10

    def test_thread_backend_async_timeout_enforced(self) -> None:
        """ThreadBackend enforces timeout on async tasks."""

        @task(timeout=0.05, backend_name="threading")
        async def slow_async_task(x: int) -> int:
            await asyncio.sleep(0.2)  # 4x timeout value for CI stability
            return x * 2

        async def run():
            return await evaluate_async(slow_async_task(x=5))

        with pytest.raises(TimeoutError, match="exceeded timeout"):
            asyncio.run(run())

    def test_thread_backend_timeout_without_retries(self) -> None:
        """ThreadBackend timeout errors are not retried."""
        attempts = []

        @task(timeout=0.05, retries=3, backend_name="threading")
        def slow_task(x: int) -> int:
            attempts.append(1)
            time.sleep(0.2)  # 4x timeout value for CI stability
            return x * 2

        with pytest.raises(TimeoutError):
            evaluate(slow_task(x=5))

        # Should only attempt once - timeouts are not retried
        assert len(attempts) == 1


class TestProcessBackendTimeout:
    """Tests for ProcessBackend timeout handling."""

    def test_process_backend_timeout_enforced(self) -> None:
        """ProcessBackend enforces timeout and raises TimeoutError."""
        with pytest.raises(TimeoutError, match="exceeded timeout"):
            evaluate(slow_task_process(x=5))

    def test_process_backend_timeout_success(self) -> None:
        """ProcessBackend allows tasks that complete within timeout."""
        result = evaluate(fast_task_process(x=5))
        assert result == 10

    def test_process_backend_async_timeout_enforced(self) -> None:
        """ProcessBackend enforces timeout on async tasks."""

        async def run():
            return await evaluate_async(slow_async_task_process(x=5))

        with pytest.raises(TimeoutError, match="exceeded timeout"):
            asyncio.run(run())


class TestAsyncTimeout:
    """Tests for timeout functionality with asynchronous tasks."""

    def test_async_task_succeeds_within_timeout(self) -> None:
        """Async task that completes quickly succeeds with timeout."""

        @task(timeout=5.0)
        async def quick_task(x: int) -> int:
            # No sleep - completes immediately, well within timeout
            return x * 2

        async def run():
            return await evaluate_async(quick_task(x=5))

        result = asyncio.run(run())
        assert result == 10

    def test_async_task_fails_on_timeout(self) -> None:
        """Async task that exceeds timeout raises TimeoutError."""

        @task(timeout=0.05, backend_name="threading")
        async def slow_task(x: int) -> int:
            await asyncio.sleep(0.2)  # 4x timeout value for CI stability
            return x * 2

        async def run():
            return await evaluate_async(slow_task(x=5))

        with pytest.raises(TimeoutError, match="exceeded timeout"):
            asyncio.run(run())

    def test_async_timeout_without_retries(self) -> None:
        """Async timeout errors are not retried."""
        attempts = []

        @task(timeout=0.05, retries=3, backend_name="threading")
        async def slow_task(x: int) -> int:
            attempts.append(1)
            await asyncio.sleep(0.2)  # 4x timeout value for CI stability
            return x * 2

        async def run():
            return await evaluate_async(slow_task(x=5))

        with pytest.raises(TimeoutError):
            asyncio.run(run())

        # Should only attempt once - timeouts are not retried
        assert len(attempts) == 1

    def test_async_timeout_with_Inline_backend(self) -> None:
        """Async tasks can use timeout even with InlineBackend."""

        @task(timeout=0.05)  # No backend_name = InlineBackend (default)
        async def slow_task(x: int) -> int:
            await asyncio.sleep(0.2)  # 4x timeout value for CI stability
            return x * 2

        async def run():
            return await evaluate_async(slow_task(x=5))

        with pytest.raises(TimeoutError):
            asyncio.run(run())


class TestInlineBackendTimeout:
    """Tests for InlineBackend timeout handling with sync tasks."""

    def test_Inline_backend_sync_timeout(self) -> None:
        """Inline backend raises TimeoutError for sync tasks that exceed timeout.

        Note that the task will still run to completion on the worker thread, but the timeout will
        be enforced and a TimeoutError will be raised in the main thread after the timeout duration
        has elapsed.
        """

        @task(timeout=0.05)  # Default backend is InlineBackend
        def slow_task(x: int) -> int:
            time.sleep(0.2)  # Exceeds timeout, but can't be interrupted
            return x * 2

        # Timeout is now enforced (task runs in thread pool)
        with pytest.raises(TimeoutError, match="exceeded timeout"):
            evaluate(slow_task(x=5))

    def test_Inline_backend_sync_timeout_success(self) -> None:
        """InlineBackend allows sync tasks that complete within timeout."""

        @task(timeout=5.0)  # Default backend is InlineBackend
        def fast_task(x: int) -> int:
            # No sleep - completes immediately, well within timeout
            return x * 2

        result = evaluate(fast_task(x=5))
        assert result == 10


class TestTimeoutPropagation:
    """Tests for timeout parameter propagation through with_options() and partial()."""

    def test_timeout_with_with_options(self) -> None:
        """Timeout can be set via with_options."""

        @task(backend_name="threading")
        def slow_task(x: int) -> int:
            time.sleep(0.2)  # 4x timeout value for CI stability
            return x * 2

        task_with_timeout = slow_task.with_options(timeout=0.05)

        with pytest.raises(TimeoutError):
            evaluate(task_with_timeout(x=5))

    def test_timeout_with_partial_task(self) -> None:
        """Timeout works with partial tasks."""

        @task(timeout=0.05, backend_name="threading")
        def slow_multiply(x: int, factor: int) -> int:
            time.sleep(0.2)  # 4x timeout value for CI stability
            return x * factor

        partial = slow_multiply.partial(factor=3)

        with pytest.raises(TimeoutError):
            evaluate(partial(x=4))


class TestMapTaskTimeout:
    """Tests for timeout functionality with map operations."""

    def test_product_task_timeout_per_iteration(self) -> None:
        """Each iteration of a product task has independent timeout enforcement."""
        attempts = []

        @task(timeout=0.1, backend_name="threading")
        def process_item(x: int) -> int:
            attempts.append(x)
            # Only slow down even numbers to exceed timeout
            if x % 2 == 0:
                time.sleep(0.3)  # Exceeds timeout
            return x * 2

        # Should fail on first even number (2)
        with pytest.raises(TimeoutError):
            evaluate(process_item.map(x=[1, 2, 3, 4]))

        # Verify that x=1 succeeded before x=2 timed out
        assert 1 in attempts
        assert 2 in attempts

    def test_async_product_task_timeout_per_iteration(self) -> None:
        """Each iteration of an async product task has independent timeout enforcement."""
        attempts = []

        @task(timeout=0.1, backend_name="threading")
        async def process_item(x: int) -> int:
            attempts.append(x)
            # Only slow down even numbers to exceed timeout
            if x % 2 == 0:
                await asyncio.sleep(0.3)  # Exceeds timeout
            return x * 2

        async def run():
            return await evaluate_async(process_item.map(x=[1, 2, 3, 4]))

        # Should fail on first even number (2)
        with pytest.raises(TimeoutError):
            asyncio.run(run())

        # Verify that x=1 succeeded before x=2 timed out
        assert 1 in attempts
        assert 2 in attempts
