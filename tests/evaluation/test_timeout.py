"""Tests for backend-level timeout enforcement."""

import asyncio
import time

import pytest

from daglite import evaluate
from daglite import evaluate_async
from daglite import task


# Module-level tasks for ProcessBackend (must be picklable)
@task(timeout=0.1, backend_name="multiprocessing")
def slow_task_process(x: int) -> int:
    """Module-level slow task for process backend timeout testing."""
    time.sleep(0.5)
    return x * 2


@task(timeout=1.0, backend_name="multiprocessing")
def fast_task_process(x: int) -> int:
    """Module-level fast task for process backend timeout testing."""
    time.sleep(0.01)
    return x * 2


@task(timeout=0.1, backend_name="multiprocessing")
async def slow_async_task_process(x: int) -> int:
    """Module-level async slow task for process backend timeout testing."""
    await asyncio.sleep(0.5)
    return x * 2


class TestThreadBackendTimeout:
    """Tests for ThreadBackend timeout handling."""

    def test_thread_backend_timeout_enforced(self) -> None:
        """ThreadBackend enforces timeout and raises TimeoutError."""

        @task(timeout=0.1, backend_name="threading")
        def slow_task(x: int) -> int:
            time.sleep(0.5)
            return x * 2

        with pytest.raises(TimeoutError, match="exceeded timeout"):
            evaluate(slow_task(x=5))

    def test_thread_backend_timeout_success(self) -> None:
        """ThreadBackend allows tasks that complete within timeout."""

        @task(timeout=1.0, backend_name="threading")
        def fast_task(x: int) -> int:
            time.sleep(0.01)
            return x * 2

        result = evaluate(fast_task(x=5))
        assert result == 10

    def test_thread_backend_async_timeout_enforced(self) -> None:
        """ThreadBackend enforces timeout on async tasks."""

        @task(timeout=0.1, backend_name="threading")
        async def slow_async_task(x: int) -> int:
            await asyncio.sleep(0.5)
            return x * 2

        async def run():
            return await evaluate_async(slow_async_task(x=5))

        with pytest.raises(TimeoutError, match="exceeded timeout"):
            asyncio.run(run())


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

        @task(timeout=1.0)
        async def quick_task(x: int) -> int:
            await asyncio.sleep(0.01)
            return x * 2

        async def run():
            return await evaluate_async(quick_task(x=5))

        result = asyncio.run(run())
        assert result == 10

    def test_async_task_fails_on_timeout(self) -> None:
        """Async task that exceeds timeout raises TimeoutError."""

        @task(timeout=0.1, backend_name="threading")
        async def slow_task(x: int) -> int:
            await asyncio.sleep(0.5)
            return x * 2

        async def run():
            return await evaluate_async(slow_task(x=5))

        with pytest.raises(TimeoutError, match="exceeded timeout"):
            asyncio.run(run())

    def test_async_timeout_without_retries(self) -> None:
        """Async timeout errors are not retried."""
        attempts = []

        @task(timeout=0.1, retries=3, backend_name="threading")
        async def slow_task(x: int) -> int:
            attempts.append(1)
            await asyncio.sleep(0.5)
            return x * 2

        async def run():
            return await evaluate_async(slow_task(x=5))

        with pytest.raises(TimeoutError):
            asyncio.run(run())

        # Should only attempt once - timeouts are not retried
        assert len(attempts) == 1

    def test_async_timeout_with_sequential_backend(self) -> None:
        """Async tasks can use timeout even with SequentialBackend."""

        @task(timeout=0.1)  # No backend_name = SequentialBackend (default)
        async def slow_task(x: int) -> int:
            await asyncio.sleep(0.5)
            return x * 2

        async def run():
            return await evaluate_async(slow_task(x=5))

        with pytest.raises(TimeoutError):
            asyncio.run(run())


class TestSequentialBackendTimeout:
    """Tests for SequentialBackend timeout handling with sync tasks."""

    def test_sequential_backend_sync_timeout_enforced(self) -> None:
        """SequentialBackend enforces timeout on sync tasks and raises TimeoutError."""

        @task(timeout=0.1)  # Default backend is SequentialBackend
        def slow_task(x: int) -> int:
            time.sleep(0.5)
            return x * 2

        with pytest.raises(TimeoutError, match="exceeded timeout"):
            evaluate(slow_task(x=5))

    def test_sequential_backend_sync_timeout_success(self) -> None:
        """SequentialBackend allows sync tasks that complete within timeout."""

        @task(timeout=1.0)  # Default backend is SequentialBackend
        def fast_task(x: int) -> int:
            time.sleep(0.01)
            return x * 2

        result = evaluate(fast_task(x=5))
        assert result == 10
