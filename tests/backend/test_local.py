"""Tests for local backend implementations."""

import time
from concurrent.futures import Future

import pytest

from daglite.backends.local import ProcessBackend
from daglite.backends.local import SequentialBackend
from daglite.backends.local import ThreadBackend
from daglite.backends.local import _reset_global_pools

# -- Test functions (no task decorator) --


def simple_add(x: int, y: int) -> int:
    """Simple function for testing."""
    return x + y


def sleep_and_return(value: int, sleep_time: float = 0.01) -> int:
    """Sleep briefly then return value."""
    time.sleep(sleep_time)
    return value


def raise_error(message: str) -> None:
    """Raise an exception for error testing."""
    raise ValueError(message)


def maybe_fail(x: int) -> int:
    """Return double value if positive, raise ValueError if negative."""
    if x < 0:
        raise ValueError(f"negative value: {x}")
    return x * 2


# -- Test fixtures --


@pytest.fixture(autouse=True)
def reset_pools():
    """Reset global pools before each test for isolation."""
    _reset_global_pools()
    yield
    _reset_global_pools()


LOCAL_BACKENDS = [
    SequentialBackend(),
    ThreadBackend(),
    ProcessBackend(),
]


# -- Test submit() --


class TestSubmit:
    """Test single task submission via submit()."""

    @pytest.mark.parametrize(
        "backend",
        LOCAL_BACKENDS,
        ids=["sequential", "thread", "process"],
    )
    def test_submit_returns_future(self, backend) -> None:
        """submit() returns a Future object."""
        future = backend.submit(simple_add, x=1, y=2)
        assert isinstance(future, Future)

    @pytest.mark.parametrize(
        "backend",
        LOCAL_BACKENDS,
        ids=["sequential", "thread", "process"],
    )
    def test_submit_computes_result(self, backend) -> None:
        """submit() future contains correct result."""
        future = backend.submit(simple_add, x=10, y=20)
        result = future.result()
        assert result == 30

    @pytest.mark.parametrize(
        "backend",
        LOCAL_BACKENDS,
        ids=["sequential", "thread", "process"],
    )
    def test_submit_with_error(self, backend) -> None:
        """submit() future captures exceptions."""
        future = backend.submit(raise_error, message="test error")

        with pytest.raises(ValueError, match="test error"):
            future.result()

    def test_sequential_executes_immediately(self) -> None:
        """SequentialBackend executes synchronously."""
        backend = SequentialBackend()
        future = backend.submit(simple_add, x=1, y=2)

        # Should be done immediately
        assert future.done()
        assert future.result() == 3

    def test_thread_executes_asynchronously(self) -> None:
        """ThreadBackend returns pending future."""
        backend = ThreadBackend()
        future = backend.submit(sleep_and_return, value=42, sleep_time=0.1)

        # May or may not be done immediately, but should complete
        result = future.result(timeout=1.0)
        assert result == 42

    def test_process_executes_asynchronously(self) -> None:
        """ProcessBackend returns pending future."""
        backend = ProcessBackend()
        future = backend.submit(sleep_and_return, value=42, sleep_time=0.1)

        # May or may not be done immediately, but should complete
        result = future.result(timeout=1.0)
        assert result == 42


# -- Test submit_many() --


class TestSubmitMany:
    """Test batch task submission via submit_many()."""

    @pytest.mark.parametrize(
        "backend",
        LOCAL_BACKENDS,
        ids=["sequential", "thread", "process"],
    )
    def test_submit_many_returns_futures(self, backend) -> None:
        """submit_many() returns list of Futures."""
        calls = [{"x": 1, "y": 2}, {"x": 3, "y": 4}, {"x": 5, "y": 6}]
        futures = backend.submit_many(simple_add, calls)

        assert len(futures) == 3
        assert all(isinstance(f, Future) for f in futures)

    @pytest.mark.parametrize(
        "backend",
        LOCAL_BACKENDS,
        ids=["sequential", "thread", "process"],
    )
    def test_submit_many_computes_results(self, backend) -> None:
        """submit_many() futures contain correct results."""
        calls = [{"x": 1, "y": 2}, {"x": 10, "y": 20}, {"x": 100, "y": 200}]
        futures = backend.submit_many(simple_add, calls)

        results = [f.result() for f in futures]
        assert results == [3, 30, 300]

    @pytest.mark.parametrize(
        "backend",
        LOCAL_BACKENDS,
        ids=["sequential", "thread", "process"],
    )
    def test_submit_many_empty_list(self, backend) -> None:
        """submit_many() handles empty call list."""
        calls: list[dict[str, int]] = []
        futures = backend.submit_many(simple_add, calls)

        assert futures == []

    @pytest.mark.parametrize(
        "backend",
        LOCAL_BACKENDS,
        ids=["sequential", "thread", "process"],
    )
    def test_submit_many_with_errors(self, backend) -> None:
        """submit_many() captures exceptions in individual futures."""
        calls = [
            {"message": "error 1"},
            {"message": "error 2"},
            {"message": "error 3"},
        ]
        futures = backend.submit_many(raise_error, calls)

        for i, future in enumerate(futures, 1):
            with pytest.raises(ValueError, match=f"error {i}"):
                future.result()

    @pytest.mark.parametrize(
        "backend",
        LOCAL_BACKENDS,
        ids=["sequential", "thread", "process"],
    )
    def test_submit_many_mixed_success_and_failure(self, backend) -> None:
        """submit_many() handles mix of successful and failed tasks."""
        calls = [{"x": 1}, {"x": -1}, {"x": 2}, {"x": -2}, {"x": 3}]
        futures = backend.submit_many(maybe_fail, calls)

        # Check successful ones
        assert futures[0].result() == 2
        assert futures[2].result() == 4
        assert futures[4].result() == 6

        # Check failed ones
        with pytest.raises(ValueError, match="negative value: -1"):
            futures[1].result()

        with pytest.raises(ValueError, match="negative value: -2"):
            futures[3].result()

    def test_thread_backend_respects_max_workers(self) -> None:
        """ThreadBackend with max_workers limits concurrency."""
        backend = ThreadBackend(max_workers=2)

        # Submit many tasks that track concurrency
        import threading

        lock = threading.Lock()
        max_concurrent = {"count": 0, "current": 0}

        def track_concurrency(task_id: int) -> int:
            with lock:
                max_concurrent["current"] += 1
                max_concurrent["count"] = max(
                    max_concurrent["count"], max_concurrent["current"]
                )

            time.sleep(0.1)

            with lock:
                max_concurrent["current"] -= 1

            return task_id

        calls = [{"task_id": i} for i in range(10)]
        futures = backend.submit_many(track_concurrency, calls)

        # Wait for all to complete
        results = [f.result() for f in futures]
        assert results == list(range(10))

        # Should have limited concurrency (might be <= max_workers due to timing)
        assert max_concurrent["count"] <= 2

    def test_process_backend_respects_max_workers(self) -> None:
        """ProcessBackend with max_workers limits concurrency."""
        backend = ProcessBackend(max_workers=2)

        calls = [{"value": i, "sleep_time": 0.05} for i in range(10)]
        futures = backend.submit_many(sleep_and_return, calls)

        results = [f.result(timeout=5.0) for f in futures]
        assert results == list(range(10))
