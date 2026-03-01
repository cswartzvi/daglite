"""
Integration tests for the eager execution model.

Exercises the full stack — `@task`, `@workflow`, `session`, `parallel_map`,
caching, retries, and async equivalents — using the public API only.
"""

from __future__ import annotations

import asyncio
import tempfile

import pytest

from daglite import async_map
from daglite import async_session
from daglite import parallel_map
from daglite import session
from daglite import task
from daglite import workflow
from daglite.cache.store import CacheStore

# region Helpers


@task
def add(x: int, y: int) -> int:
    return x + y


@task
def double(x: int) -> int:
    return x * 2


@task
async def async_add(x: int, y: int) -> int:
    await asyncio.sleep(0)
    return x + y


@task
async def async_double(x: int) -> int:
    await asyncio.sleep(0)
    return x * 2


@pytest.fixture
def temp_cache_dir():
    """Temporary directory for on-disk caching."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def cache_store(temp_cache_dir: str) -> CacheStore:
    """Pre-built ``CacheStore`` backed by a temp directory."""
    return CacheStore(temp_cache_dir)


# region Workflow integration


class TestWorkflowRun:
    """``@workflow`` + ``.run()`` exercise the full eager path."""

    def test_single_task_workflow(self) -> None:
        @workflow
        def wf(x: int, y: int) -> int:
            return add(x=x, y=y)

        assert wf.run(2, 3) == 5

    def test_chained_tasks_workflow(self) -> None:
        @workflow
        def wf(x: int, y: int) -> int:
            s = add(x=x, y=y)
            return double(x=s)

        assert wf.run(2, 3) == 10

    def test_multi_step_workflow(self) -> None:
        @workflow
        def wf(a: int, b: int, c: int) -> int:
            s = add(x=a, y=b)
            d = double(x=s)
            return add(x=d, y=c)

        assert wf.run(1, 2, 10) == 16

    def test_workflow_returns_non_task_value(self) -> None:
        """Workflows may contain plain Python alongside @task calls."""

        @workflow
        def wf(x: int) -> list[int]:
            a = add(x=x, y=1)
            b = double(x=a)
            return [a, b]

        assert wf.run(4) == [5, 10]

    def test_workflow_with_custom_name(self) -> None:
        @workflow(name="custom_name", description="A test workflow")
        def wf(x: int) -> int:
            return double(x=x)

        assert wf.name == "custom_name"
        assert wf.description == "A test workflow"
        assert wf.run(7) == 14

    def test_workflow_call_bypasses_session(self) -> None:
        """Direct ``wf(...)`` works without a session — bare eager execution."""

        @workflow
        def wf(x: int) -> int:
            return add(x=x, y=1)

        assert wf(10) == 11


class TestWorkflowRunAsync:
    """``run_async`` wraps async tasks in an ``async_session``."""

    def test_single_async_task(self) -> None:
        @workflow
        async def wf(x: int, y: int) -> int:
            return await async_add(x=x, y=y)

        assert asyncio.run(wf.run_async(3, 4)) == 7

    def test_chained_async_tasks(self) -> None:
        @workflow
        async def wf(x: int, y: int) -> int:
            s = await async_add(x=x, y=y)
            return await async_double(x=s)

        assert asyncio.run(wf.run_async(2, 3)) == 10

    def test_sync_workflow_via_run_async(self) -> None:
        """``run_async`` also accepts synchronous workflow functions."""

        @workflow
        def wf(x: int) -> int:
            return add(x=x, y=1)

        assert asyncio.run(wf.run_async(9)) == 10


# region Cache integration


class TestCacheWithSession:
    """End-to-end caching through ``session(cache=...)``."""

    def test_cache_hit_avoids_recomputation(self, cache_store: CacheStore) -> None:
        call_count = 0

        @task(cache=True)
        def expensive(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        with session(cache=cache_store):
            assert expensive(x=5) == 10
            assert expensive(x=5) == 10

        assert call_count == 1

    def test_different_args_produce_miss(self, cache_store: CacheStore) -> None:
        call_count = 0

        @task(cache=True)
        def inc(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x + 1

        with session(cache=cache_store):
            assert inc(x=1) == 2
            assert inc(x=2) == 3

        assert call_count == 2

    def test_cache_disabled_means_no_caching(self) -> None:
        call_count = 0

        @task(cache=True)
        def inc(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x + 1

        with session(cache=False):
            inc(x=1)
            inc(x=1)

        assert call_count == 2

    def test_cache_via_string_path(self, temp_cache_dir: str) -> None:
        call_count = 0

        @task(cache=True)
        def inc(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x + 1

        with session(cache=temp_cache_dir):
            assert inc(x=10) == 11
            assert inc(x=10) == 11

        assert call_count == 1

    def test_different_functions_dont_share_cache(self, cache_store: CacheStore) -> None:
        count_a = 0
        count_b = 0

        @task(cache=True)
        def fn_a(x: int) -> int:
            nonlocal count_a
            count_a += 1
            return x * 2

        @task(cache=True)
        def fn_b(x: int) -> int:
            nonlocal count_b
            count_b += 1
            return x * 2

        with session(cache=cache_store):
            assert fn_a(x=5) == 10
            assert fn_b(x=5) == 10
            # Both should have executed (no cache sharing)
            assert count_a == 1
            assert count_b == 1

            # Second round — each hits its own cache entry
            assert fn_a(x=5) == 10
            assert fn_b(x=5) == 10
            assert count_a == 1
            assert count_b == 1

    def test_none_result_is_cached(self, cache_store: CacheStore) -> None:
        call_count = 0

        @task(cache=True)
        def returns_none(x: int) -> None:
            nonlocal call_count
            call_count += 1

        with session(cache=cache_store):
            result1 = returns_none(x=1)
            result2 = returns_none(x=1)

        assert result1 is None
        assert result2 is None
        assert call_count == 1

    def test_async_task_caching(self, cache_store: CacheStore) -> None:
        call_count = 0

        @task(cache=True)
        async def async_inc(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x + 1

        async def _run() -> tuple[int, int]:
            async with async_session(cache=cache_store):
                r1 = await async_inc(x=5)
                r2 = await async_inc(x=5)
                return r1, r2

        r1, r2 = asyncio.run(_run())
        assert r1 == 6
        assert r2 == 6
        assert call_count == 1

    def test_cache_persists_across_sessions(self, cache_store: CacheStore) -> None:
        """Cache on disk survives session teardown and is reused in a new session."""
        call_count = 0

        @task(cache=True)
        def inc(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x + 1

        with session(cache=cache_store):
            assert inc(x=7) == 8

        # New session, same cache store
        with session(cache=cache_store):
            assert inc(x=7) == 8

        assert call_count == 1

    def test_task_without_cache_flag_ignores_store(self, cache_store: CacheStore) -> None:
        call_count = 0

        @task
        def plain(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x

        with session(cache=cache_store):
            plain(x=1)
            plain(x=1)

        assert call_count == 2


# region Retry integration


class TestRetriesWithSession:
    """Retry semantics exercised through sessions."""

    def test_succeeds_without_needing_retries(self) -> None:
        @task(retries=3)
        def ok(x: int) -> int:
            return x + 1

        with session():
            assert ok(x=9) == 10

    def test_retries_on_failure_then_succeeds(self) -> None:
        attempts: list[int] = []

        @task(retries=2)
        def flaky(x: int) -> int:
            attempts.append(1)
            if len(attempts) < 3:
                raise ValueError("not yet")
            return x

        with session():
            assert flaky(x=42) == 42

        assert len(attempts) == 3

    def test_exhausted_retries_raise(self) -> None:
        attempts: list[int] = []

        @task(retries=2)
        def always_fails(x: int) -> int:
            attempts.append(1)
            raise RuntimeError(f"attempt {len(attempts)}")

        with session():
            with pytest.raises(RuntimeError, match="attempt 3"):
                always_fails(x=1)

        assert len(attempts) == 3

    def test_zero_retries_fails_immediately(self) -> None:
        attempts: list[int] = []

        @task(retries=0)
        def boom(x: int) -> int:
            attempts.append(1)
            raise RuntimeError("boom")

        with session():
            with pytest.raises(RuntimeError, match="boom"):
                boom(x=1)

        assert len(attempts) == 1

    def test_async_retries(self) -> None:
        attempts: list[int] = []

        @task(retries=2)
        async def flaky(x: int) -> int:
            attempts.append(1)
            if len(attempts) < 3:
                raise ValueError("not yet")
            return x * 10

        async def _run() -> int:
            async with async_session():
                return await flaky(x=5)

        assert asyncio.run(_run()) == 50
        assert len(attempts) == 3


# region Session + parallel_map integration


class TestParallelMapInSession:
    """``parallel_map`` inside a ``session`` exercises the full stack."""

    def test_inline_backend(self) -> None:
        with session(backend="inline"):
            results = parallel_map(double, [1, 2, 3])

        assert results == [2, 4, 6]

    def test_thread_backend(self) -> None:
        with session(backend="thread"):
            results = parallel_map(double, [10, 20, 30])

        assert results == [20, 40, 60]

    def test_override_backend(self) -> None:
        """Explicit ``backend`` on ``parallel_map`` overrides the session."""
        with session(backend="inline"):
            results = parallel_map(double, [5, 6], backend="thread")

        assert results == [10, 12]

    def test_parallel_map_with_multiple_iterables(self) -> None:
        with session(backend="inline"):
            results = parallel_map(add, [1, 2, 3], [10, 20, 30])

        assert results == [11, 22, 33]


class TestAsyncMapInSession:
    """``async_map`` inside an ``async_session``."""

    def test_async_map_inline(self) -> None:
        async def _run() -> list[int]:
            async with async_session(backend="inline"):
                return await async_map(async_double, [1, 2, 3])

        assert asyncio.run(_run()) == [2, 4, 6]


# region Error propagation


class TestErrorPropagation:
    """Errors raised in eager tasks propagate cleanly through sessions and workflows."""

    def test_error_in_session_propagates(self) -> None:
        @task
        def explode(x: int) -> int:
            raise ValueError(f"bad value {x}")

        with session():
            with pytest.raises(ValueError, match="bad value 99"):
                explode(x=99)

    def test_error_in_workflow_propagates(self) -> None:
        @task
        def explode(x: int) -> int:
            raise ValueError(f"bad value {x}")

        @workflow
        def wf(x: int) -> int:
            return explode(x=x)

        with pytest.raises(ValueError, match="bad value 42"):
            wf.run(42)

    def test_async_error_propagates(self) -> None:
        @task
        async def async_explode(x: int) -> int:
            raise ValueError(f"async bad {x}")

        async def _run() -> None:
            async with async_session():
                await async_explode(x=7)

        with pytest.raises(ValueError, match="async bad 7"):
            asyncio.run(_run())


# region Mixed composition


class TestMixedComposition:
    """Complex real-world patterns: multiple tasks, branching, rejoining."""

    def test_diamond_pattern(self) -> None:
        """Two branches from one root, merged at the end."""

        @task
        def root(x: int) -> int:
            return x

        @task
        def branch_a(x: int) -> int:
            return x + 10

        @task
        def branch_b(x: int) -> int:
            return x * 10

        @task
        def merge(a: int, b: int) -> int:
            return a + b

        @workflow
        def diamond(x: int) -> int:
            r = root(x=x)
            a = branch_a(x=r)
            b = branch_b(x=r)
            return merge(a=a, b=b)

        assert diamond.run(5) == 65  # (5+10) + (5*10) = 15 + 50

    def test_parallel_map_inside_workflow(self) -> None:
        @workflow
        def wf(values: list[int]) -> list[int]:
            return parallel_map(double, values, backend="inline")

        assert wf.run([1, 2, 3]) == [2, 4, 6]

    def test_workflow_with_cache_and_retries(self, cache_store: CacheStore) -> None:
        """Cache and retries work together through a session."""
        attempts: list[int] = []

        @task(cache=True, retries=2)
        def flaky_cached(x: int) -> int:
            attempts.append(1)
            if len(attempts) < 3:
                raise ValueError("not yet")
            return x * 100

        with session(cache=cache_store):
            # First call: retries twice then succeeds, result is cached
            r1 = flaky_cached(x=5)
            assert r1 == 500
            assert len(attempts) == 3

            # Second call: cache hit, no retry needed
            r2 = flaky_cached(x=5)
            assert r2 == 500
            assert len(attempts) == 3  # No new attempts

    def test_nested_sessions(self) -> None:
        """Inner session overrides outer session backend."""

        @task
        def identity(x: int) -> int:
            return x

        with session(backend="inline"):
            r1 = identity(x=1)
            with session(backend="thread"):
                r2 = identity(x=2)
            r3 = identity(x=3)

        assert r1 == 1
        assert r2 == 2
        assert r3 == 3
