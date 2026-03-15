"""Full-stack integration tests of the execution model; exercises the public API end-to-end."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from daglite import async_session
from daglite import gather_tasks
from daglite import map_tasks
from daglite import session
from daglite import task
from daglite import workflow
from daglite.cache.store import CacheStore
from daglite.exceptions import TaskError

from .examples.tasks import add
from .examples.tasks import async_add
from .examples.tasks import async_double
from .examples.tasks import double


@pytest.fixture()
def temp_cache_dir():
    """Temporary directory for on-disk caching."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture()
def cache_store(temp_cache_dir: str) -> CacheStore:
    """Pre-built ``CacheStore`` backed by a temp directory."""
    return CacheStore(temp_cache_dir)


# region Workflow integration


class TestWorkflowCall:
    """``@workflow`` + direct ``__call__`` exercise the full eager path."""

    def test_single_task_workflow(self) -> None:
        @workflow
        def wf(x: int, y: int) -> int:
            return add(x=x, y=y)

        assert wf(2, 3) == 5

    def test_chained_tasks_workflow(self) -> None:
        @workflow
        def wf(x: int, y: int) -> int:
            s = add(x=x, y=y)
            return double(x=s)

        assert wf(2, 3) == 10

    def test_multi_step_workflow(self) -> None:
        @workflow
        def wf(a: int, b: int, c: int) -> int:
            s = add(x=a, y=b)
            d = double(x=s)
            return add(x=d, y=c)

        assert wf(1, 2, 10) == 16

    def test_workflow_returns_non_task_value(self) -> None:
        """Workflows may contain plain Python alongside @task calls."""

        @workflow
        def wf(x: int) -> list[int]:
            a = add(x=x, y=1)
            b = double(x=a)
            return [a, b]

        assert wf(4) == [5, 10]

    def test_workflow_with_custom_name(self) -> None:
        @workflow(name="custom_name", description="A test workflow")
        def wf(x: int) -> int:
            return double(x=x)

        assert wf.name == "custom_name"
        assert wf.description == "A test workflow"
        assert wf(7) == 14


class TestAsyncWorkflowCall:
    """Async ``@workflow`` + ``asyncio.run`` exercise the async eager path."""

    def test_single_async_task(self) -> None:
        @workflow
        async def wf(x: int, y: int) -> int:
            return await async_add(x=x, y=y)

        assert asyncio.run(wf(3, 4)) == 7

    def test_chained_async_tasks(self) -> None:
        @workflow
        async def wf(x: int, y: int) -> int:
            s = await async_add(x=x, y=y)
            return await async_double(x=s)

        assert asyncio.run(wf(2, 3)) == 10


# region Cache integration


class TestCacheWithSession:
    """End-to-end caching through ``session(cache_store=...)``."""

    def test_cache_hit_avoids_recomputation(self, cache_store: CacheStore) -> None:
        call_count = 0

        @task(cache=True)
        def expensive(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        with session(cache_store=cache_store):
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

        with session(cache_store=cache_store):
            assert inc(x=1) == 2
            assert inc(x=2) == 3

        assert call_count == 2

    def test_cache_disabled_means_no_caching(self) -> None:
        """A task without ``cache=True`` always re-executes."""
        call_count = 0

        @task
        def inc(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x + 1

        with session():
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

        with session(cache_store=temp_cache_dir):
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

        with session(cache_store=cache_store):
            assert fn_a(x=5) == 10
            assert fn_b(x=5) == 10
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

        with session(cache_store=cache_store):
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
            async with async_session(cache_store=cache_store):
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

        with session(cache_store=cache_store):
            assert inc(x=7) == 8

        with session(cache_store=cache_store):
            assert inc(x=7) == 8

        assert call_count == 1

    def test_task_without_cache_flag_ignores_store(self, cache_store: CacheStore) -> None:
        call_count = 0

        @task
        def plain(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x

        with session(cache_store=cache_store):
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
            with pytest.raises(TaskError, match="attempt 3"):
                always_fails(x=1)

        assert len(attempts) == 3

    def test_zero_retries_fails_immediately(self) -> None:
        attempts: list[int] = []

        @task(retries=0)
        def boom(x: int) -> int:
            attempts.append(1)
            raise RuntimeError("boom")

        with session():
            with pytest.raises(TaskError, match="boom"):
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


# region Map integration


class TestMapInSession:
    """``map_tasks`` / ``gather_tasks`` inside sessions."""

    def test_inline_backend(self) -> None:
        with session(backend="inline"):
            results = map_tasks(double, [1, 2, 3])
        assert results == [2, 4, 6]

    def test_thread_backend(self) -> None:
        with session(backend="thread"):
            results = map_tasks(double, [10, 20, 30])
        assert results == [20, 40, 60]

    def test_thread_backend_many_items(self) -> None:
        """Thread fan-out handles many items without context re-entry errors."""

        @task
        def slow_double(x: int) -> int:
            import time

            time.sleep(0.01)
            return x * 2

        values = list(range(20))
        with session(backend="thread"):
            results = map_tasks(slow_double, values)

        assert results == [x * 2 for x in values]

    def test_session_backend_used_by_default(self) -> None:
        """When backend is omitted, ``map_tasks`` uses the session default backend."""

        @task
        def backend_name(x: int) -> str:
            from daglite._resolvers import resolve_backend

            return resolve_backend()

        with session(backend="thread"):
            results = map_tasks(backend_name, [1, 2, 3])

        assert results == ["thread", "thread", "thread"]

    def test_override_backend(self) -> None:
        """Explicit ``backend`` on ``map_tasks`` overrides the session."""
        with session(backend="inline"):
            results = map_tasks(double, [5, 6], backend="thread")
        assert results == [10, 12]

    def test_multiple_iterables(self) -> None:
        with session(backend="inline"):
            results = map_tasks(add, [1, 2, 3], [10, 20, 30])
        assert results == [11, 22, 33]

    def test_async_gather_inline(self) -> None:
        async def _run() -> list[int]:
            async with async_session(backend="inline"):
                return await gather_tasks(async_double, [1, 2, 3])

        assert asyncio.run(_run()) == [2, 4, 6]

    def test_process_backend_preserves_map_index(self) -> None:
        """Process-backed mapped calls preserve per-item ``map_index`` context."""

        @task(dataset="mapped_{map_index}.pkl")
        def emit(x: int) -> int:
            return x

        with tempfile.TemporaryDirectory() as tmpdir:
            with session(backend="process", dataset_store=tmpdir):
                results = map_tasks(emit, [1, 2, 3], backend="process")

            files = sorted(Path(tmpdir).glob("mapped_*.pkl"))
            assert [path.name for path in files] == [
                "mapped_0.pkl",
                "mapped_1.pkl",
                "mapped_2.pkl",
            ]
            assert results == [1, 2, 3]


# region Error propagation


class TestErrorPropagation:
    """Errors raised in eager tasks propagate cleanly through sessions and workflows."""

    def test_error_in_session_propagates(self) -> None:
        @task
        def explode(x: int) -> int:
            raise ValueError(f"bad value {x}")

        with session():
            with pytest.raises(TaskError, match="bad value 99"):
                explode(x=99)

    def test_error_in_workflow_propagates(self) -> None:
        @task
        def explode(x: int) -> int:
            raise ValueError(f"bad value {x}")

        @workflow
        def wf(x: int) -> int:
            return explode(x=x)

        with pytest.raises(TaskError, match="bad value 42"):
            wf(42)

    def test_async_error_propagates(self) -> None:
        @task
        async def async_explode(x: int) -> int:
            raise ValueError(f"async bad {x}")

        async def _run() -> None:
            async with async_session():
                await async_explode(x=7)

        with pytest.raises(TaskError, match="async bad 7"):
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

        assert diamond(5) == 65  # (5+10) + (5*10) = 15 + 50

    def test_map_inside_workflow(self) -> None:
        @workflow
        def wf(values: list[int]) -> list[int]:
            return map_tasks(double, values, backend="inline")

        assert wf([1, 2, 3]) == [2, 4, 6]

    def test_cache_and_retries_combined(self, cache_store: CacheStore) -> None:
        """Cache and retries work together through a session."""
        attempts: list[int] = []

        @task(cache=True, retries=2)
        def flaky_cached(x: int) -> int:
            attempts.append(1)
            if len(attempts) < 3:
                raise ValueError("not yet")
            return x * 100

        with session(cache_store=cache_store):
            r1 = flaky_cached(x=5)
            assert r1 == 500
            assert len(attempts) == 3

            # Cache hit — no new attempts
            r2 = flaky_cached(x=5)
            assert r2 == 500
            assert len(attempts) == 3

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


# region Logging plugin


class TestLoggingPlugin:
    """Exercise ``LifecycleLoggingPlugin`` hook paths through real task execution."""

    def test_task_with_logging_plugin(self) -> None:
        """All lifecycle hooks fire for a normal task execution."""
        from daglite.logging.plugin import LifecycleLoggingPlugin

        plugin = LifecycleLoggingPlugin()

        @task
        def inc(x: int) -> int:
            return x + 1

        with session(plugins=[plugin]):
            assert inc(x=5) == 6

    def test_retry_with_logging_plugin(self) -> None:
        """Retry hooks fire for a flaky task."""
        from daglite.logging.plugin import LifecycleLoggingPlugin

        plugin = LifecycleLoggingPlugin()
        attempts: list[int] = []

        @task(retries=1)
        def flaky(x: int) -> int:
            attempts.append(1)
            if len(attempts) < 2:
                raise ValueError("not yet")
            return x

        with session(plugins=[plugin]):
            assert flaky(x=10) == 10

    def test_cache_hit_with_logging_plugin(self, temp_cache_dir: str) -> None:
        """Cache-hit hook fires on the second call."""
        from daglite.logging.plugin import LifecycleLoggingPlugin

        plugin = LifecycleLoggingPlugin()

        @task(cache=True)
        def square(x: int) -> int:
            return x * x

        with session(plugins=[plugin], cache_store=temp_cache_dir):
            square(x=3)
            square(x=3)

    def test_dataset_save_with_logging_plugin(self, tmp_path) -> None:
        """Dataset save hooks fire when saving via a task."""
        from daglite.datasets.store import DatasetStore
        from daglite.logging.plugin import LifecycleLoggingPlugin

        plugin = LifecycleLoggingPlugin()
        store = DatasetStore(str(tmp_path))

        @task(dataset="out.pkl", dataset_store=store)
        def produce(x: int) -> int:
            return x * 10

        with session(plugins=[plugin], dataset_store=store):
            assert produce(x=5) == 50

    def test_dataset_load_with_logging_plugin(self, tmp_path) -> None:
        """Dataset load hooks fire when loading via top-level API."""
        from daglite import load_dataset
        from daglite import save_dataset
        from daglite.datasets.store import DatasetStore
        from daglite.logging.plugin import LifecycleLoggingPlugin

        plugin = LifecycleLoggingPlugin()
        store = DatasetStore(str(tmp_path))
        save_dataset("data.pkl", [1, 2, 3], store=store)

        with session(plugins=[plugin], dataset_store=store):
            result = load_dataset("data.pkl", list)
        assert result == [1, 2, 3]
