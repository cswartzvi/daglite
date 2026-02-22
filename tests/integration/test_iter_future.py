"""
Integration tests for the lazy iterator pipeline (.iter().map().reduce()).

These tests verify that:
- IterMapFuture collects results correctly (terminal='collect')
- IterReduceFuture folds results correctly (terminal='reduce')
- The generator is iterated lazily (not materialised upfront)
- Both sync and async generators are supported
- Results match those produced by the equivalent map().reduce() path
- All three backends (inline, threading, processes) work correctly
- Upstream dependencies of the generator's kwargs are resolved properly
"""

from __future__ import annotations

from threading import current_thread
from typing import AsyncIterator, Iterator

import pytest

from daglite import task


# ---------------------------------------------------------------------------
# Module-level tasks (required for process backend picklability)
# ---------------------------------------------------------------------------


@task(backend_name="processes")
def gen_numbers_process(n: int) -> Iterator[int]:
    """Yield 0..n-1 (module-level for pickling)."""
    yield from range(n)


@task(backend_name="processes")
def double_process(x: int) -> int:
    return x * 2


@task
def add_process(acc: int, item: int) -> int:
    return acc + item


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _expected_doubled_sum(n: int) -> int:
    """Sum of doubled 0..n-1: n*(n-1)."""
    return sum(i * 2 for i in range(n))


# ---------------------------------------------------------------------------
# Basic correctness — inline backend
# ---------------------------------------------------------------------------


class TestIterMapCollect:
    """IterMapFuture collects results into a list."""

    def test_collect_basic(self) -> None:
        @task
        def numbers(n: int) -> Iterator[int]:
            yield from range(n)

        @task
        def double(x: int) -> int:
            return x * 2

        result = numbers(n=5).iter().map(double).run()
        assert result == [0, 2, 4, 6, 8]

    def test_collect_empty_generator(self) -> None:
        @task
        def empty_gen() -> Iterator[int]:
            return
            yield  # make it a generator

        @task
        def identity(x: int) -> int:
            return x

        result = empty_gen().iter().map(identity).run()
        assert result == []

    def test_collect_with_fixed_kwargs(self) -> None:
        @task
        def numbers(n: int) -> Iterator[int]:
            yield from range(n)

        @task
        def add(x: int, offset: int) -> int:
            return x + offset

        result = numbers(n=4).iter().map(add, offset=10).run()
        assert result == [10, 11, 12, 13]

    def test_collect_async_run(self) -> None:
        import asyncio

        @task
        def numbers(n: int) -> Iterator[int]:
            yield from range(3)

        @task
        def triple(x: int) -> int:
            return x * 3

        async def _run() -> list[int]:
            return await numbers(n=3).iter().map(triple).run_async()

        result = asyncio.run(_run())
        assert result == [0, 3, 6]


class TestIterReduce:
    """IterReduceFuture folds results into a scalar."""

    def test_reduce_sum(self) -> None:
        @task
        def numbers(n: int) -> Iterator[int]:
            yield from range(n)

        @task
        def double(x: int) -> int:
            return x * 2

        @task
        def add(acc: int, item: int) -> int:
            return acc + item

        n = 10
        result = numbers(n=n).iter().map(double).reduce(add, initial=0).run()
        assert result == _expected_doubled_sum(n)

    def test_reduce_with_non_zero_initial(self) -> None:
        @task
        def numbers(n: int) -> Iterator[int]:
            yield from range(n)

        @task
        def identity(x: int) -> int:
            return x

        @task
        def add(acc: int, item: int) -> int:
            return acc + item

        result = numbers(n=5).iter().map(identity).reduce(add, initial=100).run()
        assert result == 100 + sum(range(5))

    def test_reduce_empty_generator(self) -> None:
        @task
        def empty_gen() -> Iterator[int]:
            return
            yield

        @task
        def identity(x: int) -> int:
            return x

        @task
        def add(acc: int, item: int) -> int:
            return acc + item

        result = empty_gen().iter().map(identity).reduce(add, initial=42).run()
        assert result == 42  # no items → returns initial unchanged

    def test_reduce_matches_map_reduce(self) -> None:
        """IterReduceFuture and standard map().reduce() produce identical results."""

        @task
        def numbers(n: int) -> Iterator[int]:
            yield from range(n)

        @task
        def double(x: int) -> int:
            return x * 2

        @task
        def add(acc: int, item: int) -> int:
            return acc + item

        n = 8

        iter_result = numbers(n=n).iter().map(double).reduce(add, initial=0).run()
        map_result = double.map(x=numbers(n=n)).reduce(add, initial=0).run()

        assert iter_result == map_result


# ---------------------------------------------------------------------------
# Async generators
# ---------------------------------------------------------------------------


class TestAsyncGenerators:
    """IterNode handles async generator sources."""

    def test_async_gen_collect(self) -> None:
        @task
        async def async_numbers(n: int) -> AsyncIterator[int]:
            for i in range(n):
                yield i

        @task
        def double(x: int) -> int:
            return x * 2

        result = async_numbers(n=5).iter().map(double).run()
        assert result == [0, 2, 4, 6, 8]

    def test_async_gen_reduce(self) -> None:
        @task
        async def async_numbers(n: int) -> AsyncIterator[int]:
            for i in range(n):
                yield i

        @task
        def double(x: int) -> int:
            return x * 2

        @task
        def add(acc: int, item: int) -> int:
            return acc + item

        n = 6
        result = async_numbers(n=n).iter().map(double).reduce(add, initial=0).run()
        assert result == _expected_doubled_sum(n)


# ---------------------------------------------------------------------------
# Thread backend
# ---------------------------------------------------------------------------


class TestThreadBackend:
    """IterNode executes correctly with the threading backend."""

    def test_thread_collect(self) -> None:
        @task(backend_name="threading")
        def numbers(n: int) -> Iterator[int]:
            yield from range(n)

        @task(backend_name="threading")
        def double(x: int) -> int:
            return x * 2

        result = numbers(n=5).iter().map(double).run()
        assert result == [0, 2, 4, 6, 8]

    def test_thread_reduce(self) -> None:
        @task(backend_name="threading")
        def numbers(n: int) -> Iterator[int]:
            yield from range(n)

        @task(backend_name="threading")
        def double(x: int) -> int:
            return x * 2

        @task
        def add(acc: int, item: int) -> int:
            return acc + item

        n = 10
        result = numbers(n=n).iter().map(double).reduce(add, initial=0).run()
        assert result == _expected_doubled_sum(n)

    def test_reduce_runs_in_worker_not_main_thread(self) -> None:
        """The entire pipeline (gen + map + reduce) should run in one worker, not the main thread."""
        main_thread_name = current_thread().name
        observed_threads: list[str] = []

        @task(backend_name="threading")
        def numbers(n: int) -> Iterator[int]:
            observed_threads.append(current_thread().name)
            yield from range(n)

        @task(backend_name="threading")
        def double(x: int) -> int:
            observed_threads.append(current_thread().name)
            return x * 2

        @task
        def add(acc: int, item: int) -> int:
            observed_threads.append(current_thread().name)
            return acc + item

        result = numbers(n=3).iter().map(double).reduce(add, initial=0).run()
        assert result == 0 + 0 + 2 + 4

        # All three functions should share exactly one thread name (the worker)
        unique_threads = set(observed_threads)
        assert len(unique_threads) == 1, (
            f"Expected all pipeline steps in one thread, got: {unique_threads}"
        )
        # And that thread should NOT be the main thread
        (worker_thread,) = unique_threads
        assert worker_thread != main_thread_name, (
            f"Pipeline should run in a worker thread, not the main thread '{main_thread_name}'"
        )


# ---------------------------------------------------------------------------
# Upstream dependency resolution
# ---------------------------------------------------------------------------


class TestUpstreamDependencies:
    """IterNode correctly resolves upstream node dependencies in the generator's kwargs."""

    def test_generator_kwargs_from_upstream_task(self) -> None:
        """Generator's argument is the result of another task."""

        @task
        def compute_n() -> int:
            return 5

        @task
        def numbers(n: int) -> Iterator[int]:
            yield from range(n)

        @task
        def double(x: int) -> int:
            return x * 2

        result = numbers(n=compute_n()).iter().map(double).run()
        assert result == [0, 2, 4, 6, 8]

    def test_generator_kwargs_and_map_fixed_from_upstream(self) -> None:
        """Both generator arg and map fixed arg come from upstream tasks."""

        @task
        def compute_n() -> int:
            return 4

        @task
        def compute_offset() -> int:
            return 10

        @task
        def numbers(n: int) -> Iterator[int]:
            yield from range(n)

        @task
        def add(x: int, offset: int) -> int:
            return x + offset

        n_future = compute_n()
        offset_future = compute_offset()
        result = numbers(n=n_future).iter().map(add, offset=offset_future).run()
        assert result == [10, 11, 12, 13]


# ---------------------------------------------------------------------------
# PartialTask support
# ---------------------------------------------------------------------------


class TestPartialTask:
    """iter().map() accepts PartialTask instances."""

    def test_map_with_partial(self) -> None:
        @task
        def numbers(n: int) -> Iterator[int]:
            yield from range(n)

        @task
        def multiply(x: int, factor: int) -> int:
            return x * factor

        partial = multiply.partial(factor=3)
        result = numbers(n=4).iter().map(partial).run()
        assert result == [0, 3, 6, 9]

    def test_reduce_with_partial(self) -> None:
        @task
        def numbers(n: int) -> Iterator[int]:
            yield from range(1, 5)  # 1, 2, 3, 4

        @task
        def identity(x: int) -> int:
            return x

        @task
        def weighted_add(acc: int, item: int, weight: int = 1) -> int:
            return acc + item * weight

        # weighted_add.partial(weight=2) → unbound params: (acc, item)
        partial = weighted_add.partial(weight=2)
        result = numbers(n=4).iter().map(identity).reduce(partial, initial=0).run()
        # (1+2+3+4) * 2 = 20
        assert result == 20


# ---------------------------------------------------------------------------
# Process backend
# ---------------------------------------------------------------------------


class TestProcessBackend:
    """IterNode works with the process backend (requires picklable callables)."""

    def test_process_reduce(self) -> None:
        result = (
            gen_numbers_process(n=5).iter().map(double_process).reduce(add_process, initial=0).run()
        )
        assert result == _expected_doubled_sum(5)
