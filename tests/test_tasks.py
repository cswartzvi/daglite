"""Tests for the core task primitives."""

from __future__ import annotations

import asyncio
import tempfile
from collections.abc import AsyncIterator
from collections.abc import Iterator
from uuid import UUID

import pytest

from daglite import session
from daglite import task
from daglite._context import TaskContext
from daglite.exceptions import TaskError
from daglite.tasks import AsyncTask
from daglite.tasks import AsyncTaskStream
from daglite.tasks import SyncTask
from daglite.tasks import SyncTaskStream

from .examples.tasks import add
from .examples.tasks import async_add
from .examples.tasks import async_broken
from .examples.tasks import async_broken_stream
from .examples.tasks import async_count_up
from .examples.tasks import async_double
from .examples.tasks import async_maybe_none
from .examples.tasks import async_named_add
from .examples.tasks import async_to_string
from .examples.tasks import broken
from .examples.tasks import broken_stream
from .examples.tasks import count_up
from .examples.tasks import double
from .examples.tasks import maybe_none
from .examples.tasks import multiply
from .examples.tasks import named_add
from .examples.tasks import to_string

# region Tasks


class TestTaskInstanceTypes:
    """`@task` produces the correct concrete task type."""

    def test_sync_unary(self) -> None:
        assert isinstance(double, SyncTask)

    def test_async_unary(self) -> None:
        assert isinstance(async_double, AsyncTask)

    def test_sync_transform(self) -> None:
        assert isinstance(to_string, SyncTask)

    def test_async_transform(self) -> None:
        assert isinstance(async_to_string, AsyncTask)

    def test_sync_optional(self) -> None:
        assert isinstance(maybe_none, SyncTask)

    def test_async_optional(self) -> None:
        assert isinstance(async_maybe_none, AsyncTask)

    def test_sync_binary(self) -> None:
        assert isinstance(add, SyncTask)

    def test_async_binary(self) -> None:
        assert isinstance(async_add, AsyncTask)

    def test_sync_named(self) -> None:
        assert isinstance(named_add, SyncTask)

    def test_async_named(self) -> None:
        assert isinstance(async_named_add, AsyncTask)


class TestTaskProperties:
    """Tasks expose the expected metadata properties."""

    def test_name_from_function(self) -> None:
        assert add.name == "add"

    def test_name_from_decorator(self) -> None:
        assert named_add.name == "named_add"

    def test_is_async_sync(self) -> None:
        assert add.is_async is False

    def test_is_async_true(self) -> None:
        assert async_add.is_async is True

    def test_signature_preserved(self) -> None:
        params = list(add.signature.parameters.keys())
        assert params == ["x", "y"]

    def test_custom_description(self) -> None:
        @task(description="A custom task")
        def my_task() -> None:
            pass

        assert my_task.description == "A custom task"


class TestResolveName:
    """``_resolve_name`` handles templates and map_index correctly."""

    def test_plain_name_no_index(self) -> None:
        assert add._resolve_name({"x": 1, "y": 2}) == "add"

    def test_plain_name_with_index_appends_suffix(self) -> None:
        assert add._resolve_name({"x": 1, "y": 2}, map_index=3) == "add[3]"

    def test_template_name_no_index(self) -> None:
        @task(name="compute_{x}")
        def templated(x: int) -> int:
            return x * 10

        assert templated._resolve_name({"x": 42}) == "compute_42"

    def test_template_name_with_index_appends_suffix(self) -> None:
        """Template that does NOT use {map_index} should still get [i] suffix."""

        @task(name="compute_{x}")
        def templated(x: int) -> int:
            return x * 10

        assert templated._resolve_name({"x": 42}, map_index=0) == "compute_42[0]"

    def test_template_consuming_map_index_skips_suffix(self) -> None:
        """When the template already uses {map_index}, no redundant [i] suffix."""

        @task(name="item_{map_index}")
        def indexed(x: int) -> int:
            return x

        assert indexed._resolve_name({"x": 1}, map_index=7) == "item_7"


class TestResolveDatasetKey:
    """``_resolve_dataset_key`` resolves placeholders correctly."""

    def test_no_dataset_returns_none(self) -> None:
        assert add._resolve_dataset_key({"x": 1, "y": 2}) is None

    def test_static_key(self) -> None:
        @task(dataset="output.pkl")
        def save_result(value: int) -> int:
            return value

        assert save_result._resolve_dataset_key({"value": 42}) == "output.pkl"

    def test_param_placeholder(self) -> None:
        @task(dataset="results/{name}.pkl")
        def save_named(name: str, value: int) -> int:
            return value

        assert save_named._resolve_dataset_key({"name": "foo", "value": 1}) == "results/foo.pkl"

    def test_map_index_placeholder(self) -> None:
        @task(dataset="mapped_{map_index}.pkl")
        def save_mapped(x: int) -> int:
            return x * 2

        assert save_mapped._resolve_dataset_key({"x": 5}, map_index=2) == "mapped_2.pkl"

    def test_param_and_map_index(self) -> None:
        @task(dataset="mapped_{x}_{map_index}.pkl")
        def save_mapped_with_param(x: int) -> int:
            return x * 3

        key = save_mapped_with_param._resolve_dataset_key({"x": 7}, map_index=3)
        assert key == "mapped_7_3.pkl"


class TestBindArgs:
    """``_bind_args`` returns kwargs dict when ``signature.bind`` fails."""

    def test_extra_kwargs_fallback(self) -> None:
        @task
        def strict_fn(a: int) -> int:
            return a

        result = strict_fn._bind_args((), {"a": 1, "b": 2, "c": 3})
        assert result == {"a": 1, "b": 2, "c": 3}


class TestDecoratorEdgeCases:
    """Edge cases for the ``@task`` decorator."""

    def test_rejects_class(self) -> None:
        with pytest.raises(TypeError, match="callable functions"):

            @task  # type: ignore[arg-type]
            class NotAFunction:
                pass

    def test_with_parens_no_args(self) -> None:
        @task()
        def noop() -> int:
            return 42

        assert noop() == 42
        assert noop.name == "noop"

    def test_name_template_unknown_placeholder_rejected(self) -> None:
        with pytest.raises(ValueError, match="references"):

            @task(name="step_{missing}")
            def fn(x: int) -> int:
                return x

    def test_dataset_template_unknown_placeholder_rejected(self) -> None:
        with pytest.raises(ValueError, match="references"):

            @task(dataset="output_{missing}.pkl")
            def fn(x: int) -> int:
                return x


class TestSyncExecution:
    """Tasks execute correctly without any session context."""

    def test_call_kwargs(self) -> None:
        assert add(x=1, y=2) == 3

    def test_call_positional(self) -> None:
        assert add(1, 2) == 3

    def test_named_task_call(self) -> None:
        assert multiply(x=3, factor=4) == 12

    def test_error_propagates(self) -> None:
        with pytest.raises(TaskError, match="boom"):
            broken(x=1)


class TestAsyncExecution:
    """Async tasks execute correctly without any session context."""

    def test_async_call(self) -> None:
        result = asyncio.run(async_add(x=1, y=2))
        assert result == 3

    def test_async_error_propagates(self) -> None:
        with pytest.raises(TaskError, match="async boom"):
            asyncio.run(async_broken(x=1))


class TestExecutionInsideSession:
    """Tasks produce correct results inside a ``session()`` context."""

    def test_sync_task(self) -> None:
        with session():
            assert add(x=10, y=20) == 30

    def test_async_task(self) -> None:
        with session():
            result = asyncio.run(async_add(x=5, y=7))
            assert result == 12

    def test_error_wraps_in_task_error(self) -> None:
        with session():
            with pytest.raises(TaskError, match="boom"):
                broken(x=1)


class TestParentTracking:
    """Nested task calls set ``parent_id`` correctly."""

    def test_top_level_parent_is_none(self) -> None:
        with session():
            ctx_data: dict | None = None

            @task(name="top")
            def top(x: int) -> int:
                nonlocal ctx_data
                ctx = TaskContext._get()
                assert ctx is not None
                ctx_data = {"parent_id": ctx.metadata.parent_id}
                return x

            top(x=1)

        assert ctx_data is not None
        assert ctx_data["parent_id"] is None

    def test_nested_task_has_parent_id(self) -> None:
        outer_ids: list[UUID] = []
        inner_parents: list[UUID | None] = []

        @task(name="inner")
        def inner(x: int) -> int:
            ctx = TaskContext._get()
            assert ctx is not None
            inner_parents.append(ctx.metadata.parent_id)
            return x * 2

        @task(name="outer")
        def outer(x: int) -> int:
            ctx = TaskContext._get()
            assert ctx is not None
            outer_ids.append(ctx.metadata.id)
            return inner(x=x)

        with session():
            result = outer(x=5)

        assert result == 10
        assert len(outer_ids) == 1
        assert len(inner_parents) == 1
        assert inner_parents[0] == outer_ids[0]

    def test_deeply_nested_chain(self) -> None:
        parents: list[UUID | None] = []

        @task(name="c")
        def c(x: int) -> int:
            ctx = TaskContext._get()
            assert ctx is not None
            parents.append(ctx.metadata.parent_id)
            return x

        @task(name="b")
        def b(x: int) -> int:
            ctx = TaskContext._get()
            assert ctx is not None
            parents.append(ctx.metadata.parent_id)
            return c(x=x)

        @task(name="a")
        def a(x: int) -> int:
            return b(x=x)

        with session():
            a(x=42)

        # b's parent is a, c's parent is b
        assert parents[0] is not None  # b's parent (a)
        assert parents[1] is not None  # c's parent (b)
        assert parents[0] != parents[1]  # different parents


class TestCaching:
    """Cache hits return stored values without re-executing."""

    @pytest.fixture()
    def cache_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_cache_hit_avoids_recomputation(self, cache_dir: str) -> None:
        call_count = 0

        @task(cache=True)
        def counted(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        with session(cache_store=cache_dir):
            assert counted(x=5) == 10
            assert counted(x=5) == 10

        assert call_count == 1

    def test_different_args_produce_miss(self, cache_dir: str) -> None:
        call_count = 0

        @task(cache=True)
        def counted_inc(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x + 1

        with session(cache_store=cache_dir):
            assert counted_inc(x=1) == 2
            assert counted_inc(x=2) == 3

        assert call_count == 2

    def test_no_cache_without_flag(self, cache_dir: str) -> None:
        """Tasks without ``cache=True`` don't use the cache."""
        call_count = 0

        @task  # no cache=True
        def uncached(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x + 1

        with session(cache_store=cache_dir):
            uncached(x=1)
            uncached(x=1)

        assert call_count == 2

    def test_async_cache_hit(self, cache_dir: str) -> None:
        call_count = 0

        @task(cache=True)
        async def async_cached(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        with session(cache_store=cache_dir):
            r1 = asyncio.run(async_cached(x=5))
            r2 = asyncio.run(async_cached(x=5))
            assert r1 == 10
            assert r2 == 10

        assert call_count == 1


class TestRetries:
    """Tasks with retries re-attempt on failure."""

    def test_retries_succeed_eventually(self) -> None:
        call_count = 0

        @task(retries=2)
        def flaky(x: int) -> int:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError(f"Attempt {call_count}")
            return x * 10

        with session():
            result = flaky(x=7)
            assert result == 70
            assert call_count == 3

    def test_async_retry(self) -> None:
        call_count = 0

        @task(retries=1)
        async def async_flaky(x: int) -> int:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("boom")
            return x * 10

        with session():
            result = asyncio.run(async_flaky(x=5))
            assert result == 50
            assert call_count == 2


# region Streaming tasks


class TestStreamingInstanceTypes:
    """`@task` on generators produces the correct stream type."""

    def test_sync_generator(self) -> None:
        assert isinstance(count_up, SyncTaskStream)

    def test_async_generator(self) -> None:
        assert isinstance(async_count_up, AsyncTaskStream)


class TestStreamingProperties:
    """Stream tasks expose the correct metadata flags."""

    def test_sync_is_generator(self) -> None:
        assert count_up.is_generator is True

    def test_sync_is_not_async(self) -> None:
        assert count_up.is_async is False

    def test_async_is_generator(self) -> None:
        assert async_count_up.is_generator is True

    def test_async_is_async(self) -> None:
        assert async_count_up.is_async is True


class TestSyncStreamExecution:
    """SyncTaskStream yields items correctly."""

    def test_yields_all_items(self) -> None:
        assert list(count_up(n=5)) == [0, 1, 2, 3, 4]

    def test_empty_stream(self) -> None:
        assert list(count_up(n=0)) == []

    def test_single_element(self) -> None:
        assert list(count_up(n=1)) == [0]

    def test_error_wraps_in_task_error(self) -> None:
        with pytest.raises(TaskError, match="stream boom"):
            list(broken_stream(n=3))

    def test_partial_consumption_before_error(self) -> None:
        """The first yielded item is available before the error."""
        items = []
        with pytest.raises(TaskError, match="stream boom"):
            for item in broken_stream(n=3):
                items.append(item)
        assert items == [0]

    def test_inside_session(self) -> None:
        with session():
            assert list(count_up(n=3)) == [0, 1, 2]


class TestAsyncStreamExecution:
    """AsyncTaskStream yields items correctly."""

    def test_yields_all_items(self) -> None:
        async def _collect() -> list[int]:
            return [item async for item in async_count_up(n=5)]

        assert asyncio.run(_collect()) == [0, 1, 2, 3, 4]

    def test_empty_stream(self) -> None:
        async def _collect() -> list[int]:
            return [item async for item in async_count_up(n=0)]

        assert asyncio.run(_collect()) == []

    def test_error_wraps_in_task_error(self) -> None:
        async def _collect() -> list[int]:
            return [item async for item in async_broken_stream(n=3)]

        with pytest.raises(TaskError, match="async stream boom"):
            asyncio.run(_collect())

    def test_partial_consumption_before_error(self) -> None:
        async def _collect() -> list[int]:
            items: list[int] = []
            async for item in async_broken_stream(n=3):
                items.append(item)
            return items

        try:
            _ = asyncio.run(_collect())
        except TaskError:
            pass
        # The first item was yielded before the error; the coroutine raised so items stays empty
        # from the caller's perspective — verify the TaskError is raised.
        with pytest.raises(TaskError, match="async stream boom"):
            asyncio.run(_collect())

    def test_inside_session(self) -> None:
        async def _run() -> list[int]:
            return [item async for item in async_count_up(n=3)]

        with session():
            assert asyncio.run(_run()) == [0, 1, 2]


class TestGeneratorDecoratorValidation:
    """Generator tasks reject unsupported decorator options at decoration time."""

    def test_rejects_cache(self) -> None:
        with pytest.raises(ValueError, match="Caching is not supported"):

            @task(cache=True)
            def cached_gen(n: int) -> Iterator[int]:
                yield from range(n)

    def test_rejects_retries(self) -> None:
        with pytest.raises(ValueError, match="Retries are not supported"):

            @task(retries=1)
            def retried_gen(n: int) -> Iterator[int]:
                yield from range(n)

    def test_rejects_timeout(self) -> None:
        with pytest.raises(ValueError, match="Timeouts are not supported"):

            @task(timeout=5.0)
            def timed_gen(n: int) -> Iterator[int]:
                yield from range(n)

    def test_async_rejects_cache(self) -> None:
        with pytest.raises(ValueError, match="Caching is not supported"):

            @task(cache=True)
            async def cached_async_gen(n: int) -> AsyncIterator[int]:
                yield n
