"""Tests for the task composer primitives."""

from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import UUID

import pytest

from daglite import load_dataset
from daglite import save_dataset
from daglite._context import TaskContext
from daglite.composers import gather_tasks
from daglite.composers import map_tasks
from daglite.datasets.store import DatasetStore
from daglite.exceptions import TaskError
from daglite.session import async_session
from daglite.session import session
from daglite.tasks import task

from .examples.tasks import add
from .examples.tasks import async_add
from .examples.tasks import async_count_up
from .examples.tasks import async_double
from .examples.tasks import count_up
from .examples.tasks import double
from .examples.tasks import fail_on_three

CapturedMeta = dict  # {name, map_index, parent_id}


def _capture() -> CapturedMeta | None:
    """Snapshot the active TaskContext metadata inside a running task."""
    ctx = TaskContext._get()
    if ctx is None:
        return None
    m = ctx.metadata
    return {"name": m.name, "map_index": m.map_index, "parent_id": m.parent_id}


# region Map tasks


class TestMapTasks:
    """Core ``map_tasks`` behavior parameterized across backends."""

    @pytest.fixture(params=["inline", "thread", "process"])
    def backend(self, request: pytest.FixtureRequest) -> str:
        return request.param

    def test_single_iterable(self, backend: str) -> None:
        with session(backend=backend):
            result = map_tasks(double, [1, 2, 3, 4])
            assert result == [2, 4, 6, 8]

    def test_multiple_iterables(self, backend: str) -> None:
        with session(backend=backend):
            result = map_tasks(add, [1, 2, 3], [10, 20, 30])
            assert result == [11, 22, 33]

    def test_no_iterables(self, backend: str) -> None:
        with session(backend=backend):
            result = map_tasks(double)
            assert result == []

    def test_empty_list(self, backend: str) -> None:
        with session(backend=backend):
            result = map_tasks(double, [])
            assert result == []

    def test_error_propagates(self, backend: str) -> None:
        with session(backend=backend):
            with pytest.raises(TaskError, match="three is bad"):
                map_tasks(fail_on_three, [1, 2, 3, 4])


class TestBackendResolution:
    """Backend name resolution from context or override."""

    def test_defaults_to_inline_without_context(self) -> None:
        result = map_tasks(double, [5])
        assert result == [10]

    def test_inherits_from_session_context(self) -> None:
        with session(backend="thread"):
            result = map_tasks(double, [2, 3])
            assert result == [4, 6]

    def test_explicit_overrides_context(self) -> None:
        with session(backend="thread"):
            result = map_tasks(double, [4], backend="inline")
            assert result == [8]

    def test_unknown_backend_raises(self) -> None:
        from daglite.exceptions import BackendError

        with session(backend="inline"):
            with pytest.raises(BackendError, match="Unknown backend"):
                map_tasks(double, [1], backend="quantum")

    def test_thread_alias_threading(self) -> None:
        result = map_tasks(double, [1, 2], backend="threading")
        assert result == [2, 4]

    def test_thread_alias_threads(self) -> None:
        result = map_tasks(double, [3], backend="threads")
        assert result == [6]


class TestMapWithNesting:
    """Mapped tasks that call nested tasks get correct parent + map_index."""

    def test_mapped_task_calls_nested(self) -> None:
        inner_parents: list[UUID | None] = []
        outer_indices: list[int | None] = []

        @task(name="leaf")
        def leaf(x: int) -> int:
            ctx = TaskContext._get()
            assert ctx is not None
            inner_parents.append(ctx.metadata.parent_id)
            return x + 1

        @task(name="mapped")
        def mapped(x: int) -> int:
            ctx = TaskContext._get()
            assert ctx is not None
            outer_indices.append(ctx.metadata.map_index)
            return leaf(x=x)

        with session():
            results = map_tasks(mapped, [10, 20])

        assert results == [11, 21]
        assert outer_indices == [0, 1]
        # Each leaf has a parent (the mapped task)
        assert all(p is not None for p in inner_parents)


# region Gather tasks


class TestGatherTasks:
    """Tests for ``gather_tasks`` with async tasks."""

    def test_async_single_iterable(self) -> None:
        result = asyncio.run(gather_tasks(async_double, [1, 2, 3, 4]))
        assert result == [2, 4, 6, 8]

    def test_async_multiple_iterables(self) -> None:
        result = asyncio.run(gather_tasks(async_add, [1, 2], [10, 20]))
        assert result == [11, 22]

    def test_async_empty(self) -> None:
        result = asyncio.run(gather_tasks(async_double, []))
        assert result == []

    def test_async_emits_events(self) -> None:
        with session(backend="inline"):
            result = asyncio.run(gather_tasks(async_double, [2, 3]))
            assert result == [4, 6]

    def test_inline_sequential(self) -> None:
        """Inline backend processes items one at a time."""
        order: list[int] = []

        @task
        async def track(x: int) -> int:
            order.append(x)
            return x

        asyncio.run(gather_tasks(track, [1, 2, 3], backend="inline"))
        assert order == [1, 2, 3]

    def test_concurrent_default(self) -> None:
        """Non-inline backend uses asyncio.gather for concurrency."""
        result = asyncio.run(gather_tasks(async_double, [4, 5], backend="thread"))
        assert result == [8, 10]

    def test_rejects_sync_task(self) -> None:
        with pytest.raises(TypeError, match="async"):
            asyncio.run(gather_tasks(double, [1, 2, 3]))

    def test_async_gather_in_async_session(self) -> None:
        async def _run() -> list[int]:
            async with async_session(backend="inline"):
                return await gather_tasks(async_double, [1, 2, 3])

        assert asyncio.run(_run()) == [2, 4, 6]


# region Auto-wrapping


class TestAutoWrapping:
    """``map_tasks`` and ``gather_tasks`` auto-wrap plain functions."""

    def test_map_tasks_wraps_plain_function(self) -> None:
        def plain(x: int) -> int:
            return x * 2

        result = map_tasks(plain, [1, 2])
        assert result == [2, 4]

    def test_gather_tasks_wraps_plain_function(self) -> None:
        async def plain(x: int) -> int:
            return x * 2

        result = asyncio.run(gather_tasks(plain, [1, 2]))
        assert result == [2, 4]

    def test_map_tasks_rejects_non_callable(self) -> None:
        with pytest.raises(TypeError, match="expects a callable"):
            map_tasks(42, [1, 2])  # type: ignore[arg-type]

    def test_gather_tasks_rejects_non_callable(self) -> None:
        with pytest.raises(TypeError, match="expects a callable"):
            asyncio.run(gather_tasks(42, [1, 2]))  # type: ignore[arg-type]


# region Map index


class TestMapIndex:
    """``map_tasks`` correctly sets ``map_index`` and ``[i]`` suffix."""

    def test_index_suffix_on_names(self) -> None:
        captured: list[CapturedMeta] = []

        @task(name="sq")
        def sq(x: int) -> int:
            captured.append(_capture())  # type: ignore[arg-type]
            return x * x

        with session():
            results = map_tasks(sq, [2, 3, 4])

        assert results == [4, 9, 16]
        assert [c["name"] for c in captured] == ["sq[0]", "sq[1]", "sq[2]"]
        assert [c["map_index"] for c in captured] == [0, 1, 2]

    def test_map_index_is_none_for_normal_call(self) -> None:
        captured: list[CapturedMeta] = []

        @task(name="inc")
        def inc(x: int) -> int:
            captured.append(_capture())  # type: ignore[arg-type]
            return x + 1

        with session():
            inc(x=1)

        assert captured[0]["map_index"] is None
        assert captured[0]["name"] == "inc"

    def test_template_name_consumes_map_index(self) -> None:
        """If the task name template uses ``{map_index}``, no automatic suffix."""
        captured: list[CapturedMeta] = []

        @task(name="item-{map_index}")
        def item(x: int) -> int:
            captured.append(_capture())  # type: ignore[arg-type]
            return x

        with session():
            results = map_tasks(item, [10, 20, 30])

        assert results == [10, 20, 30]
        assert [c["name"] for c in captured] == ["item-0", "item-1", "item-2"]

    def test_async_gather_index(self) -> None:
        captured: list[CapturedMeta] = []

        @task(name="asq")
        async def asq(x: int) -> int:
            captured.append(_capture())  # type: ignore[arg-type]
            return x * x

        async def _run() -> list[int]:
            async with async_session():
                return await gather_tasks(asq, [2, 3, 4])

        results = asyncio.run(_run())
        assert results == [4, 9, 16]
        assert [c["map_index"] for c in captured] == [0, 1, 2]
        assert [c["name"] for c in captured] == ["asq[0]", "asq[1]", "asq[2]"]


# region Streaming composers


class TestMapStreamingTasks:
    """``map_tasks`` with generator tasks collects yields into lists."""

    @pytest.fixture(params=["inline", "thread", "process"])
    def backend(self, request: pytest.FixtureRequest) -> str:
        return request.param

    def test_sync_generator(self, backend: str) -> None:
        with session(backend=backend):
            result = map_tasks(count_up, [3, 2, 4])
        assert result == [[0, 1, 2], [0, 1], [0, 1, 2, 3]]

    def test_async_generator(self, backend: str) -> None:
        with session(backend=backend):
            result = map_tasks(async_count_up, [2, 3])
        assert result == [[0, 1], [0, 1, 2]]


class TestGatherStreamingTasks:
    """``gather_tasks`` with async generator tasks collects yields into lists."""

    def test_async_generator_gather(self) -> None:
        result = asyncio.run(gather_tasks(async_count_up, [3, 2]))
        assert result == [[0, 1, 2], [0, 1]]

    def test_async_generator_gather_in_session(self) -> None:
        async def _run() -> list[list[int]]:
            async with async_session():
                return await gather_tasks(async_count_up, [2, 4])

        assert asyncio.run(_run()) == [[0, 1], [0, 1, 2, 3]]


# region Load & save dataset


class TestLoadSaveDatasetAPI:
    """Test the top-level ``load_dataset`` / ``save_dataset`` functions."""

    def test_save_and_load_explicit_store(self, tmp_path: Path) -> None:
        store = DatasetStore(str(tmp_path))
        path = save_dataset("data.pkl", {"key": "value"}, store=store)
        assert path is not None
        result = load_dataset("data.pkl", dict, store=store)
        assert result == {"key": "value"}

    def test_save_and_load_string_store(self, tmp_path: Path) -> None:
        store_path = str(tmp_path)
        save_dataset("nums.pkl", [1, 2, 3], store=store_path)
        result = load_dataset("nums.pkl", list, store=store_path)
        assert result == [1, 2, 3]

    def test_load_from_session_context(self, tmp_path: Path) -> None:
        store = DatasetStore(str(tmp_path))
        store.save("pre.pkl", "already_there")

        with session(dataset_store=store):
            result = load_dataset("pre.pkl", str)

        assert result == "already_there"

    def test_save_from_session_context(self, tmp_path: Path) -> None:
        store = DatasetStore(str(tmp_path))

        with session(dataset_store=store):
            save_dataset("ctx.pkl", 42)

        assert store.load("ctx.pkl", int) == 42

    def test_load_no_store_falls_back_to_settings(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When no explicit store is given, _get_store falls back to settings default."""
        store = DatasetStore(str(tmp_path))
        store.save("fallback.pkl", "found_it")
        monkeypatch.setenv("DAGLITE_DATASET_STORE", str(tmp_path))

        result = load_dataset("fallback.pkl", str, store=store)
        assert result == "found_it"

    def test_save_no_store_falls_back_to_settings(self, tmp_path: Path) -> None:
        """When no explicit store is given outside a session, falls back to settings."""
        store = DatasetStore(str(tmp_path))
        save_dataset("fb.pkl", "data", store=store)
        assert load_dataset("fb.pkl", str, store=store) == "data"

    def test_save_load_text_format(self, tmp_path: Path) -> None:
        store = DatasetStore(str(tmp_path))
        save_dataset("note.txt", "some text", store=store)
        assert load_dataset("note.txt", str, store=store) == "some text"
