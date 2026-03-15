"""Tests for the dataset integration: store resolution, task decorator, and session plumbing."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from daglite import load_dataset
from daglite import save_dataset
from daglite import session
from daglite import task
from daglite._context import TaskContext
from daglite.composers import map_tasks
from daglite.datasets.store import DatasetStore
from daglite.datasets.store import _resolve_key
from daglite.session import async_session

from .examples.tasks import add


class TestTaskDecoratorDatasetFields:
    """Ensure the task decorator wires dataset fields correctly."""

    def test_defaults_are_none(self) -> None:
        assert add.dataset is None
        assert add.dataset_store is None
        assert add.dataset_format is None

    def test_dataset_key_set(self) -> None:
        @task(dataset="output.pkl")
        def save_result(value: int) -> int:
            return value

        assert save_result.dataset == "output.pkl"

    def test_dataset_store_string(self) -> None:
        @task(dataset="out.pkl", dataset_store="/tmp/ds")
        def t(x: int) -> int:
            return x

        assert t.dataset_store == "/tmp/ds"

    def test_dataset_format(self) -> None:
        @task(dataset="out.pkl", dataset_format="pickle")
        def t(x: int) -> int:
            return x

        assert t.dataset_format == "pickle"


class TestResolveKey:
    """``_resolve_key`` returns key unchanged when no task context is available."""

    def test_no_task_args(self) -> None:
        assert _resolve_key("my_dataset") == "my_dataset"

    def test_with_placeholder_no_args(self) -> None:
        # No active task context, so placeholders stay unresolved
        result = _resolve_key("{split}_data")
        assert result == "{split}_data"


class TestFormatInference:
    """Dataset format is inferred from the file extension."""

    def test_txt_extension_infers_text(self, tmp_path: Path) -> None:
        store = DatasetStore(str(tmp_path))
        store.save("note.txt", "hello world")
        loaded = store.load("note.txt", str)
        assert loaded == "hello world"

    def test_pkl_extension_infers_pickle(self, tmp_path: Path) -> None:
        store = DatasetStore(str(tmp_path))
        store.save("data.pkl", {"a": 1})
        loaded = store.load("data.pkl", dict)
        assert loaded == {"a": 1}


class TestTrySaveDataset:
    """Test the _try_save_dataset helper on _BaseTask."""

    def test_noop_when_no_dataset(self, tmp_path: Path) -> None:
        """Should silently return when task has no dataset configured."""
        store = DatasetStore(str(tmp_path))
        with session():
            result = add(x=1, y=2)
            assert result == 3
            assert store.list_keys() == []

    def test_saves_on_success(self, tmp_path: Path) -> None:
        """Task with dataset= should persist the result after execution."""
        store = DatasetStore(str(tmp_path))

        @task(dataset="my_output.pkl", dataset_store=store)
        def produce(x: int) -> dict:
            return {"value": x}

        with session():
            result = produce(42)

        assert result == {"value": 42}
        assert store.exists("my_output.pkl")
        loaded = store.load("my_output.pkl", dict)
        assert loaded == {"value": 42}

    def test_saves_with_param_template(self, tmp_path: Path) -> None:
        store = DatasetStore(str(tmp_path))

        @task(dataset="{name}_output.pkl", dataset_store=store)
        def produce(name: str) -> str:
            return f"hello {name}"

        with session():
            produce("world")

        assert store.exists("world_output.pkl")
        loaded = store.load("world_output.pkl", str)
        assert loaded == "hello world"

    def test_save_text(self, tmp_path: Path) -> None:
        store = DatasetStore(str(tmp_path))

        @task(dataset="greeting.txt", dataset_store=store)
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        with session():
            result = greet("Alice")

        assert result == "Hello, Alice!"
        assert store.exists("greeting.txt")
        loaded = store.load("greeting.txt", str)
        assert loaded == "Hello, Alice!"


class TestResolveDatasetStore:
    """Exercises ``resolve_dataset_store`` from the session context chain."""

    def test_save_resolves_store_from_session(self, tmp_path: Path) -> None:
        """``save_dataset`` without explicit ``store=`` picks up the session store."""
        store = DatasetStore(str(tmp_path))
        with session(dataset_store=store):
            save_dataset("implicit.pkl", [1, 2, 3])

        assert store.exists("implicit.pkl")
        assert store.load("implicit.pkl", list) == [1, 2, 3]

    def test_load_resolves_store_from_session(self, tmp_path: Path) -> None:
        """``load_dataset`` without explicit ``store=`` picks up the session store."""
        store = DatasetStore(str(tmp_path))
        store.save("pre.pkl", "hello")

        with session(dataset_store=store):
            result = load_dataset("pre.pkl", str)

        assert result == "hello"

    def test_bare_task_with_dataset_saves_directly(self, tmp_path: Path) -> None:
        """A task with ``dataset=`` called outside a session saves via the direct path."""
        store = DatasetStore(str(tmp_path))

        @task(dataset="bare.pkl", dataset_store=store)
        def produce(x: int) -> int:
            return x * 10

        result = produce(x=7)
        assert result == 70
        assert store.exists("bare.pkl")
        assert store.load("bare.pkl", int) == 70


class TestSessionDatasetStore:
    """The session() context manager should set dataset_store on SessionContext."""

    def test_session_with_string_store(self, tmp_path: Path) -> None:
        store_path = str(tmp_path / "datasets")
        with session(dataset_store=store_path) as ctx:
            assert ctx.dataset_store is not None
            assert isinstance(ctx.dataset_store, DatasetStore)

    def test_session_with_dataset_store(self, tmp_path: Path) -> None:
        store = DatasetStore(str(tmp_path))
        with session(dataset_store=store) as ctx:
            assert ctx.dataset_store is store

    def test_session_no_dataset_store(self) -> None:
        with session():
            pass  # Just ensure it doesn't crash

    def test_task_inherits_session_store(self, tmp_path: Path) -> None:
        """Task with dataset= but no per-task store should inherit from session."""
        store = DatasetStore(str(tmp_path))

        @task(dataset="from_session.pkl")
        def produce() -> int:
            return 99

        with session(dataset_store=store):
            produce()

        assert store.exists("from_session.pkl")
        assert store.load("from_session.pkl", int) == 99


class TestAsyncTaskDataset:
    """Verify dataset saving works with async tasks."""

    def test_async_task_saves_dataset(self, tmp_path: Path) -> None:
        store = DatasetStore(str(tmp_path))

        @task(dataset="async_out.pkl", dataset_store=store)
        async def async_produce(x: int) -> int:
            return x * 10

        async def run() -> int:
            async with async_session():
                return await async_produce(5)

        result = asyncio.run(run())
        assert result == 50
        assert store.exists("async_out.pkl")
        assert store.load("async_out.pkl", int) == 50


class TestMapTasksDataset:
    """Verify dataset saving integrates with map_tasks."""

    def test_map_saves_per_index(self, tmp_path: Path) -> None:
        store = DatasetStore(str(tmp_path))

        @task(dataset="item_{map_index}.pkl", dataset_store=store)
        def double(x: int) -> int:
            return x * 2

        with session():
            results = map_tasks(double, [10, 20, 30])

        assert results == [20, 40, 60]
        for i in range(3):
            assert store.exists(f"item_{i}.pkl")
            assert store.load(f"item_{i}.pkl", int) == (i + 1) * 20

    def test_map_saves_with_param_and_index(self, tmp_path: Path) -> None:
        store = DatasetStore(str(tmp_path))

        @task(dataset="out_{x}_{map_index}.pkl", dataset_store=store)
        def triple(x: int) -> int:
            return x * 3

        with session():
            results = map_tasks(triple, [5, 6])

        assert results == [15, 18]
        assert store.exists("out_5_0.pkl")
        assert store.exists("out_6_1.pkl")
        assert store.load("out_5_0.pkl", int) == 15
        assert store.load("out_6_1.pkl", int) == 18


class TestReporterRouting:
    """Verify dataset saves are routed through the DatasetReporter."""

    def test_session_creates_dataset_reporter(self, tmp_path: Path) -> None:
        store = DatasetStore(str(tmp_path))
        with session(dataset_store=store) as ctx:
            assert ctx.dataset_reporter is not None

    def test_session_no_store_no_reporter(self) -> None:
        with session(dataset_store=None) as ctx:
            if ctx.dataset_store is None:
                assert ctx.dataset_reporter is None

    def test_save_through_reporter(self, tmp_path: Path) -> None:
        """Task dataset save goes through reporter (DirectDatasetReporter)."""
        store = DatasetStore(str(tmp_path))

        @task(dataset="routed.pkl", dataset_store=store)
        def produce() -> int:
            return 42

        with session(dataset_store=store):
            produce()

        assert store.exists("routed.pkl")
        assert store.load("routed.pkl", int) == 42

    def test_save_dataset_api_through_reporter(self, tmp_path: Path) -> None:
        """save_dataset() should route through the session's reporter."""
        store = DatasetStore(str(tmp_path))

        with session(dataset_store=store):
            save_dataset("api_routed.pkl", {"x": 1}, store=store)

        assert store.exists("api_routed.pkl")
        assert store.load("api_routed.pkl", dict) == {"x": 1}


class TestMetadataPropagation:
    """Verify that TaskMetadata is passed through dataset hooks."""

    def test_save_inside_task_has_metadata(self, tmp_path: Path) -> None:
        """Hooks should receive metadata when save_dataset is called inside a task."""
        store = DatasetStore(str(tmp_path))
        captured: list[Any] = []

        @task
        def producer() -> int:
            ctx = TaskContext._get()
            captured.append(ctx.metadata if ctx else None)
            save_dataset("inside.pkl", 99, store=store)
            return 99

        with session(dataset_store=store):
            producer()

        assert captured[0] is not None
        assert captured[0].name == "producer"

    def test_load_inside_task_has_context(self, tmp_path: Path) -> None:
        """load_dataset inside a task should have access to TaskContext."""
        store = DatasetStore(str(tmp_path))
        store.save("pre.pkl", "preloaded")

        @task
        def consumer() -> str:
            return load_dataset("pre.pkl", str, store=store)

        with session(dataset_store=store):
            result = consumer()

        assert result == "preloaded"

    def test_nested_load_inside_task_with_dataset(self, tmp_path: Path) -> None:
        """A task that both saves its result and loads another dataset."""
        store = DatasetStore(str(tmp_path))
        store.save("input.pkl", 10)

        @task(dataset="output.pkl", dataset_store=store)
        def transform() -> int:
            val = load_dataset("input.pkl", int, store=store)
            return val * 2

        with session(dataset_store=store):
            result = transform()

        assert result == 20
        assert store.exists("output.pkl")
        assert store.load("output.pkl", int) == 20
