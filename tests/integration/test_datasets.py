"""Integration tests for dataset saving via .run()."""

import tempfile
from uuid import uuid4

import pytest

from daglite import task
from daglite.datasets.store import DatasetStore
from daglite.futures import load_dataset


class TestSaveWithEvaluate:
    """Integration tests: .save() + .run() end-to-end."""

    def test_save_simple_string(self):
        """Save a string result via .save() during evaluation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)

            @task
            def greet(name: str) -> str:
                return f"Hello, {name}!"

            future = greet(name="World").save("greeting.txt", save_store=store)
            result = future.run()

            assert result == "Hello, World!"
            assert store.exists("greeting.txt")
            assert store.load("greeting.txt", return_type=str) == "Hello, World!"

    def test_save_dict_pickle(self):
        """Save a dict result as pickle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)

            @task
            def compute(x: int) -> dict:
                return {"result": x * 2}

            future = compute(x=21).save("result.pkl", save_store=store)
            result = future.run()

            assert result == {"result": 42}
            assert store.load("result.pkl", return_type=dict) == {"result": 42}

    def test_save_with_key_formatting(self):
        """Key templates are formatted from task parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)

            @task
            def process(data_id: str) -> str:
                return f"processed_{data_id}"

            future = process(data_id="abc").save("output_{data_id}.txt", save_store=store)
            result = future.run()

            assert result == "processed_abc"
            assert store.exists("output_abc.txt")
            assert store.load("output_abc.txt", return_type=str) == "processed_abc"

    def test_save_with_extras_in_key(self):
        """Extra values are available for key formatting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)

            @task
            def process(data_id: str) -> str:
                return f"processed_{data_id}"

            future = process(data_id="abc").save(
                "output_{data_id}_{version}.txt",
                save_store=store,
                version="v1",
            )
            result = future.run()

            assert result == "processed_abc"
            assert store.exists("output_abc_v1.txt")

    def test_save_with_future_extra(self):
        """TaskFuture extras are resolved and used in key formatting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)

            @task
            def get_version() -> str:
                return "v2"

            @task
            def process(data_id: str) -> str:
                return f"processed_{data_id}"

            future = process(data_id="abc").save(
                "output_{data_id}_{version}.txt",
                save_store=store,
                version=get_version(),
            )
            result = future.run()

            assert result == "processed_abc"
            assert store.exists("output_abc_v2.txt")

    def test_save_multiple_outputs(self):
        """Multiple .save() calls create multiple outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)

            @task
            def compute(x: int) -> int:
                return x * 2

            future = (
                compute(x=5)
                .save("result_{x}.pkl", save_store=store)
                .save("backup_{x}.pkl", save_store=store)
            )
            result = future.run()

            assert result == 10
            assert store.exists("result_5.pkl")
            assert store.exists("backup_5.pkl")

    def test_save_with_explicit_format(self):
        """Explicit format overrides extension-based inference."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)

            @task
            def make_text() -> str:
                return "hello"

            future = make_text().save("output.dat", save_store=store, save_format="text")
            future.run()

            data = store._driver.load("output.dat")
            assert data == b"hello"

    def test_save_does_not_alter_result(self):
        """save() is a side effect; .run() returns the original result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)

            @task
            def compute() -> int:
                return 42

            result = compute().save("answer.pkl", save_store=store).run()
            assert result == 42

    def test_save_in_chain(self):
        """save() works in a chained pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)

            @task
            def step1(x: int) -> int:
                return x + 1

            @task
            def step2(y: int) -> int:
                return y * 2

            future = step1(x=5).save("step1.pkl", save_store=store).then(step2)
            result = future.run()

            assert result == 12
            assert store.exists("step1.pkl")
            loaded = store.load("step1.pkl", return_type=int)
            assert loaded == 6

    def test_save_with_checkpoint_true(self):
        """save_checkpoint=True stores the output and marks it as a checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)

            @task
            def compute(x: int) -> int:
                return x * 2

            future = compute(x=5).save("result.pkl", save_store=store, save_checkpoint=True)
            result = future.run()
            assert result == 10
            assert store.exists("result.pkl")


class TestSaveWithMapTasks:
    """Integration tests for .save() with map/product/zip tasks."""

    def test_save_with_product(self):
        """save() works with product fan-out tasks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)

            @task
            def process(x: int) -> int:
                return x * 2

            @task
            def collect(values: list[int]) -> list[int]:
                return sorted(values)

            future = (
                process.map(x=[1, 2, 3], map_mode="product")
                .save("item_{x}_{iteration_index}.pkl", save_store=store)
                .join(collect)
            )
            result = future.run()

            assert result == [2, 4, 6]
            # Check that at least some outputs were saved
            keys = store.list_keys()
            assert len(keys) == 3
            assert keys == ["item_1_0.pkl", "item_2_1.pkl", "item_3_2.pkl"]

    def test_save_with_zip(self):
        """save() works with zip fan-out tasks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)

            @task
            def add(x: int, y: int) -> int:
                return x + y

            @task
            def collect(values: list[int]) -> list[int]:
                return sorted(values)

            future = (
                add.map(x=[1, 2, 3], y=[10, 20, 30], map_mode="zip")
                .save("sum_{iteration_index}.pkl", save_store=store)
                .join(collect)
            )
            result = future.run()

            assert result == [11, 22, 33]
            keys = store.list_keys()
            assert len(keys) == 3


class TestSaveWithTaskStore:
    """Integration tests for task_store fallback."""

    def test_task_store_used_when_no_explicit_store(self):
        """task_store is used when save_store is not specified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)

            @task(store=store)
            def compute(x: int) -> int:
                return x * 2

            future = compute(x=5).save("result.pkl")
            result = future.run()

            assert result == 10
            assert store.exists("result.pkl")


class TestSaveWithMapTaskFutureExtras:
    """Integration tests: .save() on map tasks with future-based extras."""

    def test_map_save_with_future_extra_in_key(self):
        """Map task save with a future extra used in the key template."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)

            @task
            def make_prefix() -> str:
                return "run1"

            @task
            def double(x: int) -> int:
                return x * 2

            prefix_future = make_prefix()
            future = double.map(x=[1, 2]).save(
                "{prefix}_{iteration_index}.pkl",
                save_store=store,
                prefix=prefix_future,
            )

            @task
            def collect(values: list[int]) -> list[int]:
                return sorted(values)

            final = future.join(collect)
            result = final.run()

            assert result == [2, 4]
            assert store.exists("run1_0.pkl")
            assert store.exists("run1_1.pkl")

    def test_map_save_with_plain_string_extra(self):
        """Map task save with a plain (non-future) extra in the key template."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)

            @task
            def triple(x: int) -> int:
                return x * 3

            future = triple.map(x=[1, 2]).save(
                "{label}_{iteration_index}.pkl",
                save_store=store,
                label="batch",
            )

            @task
            def collect(values: list[int]) -> list[int]:
                return sorted(values)

            final = future.join(collect)
            result = final.run()

            assert result == [3, 6]
            assert store.exists("batch_0.pkl")
            assert store.exists("batch_1.pkl")


class TestLoadDataset:
    """Tests for DatasetNode.run() execution."""

    def test_load_simple_value(self):
        """DatasetNode loads a value from the store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)
            store.save("data.pkl", {"key": "value"})

            future = load_dataset("data.pkl", load_store=store, load_type=dict)
            result = future.run()
            assert result == {"key": "value"}

    def test_load_string_value(self):
        """DatasetNode loads a string using text format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)
            store.save("hello.txt", "hello world", format="text")

            future = load_dataset("hello.txt", load_store=store, load_type=str)
            result = future.run()
            assert result == "hello world"

    def test_load_with_key_template(self):
        """Key templates are resolved from extras at runtime."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)
            store.save("data_v1.pkl", [1, 2, 3])

            future = load_dataset(
                "data_{version}.pkl", load_store=store, load_type=list, version="v1"
            )
            result = future.run()
            assert result == [1, 2, 3]

    def test_load_with_future_extra(self):
        """Key template extras can be other task futures."""

        @task
        def get_version() -> str:
            return "v2"

        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)
            store.save("data_v2.pkl", {"version": 2})

            future = load_dataset(
                "data_{version}.pkl",
                load_store=store,
                load_type=dict,
                version=get_version(),
            )
            result = future.run()
            assert result == {"version": 2}

    def test_load_then_process(self):
        """DatasetFuture can be chained with .then()."""

        @task
        def double(values: list) -> list:
            return [v * 2 for v in values]

        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)
            store.save("input.pkl", [1, 2, 3])

            future = load_dataset("input.pkl", load_store=store, load_type=list).then(double)
            result = future.run()
            assert result == [2, 4, 6]

    def test_load_then_save(self):
        """DatasetFuture can chain .save() to re-save loaded data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)
            store.save("input.pkl", {"data": 42})

            future = load_dataset("input.pkl", load_store=store, load_type=dict).save(
                "output.pkl", save_store=store
            )
            result = future.run()

            assert result == {"data": 42}
            assert store.exists("output.pkl")
            assert store.load("output.pkl", dict) == {"data": 42}

    def test_load_with_format(self):
        """Explicit load_format is threaded through to store.load()."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)
            store.save("data.txt", "hello", format="text")

            future = load_dataset("data.txt", load_store=store, load_type=str, load_format="text")
            result = future.run()
            assert result == "hello"

    def test_bad_key_template_raises_at_runtime(self):
        """Missing key template variable raises ValueError at runtime."""
        from daglite.graph.nodes import DatasetNode

        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)
            node = DatasetNode(
                id=uuid4(),
                name="test",
                store=store,
                load_key="data_{missing}.pkl",
                kwargs={},
            )
            import asyncio

            ## _prepare returns submissions; calling a submission triggers the key formatting
            submissions = node._prepare({})
            with pytest.raises(ValueError, match="missing"):
                asyncio.run(submissions[0]())


class TestLoadDatasetHooks:
    """Tests for before/after_dataset_load hook firing."""

    def test_hooks_fire_during_load(self):
        """Both before and after hooks fire when a DatasetNode executes."""
        from daglite.plugins.hooks.markers import hook_impl

        hook_calls: list[str] = []

        class HookTracker:
            @hook_impl
            def before_dataset_load(self, **kwargs):
                hook_calls.append("before")

            @hook_impl
            def after_dataset_load(self, **kwargs):
                hook_calls.append("after")

        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)
            store.save("data.pkl", {"ok": True})

            future = load_dataset("data.pkl", load_store=store, load_type=dict)
            result = future.run(plugins=[HookTracker()])

            assert result == {"ok": True}
            assert "before" in hook_calls
            assert "after" in hook_calls
