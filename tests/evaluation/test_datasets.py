"""Integration tests for dataset saving via evaluate()."""

import tempfile

from daglite import evaluate
from daglite import task
from daglite.datasets.store import DatasetStore


class TestSaveWithEvaluate:
    """Integration tests: .save() + evaluate() end-to-end."""

    def test_save_simple_string(self):
        """Save a string result via .save() during evaluation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)

            @task
            def greet(name: str) -> str:
                return f"Hello, {name}!"

            future = greet(name="World").save("greeting.txt", save_store=store)
            result = evaluate(future)

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
            result = evaluate(future)

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
            result = evaluate(future)

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
            result = evaluate(future)

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
            result = evaluate(future)

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
            result = evaluate(future)

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
            evaluate(future)

            data = store._driver.load("output.dat")
            assert data == b"hello"

    def test_save_does_not_alter_result(self):
        """save() is a side effect; evaluate() returns the original result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)

            @task
            def compute() -> int:
                return 42

            result = evaluate(compute().save("answer.pkl", save_store=store))
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
            result = evaluate(future)

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
            result = evaluate(future)
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
                process.product(x=[1, 2, 3])
                .save("item_{x}_{iteration_index}.pkl", save_store=store)
                .join(collect)
            )
            result = evaluate(future)

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
                add.zip(x=[1, 2, 3], y=[10, 20, 30])
                .save("sum_{iteration_index}.pkl", save_store=store)
                .join(collect)
            )
            result = evaluate(future)

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
            result = evaluate(future)

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
            future = double.product(x=[1, 2]).save(
                "{prefix}_{iteration_index}.pkl",
                save_store=store,
                prefix=prefix_future,
            )

            @task
            def collect(values: list[int]) -> list[int]:
                return sorted(values)

            final = future.join(collect)
            result = evaluate(final)

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

            future = triple.product(x=[1, 2]).save(
                "{label}_{iteration_index}.pkl",
                save_store=store,
                label="batch",
            )

            @task
            def collect(values: list[int]) -> list[int]:
                return sorted(values)

            final = future.join(collect)
            result = evaluate(final)

            assert result == [3, 6]
            assert store.exists("batch_0.pkl")
            assert store.exists("batch_1.pkl")
