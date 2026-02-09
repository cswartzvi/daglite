"""Integration tests for dataset saving via the processes backend.

Tasks MUST be defined at module level to be picklable by multiprocessing.
This ensures Windows compatibility (spawn start method).
"""

import tempfile

from daglite import evaluate
from daglite import task
from daglite.datasets.store import DatasetStore

# ── Module-level tasks (required for processes backend pickling) ──────


@task
def greet(name: str) -> str:
    return f"Hello, {name}!"


@task
def compute_dict(x: int) -> dict:
    return {"result": x * 2}


@task
def process_id(data_id: str) -> str:
    return f"processed_{data_id}"


@task
def get_version() -> str:
    return "v2"


@task
def make_text() -> str:
    return "hello"


@task
def compute_int(x: int) -> int:
    return x * 2


@task
def step1(x: int) -> int:
    return x + 1


@task
def step2(y: int) -> int:
    return y * 2


@task
def add(x: int, y: int) -> int:
    return x + y


@task
def triple(x: int) -> int:
    return x * 3


@task
def collect_ints(values: list[int]) -> list[int]:
    return sorted(values)


@task
def make_prefix() -> str:
    return "run1"


# ── Integration tests ────────────────────────────────────────────────


class TestSaveWithProcessBackend:
    """End-to-end tests for .save() using the processes backend."""

    def test_save_simple_string(self):
        """Save a string result via .save() with processes backend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)

            future = greet.with_options(backend_name="processes")(name="World").save(
                "greeting.txt", save_store=store
            )
            result = evaluate(future)

            assert result == "Hello, World!"
            assert store.exists("greeting.txt")
            assert store.load("greeting.txt", return_type=str) == "Hello, World!"

    def test_save_dict_pickle(self):
        """Save a dict result as pickle via processes backend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)

            future = compute_dict.with_options(backend_name="processes")(x=21).save(
                "result.pkl", save_store=store
            )
            result = evaluate(future)

            assert result == {"result": 42}
            assert store.load("result.pkl", return_type=dict) == {"result": 42}

    def test_save_with_key_formatting(self):
        """Key templates are formatted from task parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)

            future = process_id.with_options(backend_name="processes")(data_id="abc").save(
                "output_{data_id}.txt", save_store=store
            )
            result = evaluate(future)

            assert result == "processed_abc"
            assert store.exists("output_abc.txt")
            assert store.load("output_abc.txt", return_type=str) == "processed_abc"

    def test_save_with_extras_in_key(self):
        """Extra values are available for key formatting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)

            future = process_id.with_options(backend_name="processes")(data_id="abc").save(
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

            version_future = get_version.with_options(backend_name="processes")()
            future = process_id.with_options(backend_name="processes")(data_id="abc").save(
                "output_{data_id}_{version}.txt",
                save_store=store,
                version=version_future,
            )
            result = evaluate(future)

            assert result == "processed_abc"
            assert store.exists("output_abc_v2.txt")

    def test_save_multiple_outputs(self):
        """Multiple .save() calls create multiple outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)

            future = (
                compute_int.with_options(backend_name="processes")(x=5)
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

            future = make_text.with_options(backend_name="processes")().save(
                "output.dat", save_store=store, save_format="text"
            )
            evaluate(future)

            data = store._driver.load("output.dat")
            assert data == b"hello"

    def test_save_does_not_alter_result(self):
        """save() is a side effect; evaluate() returns the original result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)

            result = evaluate(
                compute_int.with_options(backend_name="processes")(x=21).save(
                    "answer.pkl", save_store=store
                )
            )
            assert result == 42

    def test_save_in_chain(self):
        """save() works in a chained pipeline with processes backend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)

            s1 = step1.with_options(backend_name="processes")
            s2 = step2.with_options(backend_name="processes")

            future = s1(x=5).save("step1.pkl", save_store=store).then(s2)
            result = evaluate(future)

            assert result == 12
            assert store.exists("step1.pkl")
            loaded = store.load("step1.pkl", return_type=int)
            assert loaded == 6


class TestSaveWithMapTasksProcessBackend:
    """Integration tests for .save() with map tasks on processes backend."""

    def test_save_with_product(self):
        """save() works with product fan-out tasks on processes backend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)

            proc = compute_int.with_options(backend_name="processes")
            future = (
                proc.product(x=[1, 2, 3])
                .save("item_{x}_{iteration_index}.pkl", save_store=store)
                .join(collect_ints)
            )
            result = evaluate(future)

            assert result == [2, 4, 6]
            keys = store.list_keys()
            assert len(keys) == 3
            assert keys == ["item_1_0.pkl", "item_2_1.pkl", "item_3_2.pkl"]

    def test_save_with_zip(self):
        """save() works with zip fan-out tasks on processes backend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)

            proc = add.with_options(backend_name="processes")
            future = (
                proc.zip(x=[1, 2, 3], y=[10, 20, 30])
                .save("sum_{iteration_index}.pkl", save_store=store)
                .join(collect_ints)
            )
            result = evaluate(future)

            assert result == [11, 22, 33]
            keys = store.list_keys()
            assert len(keys) == 3


class TestMapSaveFutureExtrasProcessBackend:
    """Integration tests: .save() on map tasks with future-based extras via processes."""

    def test_map_save_with_future_extra_in_key(self):
        """Map task save with a future extra used in the key template."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)

            dbl = compute_int.with_options(backend_name="processes")
            pfx = make_prefix.with_options(backend_name="processes")

            prefix_future = pfx()
            future = (
                dbl.product(x=[1, 2])
                .save(
                    "{prefix}_{iteration_index}.pkl",
                    save_store=store,
                    prefix=prefix_future,
                )
                .join(collect_ints)
            )
            result = evaluate(future)

            assert result == [2, 4]
            assert store.exists("run1_0.pkl")
            assert store.exists("run1_1.pkl")

    def test_map_save_with_plain_string_extra(self):
        """Map task save with a plain (non-future) extra in the key template."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)

            t = triple.with_options(backend_name="processes")
            future = (
                t.product(x=[1, 2])
                .save(
                    "{label}_{iteration_index}.pkl",
                    save_store=store,
                    label="batch",
                )
                .join(collect_ints)
            )
            result = evaluate(future)

            assert result == [3, 6]
            assert store.exists("batch_0.pkl")
            assert store.exists("batch_1.pkl")


class TestSaveWithTaskStoreProcessBackend:
    """Integration tests for task_store fallback via processes backend."""

    def test_task_store_used_when_no_explicit_store(self):
        """task_store is used when save_store is not specified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)

            proc = compute_int.with_options(backend_name="processes", store=store)
            future = proc(x=5).save("result.pkl")
            result = evaluate(future)

            assert result == 10
            assert store.exists("result.pkl")
