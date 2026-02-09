"""Tests for load_dataset / DatasetFuture / DatasetNode."""

import tempfile
from uuid import uuid4

import pytest

from daglite import load_dataset
from daglite import task
from daglite.datasets.store import DatasetStore
from daglite.futures import DatasetFuture
from daglite.futures import MapTaskFuture
from daglite.futures import TaskFuture


class TestLoadDatasetFunction:
    """Tests for the load_dataset() factory function."""

    def test_returns_dataset_future(self):
        future = load_dataset("data.pkl")
        assert isinstance(future, DatasetFuture)

    def test_key_stored(self):
        future = load_dataset("data.pkl")
        assert future.load_key == "data.pkl"

    def test_options_default_empty_dict(self):
        future = load_dataset("data.pkl")
        assert future.load_options == {}

    def test_explicit_params(self):
        store = DatasetStore(tempfile.mkdtemp())
        future = load_dataset(
            "data.csv",
            load_type=dict,
            load_format="pandas/csv",
            load_store=store,
            load_options={"use_tabs": True},
        )
        assert future.load_type is dict
        assert future.load_format == "pandas/csv"
        assert future.load_store is store
        assert future.load_options == {"use_tabs": True}

    def test_store_string_resolved(self):
        """String store path is wrapped in DatasetStore."""
        with tempfile.TemporaryDirectory() as tmpdir:
            future = load_dataset("data.pkl", load_store=tmpdir)
            assert isinstance(future.load_store, DatasetStore)
            assert future.load_store.base_path == tmpdir

    def test_extras_stored(self):
        future = load_dataset("data_{date}.pkl", date="2024-01-01")
        assert future.kwargs == {"date": "2024-01-01"}

    def test_extras_with_future_dependency(self):
        @task
        def get_date() -> str:
            return "2024-01-01"

        date_future = get_date()
        future = load_dataset("data_{date}.pkl", date=date_future)
        assert future.kwargs["date"] is date_future

    def test_invalid_key_template_raises(self):
        with pytest.raises(ValueError, match="Invalid key template"):
            load_dataset("data_{}.pkl")


class TestDatasetFutureThen:
    """Tests for DatasetFuture.then() chaining."""

    def test_then_returns_task_future(self):
        @task
        def process(data: str) -> str:
            return data.upper()

        future = load_dataset("data.pkl").then(process)
        assert isinstance(future, TaskFuture)

    def test_then_with_extra_kwargs(self):
        @task
        def combine(data: str, suffix: str) -> str:
            return data + suffix

        future = load_dataset("data.pkl").then(combine, suffix="!")
        assert isinstance(future, TaskFuture)

    def test_then_product_returns_map_future(self):
        @task
        def multiply(data: str, factor: int) -> str:
            return data * factor

        future = load_dataset("data.pkl").then_product(multiply, factor=[1, 2, 3])
        assert isinstance(future, MapTaskFuture)

    def test_then_zip_returns_map_future(self):
        @task
        def annotate(data: str, label: str) -> str:
            return f"{label}: {data}"

        future = load_dataset("data.pkl").then_zip(annotate, label=["a", "b"])
        assert isinstance(future, MapTaskFuture)


class TestDatasetFutureSave:
    """Tests for DatasetFuture.save() chaining."""

    def test_save_returns_dataset_future(self):
        future = load_dataset("input.pkl").save("output.pkl")
        assert isinstance(future, DatasetFuture)

    def test_save_preserves_load_key(self):
        future = load_dataset("input.pkl").save("output.pkl")
        assert future.load_key == "input.pkl"


class TestDatasetFutureToGraph:
    """Tests for DatasetFuture.to_graph() conversion."""

    def test_to_graph_returns_dataset_node(self):
        from daglite.graph.nodes import DatasetNode

        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)
            future = load_dataset("data.pkl", load_store=store)
            node = future.to_graph()
            assert isinstance(node, DatasetNode)

    def test_node_has_correct_kind(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)
            future = load_dataset("data.pkl", load_store=store)
            node = future.to_graph()
            assert node.to_metadata().kind == "dataset"

    def test_node_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)
            future = load_dataset("data.pkl", load_store=store)
            node = future.to_graph()
            assert node.name == "load(data.pkl)"

    def test_node_store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)
            future = load_dataset("data.pkl", load_store=store)
            node = future.to_graph()
            assert node.store is store

    def test_node_extras_as_kwargs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)
            future = load_dataset("data_{v}.pkl", load_store=store, v="1.0")
            node = future.to_graph()
            assert "v" in node.kwargs

    def test_node_dependencies_from_future_extras(self):
        @task
        def get_version() -> str:
            return "1.0"

        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)
            version = get_version()
            future = load_dataset("data_{v}.pkl", load_store=store, v=version)
            node = future.to_graph()
            assert version.id in node.dependencies()

    def test_invalid_placeholder_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)
            future = load_dataset("data_{missing}.pkl", load_store=store)
            with pytest.raises(ValueError, match="missing"):
                future.to_graph()

    def test_output_configs_from_save(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)
            future = load_dataset("input.pkl", load_store=store).save(
                "output.pkl", save_store=store
            )
            node = future.to_graph()
            assert len(node.output_configs) == 1
            assert node.output_configs[0].key == "output.pkl"


class TestDatasetFutureToGraphWithSaveExtras:
    """Tests for DatasetFuture.to_graph() when .save() extras contain futures."""

    def test_save_with_future_extra_creates_dependency(self):
        """A future extra on .save() becomes a dependency ref in output config."""

        @task
        def get_version() -> str:
            return "v1"

        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)
            version = get_version()
            future = load_dataset("input.pkl", load_store=store).save(
                "output_{version}.pkl", save_store=store, version=version
            )
            node = future.to_graph()

            assert len(node.output_configs) == 1
            dep = node.output_configs[0].dependencies["version"]
            assert dep.is_ref
            assert dep.ref == version.id

    def test_save_with_plain_extra_creates_value(self):
        """A plain (non-future) extra on .save() becomes a value param."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)
            future = load_dataset("input.pkl", load_store=store).save(
                "output_{label}.pkl", save_store=store, label="batch1"
            )
            node = future.to_graph()

            assert len(node.output_configs) == 1
            dep = node.output_configs[0].dependencies["label"]
            assert not dep.is_ref
            assert dep.value == "batch1"


class TestDatasetNodeOutputDependencies:
    """Tests for DatasetNode.dependencies() with output config refs."""

    def test_output_config_future_deps_included(self):
        """Dependencies from output configs are included in DatasetNode.dependencies()."""
        from daglite.graph.base import OutputConfig
        from daglite.graph.base import ParamInput
        from daglite.graph.nodes import DatasetNode

        dep_id_1 = uuid4()
        dep_id_2 = uuid4()
        output_config = OutputConfig(
            key="out_{v}_{w}.pkl",
            name=None,
            format=None,
            store=None,
            dependencies={
                "v": ParamInput.from_ref(dep_id_1),
                "w": ParamInput.from_ref(dep_id_2),
                "x": ParamInput.from_value("static"),
            },
            options={},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)
            node = DatasetNode(
                id=uuid4(),
                name="test",
                store=store,
                load_key="data.pkl",
                kwargs={},
                output_configs=(output_config,),
            )
            deps = node.dependencies()
            assert dep_id_1 in deps
            assert dep_id_2 in deps


class TestResolveDefaultStore:
    """Tests for DatasetFuture._resolve_default_store() fallback."""

    def test_fallback_to_global_string_store(self):
        """When no load_store is provided, falls back to global settings (string path)."""
        from daglite.settings import DagliteSettings
        from daglite.settings import set_global_settings

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                set_global_settings(DagliteSettings(datastore_store=tmpdir))
                future = load_dataset("data.pkl")
                node = future.to_graph()
                assert isinstance(node.store, DatasetStore)
                assert node.store.base_path == tmpdir
            finally:
                set_global_settings(DagliteSettings())

    def test_fallback_to_global_datasetstore_instance(self):
        """When global settings has a DatasetStore instance, it is used directly."""
        from daglite.settings import DagliteSettings
        from daglite.settings import set_global_settings

        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)
            try:
                set_global_settings(DagliteSettings(datastore_store=store))
                future = load_dataset("data.pkl")
                node = future.to_graph()
                assert node.store is store
            finally:
                set_global_settings(DagliteSettings())


class TestDatasetLoadLogging:
    """Tests for dataset load hook implementations in the logging plugin.

    These tests mock ``_apply_logging_config`` to avoid calling
    ``logging.config.dictConfig`` which would globally reconfigure the
    logging system and pollute subsequent tests (e.g. centralized logging
    integration tests that rely on clean handler state).
    """

    @pytest.fixture(autouse=True)
    def _no_dictconfig(self):
        """Prevent LifecycleLoggingPlugin from applying global dictConfig."""
        from unittest.mock import patch as _patch

        with _patch("daglite.plugins.builtin.logging.LifecycleLoggingPlugin._apply_logging_config"):
            yield

    def test_before_dataset_load_with_format(self):
        from unittest.mock import patch as _patch

        from daglite.plugins.builtin.logging import LifecycleLoggingPlugin

        plugin = LifecycleLoggingPlugin()
        with _patch.object(plugin._logger, "debug") as mock_debug:
            plugin.before_dataset_load(
                key="data.csv", return_type=dict, format="pandas/csv", options=None
            )
        msg = mock_debug.call_args[0][0]
        assert "data.csv" in msg
        assert "(format=pandas/csv)" in msg

    def test_before_dataset_load_without_format(self):
        from unittest.mock import patch as _patch

        from daglite.plugins.builtin.logging import LifecycleLoggingPlugin

        plugin = LifecycleLoggingPlugin()
        with _patch.object(plugin._logger, "debug") as mock_debug:
            plugin.before_dataset_load(key="data.pkl", return_type=dict, format=None, options=None)
        msg = mock_debug.call_args[0][0]
        assert "data.pkl" in msg
        assert "format=" not in msg

    def test_after_dataset_load_with_format(self):
        from unittest.mock import patch as _patch

        from daglite.plugins.builtin.logging import LifecycleLoggingPlugin

        plugin = LifecycleLoggingPlugin()
        with _patch.object(plugin._logger, "info") as mock_info:
            plugin.after_dataset_load(
                key="data.csv",
                return_type=dict,
                format="pandas/csv",
                options=None,
                result={"data": 1},
                duration=0.123,
            )
        msg = mock_info.call_args[0][0]
        assert "data.csv" in msg
        assert "(format=pandas/csv)" in msg

    def test_after_dataset_load_without_format(self):
        from unittest.mock import patch as _patch

        from daglite.plugins.builtin.logging import LifecycleLoggingPlugin

        plugin = LifecycleLoggingPlugin()
        with _patch.object(plugin._logger, "info") as mock_info:
            plugin.after_dataset_load(
                key="data.pkl",
                return_type=dict,
                format=None,
                options=None,
                result={"data": 1},
                duration=0.5,
            )
        msg = mock_info.call_args[0][0]
        assert "data.pkl" in msg
        assert "format=" not in msg
