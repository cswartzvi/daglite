"""Unit tests for key template validation and save() method on futures."""

import pytest

from daglite._validation import check_key_placeholders
from daglite._validation import check_key_template
from daglite.datasets.store import DatasetStore
from daglite.futures import TaskFuture
from daglite.tasks import task


class TestValidateKeyTemplate:
    """Tests for check_key_template() syntax validation."""

    def test_valid_template_no_placeholders(self):
        """Plain strings pass validation."""
        check_key_template("simple_key")

    def test_valid_template_with_placeholders(self):
        """Templates with named placeholders pass."""
        check_key_template("output_{data_id}_{version}")

    def test_valid_template_nested_path(self):
        """Templates with path separators pass."""
        check_key_template("outputs/{data_id}/result.pkl")

    def test_empty_placeholder_raises(self):
        """Empty {} placeholders are rejected."""
        with pytest.raises(ValueError, match="empty placeholder"):
            check_key_template("output_{}")

    def test_malformed_template_unclosed_brace(self):
        """Unclosed brace raises ValueError."""
        with pytest.raises(ValueError, match="Invalid key template"):
            check_key_template("output_{unclosed")

    def test_multiple_valid_placeholders(self):
        """Multiple named placeholders all pass."""
        check_key_template("{a}_{b}_{c}")

    def test_mixed_literal_and_placeholder(self):
        """Mix of literal text and placeholders passes."""
        check_key_template("prefix_{name}_suffix")

    def test_single_placeholder_only(self):
        """A key that is just a single placeholder passes."""
        check_key_template("{key}")


class TestValidateKeyPlaceholders:
    """Tests for check_key_placeholders() placeholder-name validation."""

    def test_all_placeholders_available(self):
        """No error when all placeholders match available names."""
        check_key_placeholders("output_{a}_{b}", {"a", "b", "c"})

    def test_missing_placeholder_raises(self):
        """Missing placeholder raises ValueError."""
        with pytest.raises(ValueError, match="won't be available"):
            check_key_placeholders("output_{missing}", {"a", "b"})

    def test_partial_match_raises(self):
        """One matching and one missing still raises."""
        with pytest.raises(ValueError, match="won't be available"):
            check_key_placeholders("output_{a}_{missing}", {"a", "b"})

    def test_no_placeholders_always_passes(self):
        """A literal key with no placeholders always passes."""
        check_key_placeholders("literal_key", set())

    def test_empty_available_with_placeholder_raises(self):
        """Placeholders with empty available set raises."""
        with pytest.raises(ValueError, match="won't be available"):
            check_key_placeholders("{x}", set())

    def test_error_message_lists_available(self):
        """Error message includes the sorted available variables."""
        with pytest.raises(ValueError, match="Available placeholders: \\['alpha', 'beta'\\]"):
            check_key_placeholders("{gamma}", {"alpha", "beta"})


class TestFutureSave:
    """Tests for the .save() method on TaskFuture."""

    def test_save_returns_self_type(self):
        """save() returns the same type for chaining."""

        @task
        def work(x: int) -> int:
            return x  # pragma: no cover

        future = work(x=1).save("output_{x}")
        assert isinstance(future, TaskFuture)

    def test_save_preserves_id(self):
        """save() preserves the original future's ID."""

        @task
        def work(x: int) -> int:
            return x  # pragma: no cover

        original = work(x=1)
        saved = original.save("output_{x}")
        assert saved.id == original.id

    def test_save_chaining_accumulates_configs(self):
        """Multiple save() calls accumulate output configurations."""

        @task
        def work(x: int) -> int:
            return x  # pragma: no cover

        future = work(x=1).save("out1_{x}").save("out2_{x}")
        assert len(future._output_futures) == 2

    def test_save_with_checkpoint_true(self):
        """save_checkpoint=True uses the key as the checkpoint name."""

        @task
        def work(x: int) -> int:
            return x  # pragma: no cover

        future = work(x=1).save("my_key", save_checkpoint=True)
        assert future._output_futures[0].name == "my_key"

    def test_save_with_checkpoint_string(self):
        """save_checkpoint=str uses the string as checkpoint name."""

        @task
        def work(x: int) -> int:
            return x  # pragma: no cover

        future = work(x=1).save("my_key", save_checkpoint="cp_name")
        assert future._output_futures[0].name == "cp_name"

    def test_save_without_checkpoint(self):
        """save_checkpoint=None means no checkpoint."""

        @task
        def work(x: int) -> int:
            return x  # pragma: no cover

        future = work(x=1).save("my_key")
        assert future._output_futures[0].name is None

    def test_save_with_format(self):
        """save_format is stored in the future output config."""

        @task
        def work(x: int) -> int:
            return x  # pragma: no cover

        future = work(x=1).save("data.pkl", save_format="pickle")
        assert future._output_futures[0].format == "pickle"

    def test_save_with_store_string(self):
        """String save_store is converted to DatasetStore."""

        @task
        def work(x: int) -> int:
            return x  # pragma: no cover

        future = work(x=1).save("data.pkl", save_store="/tmp/test_store")
        assert isinstance(future._output_futures[0].store, DatasetStore)

    def test_save_with_store_instance(self):
        """DatasetStore instance is stored directly."""
        import tempfile

        @task
        def work(x: int) -> int:
            return x  # pragma: no cover

        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)
            future = work(x=1).save("data.pkl", save_store=store)
            assert future._output_futures[0].store is store

    def test_save_falls_back_to_task_store(self):
        """Without explicit save_store, uses task_store."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)

            @task(store=store)
            def work(x: int) -> int:
                return x  # pragma: no cover

            future = work(x=1)
            saved = future.save("data.pkl")
            assert saved._output_futures[0].store is store

    def test_save_with_options(self):
        """save_options are stored in the config."""

        @task
        def work(x: int) -> int:
            return x  # pragma: no cover

        future = work(x=1).save("data.pkl", save_options={"protocol": 5})
        assert future._output_futures[0].options == {"protocol": 5}

    def test_save_with_extras(self):
        """Extra kwargs are stored as extras for key formatting."""

        @task
        def work(x: int) -> int:
            return x  # pragma: no cover

        future = work(x=1).save("output_{x}_{version}", version="v1")
        assert future._output_futures[0].extras == {"version": "v1"}

    def test_save_with_future_as_extra(self):
        """TaskFuture extras are stored for dependency resolution."""

        @task
        def work(x: int) -> int:
            return x  # pragma: no cover

        @task
        def get_version() -> str:
            return "v1"  # pragma: no cover

        future = work(x=1).save("output_{version}", version=get_version())
        assert isinstance(future._output_futures[0].extras["version"], TaskFuture)

    def test_save_rejects_empty_placeholder(self):
        """save() rejects keys with empty {} placeholders."""

        @task
        def work(x: int) -> int:
            return x  # pragma: no cover

        with pytest.raises(ValueError, match="empty placeholder"):
            work(x=1).save("output_{}")

    def test_save_rejects_malformed_template(self):
        """save() rejects malformed key templates."""

        @task
        def work(x: int) -> int:
            return x  # pragma: no cover

        with pytest.raises(ValueError, match="Invalid key template"):
            work(x=1).save("output_{unclosed")


class TestToGraphValidation:
    """Tests for placeholder validation during to_graph()."""

    def test_task_future_to_graph_valid_placeholders(self):
        """to_graph() succeeds when all placeholders match kwargs."""

        @task
        def work(data_id: str, version: str) -> str:
            return ""  # pragma: no cover

        future = work(data_id="abc", version="v1").save("output_{data_id}_{version}")
        node = future.to_graph()
        assert len(node.output_configs) == 1

    def test_task_future_to_graph_invalid_placeholder_raises(self):
        """to_graph() raises when a placeholder doesn't match kwargs or extras."""

        @task
        def work(data_id: str) -> str:
            return ""  # pragma: no cover

        future = work(data_id="abc").save("output_{data_id}_{missing}")
        with pytest.raises(ValueError, match="won't be available"):
            future.to_graph()

    def test_task_future_to_graph_extras_count_as_available(self):
        """Extras provided to save() are available as placeholders."""

        @task
        def work(data_id: str) -> str:
            return ""  # pragma: no cover

        future = work(data_id="abc").save("output_{data_id}_{version}", version="v1")
        node = future.to_graph()
        assert len(node.output_configs) == 1

    def test_map_future_to_graph_valid_placeholders(self):
        """MapTaskFuture.to_graph() validates against fixed + mapped kwargs."""

        @task
        def work(x: int, y: int) -> int:
            return x + y  # pragma: no cover

        future = work.product(x=[1, 2], y=[3, 4]).save("out_{x}_{y}_{iteration_index}")
        node = future.to_graph()
        assert len(node.output_configs) == 1

    def test_map_future_to_graph_invalid_placeholder_raises(self):
        """MapTaskFuture.to_graph() raises for unknown placeholders."""

        @task
        def work(x: int) -> int:
            return x  # pragma: no cover

        future = work.product(x=[1, 2]).save("out_{x}_{missing}")
        with pytest.raises(ValueError, match="won't be available"):
            future.to_graph()

    def test_map_future_iteration_index_available(self):
        """MapTaskFuture has iteration_index as an available placeholder."""

        @task
        def work(x: int) -> int:
            return x  # pragma: no cover

        future = work.product(x=[1, 2]).save("out_{iteration_index}")
        node = future.to_graph()
        assert len(node.output_configs) == 1

    def test_to_graph_no_outputs_passes(self):
        """to_graph() works fine with no save calls."""

        @task
        def work(x: int) -> int:
            return x  # pragma: no cover

        node = work(x=1).to_graph()
        assert len(node.output_configs) == 0

    def test_to_graph_format_threaded(self):
        """save_format is threaded through to OutputConfig.format."""

        @task
        def work(x: int) -> int:
            return x  # pragma: no cover

        future = work(x=1).save("data.pkl", save_format="pickle")
        node = future.to_graph()
        assert node.output_configs[0].format == "pickle"

    def test_to_graph_options_threaded(self):
        """save_options are threaded through to OutputConfig.options."""

        @task
        def work(x: int) -> int:
            return x  # pragma: no cover

        future = work(x=1).save("data.pkl", save_options={"protocol": 5})
        node = future.to_graph()
        assert node.output_configs[0].options == {"protocol": 5}

    def test_to_graph_checkpoint_name_threaded(self):
        """Checkpoint name is threaded through to OutputConfig.name."""

        @task
        def work(x: int) -> int:
            return x  # pragma: no cover

        future = work(x=1).save("out", save_checkpoint="cp")
        node = future.to_graph()
        assert node.output_configs[0].name == "cp"

    def test_to_graph_multiple_outputs(self):
        """Multiple save() calls produce multiple output configs."""

        @task
        def work(x: int) -> int:
            return x  # pragma: no cover

        future = work(x=1).save("out1_{x}").save("out2_{x}")
        node = future.to_graph()
        assert len(node.output_configs) == 2
        assert node.output_configs[0].key == "out1_{x}"
        assert node.output_configs[1].key == "out2_{x}"

    def test_to_graph_future_extra_becomes_dependency(self):
        """TaskFuture extras become InputParam refs in output dependencies."""

        @task
        def work(x: int) -> int:
            return x  # pragma: no cover

        @task
        def get_version() -> str:
            return "v1"  # pragma: no cover

        version_future = get_version()
        future = work(x=1).save("out_{version}", version=version_future)
        node = future.to_graph()
        config = node.output_configs[0]
        assert "version" in config.dependencies
        assert config.dependencies["version"].reference
