"""Integration tests for output functionality with evaluate().

Tests the complete output pipeline including .save() (with optional checkpoint) and OutputPlugin.
"""

import tempfile

import pytest

from daglite import evaluate
from daglite import task
from daglite.outputs.store import FileOutputStore
from daglite.plugins.builtin.output import OutputPlugin


class TestOutputPluginIntegration:
    """Integration tests for OutputPlugin with evaluate()."""

    def test_save_with_explicit_store(self):
        """Test .save() with explicit store parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileOutputStore(tmpdir)

            @task
            def add(x: int, y: int) -> int:
                return x + y

            result = add(x=5, y=10).save("result", store=store)
            output = evaluate(result, plugins=[OutputPlugin()])

            assert output == 15
            assert store.load("result", int) == 15

    def test_save_with_plugin_default_store(self):
        """Test .save() using plugin's default store."""
        with tempfile.TemporaryDirectory() as tmpdir:

            @task
            def multiply(a: int, b: int) -> int:
                return a * b

            result = multiply(a=3, b=7).save("product")
            output = evaluate(result, plugins=[OutputPlugin(store=tmpdir)])

            assert output == 21
            assert FileOutputStore(tmpdir).load("product", int) == 21

    def test_save_with_task_level_store(self):
        """Test that task-level store is used by .save()."""
        with tempfile.TemporaryDirectory() as task_dir, tempfile.TemporaryDirectory() as plugin_dir:
            task_store = FileOutputStore(task_dir)

            @task(store=task_store)
            def process(x: int) -> int:
                return x * 2

            result = process(x=5).save("output")
            output = evaluate(result, plugins=[OutputPlugin(store=plugin_dir)])

            assert output == 10
            assert task_store.load("output", int) == 10
            # Verify it was saved to task store, not plugin store
            assert not FileOutputStore(plugin_dir).exists("output")

    def test_save_with_format_string_key(self):
        """Test .save() with parameter formatting in key."""
        with tempfile.TemporaryDirectory() as tmpdir:

            @task
            def process(data_id: str, value: int) -> int:
                return value * 2

            result = process(data_id="abc123", value=10).save("output_{data_id}")
            output = evaluate(result, plugins=[OutputPlugin(store=tmpdir)])

            assert output == 20
            assert FileOutputStore(tmpdir).load("output_abc123", int) == 20

    def test_save_with_checkpoint_creates_named_output(self):
        """Test .save(checkpoint=...) saves output with name."""
        with tempfile.TemporaryDirectory() as tmpdir:

            @task
            def train(model: str) -> str:
                return f"trained_{model}"

            result = train(model="linear").save("checkpoint", checkpoint="model_v1")
            output = evaluate(result, plugins=[OutputPlugin(store=tmpdir)])

            assert output == "trained_linear"
            assert FileOutputStore(tmpdir).load("checkpoint", str) == "trained_linear"

    def test_multiple_saves_on_same_task(self):
        """Test that multiple .save() calls work correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:

            @task
            def compute(x: int) -> int:
                return x**2

            result = compute(x=4).save("v1").save("v2")
            output = evaluate(result, plugins=[OutputPlugin(store=tmpdir)])

            assert output == 16
            store = FileOutputStore(tmpdir)
            assert store.load("v1", int) == 16
            assert store.load("v2", int) == 16

    def test_save_with_literal_extras(self):
        """Test .save() with literal extra parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:

            @task
            def process(x: int) -> int:
                return x * 2

            result = process(x=5).save("output", version="1.0", author="test")
            output = evaluate(result, plugins=[OutputPlugin(store=tmpdir)])

            assert output == 10

    def test_save_with_task_future_extras(self):
        """Test .save() with TaskFuture as extra parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:

            @task
            def get_version() -> str:
                return "v2.0"

            @task
            def train(model: str, version: str) -> str:
                return f"{model}_{version}"

            version_future = get_version()
            result = train(model="linear", version=version_future).save(
                "model.pkl", version=version_future
            )
            output = evaluate(result, plugins=[OutputPlugin(store=tmpdir)])

            assert output == "linear_v2.0"

    def test_save_in_task_chain(self):
        """Test .save() in a chain of tasks."""
        with tempfile.TemporaryDirectory() as tmpdir:

            @task
            def step1(x: int) -> int:
                return x * 2

            @task
            def step2(x: int) -> int:
                return x + 10

            result = step1(x=5).save("step1_output").then(step2)
            output = evaluate(result, plugins=[OutputPlugin(store=tmpdir)])

            assert output == 20
            assert FileOutputStore(tmpdir).load("step1_output", int) == 10

    @pytest.mark.parametrize(
        "key_pattern,input_values,expected_key",
        [
            ("output_{x}", {"x": 5, "y": 1}, "output_5"),
            ("data_{x}_{y}", {"x": 3, "y": 7}, "data_3_7"),
            ("simple", {"x": 10, "y": 1}, "simple"),
        ],
    )
    def test_save_variations(self, key_pattern, input_values, expected_key):
        """Test various combinations of key patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:

            @task
            def compute(x: int, y: int = 1) -> int:
                return x * y

            result = compute(**input_values).save(key_pattern)
            output = evaluate(result, plugins=[OutputPlugin(store=tmpdir)])

            expected_output = input_values["x"] * input_values.get("y", 1)
            assert output == expected_output
            assert FileOutputStore(tmpdir).load(expected_key, int) == expected_output


class TestOutputPluginStringShortcuts:
    """Integration tests for string shortcut conversions."""

    def test_plugin_string_shortcut(self):
        """Test that OutputPlugin converts string to FileOutputStore."""
        with tempfile.TemporaryDirectory() as tmpdir:

            @task
            def process(x: int) -> int:
                return x * 2

            result = process(x=5).save("output")
            output = evaluate(result, plugins=[OutputPlugin(store=tmpdir)])

            assert output == 10
            assert FileOutputStore(tmpdir).load("output", int) == 10

    def test_task_decorator_string_shortcut(self):
        """Test that @task(store=...) converts string to FileOutputStore."""
        with tempfile.TemporaryDirectory() as tmpdir:

            @task(store=tmpdir)
            def process(x: int) -> int:
                return x * 2

            result = process(x=5).save("output")
            output = evaluate(result, plugins=[OutputPlugin()])

            assert output == 10
            assert FileOutputStore(tmpdir).load("output", int) == 10

    def test_save_method_string_shortcut(self):
        """Test that .save(store=...) converts string to FileOutputStore."""
        with tempfile.TemporaryDirectory() as tmpdir:

            @task
            def process(x: int) -> int:
                return x * 2

            result = process(x=5).save("output", store=tmpdir)
            output = evaluate(result, plugins=[OutputPlugin()])

            assert output == 10
            assert FileOutputStore(tmpdir).load("output", int) == 10

    def test_save_with_checkpoint_string_shortcut(self):
        """Test that .save(checkpoint=..., store=...) converts string to FileOutputStore."""
        with tempfile.TemporaryDirectory() as tmpdir:

            @task
            def process(x: int) -> int:
                return x * 2

            result = process(x=5).save("output", checkpoint="step1", store=tmpdir)
            output = evaluate(result, plugins=[OutputPlugin()])

            assert output == 10
            assert FileOutputStore(tmpdir).load("output", int) == 10

    def test_save_with_checkpoint_true_uses_key_as_name(self):
        """Test that .save(checkpoint=True) uses the key as checkpoint name."""
        with tempfile.TemporaryDirectory() as tmpdir:

            @task
            def process(x: int) -> int:
                return x * 2

            result = process(x=5).save("my_checkpoint", checkpoint=True, store=tmpdir)
            output = evaluate(result, plugins=[OutputPlugin()])

            assert output == 10
            assert FileOutputStore(tmpdir).load("my_checkpoint", int) == 10


class TestMapTaskFutureOutputs:
    """Tests for MapTaskFuture with .save() (with optional checkpoint)."""

    def test_map_task_with_save(self):
        """Test MapTaskFuture.save() works correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:

            @task
            def square(x: int) -> int:
                return x * x

            # Create a map operation and save (saves individual results, last one wins)
            result = square.product(x=[1, 2, 3]).save("squares")
            output = evaluate(result, plugins=[OutputPlugin(store=tmpdir)])

            assert output == [1, 4, 9]
            # MapTask saves the last individual result (not the aggregated list)
            assert FileOutputStore(tmpdir).load("squares", int) == 9

    def test_map_task_with_save_and_extras(self):
        """Test MapTaskFuture.save() with literal extras."""
        with tempfile.TemporaryDirectory() as tmpdir:

            @task
            def process(x: int) -> int:
                return x * 2

            # Save with literal extras
            result = process.product(x=[1, 2, 3]).save("results", version="v1.0", author="test")
            output = evaluate(result, plugins=[OutputPlugin(store=tmpdir)])

            assert output == [2, 4, 6]
            # MapTask saves the last individual result
            assert FileOutputStore(tmpdir).load("results", int) == 6

    def test_map_task_with_save_checkpoint(self):
        """Test MapTaskFuture.save(checkpoint=...) works correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:

            @task
            def double(x: int) -> int:
                return x * 2

            result = double.product(x=[1, 2, 3]).save("doubled", checkpoint="processing")
            output = evaluate(result, plugins=[OutputPlugin(store=tmpdir)])

            assert output == [2, 4, 6]
            # MapTask saves the last individual result
            assert FileOutputStore(tmpdir).load("doubled", int) == 6


class TestOutputWithExternalTaskFutureRefs:
    """Tests for output extras with external TaskFuture refs (not task parameters)."""

    def test_save_with_external_version_task(self):
        """Test .save() with version from an external task (not a parameter)."""
        with tempfile.TemporaryDirectory() as tmpdir:

            @task
            def get_version() -> str:
                return "v1.2.3"

            @task
            def process_data(x: int) -> int:
                return x * 2

            version = get_version()
            result = process_data(x=10).save("data", version=version)
            output = evaluate(result, plugins=[OutputPlugin(store=tmpdir)])

            assert output == 20
            # Can load the saved result
            assert FileOutputStore(tmpdir).load("data", int) == 20

    def test_save_with_multiple_external_refs(self):
        """Test .save() with multiple external task refs."""
        with tempfile.TemporaryDirectory() as tmpdir:

            @task
            def get_version() -> str:
                return "v2.0"

            @task
            def get_author() -> str:
                return "alice"

            @task
            def compute(a: int, b: int) -> int:
                return a + b

            version = get_version()
            author = get_author()
            result = compute(a=5, b=7).save("result", version=version, author=author)
            output = evaluate(result, plugins=[OutputPlugin(store=tmpdir)])

            assert output == 12
            assert FileOutputStore(tmpdir).load("result", int) == 12

    def test_save_with_mixed_external_and_param_refs(self):
        """Test .save() with both external refs and parameter refs."""
        with tempfile.TemporaryDirectory() as tmpdir:

            @task
            def get_build_id() -> str:
                return "build-123"

            @task
            def transform(data: int, multiplier: int) -> int:
                return data * multiplier

            build_id = get_build_id()
            result = transform(data=10, multiplier=3).save(
                "output_{data}",
                build_id=build_id,
                multiplier_used=3,  # Literal value
            )
            output = evaluate(result, plugins=[OutputPlugin(store=tmpdir)])

            assert output == 30
            assert FileOutputStore(tmpdir).load("output_10", int) == 30

    def test_save_with_computed_external_ref(self):
        """Test .save() where external ref is result of computation."""
        with tempfile.TemporaryDirectory() as tmpdir:

            @task
            def get_timestamp() -> int:
                return 1234567890

            @task
            def calculate(x: int) -> int:
                return x**2

            timestamp = get_timestamp()
            result = calculate(x=5).save("calculation", timestamp=timestamp)
            output = evaluate(result, plugins=[OutputPlugin(store=tmpdir)])

            assert output == 25
            assert FileOutputStore(tmpdir).load("calculation", int) == 25

    def test_save_checkpoint_with_external_version_ref(self):
        """Test .save(checkpoint=...) with external version task."""
        with tempfile.TemporaryDirectory() as tmpdir:

            @task
            def get_schema_version() -> str:
                return "schema_v3"

            @task
            def validate(data: str) -> str:
                return data.upper()

            schema_version = get_schema_version()
            result = validate(data="test").save(
                "validated_data", checkpoint="validation", schema=schema_version
            )
            output = evaluate(result, plugins=[OutputPlugin(store=tmpdir)])

            assert output == "TEST"
            assert FileOutputStore(tmpdir).load("validated_data", str) == "TEST"

    def test_map_task_with_external_refs_in_extras(self):
        """Test MapTask with external TaskFuture refs in output extras."""
        with tempfile.TemporaryDirectory() as tmpdir:

            @task
            def get_batch_id() -> str:
                return "batch-456"

            @task
            def process_item(x: int) -> int:
                return x * 10

            batch_id = get_batch_id()
            result = process_item.product(x=[1, 2, 3]).save("processed", batch=batch_id)
            output = evaluate(result, plugins=[OutputPlugin(store=tmpdir)])

            assert output == [10, 20, 30]
            # MapTask saves the last individual result
            assert FileOutputStore(tmpdir).load("processed", int) == 30
