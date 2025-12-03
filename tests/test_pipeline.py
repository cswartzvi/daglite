"""Tests for pipeline functionality."""

from __future__ import annotations

import pytest

from daglite import pipeline
from daglite import task
from daglite.pipelines import Pipeline
from daglite.pipelines import load_pipeline
from daglite.tasks import MapTaskFuture
from daglite.tasks import TaskFuture


# Test fixtures
@task
def add(x: int, y: int) -> int:  # pragma: no cover
    """Add two numbers."""
    return x + y


@task
def multiply(x: int, factor: int) -> int:  # pragma: no cover
    """Multiply by a factor."""
    return x * factor


class TestPipelineDecorator:
    """Tests for the @pipeline decorator."""

    def test_pipeline_decorator_basic(self):
        """Test basic pipeline decorator usage."""

        @pipeline
        def simple_pipeline(x: int, y: int):  # pragma: no cover
            return add.bind(x=x, y=y)

        assert isinstance(simple_pipeline, Pipeline)
        assert simple_pipeline.name == "simple_pipeline"
        assert simple_pipeline.description is None or simple_pipeline.description == ""

    def test_pipeline_decorator_with_docstring(self):
        """Test pipeline decorator preserves docstring."""

        @pipeline
        def documented_pipeline(x: int):  # pragma: no cover
            """This is a documented pipeline."""
            return add.bind(x=x, y=10)

        assert documented_pipeline.description == "This is a documented pipeline."

    def test_pipeline_decorator_with_name(self):
        """Test pipeline decorator with custom name."""

        @pipeline(name="custom_name")
        def my_pipeline(x: int):  # pragma: no cover
            return add.bind(x=x, y=5)

        assert my_pipeline.name == "custom_name"

    def test_pipeline_decorator_with_description(self):
        """Test pipeline decorator with custom description."""

        @pipeline(description="Custom description")
        def my_pipeline(x: int):  # pragma: no cover
            return add.bind(x=x, y=5)

        assert my_pipeline.description == "Custom description"

    def test_pipeline_decorator_rejects_non_callable(self):
        """Test that pipeline decorator rejects non-callable objects."""
        with pytest.raises(TypeError, match="can only be applied to callable functions"):
            pipeline(42)  # pyright: ignore

    def test_pipeline_decorator_rejects_class(self):
        """Test that pipeline decorator rejects classes."""
        with pytest.raises(TypeError, match="can only be applied to callable functions"):

            @pipeline
            class NotAPipeline:
                pass


class TestPipelineClass:
    """Tests for the Pipeline class."""

    def test_pipeline_callable(self):
        """Test that pipeline instances are callable."""

        @pipeline
        def simple_pipeline(x: int, y: int):
            return add.bind(x=x, y=y)

        result = simple_pipeline(5, 10)
        assert isinstance(result, TaskFuture)

    def test_pipeline_signature(self):
        """Test pipeline signature property."""

        @pipeline
        def my_pipeline(x: int, y: int, z: str = "default"):  # pragma: no cover
            return add.bind(x=x, y=y)

        sig = my_pipeline.signature
        assert "x" in sig.parameters
        assert "y" in sig.parameters
        assert "z" in sig.parameters
        assert sig.parameters["z"].default == "default"

    def test_pipeline_get_typed_params(self):
        """Test getting typed parameters."""

        @pipeline
        def typed_pipeline(x: int, y: float, z: str):  # pragma: no cover
            return add.bind(x=x, y=int(y))

        typed_params = typed_pipeline.get_typed_params()
        # Annotations can be strings due to __future__ annotations
        assert set(typed_params.keys()) == {"x", "y", "z"}
        assert all(v is not None for v in typed_params.values())

    def test_pipeline_get_typed_params_with_untyped(self):
        """Test getting typed parameters with some untyped."""

        @pipeline
        def mixed_pipeline(x: int, y, z: str):  # noqa: ANN001 # pragma: no cover
            return add.bind(x=x, y=0)

        typed_params = mixed_pipeline.get_typed_params()
        # Annotations can be strings due to __future__ annotations
        assert set(typed_params.keys()) == {"x", "y", "z"}
        assert typed_params["y"] is None
        assert typed_params["x"] is not None
        assert typed_params["z"] is not None

    def test_pipeline_has_typed_params_all_typed(self):
        """Test has_typed_params returns True when all params are typed."""

        @pipeline
        def typed_pipeline(x: int, y: float):  # pragma: no cover
            return add.bind(x=x, y=int(y))

        assert typed_pipeline.has_typed_params() is True

    def test_pipeline_has_typed_params_some_untyped(self):
        """Test has_typed_params returns False when some params are untyped."""

        @pipeline
        def mixed_pipeline(x: int, y):  # noqa: ANN001 # pragma: no cover
            return add.bind(x=x, y=y)

        assert mixed_pipeline.has_typed_params() is False

    def test_pipeline_has_typed_params_none_typed(self):
        """Test has_typed_params returns False when no params are typed."""

        @pipeline
        def untyped_pipeline(x, y):  # noqa: ANN001  # pragma: no cover
            return add.bind(x=x, y=y)

        assert untyped_pipeline.has_typed_params() is False


class TestPipelineReturnsTaskFuture:
    """Tests for pipelines returning TaskFuture."""

    def test_pipeline_returns_task_future(self):
        """Test pipeline that returns a TaskFuture."""

        @pipeline
        def simple_pipeline(x: int, y: int):
            return add.bind(x=x, y=y)

        result = simple_pipeline(5, 10)
        assert isinstance(result, TaskFuture)

    def test_pipeline_with_chained_tasks(self):
        """Test pipeline with chained tasks."""

        @pipeline
        def chained_pipeline(x: int, y: int, factor: int):
            sum_result = add.bind(x=x, y=y)
            return multiply.bind(x=sum_result, factor=factor)

        result = chained_pipeline(5, 10, 2)
        assert isinstance(result, TaskFuture)


class TestPipelineReturnsMapTaskFuture:
    """Tests for pipelines returning MapTaskFuture."""

    def test_pipeline_returns_map_task_future(self):
        """Test pipeline that returns a MapTaskFuture."""

        @pipeline
        def extend_pipeline(values: list[int]):
            return add.product(x=values, y=[1, 2, 3])

        result = extend_pipeline([10, 20])
        assert isinstance(result, MapTaskFuture)

    def test_pipeline_with_map_and_join(self):
        """Test pipeline with map and join operations."""

        @task
        def sum_all(values: list[int]) -> int:  # pragma: no cover
            return sum(values)

        @pipeline
        def map_reduce_pipeline(values: list[int]):
            doubled = multiply.product(x=values, factor=[2])
            return doubled.join(sum_all)

        result = map_reduce_pipeline([1, 2, 3, 4])
        assert isinstance(result, TaskFuture)


class TestLoadPipeline:
    """Tests for load_pipeline function."""

    def test_load_pipeline_invalid_path_no_dot(self):
        """Test load_pipeline with invalid path (no dot)."""
        with pytest.raises(ValueError, match="Invalid pipeline path"):
            load_pipeline("invalid")

    def test_load_pipeline_module_not_found(self):
        """Test load_pipeline with non-existent module."""
        with pytest.raises(ModuleNotFoundError):
            load_pipeline("nonexistent.module.pipeline")

    def test_load_pipeline_attribute_not_found(self):
        """Test load_pipeline with non-existent attribute."""
        with pytest.raises(AttributeError, match="not found in module"):
            load_pipeline("daglite.nonexistent_pipeline")

    def test_load_pipeline_not_a_pipeline(self):
        """Test load_pipeline with non-Pipeline object."""
        with pytest.raises(TypeError, match="is not a Pipeline"):
            load_pipeline("daglite.task")  # task is a function, not a Pipeline

    def test_load_pipeline_success(self):
        """Test successfully loading a pipeline."""
        # Load from examples
        pipeline_obj = load_pipeline("tests.examples.pipelines.math_pipeline")
        assert isinstance(pipeline_obj, Pipeline)
        assert pipeline_obj.name == "math_pipeline"
