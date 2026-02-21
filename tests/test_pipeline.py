"""
Unit tests for pipeline declaration and loading.

Tests in this file should NOT focus on evaluation. Evaluation tests are in tests/evaluation/.
"""

from __future__ import annotations

import pytest

from daglite import pipeline
from daglite import task
from daglite.futures import TaskFuture
from daglite.pipelines import Pipeline
from daglite.pipelines import load_pipeline


# Test fixtures
@task
def add(x: int, y: int) -> int:  # pragma: no cover
    """Add two numbers."""
    return x + y


@task
def multiply(x: int, factor: int) -> int:  # pragma: no cover
    """Multiply by a factor."""
    return x * factor


class TestPipeline:
    """Tests for the @pipeline decorator."""

    def test_pipeline_decorator_basic(self):
        """Test basic pipeline decorator usage."""

        @pipeline
        def simple_pipeline(x: int, y: int):  # pragma: no cover
            return add(x=x, y=y)

        assert isinstance(simple_pipeline, Pipeline)
        assert simple_pipeline.name == "simple_pipeline"
        assert simple_pipeline.description is None or simple_pipeline.description == ""

    def test_pipeline_decorator_with_docstring(self):
        """Test pipeline decorator preserves docstring."""

        @pipeline
        def documented_pipeline(x: int):  # pragma: no cover
            """This is a documented pipeline."""
            return add(x=x, y=10)

        assert documented_pipeline.description == "This is a documented pipeline."

    def test_pipeline_decorator_with_name(self):
        """Test pipeline decorator with custom name."""

        @pipeline(name="custom_name")
        def my_pipeline(x: int):  # pragma: no cover
            return add(x=x, y=5)

        assert my_pipeline.name == "custom_name"

    def test_pipeline_decorator_with_description(self):
        """Test pipeline decorator with custom description."""

        @pipeline(description="Custom description")
        def my_pipeline(x: int):  # pragma: no cover
            return add(x=x, y=5)

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


class TestPipelineSignature:
    """Tests for Pipeline signature and type inspection methods."""

    def test_pipeline_signature(self):
        """Test that pipeline.signature returns correct signature."""
        from typing import get_type_hints

        @pipeline
        def typed_pipeline(x: int, y: str) -> TaskFuture[int]:  # pragma: no cover
            return add(x=x, y=5)  # type: ignore

        sig = typed_pipeline.signature
        assert "x" in sig.parameters
        assert "y" in sig.parameters

        # Get actual type hints (resolves forward references)
        hints = get_type_hints(typed_pipeline.func)
        assert hints["x"] == int  # noqa: E721
        assert hints["y"] == str  # noqa: E721

    def test_get_typed_params_all_typed(self):
        """Test get_typed_params with fully typed parameters."""

        @pipeline
        def typed_pipeline(x: int, y: str, z: float) -> TaskFuture[int]:  # pragma: no cover
            return add(x=x, y=5)  # type: ignore

        typed_params = typed_pipeline.get_typed_params()
        # With __future__ annotations, these are strings or type objects depending on Python version
        assert "x" in typed_params
        assert "y" in typed_params
        assert "z" in typed_params
        assert all(v is not None for v in typed_params.values())

    def test_get_typed_params_partially_typed(self):
        """Test get_typed_params with partially typed parameters."""

        @pipeline
        def partial_pipeline(x: int, y, z: str):  # pragma: no cover
            return add(x=x, y=5)

        typed_params = partial_pipeline.get_typed_params()
        assert typed_params["x"] is not None  # Has annotation
        assert typed_params["y"] is None  # No annotation
        assert typed_params["z"] is not None  # Has annotation

    def test_get_typed_params_no_types(self):
        """Test get_typed_params with no type annotations."""

        @pipeline
        def untyped_pipeline(x, y):  # pragma: no cover
            return add(x=x, y=y)

        typed_params = untyped_pipeline.get_typed_params()
        assert typed_params == {"x": None, "y": None}

    def test_has_typed_params_all_typed(self):
        """Test has_typed_params returns True when all params typed."""

        @pipeline
        def typed_pipeline(x: int, y: str) -> TaskFuture[int]:  # pragma: no cover
            return add(x=x, y=5)

        assert typed_pipeline.has_typed_params() is True

    def test_has_typed_params_partially_typed(self):
        """Test has_typed_params returns False when some params untyped."""

        @pipeline
        def partial_pipeline(x: int, y) -> TaskFuture[int]:  # pragma: no cover
            return add(x=x, y=5)

        assert partial_pipeline.has_typed_params() is False

    def test_has_typed_params_no_types(self):
        """Test has_typed_params returns False when no params typed."""

        @pipeline
        def untyped_pipeline(x, y):  # pragma: no cover
            return add(x=x, y=y)

        assert untyped_pipeline.has_typed_params() is False
