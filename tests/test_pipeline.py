"""
Unit tests for pipeline declaration and loading.

Tests in this file should NOT focus on evaluation. Evaluation tests are in tests/evaluation/.
"""

from __future__ import annotations

import pytest

from daglite import pipeline
from daglite import task
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
