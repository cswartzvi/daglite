"""Tests for the workflow decorator, workflow types, and load_workflow."""

from __future__ import annotations

import asyncio

import pytest

from daglite import workflow
from daglite.workflows import AsyncWorkflow
from daglite.workflows import SyncWorkflow
from daglite.workflows import load_workflow

from .examples.workflows import async_workflow
from .examples.workflows import math_workflow
from .examples.workflows import named_async_workflow
from .examples.workflows import named_sync_workflow
from .examples.workflows import sync_workflow
from .examples.workflows import untyped_workflow


class TestWorkflowInstanceTypes:
    """`@workflow` produces the correct concrete workflow type."""

    def test_sync_workflow(self) -> None:
        assert isinstance(sync_workflow, SyncWorkflow)

    def test_named_sync_workflow(self) -> None:
        assert isinstance(named_sync_workflow, SyncWorkflow)

    def test_async_workflow(self) -> None:
        assert isinstance(async_workflow, AsyncWorkflow)

    def test_named_async_workflow(self) -> None:
        assert isinstance(named_async_workflow, AsyncWorkflow)


class TestWorkflowProperties:
    """Workflows expose the expected metadata properties."""

    def test_name_from_function(self) -> None:
        assert sync_workflow.name == "sync_workflow"

    def test_name_from_decorator(self) -> None:
        assert named_sync_workflow.name == "custom_sync_workflow"

    def test_async_name_from_decorator(self) -> None:
        assert named_async_workflow.name == "custom_async_workflow"

    def test_description_from_docstring(self) -> None:
        assert "Sync workflow" in sync_workflow.description

    def test_signature_preserved(self) -> None:
        params = list(math_workflow.signature.parameters.keys())
        assert params == ["x", "y", "factor"]


class TestTypedParams:
    """get_typed_params and has_typed_params work correctly."""

    def test_typed_params(self) -> None:
        params = sync_workflow.get_typed_params()
        assert params == {"x": int, "y": int}

    def test_has_typed_params_true(self) -> None:
        assert sync_workflow.has_typed_params() is True

    def test_has_typed_params_false(self) -> None:
        assert untyped_workflow.has_typed_params() is False

    def test_untyped_params_are_none(self) -> None:
        params = untyped_workflow.get_typed_params()
        assert params == {"x": None, "y": None}

    def test_get_typed_params_exception_fallback(self) -> None:
        """get_typed_params falls back to empty hints on exception."""
        from unittest.mock import patch

        def normal_fn(x: int) -> None:
            pass

        wf = SyncWorkflow(func=normal_fn, name="test", description="")

        with patch("daglite.workflows.typing.get_type_hints", side_effect=Exception("fail")):
            params = wf.get_typed_params()
        assert params == {"x": None}


class TestDecoratorForms:
    """The @workflow decorator works with and without parentheses."""

    def test_bare_decorator(self) -> None:
        @workflow
        def bare(x: int) -> int:
            return x

        assert isinstance(bare, SyncWorkflow)
        assert bare.name == "bare"

    def test_parens_no_args(self) -> None:
        @workflow()
        def with_parens(x: int) -> int:
            return x

        assert isinstance(with_parens, SyncWorkflow)

    def test_custom_name(self) -> None:
        @workflow(name="custom")
        def named(x: int) -> int:
            return x

        assert named.name == "custom"

    def test_custom_description(self) -> None:
        @workflow(description="A custom description")
        def described(x: int) -> int:
            return x

        assert described.description == "A custom description"

    def test_async_bare(self) -> None:
        @workflow
        async def async_bare(x: int) -> int:
            return x

        assert isinstance(async_bare, AsyncWorkflow)


class TestDecoratorValidation:
    """The @workflow decorator rejects invalid targets."""

    def test_rejects_class(self) -> None:
        with pytest.raises(TypeError, match="callable functions"):

            @workflow  # type: ignore[arg-type]
            class NotAFunction:
                pass

    def test_rejects_sync_generator(self) -> None:
        with pytest.raises(TypeError, match="generator functions"):

            @workflow  # type: ignore[arg-type]
            def gen_wf():
                yield 1

    def test_rejects_async_generator(self) -> None:
        with pytest.raises(TypeError, match="generator functions"):

            @workflow  # type: ignore[arg-type]
            async def async_gen_wf():
                yield 1


class TestWorkflowExecution:
    """Workflows execute and return correct results."""

    def test_sync_call(self) -> None:
        assert sync_workflow(x=2, y=3) == 5

    def test_sync_with_defaults(self) -> None:
        assert math_workflow(x=2, y=3) == 10  # (2+3)*2

    def test_sync_override_default(self) -> None:
        assert math_workflow(x=2, y=3, factor=3) == 15  # (2+3)*3

    def test_async_call(self) -> None:
        result = asyncio.run(async_workflow(x=2, y=3))
        assert result == 5

    def test_async_workflow_non_awaitable(self) -> None:
        """AsyncWorkflow._run with a sync function returns result directly."""

        def sync_fn(x: int) -> int:
            return x * 2

        wf = AsyncWorkflow(func=sync_fn, name="test", description="")
        result = asyncio.run(wf(3))
        assert result == 6


class TestLoadWorkflow:
    """load_workflow loads workflows from dotted module paths."""

    def test_valid_path(self) -> None:
        wf = load_workflow("tests.examples.workflows.math_workflow")
        assert isinstance(wf, SyncWorkflow)
        assert wf.name == "math_workflow"

    def test_async_path(self) -> None:
        wf = load_workflow("tests.examples.workflows.async_workflow")
        assert isinstance(wf, AsyncWorkflow)

    def test_invalid_no_dot(self) -> None:
        with pytest.raises(ValueError, match="Invalid workflow path"):
            load_workflow("nodot")

    def test_module_not_found(self) -> None:
        with pytest.raises(ModuleNotFoundError):
            load_workflow("nonexistent.module.wf")

    def test_attr_not_found(self) -> None:
        with pytest.raises(AttributeError, match="not found"):
            load_workflow("tests.examples.workflows.no_such_workflow")

    def test_not_a_workflow(self) -> None:
        with pytest.raises(TypeError, match="not a Workflow"):
            load_workflow("tests.examples.tasks.double")
