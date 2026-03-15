"""
Reusable workflow definitions for daglite tests.

This module provides canonical ``@workflow``-decorated functions used across
unit, integration, and type-assertion tests.
"""

from __future__ import annotations

from daglite import workflow

from .tasks import add
from .tasks import async_add
from .tasks import failing_task
from .tasks import multiply


@workflow
def sync_workflow(x: int, y: int) -> int:
    """Sync workflow for type assertions."""
    return add(x=x, y=y)


@workflow(name="custom_sync_workflow")
def named_sync_workflow(x: int, y: int) -> int:
    """Sync workflow with custom name — exercises decorator-with-kwargs path."""
    return add(x=x, y=y)


@workflow
async def async_workflow(x: int, y: int) -> int:
    """Async workflow for type assertions."""
    return await async_add(x=x, y=y)


@workflow(name="custom_async_workflow")
async def named_async_workflow(x: int, y: int) -> int:
    """Async workflow with custom name — exercises decorator-with-kwargs path."""
    return await async_add(x=x, y=y)


@workflow
def math_workflow(x: int, y: int, factor: int = 2):
    """
    A workflow demonstrating basic arithmetic operations.

    This workflow adds two numbers and multiplies the result by a factor.

    Args:
        x: First number to add.
        y: Second number to add.
        factor: Multiplication factor (default: 2).

    Returns:
        The result of (x + y) * factor.
    """
    sum_result = add(x=x, y=y)
    return multiply(x=sum_result, factor=factor)


@workflow
def untyped_workflow(x, y):  # noqa: ANN001
    """
    A workflow with untyped parameters for testing warnings.

    This workflow demonstrates what happens when parameters lack type annotations.
    The CLI will issue a warning and treat all values as strings.

    Args:
        x: First number (untyped).
        y: Second number (untyped).

    Returns:
        The result of adding x and y.
    """
    # Note: Since x and y are untyped strings from CLI, we need to handle them
    # This is intentionally problematic to demonstrate the issue
    return add(x=int(x) if isinstance(x, str) else x, y=int(y) if isinstance(y, str) else y)


@workflow
def failing_workflow(x: int):
    """A workflow that will fail during execution."""
    return failing_task(x=x)


@workflow
def empty_workflow():
    """A workflow with no parameters for testing output branches."""
    return add(x=1, y=2)


@workflow
def verbose_workflow():
    """
    This is a workflow with a very long description that exceeds the display column limit and
    should be truncated when listed."""
    return add(x=1, y=2)


@workflow
def no_description_workflow():
    return add(x=1, y=2)
