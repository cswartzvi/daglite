"""Example workflows for testing."""

from daglite import task
from daglite import workflow


@task
def add(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y


@task
def multiply(x: int, factor: int) -> int:
    """Multiply a number by a factor."""
    return x * factor


@task
def failing_task(x: int) -> int:
    """A task that always raises an exception."""
    raise RuntimeError(f"Intentional failure with input: {x}")


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
        TaskFuture[int]: The result of (x + y) * factor.
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
        TaskFuture: The result of adding x and y (will be string concatenation!).
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
