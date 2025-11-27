from daglite.backends.local import SequentialBackend
from daglite.backends.local import ThreadBackend
from daglite.tasks import FixedParamTask
from daglite.tasks import Task
from daglite.tasks import task


class TestTaskValidDefinitions:
    """
    Test the @task decorator with valid task definitions.

    NOTE: Tests in this class should focus on valid usage of the @task decorator and not
    evaluation of the tasks themselves. See tests_evaluation.py for tests related to task
    evaluation.
    """

    def test_task_decorator_with_defaults(self) -> None:
        """Decorating a function without parameters uses sensible defaults."""

        @task
        def add(x: int, y: int) -> int:
            """Simple addition function."""
            return x + y

        assert isinstance(add, Task)
        assert add.name == "add"
        assert add.description == "Simple addition function."
        assert isinstance(add.backend, SequentialBackend)
        assert add.func(1, 2) == 3

    def test_task_decorator_with_params(self) -> None:
        """Decorator accepts custom name, description, and backend configuration."""

        @task(name="custom_add", description="Custom addition task", backend=ThreadBackend())
        def add(x: int, y: int) -> int:  # pragma: no cover
            """Not used docstring."""
            return x + y

        assert add.name == "custom_add"
        assert add.description == "Custom addition task"
        assert isinstance(add.backend, ThreadBackend)

    def test_fixed_param_task(self) -> None:
        """Fixing parameters creates a partially bound task."""

        @task
        def multiply(x: int, y: int) -> int:
            """Simple multiplication function."""
            return x * y

        fixed_task = multiply.fix(y=5)

        assert isinstance(fixed_task, FixedParamTask)
        assert isinstance(fixed_task.task, Task)
        assert fixed_task.task.func(2, 5) == 10

    def test_fixed_param_task_with_params(self) -> None:
        """Fixing parameters preserves task metadata."""

        @task(name="multiply_task", description="Multiplication task")
        def multiply(x: int, y: int) -> int:
            """Simple multiplication function."""
            return x * y

        fixed_task = multiply.fix(y=10)

        assert isinstance(fixed_task, FixedParamTask)
        assert fixed_task.task.name == "multiply_task"
        assert fixed_task.task.description == "Multiplication task"
        assert fixed_task.task.func(3, 10) == 30

    def test_task_with_options(self) -> None:
        """Task metadata can be updated after creation."""

        @task
        def power(base: int, exponent: int) -> int:
            """Simple power function."""
            return base**exponent

        task_with_options = power.with_options(
            name="power_task", description="Power calculation task"
        )

        assert task_with_options.name == "power_task"
        assert task_with_options.description == "Power calculation task"
        assert task_with_options.func(2, 3) == 8
