import pytest

from daglite.backends.local import SequentialBackend
from daglite.backends.local import ThreadBackend
from daglite.exceptions import ParameterError
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
        """@task decorator creates a Task subclass."""

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
        """@task decorator with parameters creates Task subclass."""

        @task(name="custom_add", description="Custom addition task", backend=ThreadBackend())
        def add(x: int, y: int) -> int:  # pragma: no cover
            """Not used docstring."""
            return x + y

        assert add.name == "custom_add"
        assert add.description == "Custom addition task"
        assert isinstance(add.backend, ThreadBackend)

    def test_fixed_param_task(self) -> None:
        """FixedParamTask creates a task with fixed parameters."""

        @task
        def multiply(x: int, y: int) -> int:
            """Simple multiplication function."""
            return x * y

        fixed_task = multiply.fix(y=5)

        assert isinstance(fixed_task, FixedParamTask)
        assert isinstance(fixed_task.task, Task)
        assert fixed_task.task.func(2, 5) == 10

    def test_fixed_param_task_with_params(self) -> None:
        """FixedParamTask with parameters."""

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
        """Task created with options."""

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


class TestTaskInvalidDefinitions:
    """
    Test the @task decorator with invalid task definitions.

    NOTE: Tests in this class should focus on invalid usage of the @task decorator and not
    evaluation of the tasks themselves. See tests_evaluation.py for tests related to task
    evaluation.
    """

    def test_task_decorator_with_non_callable(self) -> None:
        """@task decorator raises TypeError for non-callable."""

        with pytest.raises(TypeError, match="can only be applied to callable functions"):

            @task
            class NotCallable:
                pass

    def test_task_bind_with_invalid_params(self) -> None:
        """.bind() raises ParameterError for invalid parameter types."""

        @task
        def subtract(x: int, y: int) -> int:  # pragma: no cover
            """Simple subtraction function."""
            return x - y

        with pytest.raises(ParameterError, match="Invalid parameters for task"):
            subtract.bind(z=10)

    def test_task_bind_with_missing_params(self) -> None:
        """.bind() raises ParameterError for missing required parameters."""

        @task
        def power(base: int, exponent: int) -> int:  # pragma: no cover
            """Simple power function."""
            return base**exponent

        with pytest.raises(ParameterError, match="Missing parameters for task"):
            power.bind(base=2)

    def test_task_bind_with_overlapping_params(self) -> None:
        """.bind() raises ParameterError for overlapping parameters."""

        @task
        def multiply(x: int, y: int) -> int:  # pragma: no cover
            """Simple multiplication function."""
            return x * y

        fixed = multiply.fix(x=4)

        with pytest.raises(ParameterError, match="Overlapping parameters"):
            fixed.bind(x=5, y=10)

    def test_task_fix_with_invalid_params(self) -> None:
        """.fix() raises TypeError for invalid parameter types."""

        @task
        def divide(x: int, y: int) -> float:  # pragma: no cover
            """Simple division function."""
            return x / y

        with pytest.raises(ParameterError, match="Invalid parameters for task"):
            divide.fix(z=5)

    def test_task_extend_with_non_iterable_params(self) -> None:
        """.extend() raises ParameterError for overlapping parameters."""

        @task
        def add(x: int, y: int) -> int:  # pragma: no cover
            """Simple addition function."""
            return x + y

        with pytest.raises(ParameterError, match="Non-iterable parameters"):
            add.extend(y=20, z=5)

    def test_task_extend_with_overlapping_params(self) -> None:
        """.extend() raises ParameterError for overlapping parameters."""

        @task
        def multiply(x: int, y: int) -> int:  # pragma: no cover
            """Simple multiplication function."""
            return x * y

        fixed = multiply.fix(x=3)

        with pytest.raises(ParameterError, match="Overlapping parameters"):
            fixed.extend(y=[1, 2, 3], x=[4, 5, 6])

    def test_task_extend_invalid_params(self) -> None:
        """FixedParamTask raises ParameterError for invalid fixed parameters."""

        @task
        def subtract(x: int, y: int) -> int:  # pragma: no cover
            """Simple subtraction function."""
            return x - y

        with pytest.raises(ParameterError, match="Invalid parameters"):
            subtract.extend(z=[10, 2, 3])

    def test_task_extend_missing_params(self) -> None:
        """FixedParamTask raises ParameterError for missing fixed parameters."""

        @task
        def power(base: int, exponent: int) -> int:  # pragma: no cover
            """Simple power function."""
            return base**exponent

        fixed = power.fix(base=2)

        with pytest.raises(ParameterError, match="Missing parameters"):
            fixed.extend()

    def test_task_zip_with_non_iterable_params(self) -> None:
        """.zip() raises ParameterError for non-iterable parameters."""

        @task
        def divide(x: int, y: int) -> float:  # pragma: no cover
            """Simple division function."""
            return x / y

        with pytest.raises(ParameterError, match="Non-iterable parameters"):
            divide.zip(x=10, y=5)

    def test_task_zip_with_overlapping_params(self) -> None:
        """.zip() raises ParameterError for overlapping parameters."""

        @task
        def add(x: int, y: int) -> int:  # pragma: no cover
            """Simple addition function."""
            return x + y

        fixed = add.fix(y=2)

        with pytest.raises(ParameterError, match="Overlapping parameters"):
            fixed.zip(y=[3, 4, 5], x=[1, 2, 3])

    def test_task_zip_invalid_params(self) -> None:
        """FixedParamTask raises ParameterError for invalid fixed parameters."""

        @task
        def multiply(x: int, y: int) -> int:  # pragma: no cover
            """Simple multiplication function."""
            return x * y

        with pytest.raises(ParameterError, match="Invalid parameters"):
            multiply.zip(z=[10, 2, 3])

    def test_task_zip_missing_params(self) -> None:
        """FixedParamTask raises ParameterError for missing fixed parameters."""

        @task
        def subtract(x: int, y: int) -> int:  # pragma: no cover
            """Simple subtraction function."""
            return x - y

        fixed = subtract.fix(x=10)

        with pytest.raises(ParameterError, match="Missing parameters"):
            fixed.zip()

    def test_task_map_wiht_invalid_signature(self) -> None:
        """.map() raises ParameterError for invalid parameter types."""

        @task
        def prepare(data: int) -> int:  # pragma: no cover
            """Simple prepare function."""
            return data + 1

        @task
        def mapping(a: int, b: int) -> int:  # pragma: no cover
            return a + b

        prepared = prepare.extend(data=[1, 2, 3])
        with pytest.raises(ParameterError, match="must have exactly one parameter"):
            prepared.map(mapping)

    def test_fixed_task_map_with_invalid_signature(self) -> None:
        """.map() on FixedParamTask raises ParameterError for invalid parameter types."""

        @task
        def prepare(data: int) -> int:  # pragma: no cover
            """Simple prepare function."""
            return data + 1

        @task
        def mapping(a: int, b: int, c: int) -> int:  # pragma: no cover
            return a + b + c

        prepared = prepare.extend(data=[1, 2, 3])
        fixed_mapping = mapping.fix(c=20)
        with pytest.raises(ParameterError, match="must have exactly one unbound parameter"):
            prepared.map(fixed_mapping)

    def test_task_join_with_invalid_signature(self) -> None:
        """.join() raises ParameterError for invalid parameter types."""

        @task
        def prepare(data: int) -> int:  # pragma: no cover
            """Simple prepare function."""
            return data + 1

        @task
        def mapping(a: int) -> int:  # pragma: no cover
            return a * 2

        @task
        def joining(a: int, b: int) -> int:  # pragma: no cover
            return a * 2

        prepared = prepare.extend(data=[1, 2, 3])
        mapped = prepared.map(mapping)
        with pytest.raises(ParameterError, match="must have exactly one parameter"):
            mapped.join(joining)

    def test_fixed_task_join_with_invalid_signature(self) -> None:
        """.join() on FixedParamTask raises ParameterError for invalid parameter types."""

        @task
        def prepare(data: int) -> int:  # pragma: no cover
            """Simple prepare function."""
            return data + 1

        @task
        def mapping(a: int) -> int:  # pragma: no cover
            return a * 2

        @task
        def joining(a: int, b: int, c: int) -> int:  # pragma: no cover
            return a + b + c

        prepared = prepare.extend(data=[1, 2, 3])
        mapped = prepared.map(mapping)
        fixed_joining = joining.fix(c=10)
        with pytest.raises(ParameterError, match="must have exactly one unbound parameter"):
            mapped.join(fixed_joining)


class TestBaseTaskFuture:
    """Test core BaseTaskFuture behavior."""

    def test_futures_have_unique_ids(self) -> None:
        """BaseTaskFuture instances have unique IDs."""

        def add(x: int, y: int) -> int:
            return x + y

        future1 = task(add).bind(x=1, y=2)
        future2 = task(add).bind(x=1, y=2)

        assert future1.id != future2.id

    def test_future_len_raises_type_error(self) -> None:
        """BaseTaskFuture.__len__() raises TypeError."""

        def multiply(x: int, y: int) -> int:
            return x * y

        future = task(multiply).bind(x=3, y=4)

        with pytest.raises(TypeError, match="do not support len()"):
            len(future)

    def test_future_bool_raises_type_error(self) -> None:
        """BaseTaskFuture.__bool__() raises TypeError."""

        def divide(x: int, y: int) -> float:
            return x / y

        future = task(divide).bind(x=10, y=2)

        with pytest.raises(TypeError, match="cannot be used in boolean context."):
            if future:
                pass
