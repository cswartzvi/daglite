import pytest

from daglite.exceptions import ParameterError
from daglite.tasks import task


class TestInvalidTaskFutureDefinitions:
    """
    Test the @task decorator with invalid task definitions.

    NOTE: Tests in this class should focus on invalid usage of the @task decorator and not
    evaluation of the tasks themselves. See tests_evaluation.py for tests related to task
    evaluation.
    """

    def test_task_decorator_with_non_callable(self) -> None:
        """Decorator rejects non-callable objects."""

        with pytest.raises(TypeError, match="can only be applied to callable functions"):

            @task
            class NotCallable:
                pass

    def test_task_bind_with_invalid_params(self) -> None:
        """Binding fails when given parameters that don't exist."""

        @task
        def subtract(x: int, y: int) -> int:  # pragma: no cover
            """Simple subtraction function."""
            return x - y

        with pytest.raises(ParameterError, match="Invalid parameters for task"):
            subtract.bind(z=10)

    def test_task_bind_with_missing_params(self) -> None:
        """Binding fails when required parameters are omitted."""

        @task
        def power(base: int, exponent: int) -> int:  # pragma: no cover
            """Simple power function."""
            return base**exponent

        with pytest.raises(ParameterError, match="Missing parameters for task"):
            power.bind(base=2)

    def test_task_bind_with_overlapping_params(self) -> None:
        """Binding fails when attempting to rebind already-fixed parameters."""

        @task
        def multiply(x: int, y: int) -> int:  # pragma: no cover
            """Simple multiplication function."""
            return x * y

        fixed = multiply.fix(x=4)

        with pytest.raises(ParameterError, match="Overlapping parameters"):
            fixed.bind(x=5, y=10)

    def test_task_then_with_invalid_params(self) -> None:
        """Chaining fails when given parameters that don't exist."""

        @task
        def prepare(data: int) -> int:  # pragma: no cover
            """Simple prepare function."""
            return data + 1

        @task
        def add(x: int, y: int) -> int:  # pragma: no cover
            """Simple addition function."""
            return x + y

        prepared = prepare.bind(data=10)
        added = add.fix(x=5)

        with pytest.raises(ParameterError, match="Invalid parameters for task"):
            prepared.then(added, z=5)

    def test_task_then_with_multiple_ubound_params(self) -> None:
        """Chaining with partially bound tasks fails when given invalid parameters."""

        @task
        def prepare(data: int) -> int:  # pragma: no cover
            """Simple prepare function."""
            return data + 1

        @task
        def add(x: int, y: int, z: int) -> int:  # pragma: no cover
            """Simple addition function."""
            return x + y + z

        prepared = prepare.bind(data=10)

        with pytest.raises(ParameterError, match="must have exactly one unbound parameter"):
            prepared.then(add)

    def test_fixed_task_then_with_multiple_ubound_params(self) -> None:
        """Chaining with partially bound tasks fails when given invalid parameters."""

        @task
        def prepare(data: int) -> int:  # pragma: no cover
            """Simple prepare function."""
            return data + 1

        @task
        def add(x: int, y: int, z: int) -> int:  # pragma: no cover
            """Simple addition function."""
            return x + y + z

        prepared = prepare.bind(data=10)
        fixed_add = add.fix(z=20)

        with pytest.raises(ParameterError, match="must have exactly one unbound parameter"):
            prepared.then(fixed_add)

    def test_fixed_task_then_with_no_unbound_params(self) -> None:
        """Chaining with fully bound tasks fails when there are no unbound parameters."""

        @task
        def prepare(data: int) -> int:  # pragma: no cover
            """Simple prepare function."""
            return data + 1

        @task
        def add(x: int, y: int) -> int:  # pragma: no cover
            """Simple addition function."""
            return x + y

        prepared = prepare.bind(data=10)
        added = add.fix(x=5, y=15)

        with pytest.raises(ParameterError, match="has no unbound parameters"):
            prepared.then(added)

    def test_fixe_task_then_with_overlapping_params(self) -> None:
        """Chaining with partially bound tasks fails when given overlapping parameters."""

        @task
        def prepare(data: int) -> int:  # pragma: no cover
            """Simple prepare function."""
            return data + 1

        @task
        def add(x: int, y: int) -> int:  # pragma: no cover
            """Simple addition function."""
            return x + y

        prepared = prepare.bind(data=10)
        fixed = add.fix(y=5)

        with pytest.raises(ParameterError, match="Overlapping parameters"):
            prepared.then(fixed, y=20)

    def test_task_fix_with_invalid_params(self) -> None:
        """Fixing fails when given parameters that don't exist."""

        @task
        def divide(x: int, y: int) -> float:  # pragma: no cover
            """Simple division function."""
            return x / y

        with pytest.raises(ParameterError, match="Invalid parameters for task"):
            divide.fix(z=5)

    def test_task_extend_with_non_iterable_params(self) -> None:
        """Cartesian product operations require iterable parameters."""

        @task
        def add(x: int, y: int) -> int:  # pragma: no cover
            """Simple addition function."""
            return x + y

        with pytest.raises(ParameterError, match="Non-iterable parameters"):
            add.extend(x=20, y=5)

    def test_task_extend_with_overlapping_params(self) -> None:
        """Cartesian product fails when attempting to rebind fixed parameters."""

        @task
        def multiply(x: int, y: int) -> int:  # pragma: no cover
            """Simple multiplication function."""
            return x * y

        fixed = multiply.fix(x=3)

        with pytest.raises(ParameterError, match="Overlapping parameters"):
            fixed.extend(y=[1, 2, 3], x=[4, 5, 6])

    def test_task_extend_invalid_params(self) -> None:
        """Cartesian product fails when given parameters that don't exist."""

        @task
        def subtract(x: int, y: int) -> int:  # pragma: no cover
            """Simple subtraction function."""
            return x - y

        with pytest.raises(ParameterError, match="Invalid parameters"):
            subtract.extend(z=[10, 2, 3])

    def test_task_extend_missing_params(self) -> None:
        """Cartesian product fails when required parameters are omitted."""

        @task
        def power(base: int, exponent: int) -> int:  # pragma: no cover
            """Simple power function."""
            return base**exponent

        fixed = power.fix(base=2)

        with pytest.raises(ParameterError, match="Missing parameters"):
            fixed.extend()

    def test_task_zip_with_non_iterable_params(self) -> None:
        """Pairwise operations require iterable parameters."""

        @task
        def divide(x: int, y: int) -> float:  # pragma: no cover
            """Simple division function."""
            return x / y

        with pytest.raises(ParameterError, match="Non-iterable parameters"):
            divide.zip(x=10, y=5)

    def test_task_zip_with_overlapping_params(self) -> None:
        """Pairwise operations fail when attempting to rebind fixed parameters."""

        @task
        def add(x: int, y: int) -> int:  # pragma: no cover
            """Simple addition function."""
            return x + y

        fixed = add.fix(y=2)

        with pytest.raises(ParameterError, match="Overlapping parameters"):
            fixed.zip(y=[3, 4, 5], x=[1, 2, 3])

    def test_task_zip_invalid_params(self) -> None:
        """Pairwise operations fail when given parameters that don't exist."""

        @task
        def multiply(x: int, y: int) -> int:  # pragma: no cover
            """Simple multiplication function."""
            return x * y

        with pytest.raises(ParameterError, match="Invalid parameters"):
            multiply.zip(z=[10, 2, 3])

    def test_task_zip_missing_params(self) -> None:
        """Pairwise operations fail when required parameters are omitted."""

        @task
        def subtract(x: int, y: int) -> int:  # pragma: no cover
            """Simple subtraction function."""
            return x - y

        fixed = subtract.fix(x=10)

        with pytest.raises(ParameterError, match="Missing parameters"):
            fixed.zip()

    def test_task_zip_with_mismatched_lengths(self) -> None:
        """Pairwise operations require all iterable parameters to have the same length."""

        @task
        def add(x: int, y: int) -> int:  # pragma: no cover
            """Simple addition function."""
            return x + y

        with pytest.raises(ParameterError, match="Mixed lengths for scalar parameters"):
            add.zip(x=[1, 2, 3], y=[4, 5])

    def test_task_map_with_invalid_signature(self) -> None:
        """Mapping over results requires a single-parameter function."""

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
        """Mapping with partially bound tasks requires exactly one unbound parameter."""

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
        """Reducing results requires a single-parameter function."""

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
        """Reducing with partially bound tasks requires exactly one unbound parameter."""

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
        """Each future receives a unique identifier."""

        def add(x: int, y: int) -> int:  # pragma: no cover
            return x + y

        future1 = task(add).bind(x=1, y=2)
        future2 = task(add).bind(x=1, y=2)

        assert future1.id != future2.id

    def test_future_len_raises_type_error(self) -> None:
        """Unevaluated futures prevent accidental length operations."""

        def multiply(x: int, y: int) -> int:  # pragma: no cover
            return x * y

        future = task(multiply).bind(x=3, y=4)

        with pytest.raises(TypeError, match="do not support len()"):
            len(future)

    def test_future_bool_raises_type_error(self) -> None:
        """Unevaluated futures prevent accidental boolean operations."""

        def divide(x: int, y: int) -> float:  # pragma: no cover
            return x / y

        future = task(divide).bind(x=10, y=2)

        with pytest.raises(TypeError, match="cannot be used in boolean context."):
            bool(future)
