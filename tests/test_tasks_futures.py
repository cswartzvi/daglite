"""
Unit tests for task and task future definitions and parameter handling.

These tests focus on validating the behavior of the @task decorator, Task, and TaskFuture classes
when defining tasks and binding parameters. They ensure that invalid usages raise appropriate
exceptions.

Tests in this file should NOT focus on evaluation. Evaluation tests are in tests/evaluation/.
"""

import pytest

from daglite.exceptions import DagliteError
from daglite.exceptions import ParameterError
from daglite.futures import MapTaskFuture
from daglite.futures import TaskFuture
from daglite.tasks import PartialTask
from daglite.tasks import Task
from daglite.tasks import task


class TestTaskDefinition:
    """Test the @task decorator definition and metadata handling."""

    def test_task_decorator_with_defaults(self) -> None:
        """Decorating a function without parameters uses sensible defaults."""

        @task
        def add(x: int, y: int) -> int:
            """Simple addition function."""
            return x + y

        assert isinstance(add, Task)
        assert add.name == "add"
        assert add.description == "Simple addition function."
        assert add.backend_name is None  # Default is None (uses engine's default)
        assert add.func(1, 2) == 3

    def test_task_decorator_with_params(self) -> None:
        """Decorator accepts custom name, description, and backend configuration."""

        @task(name="custom_add", description="Custom addition task", backend_name="threading")
        def add(x: int, y: int) -> int:  # pragma: no cover
            """Not used docstring."""
            return x + y

        assert add.name == "custom_add"
        assert add.description == "Custom addition task"
        assert add.backend_name == "threading"

    def test_async_task_is_async_attribute(self) -> None:
        """Task.is_async correctly identifies async functions."""

        @task
        def sync_func(x: int) -> int:  # pragma: no cover
            return x

        @task
        async def async_func(x: int) -> int:  # pragma: no cover
            return x

        assert not sync_func.is_async
        assert async_func.is_async

    def test_partial_task(self) -> None:
        """Fixing parameters creates a partially bound task."""

        @task
        def multiply(x: int, y: int) -> int:
            """Simple multiplication function."""
            return x * y

        fixed_task = multiply.partial(y=5)

        assert isinstance(fixed_task, PartialTask)
        assert isinstance(fixed_task.task, Task)
        assert fixed_task.task.func(2, 5) == 10

    def test_partial_task_with_params(self) -> None:
        """Fixing parameters preserves task metadata."""

        @task(name="multiply_task", description="Multiplication task")
        def multiply(x: int, y: int) -> int:
            """Simple multiplication function."""
            return x * y

        fixed_task: PartialTask = multiply.partial(y=10)

        assert isinstance(fixed_task, PartialTask)
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

    def test_task_decorator_with_non_callable(self) -> None:
        """Decorator rejects non-callable objects."""

        with pytest.raises(TypeError, match="can only be applied to callable functions"):

            @task
            class NotCallable:
                pass

    def test_task_with_negative_retries(self) -> None:
        """Defining a task with negative retries raises ParameterError."""

        with pytest.raises(ParameterError, match="invalid retries=-1"):

            @task(retries=-1)
            def faulty_task(x: int) -> int:
                return x

    def test_task_with_negative_timeout(self) -> None:
        """Defining a task with negative timeout raises ParameterError."""

        with pytest.raises(ParameterError, match="invalid timeout=-5"):

            @task(timeout=-5)
            def faulty_task(x: int) -> int:
                return x

    def test_task_with_negative_cache_ttl(self) -> None:
        """Defining a task with negative cache_ttl raises ParameterError."""

        with pytest.raises(ParameterError, match="invalid cache_ttl=-10"):

            @task(cache_ttl=-10)
            def faulty_task(x: int) -> int:
                return x


class TestFutureRepr:
    """Test __repr__ output for TaskFuture, MapTaskFuture, and DatasetFuture."""

    def test_task_future_repr(self) -> None:
        """TaskFuture repr shows task name and kwargs."""

        @task
        def add(x: int, y: int) -> int:  # pragma: no cover
            return x + y

        future = add(x=1, y=2)

        assert repr(future) == "TaskFuture(add, x=1, y=2)"

    def test_task_future_repr_no_kwargs(self) -> None:
        """TaskFuture repr works for zero-parameter tasks."""

        @task
        def noop() -> None:  # pragma: no cover
            pass

        future = noop()

        assert repr(future) == "TaskFuture(noop)"

    def test_task_future_repr_with_custom_name(self) -> None:
        """TaskFuture repr uses the task's custom name."""

        @task(name="custom_add")
        def add(x: int, y: int) -> int:  # pragma: no cover
            return x + y

        future = add(x=1, y=2)

        assert repr(future) == "TaskFuture(custom_add, x=1, y=2)"

    def test_task_future_repr_truncates_long_values(self) -> None:
        """TaskFuture repr truncates values exceeding 50 characters."""

        @task
        def process(data: str) -> str:  # pragma: no cover
            return data

        long_value = "a" * 100
        future = process(data=long_value)

        r = repr(future)
        assert r.startswith("TaskFuture(process, data=")
        assert "..." in r

    def test_map_task_future_product_repr(self) -> None:
        """MapTaskFuture repr shows task name, mode, and kwargs."""

        @task
        def double(x: int) -> int:  # pragma: no cover
            return x * 2

        future = double.product(x=[1, 2, 3])

        assert isinstance(future, MapTaskFuture)
        assert repr(future) == "MapTaskFuture(double, mode=product, x=[1, 2, 3])"

    def test_map_task_future_zip_repr(self) -> None:
        """MapTaskFuture repr shows zip mode."""

        @task
        def add(x: int, y: int) -> int:  # pragma: no cover
            return x + y

        future = add.zip(x=[1, 2], y=[3, 4])

        assert isinstance(future, MapTaskFuture)
        assert repr(future) == "MapTaskFuture(add, mode=zip, x=[1, 2], y=[3, 4])"

    def test_map_task_future_repr_with_fixed_kwargs(self) -> None:
        """MapTaskFuture repr includes both fixed and mapped kwargs."""

        @task
        def scale(x: int, factor: int) -> int:  # pragma: no cover
            return x * factor

        fixed = scale.partial(factor=10)
        future = fixed.product(x=[1, 2, 3])

        assert isinstance(future, MapTaskFuture)
        assert repr(future) == "MapTaskFuture(scale, mode=product, factor=10, x=[1, 2, 3])"


class TestParameterValidation:
    """Test parameter validation for task calls and partial() operations."""

    def test_task_call_with_invalid_params(self) -> None:
        """Calling fails when given parameters that don't exist."""

        @task
        def subtract(x: int, y: int) -> int:  # pragma: no cover
            return x - y

        with pytest.raises(ParameterError, match="Invalid parameters for task"):
            subtract(z=10)

    def test_task_call_with_missing_params(self) -> None:
        """Calling fails when required parameters are omitted."""

        @task
        def power(base: int, exponent: int) -> int:  # pragma: no cover
            return base**exponent

        with pytest.raises(ParameterError, match="Missing parameters for task"):
            power(base=2)

    def test_task_call_with_overlapping_params(self) -> None:
        """Calling fails when attempting to re-bind already-partial parameters."""

        @task
        def multiply(x: int, y: int) -> int:  # pragma: no cover
            return x * y

        fixed = multiply.partial(x=4)

        with pytest.raises(ParameterError, match="Overlapping parameters"):
            fixed(x=5, y=10)

    def test_partial_task_with_invalid_params(self) -> None:
        """Fixing fails when given parameters that don't exist."""

        @task
        def divide(x: int, y: int) -> float:  # pragma: no cover
            return x / y

        with pytest.raises(ParameterError, match="Invalid parameters for task"):
            divide.partial(z=5)


class TestProductOperationErrors:
    """Test error handling for product() and then_product() operations."""

    def test_task_product_with_non_iterable_params(self) -> None:
        """product() requires iterable parameters."""

        @task
        def add(x: int, y: int) -> int:  # pragma: no cover
            return x + y

        with pytest.raises(ParameterError, match="Non-iterable parameters"):
            add.product(x=20, y=5)

    def test_task_product_with_overlapping_params(self) -> None:
        """product() fails when attempting to re-bind partially-applied parameters."""

        @task
        def multiply(x: int, y: int) -> int:  # pragma: no cover
            return x * y

        fixed = multiply.partial(x=3)

        with pytest.raises(ParameterError, match="Overlapping parameters"):
            fixed.product(y=[1, 2, 3], x=[4, 5, 6])

    def test_task_product_invalid_params(self) -> None:
        """product() fails when given parameters that don't exist."""

        @task
        def subtract(x: int, y: int) -> int:  # pragma: no cover
            return x - y

        with pytest.raises(ParameterError, match="Invalid parameters"):
            subtract.product(z=[10, 2, 3])

    def test_task_product_missing_params(self) -> None:
        """product() fails when required parameters are omitted."""

        @task
        def power(base: int, exponent: int) -> int:  # pragma: no cover
            return base**exponent

        fixed = power.partial(base=2)

        with pytest.raises(ParameterError, match="Missing parameters"):
            fixed.product()

    def test_then_product_with_invalid_params(self) -> None:
        """then_product() fails when given parameters that don't exist."""

        @task
        def start() -> int:
            return 5

        @task
        def combine(x: int, y: int) -> int:  # pragma: no cover
            return x + y

        with pytest.raises(ParameterError, match="Invalid parameters for task"):
            start().then_product(combine, z=[1, 2, 3])

    def test_then_product_with_non_iterable_params(self) -> None:
        """then_product() requires mapped parameters to be iterable."""

        @task
        def start() -> int:
            return 5

        @task
        def combine(x: int, y: int) -> int:  # pragma: no cover
            return x + y

        with pytest.raises(ParameterError, match="Non-iterable parameters"):
            start().then_product(combine, y=10)

    def test_then_product_with_overlapping_params(self) -> None:
        """then_product() fails when trying to re-bind partially-applied parameters."""

        @task
        def start() -> int:
            return 5

        @task
        def combine(x: int, y: int, z: int) -> int:  # pragma: no cover
            return x + y + z

        fixed = combine.partial(y=10)

        with pytest.raises(ParameterError, match="Overlapping parameters"):
            start().then_product(fixed, y=[1, 2, 3])

    def test_then_product_with_no_mapped_params(self) -> None:
        """then_product() fails when no mapped parameters are provided."""

        @task
        def start() -> int:
            return 5

        @task
        def identity(x: int) -> int:  # pragma: no cover
            return x

        with pytest.raises(ParameterError, match="At least one mapped parameter required"):
            start().then_product(identity)


class TestZipOperationErrors:
    """Test error handling for zip() and then_zip() operations."""

    def test_task_zip_with_non_iterable_params(self) -> None:
        """zip() requires iterable parameters."""

        @task
        def divide(x: int, y: int) -> float:  # pragma: no cover
            return x / y

        with pytest.raises(ParameterError, match="Non-iterable parameters"):
            divide.zip(x=10, y=5)

    def test_task_zip_with_overlapping_params(self) -> None:
        """zip() fails when attempting to re-bind partially-applied parameters."""

        @task
        def add(x: int, y: int) -> int:  # pragma: no cover
            return x + y

        fixed = add.partial(y=2)

        with pytest.raises(ParameterError, match="Overlapping parameters"):
            fixed.zip(y=[3, 4, 5], x=[1, 2, 3])

    def test_task_zip_invalid_params(self) -> None:
        """zip() fails when given parameters that don't exist."""

        @task
        def multiply(x: int, y: int) -> int:  # pragma: no cover
            return x * y

        with pytest.raises(ParameterError, match="Invalid parameters"):
            multiply.zip(z=[10, 2, 3])

    def test_task_zip_missing_params(self) -> None:
        """zip() fails when required parameters are omitted."""

        @task
        def subtract(x: int, y: int) -> int:  # pragma: no cover
            return x - y

        fixed = subtract.partial(x=10)

        with pytest.raises(ParameterError, match="Missing parameters"):
            fixed.zip()

    def test_task_zip_with_mismatched_lengths(self) -> None:
        """zip() requires all iterable parameters to have the same length."""

        @task
        def add(x: int, y: int) -> int:  # pragma: no cover
            return x + y

        with pytest.raises(ParameterError, match="Mixed lengths for task 'add'"):
            add.zip(x=[1, 2, 3], y=[4, 5])

    def test_then_zip_with_invalid_params(self) -> None:
        """then_zip() fails when given parameters that don't exist."""

        @task
        def start() -> int:
            return 5

        @task
        def combine(x: int, y: int) -> int:  # pragma: no cover
            return x + y

        with pytest.raises(ParameterError, match="Invalid parameters for task"):
            start().then_zip(combine, z=[1, 2, 3])

    def test_then_zip_with_non_iterable_params(self) -> None:
        """then_zip() requires mapped parameters to be iterable."""

        @task
        def start() -> int:
            return 5

        @task
        def combine(x: int, y: int) -> int:  # pragma: no cover
            return x + y

        with pytest.raises(ParameterError, match="Non-iterable parameters"):
            start().then_zip(combine, y=10)

    def test_then_zip_with_mismatched_lengths(self) -> None:
        """then_zip() fails when mapped parameter lengths don't match."""

        @task
        def start() -> int:
            return 5

        @task
        def compute(x: int, y: int, z: int) -> int:  # pragma: no cover
            return x + y + z

        with pytest.raises(ParameterError, match="Mixed lengths"):
            start().then_zip(compute, y=[1, 2, 3], z=[10, 20])

    def test_then_zip_with_overlapping_params(self) -> None:
        """then_zip() fails when trying to re-bind partially-applied parameters."""

        @task
        def start() -> int:
            return 5

        @task
        def combine(x: int, y: int, z: int) -> int:  # pragma: no cover
            return x + y + z

        fixed = combine.partial(y=10)

        with pytest.raises(ParameterError, match="Overlapping parameters"):
            start().then_zip(fixed, y=[1, 2, 3])

    def test_then_zip_with_no_mapped_params(self) -> None:
        """then_zip() fails when no mapped parameters are provided."""

        @task
        def start() -> int:
            return 5

        @task
        def identity(x: int) -> int:  # pragma: no cover
            return x

        with pytest.raises(ParameterError, match="At least one mapped parameter required"):
            start().then_zip(identity)


class TestFluentAPIErrors:
    """Test error handling for fluent API methods: then(), map(), join()."""

    def test_task_then_with_invalid_params(self) -> None:
        """then() fails when given parameters that don't exist."""

        @task
        def prepare(data: int) -> int:  # pragma: no cover
            return data + 1

        @task
        def add(x: int, y: int) -> int:  # pragma: no cover
            return x + y

        prepared = prepare(data=10)
        added = add.partial(x=5)

        with pytest.raises(ParameterError, match="Invalid parameters for task"):
            prepared.then(added, z=5)

    def test_task_then_with_multiple_unbound_params(self) -> None:
        """then() requires target task to have exactly one unbound parameter."""

        @task
        def prepare(data: int) -> int:  # pragma: no cover
            return data + 1

        @task
        def add(x: int, y: int, z: int) -> int:  # pragma: no cover
            return x + y + z

        prepared = prepare(data=10)

        with pytest.raises(ParameterError, match="must have exactly one unbound parameter"):
            prepared.then(add)

    def test_task_map_with_invalid_signature(self) -> None:
        """map() requires a single-parameter function."""

        @task
        def prepare(data: int) -> int:  # pragma: no cover
            return data + 1

        @task
        def mapping(a: int, b: int) -> int:  # pragma: no cover
            return a + b

        prepared = prepare.product(data=[1, 2, 3])
        with pytest.raises(ParameterError, match="must have exactly one unbound parameter"):
            prepared.then(mapping)

    def test_task_map_with_kwargs(self) -> None:
        """map() with inline kwargs works correctly."""

        @task
        def prepare(data: int) -> int:  # pragma: no cover
            return data + 1

        @task
        def scale(x: int, factor: int) -> int:  # pragma: no cover
            return x * factor

        prepared = prepare.product(data=[1, 2, 3])
        # Should work with inline kwargs
        scaled = prepared.then(scale, factor=10)
        assert scaled is not None

    def test_task_map_with_kwargs_multiple_unbound(self) -> None:
        """map() with kwargs fails when multiple parameters remain unbound."""

        @task
        def prepare(data: int) -> int:  # pragma: no cover
            return data + 1

        @task
        def add(x: int, y: int, z: int) -> int:  # pragma: no cover
            return x + y + z

        prepared = prepare.product(data=[1, 2, 3])
        with pytest.raises(ParameterError, match="must have exactly one unbound parameter"):
            prepared.then(add, z=5)

    def test_task_map_with_overlapping_kwargs(self) -> None:
        """map() with overlapping kwargs fails."""

        @task
        def prepare(data: int) -> int:  # pragma: no cover
            return data + 1

        @task
        def scale(x: int, factor: int) -> int:  # pragma: no cover
            return x * factor

        prepared = prepare.product(data=[1, 2, 3])
        fixed_scale = scale.partial(factor=10)
        with pytest.raises(ParameterError, match="Overlapping parameters"):
            prepared.then(fixed_scale, factor=20)

    def test_task_join_with_kwargs(self) -> None:
        """join() with inline kwargs works correctly."""

        @task
        def prepare(data: int) -> int:  # pragma: no cover
            return data + 1

        @task
        def weighted_sum(xs: list[int], weight: float) -> float:  # pragma: no cover
            return sum(xs) * weight

        prepared = prepare.product(data=[1, 2, 3])
        # Should work with inline kwargs
        total = prepared.join(weighted_sum, weight=2.5)
        assert total is not None

    def test_task_join_with_invalid_signature(self) -> None:
        """join() requires a single-parameter function."""

        @task
        def prepare(data: int) -> int:  # pragma: no cover
            return data + 1

        @task
        def mapping(a: int) -> int:  # pragma: no cover
            return a * 2

        @task
        def joining(a: int, b: int) -> int:  # pragma: no cover
            return a * 2

        prepared = prepare.product(data=[1, 2, 3])
        mapped = prepared.then(mapping)
        with pytest.raises(ParameterError, match="must have exactly one unbound parameter"):
            mapped.join(joining)

    def test_task_join_with_kwargs_multiple_unbound(self) -> None:
        """join() with kwargs fails when multiple parameters remain unbound."""

        @task
        def prepare(data: int) -> int:  # pragma: no cover
            return data + 1

        @task
        def reduce_three(xs: list[int], y: int, z: int) -> int:  # pragma: no cover
            return sum(xs) + y + z

        prepared = prepare.product(data=[1, 2, 3])
        with pytest.raises(ParameterError, match="must have exactly one unbound parameter"):
            prepared.join(reduce_three, z=5)

    def test_task_join_with_overlapping_kwargs(self) -> None:
        """join() with overlapping kwargs fails."""

        @task
        def prepare(data: int) -> int:  # pragma: no cover
            return data + 1

        @task
        def weighted_sum(xs: list[int], weight: float) -> float:  # pragma: no cover
            return sum(xs) * weight

        prepared = prepare.product(data=[1, 2, 3])
        fixed_sum = weighted_sum.partial(weight=1.5)
        with pytest.raises(ParameterError, match="Overlapping parameters"):
            prepared.join(fixed_sum, weight=2.5)


class TestPartialTaskErrors:
    """Test error handling for PartialTask operations."""

    def test_partial_task_then_with_multiple_unbound_params(self) -> None:
        """then() with PartialTask fails when multiple parameters remain unbound."""

        @task
        def prepare(data: int) -> int:  # pragma: no cover
            return data + 1

        @task
        def add(x: int, y: int, z: int) -> int:  # pragma: no cover
            return x + y + z

        prepared = prepare(data=10)
        fixed_add = add.partial(z=20)

        with pytest.raises(ParameterError, match="must have exactly one unbound parameter"):
            prepared.then(fixed_add)

    def test_partial_task_then_with_no_unbound_params(self) -> None:
        """then() with fully bound PartialTask fails."""

        @task
        def prepare(data: int) -> int:  # pragma: no cover
            return data + 1

        @task
        def add(x: int, y: int) -> int:  # pragma: no cover
            return x + y

        prepared = prepare(data=10)
        added = add.partial(x=5, y=15)

        with pytest.raises(ParameterError, match="has no unbound parameters"):
            prepared.then(added)

    def test_partial_task_then_with_overlapping_params(self) -> None:
        """then() with PartialTask fails when given overlapping parameters."""

        @task
        def prepare(data: int) -> int:  # pragma: no cover
            return data + 1

        @task
        def add(x: int, y: int) -> int:  # pragma: no cover
            return x + y

        prepared = prepare(data=10)
        fixed = add.partial(y=5)

        with pytest.raises(ParameterError, match="Overlapping parameters"):
            prepared.then(fixed, y=20)

    def test_partial_task_map_with_invalid_signature(self) -> None:
        """map() with partially bound PartialTask requires exactly one unbound parameter."""

        @task
        def prepare(data: int) -> int:  # pragma: no cover
            return data + 1

        @task
        def mapping(a: int, b: int, c: int) -> int:  # pragma: no cover
            return a + b + c

        prepared = prepare.product(data=[1, 2, 3])
        fixed_mapping = mapping.partial(c=20)
        with pytest.raises(ParameterError, match="must have exactly one unbound parameter"):
            prepared.then(fixed_mapping)

    def test_partial_task_join_with_invalid_signature(self) -> None:
        """join() with partially bound PartialTask requires exactly one unbound parameter."""

        @task
        def prepare(data: int) -> int:  # pragma: no cover
            return data + 1

        @task
        def mapping(a: int) -> int:  # pragma: no cover
            return a * 2

        @task
        def joining(a: int, b: int, c: int) -> int:  # pragma: no cover
            return a + b + c

        prepared = prepare.product(data=[1, 2, 3])
        mapped = prepared.then(mapping)
        fixed_joining = joining.partial(c=10)
        with pytest.raises(ParameterError, match="must have exactly one unbound parameter"):
            mapped.join(fixed_joining)


class TestBaseTaskFuture:
    """Test core BaseTaskFuture behavior."""

    def test_futures_have_unique_ids(self) -> None:
        """Each future receives a unique identifier."""

        def add(x: int, y: int) -> int:  # pragma: no cover
            return x + y

        future1 = task(add)(x=1, y=2)
        future2 = task(add)(x=1, y=2)

        assert future1.id != future2.id

    def test_future_len_raises_type_error(self) -> None:
        """Unevaluated futures prevent accidental length operations."""

        def multiply(x: int, y: int) -> int:  # pragma: no cover
            return x * y

        future = task(multiply)(x=3, y=4)

        with pytest.raises(TypeError, match="do not support len()"):
            len(future)

    def test_future_bool_raises_type_error(self) -> None:
        """Unevaluated futures prevent accidental boolean operations."""

        def divide(x: int, y: int) -> float:  # pragma: no cover
            return x / y

        future = task(divide)(x=10, y=2)

        with pytest.raises(TypeError, match="cannot be used in boolean context."):
            bool(future)


class TestSplitMethod:
    """Tests for TaskFuture.split() method construction."""

    def test_split_method_with_annotations(self) -> None:
        """TaskFuture.split() method should work with type annotations."""

        @task
        def make_pair() -> tuple[int, str]:
            return (1, "a")

        futures = make_pair().split()

        assert len(futures) == 2
        assert all(isinstance(f, TaskFuture) for f in futures)

    def test_split_method_with_size_parameter(self) -> None:
        """TaskFuture.split() method should accept explicit size."""

        @task
        def make_triple():
            return (1, 2, 3)

        futures = make_triple().split(size=3)

        assert len(futures) == 3
        assert all(isinstance(f, TaskFuture) for f in futures)

    def test_split_method_raises_without_size(self) -> None:
        """TaskFuture.split() method should raise when size cannot be inferred."""

        @task
        def make_untyped():
            return (1, 2, 3)

        with pytest.raises(DagliteError, match="Cannot infer tuple size"):
            make_untyped().split()

    def test_split_method_with_large_tuple(self) -> None:
        """TaskFuture.split() should handle larger tuples."""

        @task
        def make_five() -> tuple[int, int, int, int, int]:
            return (1, 2, 3, 4, 5)

        futures = make_five().split()

        assert len(futures) == 5
        assert all(isinstance(f, TaskFuture) for f in futures)
