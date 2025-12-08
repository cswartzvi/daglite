"""Unit tests for composer functions (construction, validation, graph building)."""

import pytest

from daglite import task
from daglite.composers import ConditionalFuture
from daglite.composers import LoopFuture
from daglite.composers import loop
from daglite.composers import split
from daglite.composers import when
from daglite.exceptions import DagliteError
from daglite.graph.nodes import ConditionalNode
from daglite.graph.nodes import LoopNode
from daglite.tasks import TaskFuture


class TestSplitConstruction:
    """Tests for split() composer construction."""

    def test_split_creates_correct_number_of_futures(self) -> None:
        """split() should create the correct number of TaskFutures."""

        @task
        def make_pair() -> tuple[int, str]:
            return (1, "a")

        result = make_pair.bind()
        futures = split(result)

        assert len(futures) == 2
        assert all(isinstance(f, TaskFuture) for f in futures)

    def test_split_with_size_parameter(self) -> None:
        """split() should respect explicit size parameter."""

        @task
        def make_variadic() -> tuple[int, ...]:
            return (1, 2, 3, 4)

        result = make_variadic.bind()
        futures = split(result, size=4)

        assert len(futures) == 4

    def test_split_raises_without_annotation_or_size(self) -> None:
        """split() should raise ValueError when size cannot be inferred."""

        @task
        def make_untyped():
            return (1, 2, 3)

        result = make_untyped.bind()

        with pytest.raises(DagliteError, match="Cannot infer tuple size"):
            split(result)

    def test_split_method_with_annotations(self) -> None:
        """TaskFuture.split() method should work with type annotations."""

        @task
        def make_pair() -> tuple[int, str]:
            return (1, "a")

        futures = make_pair.bind().split()

        assert len(futures) == 2
        assert all(isinstance(f, TaskFuture) for f in futures)

    def test_split_method_with_size_parameter(self) -> None:
        """TaskFuture.split() method should accept explicit size."""

        @task
        def make_triple():
            return (1, 2, 3)

        futures = make_triple.bind().split(size=3)

        assert len(futures) == 3
        assert all(isinstance(f, TaskFuture) for f in futures)

    def test_split_method_raises_without_size(self) -> None:
        """TaskFuture.split() method should raise when size cannot be inferred."""

        @task
        def make_untyped():
            return (1, 2, 3)

        with pytest.raises(DagliteError, match="Cannot infer tuple size"):
            make_untyped.bind().split()


class TestWhenConstruction:
    """Tests for when() composer construction."""

    def test_when_creates_conditional_future(self) -> None:
        """when() should create a ConditionalFuture."""

        @task
        def check() -> bool:
            return True

        @task
        def get_value() -> int:
            return 42

        condition = check.bind()
        result = when(condition, get_value.bind(), get_value.bind())

        assert isinstance(result, ConditionalFuture)
        assert result.condition is condition


class TestLoopConstruction:
    """Tests for loop() composer construction."""

    def test_loop_creates_loop_future(self) -> None:
        """loop() should create a LoopFuture."""

        @task
        def increment(n: int) -> tuple[int, bool]:
            return (n + 1, n < 10)

        result = loop(initial=0, body=increment)

        assert isinstance(result, LoopFuture)
        assert result.initial_state == 0
        assert result.body is increment

    def test_loop_with_task_future_initial(self) -> None:
        """loop() should accept TaskFuture as initial state."""

        @task
        def get_start() -> int:
            return 5

        @task
        def increment(n: int) -> tuple[int, bool]:
            return (n + 1, n < 10)

        start = get_start.bind()
        result = loop(initial=start, body=increment)

        assert isinstance(result, LoopFuture)
        assert isinstance(result.initial_state, TaskFuture)

    def test_loop_with_body_kwargs(self) -> None:
        """loop() should accept body kwargs."""

        @task
        def add_factor(state: int, factor: int) -> tuple[int, bool]:
            return (state + factor, state < 50)

        result = loop(initial=0, body=add_factor, factor=5)

        assert isinstance(result, LoopFuture)
        assert result.body_kwargs == {"factor": 5}

    def test_loop_with_custom_max_iterations(self) -> None:
        """loop() should respect max_iterations parameter."""

        @task
        def increment(n: int) -> tuple[int, bool]:
            return (n + 1, True)

        result = loop(initial=0, body=increment, max_iterations=100)

        assert result.max_iterations == 100

    def test_loop_raises_if_body_has_no_unbound_param(self) -> None:
        """loop() should raise if body has no unbound parameters."""

        @task
        def no_params() -> tuple[int, bool]:
            return (0, False)

        with pytest.raises(ValueError, match="must have at least one unbound parameter"):
            loop(initial=0, body=no_params)

    def test_loop_raises_if_body_has_multiple_unbound_params(self) -> None:
        """loop() should raise if body has multiple unbound parameters."""

        @task
        def two_params(state: int, other: int) -> tuple[int, bool]:
            return (state + other, state < 10)

        with pytest.raises(ValueError, match="multiple unbound parameters"):
            loop(initial=0, body=two_params)


class TestComposerThenChaining:
    """Tests for .then() chaining with composers."""

    def test_conditional_future_then_with_fixed_param_task(self) -> None:
        """ConditionalFuture.then() works with FixedParamTask."""

        @task
        def check() -> bool:
            return True

        @task
        def get_value() -> int:
            return 42

        @task
        def scale(x: int, factor: int) -> int:
            return x * factor

        condition = check.bind()
        conditional = when(condition, get_value.bind(), get_value.bind())

        # Use .then() with a FixedParamTask
        fixed_scale = scale.fix(factor=2)
        result = conditional.then(fixed_scale)

        from daglite.tasks import TaskFuture

        assert isinstance(result, TaskFuture)
        assert result.task == scale

    def test_loop_future_then_with_fixed_param_task(self) -> None:
        """LoopFuture.then() works with FixedParamTask."""

        @task
        def increment(n: int) -> tuple[int, bool]:
            return (n + 1, n < 10)

        @task
        def multiply(x: int, factor: int) -> int:
            return x * factor

        loop_future = loop(initial=0, body=increment)

        # Use .then() with a FixedParamTask
        fixed_multiply = multiply.fix(factor=3)
        result = loop_future.then(fixed_multiply)

        from daglite.tasks import TaskFuture

        assert isinstance(result, TaskFuture)
        assert result.task == multiply


class TestComposerGraphConstruction:
    """Tests for graph node construction from composers."""

    def test_conditional_future_to_graph(self) -> None:
        """ConditionalFuture should convert to ConditionalNode."""

        @task
        def check() -> bool:
            return True

        @task
        def get_value() -> int:
            return 42

        condition = check.bind()
        conditional = when(condition, get_value.bind(), get_value.bind())
        node = conditional.to_graph()

        assert isinstance(node, ConditionalNode)
        assert node.name == "conditional"

    def test_loop_future_to_graph(self) -> None:
        """LoopFuture should convert to LoopNode."""

        @task
        def increment(n: int) -> tuple[int, bool]:
            return (n + 1, n < 10)

        loop_future = loop(initial=0, body=increment)
        node = loop_future.to_graph()

        assert isinstance(node, LoopNode)
        assert node.name == "loop_increment"
        assert node.max_iterations == 1000

    def test_conditional_dependencies(self) -> None:
        """ConditionalFuture should report all three futures as dependencies."""

        @task
        def check() -> bool:
            return True

        @task
        def get_value() -> int:
            return 42

        condition = check.bind()
        then_branch = get_value.bind()
        else_branch = get_value.bind()

        conditional = when(condition, then_branch, else_branch)
        deps = conditional.get_dependencies()

        assert len(deps) == 3
        assert condition in deps
        assert then_branch in deps
        assert else_branch in deps

    def test_loop_dependencies(self) -> None:
        """LoopFuture should report initial state as dependency when it's a TaskFuture."""

        @task
        def get_start() -> int:
            return 5

        @task
        def increment(n: int) -> tuple[int, bool]:
            return (n + 1, n < 10)

        start = get_start.bind()
        loop_future = loop(initial=start, body=increment)
        deps = loop_future.get_dependencies()

        assert len(deps) == 1
        assert start in deps

    def test_loop_dependencies_with_kwargs(self) -> None:
        """LoopFuture should include TaskFuture kwargs in dependencies."""

        @task
        def get_factor() -> int:
            return 7

        @task
        def add_factor(state: int, factor: int) -> tuple[int, bool]:
            return (state + factor, state < 50)

        factor = get_factor.bind()
        loop_future = loop(initial=0, body=add_factor, factor=factor)
        deps = loop_future.get_dependencies()

        assert len(deps) == 1
        assert factor in deps
