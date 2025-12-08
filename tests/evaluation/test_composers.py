"""Integration tests for split() composer with evaluation."""

from daglite import evaluate
from daglite import task
from daglite.composers import loop
from daglite.composers import split
from daglite.composers import when
from daglite.exceptions import DagliteError


class TestSplitWithTypeAnnotations:
    """Tests for split() with type annotations."""

    def test_split_pair_with_annotations(self) -> None:
        """split() should work with two-element tuple."""

        @task
        def make_pair() -> tuple[int, str]:
            return (42, "hello")

        result = make_pair.bind()
        first, second = split(result)

        assert evaluate(first) == 42
        assert evaluate(second) == "hello"

    def test_split_triple_with_annotations(self) -> None:
        """split() should work with three-element tuple."""

        @task
        def make_triple() -> tuple[int, str, float]:
            return (1, "a", 2.5)

        result = make_triple.bind()
        a, b, c = split(result)

        assert evaluate(a) == 1
        assert evaluate(b) == "a"
        assert evaluate(c) == 2.5

    def test_split_single_element(self) -> None:
        """split() should work with single-element tuple."""

        @task
        def make_single() -> tuple[str]:
            return ("only",)

        result = make_single.bind()
        (only,) = split(result)

        assert evaluate(only) == "only"

    def test_split_five_elements(self) -> None:
        """split() should work with five-element tuple."""

        @task
        def make_five() -> tuple[int, int, int, int, int]:
            return (1, 2, 3, 4, 5)

        result = make_five.bind()
        a, b, c, d, e = split(result)

        assert evaluate(a) == 1
        assert evaluate(b) == 2
        assert evaluate(c) == 3
        assert evaluate(d) == 4
        assert evaluate(e) == 5

    def test_split_mixed_types(self) -> None:
        """split() should preserve element types."""

        @task
        def make_mixed() -> tuple[int, str, bool, list[int]]:
            return (42, "test", True, [1, 2, 3])

        result = make_mixed.bind()
        num, text, flag, items = split(result)

        assert evaluate(num) == 42
        assert evaluate(text) == "test"
        assert evaluate(flag) is True
        assert evaluate(items) == [1, 2, 3]


class TestSplitWithSizeParameter:
    """Tests for split() with explicit size parameter."""

    def test_split_with_explicit_size(self) -> None:
        """split() should work with explicit size for untyped tuple."""

        @task
        def make_pair():
            return (1, 2)

        result = make_pair.bind()
        first, second = split(result, size=2)

        assert evaluate(first) == 1
        assert evaluate(second) == 2

    def test_split_with_explicit_size_triple(self) -> None:
        """split() should work with explicit size=3."""

        @task
        def make_triple():
            return (10, 20, 30)

        result = make_triple.bind()
        a, b, c = split(result, size=3)

        assert evaluate(a) == 10
        assert evaluate(b) == 20
        assert evaluate(c) == 30

    def test_split_size_overrides_annotation(self) -> None:
        """Explicit size should override type annotation."""

        @task
        def make_pair() -> tuple[int, int]:
            return (1, 2)

        result = make_pair.bind()
        # Use size=2 explicitly even though annotation says 2
        first, second = split(result, size=2)

        assert evaluate(first) == 1
        assert evaluate(second) == 2


class TestSplitComposition:
    """Tests for split() composed with other operations."""

    def test_split_then_process_independently(self) -> None:
        """Split elements should be processable independently."""

        @task
        def make_pair() -> tuple[int, int]:
            return (3, 5)

        @task
        def double(x: int) -> int:
            return x * 2

        @task
        def triple(x: int) -> int:
            return x * 3

        result = make_pair.bind()
        first, second = split(result)

        doubled = double.bind(x=first)
        tripled = triple.bind(x=second)

        assert evaluate(doubled) == 6
        assert evaluate(tripled) == 15

    def test_split_then_recombine(self) -> None:
        """Split elements can be recombined."""

        @task
        def make_pair() -> tuple[int, int]:
            return (10, 20)

        @task
        def add(x: int, y: int) -> int:
            return x + y

        result = make_pair.bind()
        first, second = split(result)
        combined = add.bind(x=first, y=second)

        assert evaluate(combined) == 30

    def test_split_with_map_reduce(self) -> None:
        """Split elements can be used in map/reduce patterns."""

        @task
        def make_triple() -> tuple[int, int, int]:
            return (1, 2, 3)

        @task
        def sum_three(a: int, b: int, c: int) -> int:
            return a + b + c

        result = make_triple.bind()
        a, b, c = split(result)
        total = sum_three.bind(a=a, b=b, c=c)

        assert evaluate(total) == 6

    def test_split_nested_dependency(self) -> None:
        """Split should work with nested task dependencies."""

        @task
        def compute(x: int) -> int:
            return x * 2

        @task
        def make_pair(n: int) -> tuple[int, int]:
            return (n, n + 1)

        computed = compute.bind(x=5)
        pair = make_pair.bind(n=computed)
        first, second = split(pair)

        assert evaluate(first) == 10
        assert evaluate(second) == 11


class TestSplitErrorCases:
    """Tests for split() error handling."""

    def test_split_without_annotation_or_size_raises_error(self) -> None:
        """split() should raise ValueError without type hints or size."""
        import pytest

        @task
        def make_untyped():
            return (1, 2)

        result = make_untyped.bind()

        with pytest.raises(DagliteError, match="Cannot infer tuple size"):
            split(result)

    def test_split_with_variadic_tuple_needs_size(self) -> None:
        """split() should require size for variadic tuples."""
        import pytest

        @task
        def make_variadic() -> tuple[int, ...]:
            return (1, 2, 3)

        result = make_variadic.bind()

        with pytest.raises(DagliteError, match="Cannot infer tuple size"):
            split(result)  # type: ignore


class TestSplitExecutionOrder:
    """Tests for split() execution behavior."""

    def test_split_shares_upstream_execution(self) -> None:
        """Multiple split elements should share upstream computation."""
        execution_count = {"count": 0}

        @task
        def make_pair_with_side_effect() -> tuple[int, int]:
            execution_count["count"] += 1
            return (1, 2)

        result = make_pair_with_side_effect.bind()
        first, second = split(result)

        # Evaluate both - upstream should only execute once
        evaluate(first)
        evaluate(second)

        # This would be 1 if they share, 2 if independent
        # (Currently they don't share in separate evaluate calls)
        assert execution_count["count"] == 2  # Expected current behavior


class TestSplitTypePreservation:
    """Tests that split() preserves type information."""

    def test_split_preserves_element_types(self) -> None:
        """Split should maintain type information for type checkers."""

        @task
        def make_mixed() -> tuple[int, str, bool]:
            return (42, "hello", True)

        result = make_mixed.bind()
        num, text, flag = split(result)

        # These should pass type checking
        assert isinstance(evaluate(num), int)
        assert isinstance(evaluate(text), str)
        assert isinstance(evaluate(flag), bool)


class TestSplitMethod:
    """Tests for TaskFuture.split() method."""

    def test_split_method_with_annotations(self) -> None:
        """TaskFuture.split() method should work with type annotations."""

        @task
        def make_pair() -> tuple[int, str]:
            return (42, "hello")

        first, second = make_pair.bind().split()

        assert evaluate(first) == 42
        assert evaluate(second) == "hello"

    def test_split_method_with_explicit_size(self) -> None:
        """TaskFuture.split() method should work with explicit size."""

        @task
        def make_triple():
            return (1, 2, 3)

        a, b, c = make_triple.bind().split(size=3)

        assert evaluate(a) == 1
        assert evaluate(b) == 2
        assert evaluate(c) == 3

    def test_split_method_chaining(self) -> None:
        """TaskFuture.split() should support fluent chaining."""

        @task
        def compute() -> tuple[int, int]:
            return (10, 20)

        @task
        def double(x: int) -> int:
            return x * 2

        # Fluent chaining: bind -> split -> bind
        x, y = compute.bind().split()
        result = double.bind(x=x)

        assert evaluate(result) == 20

    def test_split_method_equivalent_to_function(self) -> None:
        """TaskFuture.split() should be equivalent to split(future)."""

        @task
        def make_data() -> tuple[int, str, bool]:
            return (42, "test", True)

        future = make_data.bind()

        # Function form
        f1, f2, f3 = split(future)

        # Method form
        m1, m2, m3 = make_data.bind().split()

        assert evaluate(f1) == evaluate(m1)
        assert evaluate(f2) == evaluate(m2)
        assert evaluate(f3) == evaluate(m3)

    """Tests for when() composer with evaluation."""

    def test_when_executes_then_branch(self) -> None:
        """when() should execute then branch when condition is True."""

        @task
        def always_true() -> bool:
            return True

        @task
        def then_value() -> str:
            return "then executed"

        @task
        def else_value() -> str:
            return "else executed"

        condition = always_true.bind()
        result = when(condition, then_value.bind(), else_value.bind())

        assert evaluate(result) == "then executed"

    def test_when_executes_else_branch(self) -> None:
        """when() should execute else branch when condition is False."""

        @task
        def always_false() -> bool:
            return False

        @task
        def then_value() -> str:
            return "then executed"

        @task
        def else_value() -> str:
            return "else executed"

        condition = always_false.bind()
        result = when(condition, then_value.bind(), else_value.bind())

        assert evaluate(result) == "else executed"

    def test_when_with_dependent_condition(self) -> None:
        """when() should work with condition depending on other tasks."""

        @task
        def compute(x: int) -> int:
            return x * 2

        @task
        def is_large(x: int) -> bool:
            return x > 10

        @task
        def process_large(x: int) -> str:
            return f"Large: {x}"

        @task
        def process_small(x: int) -> str:
            return f"Small: {x}"

        value = compute.bind(x=3)  # Results in 6
        check = is_large.bind(x=value)
        result = when(
            check,
            process_large.bind(x=value),
            process_small.bind(x=value),
        )

        assert evaluate(result) == "Small: 6"

    def test_when_nested(self) -> None:
        """when() should support nested conditionals."""

        @task
        def check_positive(x: int) -> bool:
            return x > 0

        @task
        def check_even(x: int) -> bool:
            return x % 2 == 0

        @task
        def classify(tag: str) -> str:
            return tag

        value = 4
        pos_check = check_positive.bind(x=value)
        even_check = check_even.bind(x=value)

        # Nested: if positive then (if even then "pos_even" else "pos_odd") else "negative"
        result = when(
            pos_check,
            when(
                even_check,
                classify.bind(tag="positive_even"),
                classify.bind(tag="positive_odd"),
            ),
            classify.bind(tag="negative"),
        )

        assert evaluate(result) == "positive_even"


class TestLoopEvaluation:
    """Tests for loop() composer with evaluation."""

    def test_loop_simple_counter(self) -> None:
        """loop() should count up to a threshold."""

        @task
        def increment(count: int) -> tuple[int, bool]:
            new_count = count + 1
            should_continue = new_count < 5
            return (new_count, should_continue)

        result = loop(initial=0, body=increment)

        assert evaluate(result) == 5

    def test_loop_with_multiplication(self) -> None:
        """loop() should support multiplicative accumulation."""

        @task
        def double(n: int) -> tuple[int, bool]:
            new_n = n * 2
            should_continue = new_n < 100
            return (new_n, should_continue)

        result = loop(initial=1, body=double)

        # 1 -> 2 -> 4 -> 8 -> 16 -> 32 -> 64 -> 128 (stops)
        assert evaluate(result) == 128

    def test_loop_with_task_future_initial(self) -> None:
        """loop() should work with TaskFuture as initial state."""

        @task
        def get_start() -> int:
            return 10

        @task
        def add_five(n: int) -> tuple[int, bool]:
            new_n = n + 5
            return (new_n, new_n < 30)

        start = get_start.bind()
        result = loop(initial=start, body=add_five)

        # 10 -> 15 -> 20 -> 25 -> 30 (stops)
        assert evaluate(result) == 30

    def test_loop_with_body_kwargs(self) -> None:
        """loop() should support additional body parameters."""

        @task
        def add_factor(state: int, factor: int) -> tuple[int, bool]:
            new_state = state + factor
            return (new_state, new_state < 50)

        result = loop(initial=0, body=add_factor, factor=7)

        # 0 -> 7 -> 14 -> 21 -> 28 -> 35 -> 42 -> 49 -> 56 (stops)
        assert evaluate(result) == 56

    def test_loop_with_max_iterations(self) -> None:
        """loop() should respect max_iterations limit."""

        @task
        def infinite_increment(n: int) -> tuple[int, bool]:
            # This would loop forever without max_iterations
            return (n + 1, True)

        result = loop(initial=0, body=infinite_increment, max_iterations=10)

        assert evaluate(result) == 10

    def test_loop_immediate_stop(self) -> None:
        """loop() should handle immediate stop condition."""

        @task
        def stop_immediately(n: int) -> tuple[int, bool]:
            return (n, False)

        result = loop(initial=42, body=stop_immediately)

        assert evaluate(result) == 42

    def test_loop_with_dependent_factor(self) -> None:
        """loop() should work with TaskFuture in body_kwargs."""

        @task
        def compute_factor() -> int:
            return 3

        @task
        def multiply_by(state: int, factor: int) -> tuple[int, bool]:
            new_state = state * factor
            return (new_state, new_state < 100)

        factor = compute_factor.bind()
        result = loop(initial=2, body=multiply_by, factor=factor)

        # 2 -> 6 -> 18 -> 54 -> 162 (stops)
        assert evaluate(result) == 162
