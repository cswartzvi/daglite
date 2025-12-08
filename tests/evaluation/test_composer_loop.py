"""Integration tests for when() and loop() composers with evaluate()."""

from daglite import evaluate
from daglite import task
from daglite.composers import loop
from daglite.composers import when


class TestWhenEvaluation:
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


class TestComposerCombinations:
    """Test combining different composers."""

    def test_when_with_loop_in_branch(self) -> None:
        """when() branches can contain loop() composers."""

        @task
        def should_loop(x: int) -> bool:
            return x < 5

        @task
        def increment(n: int) -> tuple[int, bool]:
            return (n + 1, n < 10)

        @task
        def identity(n: int) -> int:
            return n

        value = 3
        condition = should_loop.bind(x=value)

        result = when(
            condition,
            loop(initial=value, body=increment),
            identity.bind(n=value),
        )

        # value=3, condition=True, so loop: 3->4->5->6->7->8->9->10->11
        assert evaluate(result) == 11

    def test_loop_with_conditional_body(self) -> None:
        """loop() can have conditional logic using when()."""

        @task
        def is_even(n: int) -> bool:
            return n % 2 == 0

        @task
        def add_one(n: int) -> int:
            return n + 1

        @task
        def add_two(n: int) -> int:
            return n + 2

        @task
        def conditional_add(state: int) -> tuple[int, bool]:
            check = is_even.bind(n=state)
            new_state_future = when(
                check,
                add_one.bind(n=state),
                add_two.bind(n=state),
            )
            new_state = evaluate(new_state_future)
            return (new_state, new_state < 20)

        result = loop(initial=0, body=conditional_add)

        # 0(even)->1->3->5->7->9->11->13->15->17->19->21(stop)
        assert evaluate(result) == 21
