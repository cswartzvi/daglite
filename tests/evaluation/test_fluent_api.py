"""Tests for fluent API (.then(), .map(), .join() with kwargs)."""

import pytest

from daglite import evaluate
from daglite import task
from daglite.composers import loop
from daglite.composers import split
from daglite.composers import when


class TestTaskFutureThen:
    """Tests for TaskFuture.then() chaining."""

    def test_then_simple_chain(self) -> None:
        """TaskFuture.then() chains tasks linearly."""

        @task
        def fetch(source: str) -> int:
            return len(source)

        @task
        def double(x: int) -> int:
            return x * 2

        @task
        def add(x: int, y: int) -> int:
            return x + y

        result = evaluate(fetch.bind(source="hello").then(double).then(add, y=10))
        assert result == 20  # len("hello")*2 + 10 = 5*2 + 10 = 20

    def test_then_with_multiple_kwargs(self) -> None:
        """TaskFuture.then() handles multiple inline kwargs."""

        @task
        def start() -> int:
            return 5

        @task
        def compute(x: int, factor: int, offset: int) -> int:
            return x * factor + offset

        result = evaluate(start.bind().then(compute, factor=3, offset=7))
        assert result == 22  # 5*3 + 7

    def test_then_with_fixed_task(self) -> None:
        """TaskFuture.then() works with pre-fixed tasks."""

        @task
        def start() -> int:
            return 10

        @task
        def scale(x: int, factor: int, offset: int) -> int:
            return x * factor + offset

        fixed_scale = scale.fix(offset=5)
        result = evaluate(start.bind().then(fixed_scale, factor=2))
        assert result == 25  # 10*2 + 5

    def test_then_basic_chain(self) -> None:
        """TaskFuture.then() chains correctly."""

        @task
        def double(x: int) -> int:
            return x * 2

        @task
        def add_ten(x: int) -> int:
            return x + 10

        @task
        def to_string(x: int) -> str:
            return f"Result: {x}"

        result = double.bind(x=5).then(add_ten).then(to_string)
        assert evaluate(result) == "Result: 20"

    def test_then_return_type_preservation(self) -> None:
        """TaskFuture.then() preserves return types correctly."""

        @task
        def double(x: int) -> int:
            return x * 2

        @task
        def to_string(x: int) -> str:
            return f"Result: {x}"

        result = double.bind(x=5).then(to_string)
        from daglite.tasks import TaskFuture

        assert isinstance(result, TaskFuture)
        assert isinstance(evaluate(result), str)


class TestMapTaskFutureThen:
    """Tests for MapTaskFuture.then() chaining."""

    def test_map_task_future_then(self) -> None:
        """MapTaskFuture.then() passes entire list to next task."""

        @task
        def double(x: int) -> int:
            return x * 2

        @task
        def sum_list(xs: list[int]) -> int:
            return sum(xs)

        numbers = double.product(x=[1, 2, 3, 4])
        result = numbers.then(sum_list)
        assert evaluate(result) == 20  # (2 + 4 + 6 + 8)

    def test_map_task_future_then_with_kwargs(self) -> None:
        """MapTaskFuture.then() with additional kwargs."""

        @task
        def double(x: int) -> int:
            return x * 2

        @task
        def weighted_sum(xs: list[int], weight: float) -> float:
            return sum(xs) * weight

        numbers = double.product(x=[1, 2, 3])
        result = numbers.then(weighted_sum, weight=0.5)
        assert evaluate(result) == 6.0  # (2 + 4 + 6) * 0.5

    def test_map_task_future_then_equivalent_to_join(self) -> None:
        """MapTaskFuture.then() is equivalent to .join()."""

        @task
        def double(x: int) -> int:
            return x * 2

        @task
        def sum_list(xs: list[int]) -> int:
            return sum(xs)

        numbers = double.product(x=[1, 2, 3, 4])
        result_then = numbers.then(sum_list)
        result_join = numbers.join(sum_list)

        assert evaluate(result_then) == evaluate(result_join)

    def test_then_return_type_preservation(self) -> None:
        """MapTaskFuture.then() preserves return types correctly."""

        @task
        def double(x: int) -> int:
            return x * 2

        @task
        def sum_list(xs: list[int]) -> int:
            return sum(xs)

        numbers = double.product(x=[1, 2, 3])
        total = numbers.then(sum_list)
        from daglite.tasks import TaskFuture

        assert isinstance(total, TaskFuture)
        assert isinstance(evaluate(total), int)


class TestConditionalFutureThen:
    """Tests for ConditionalFuture.then() chaining."""

    def test_conditional_future_then_true_branch(self) -> None:
        """ConditionalFuture.then() chains from true branch."""

        @task
        def double(x: int) -> int:
            return x * 2

        @task
        def add_ten(x: int) -> int:
            return x + 10

        @task
        def is_positive(x: int) -> bool:
            return x > 0

        @task
        def to_string(x: int) -> str:
            return f"Result: {x}"

        value = double.bind(x=10)  # 20
        check = is_positive.bind(x=value)
        conditional = when(
            check,
            then_branch=add_ten.bind(x=value),  # 30
            else_branch=double.bind(x=value),  # 40
        )
        result = conditional.then(to_string)
        assert evaluate(result) == "Result: 30"

    def test_conditional_future_then_false_branch(self) -> None:
        """ConditionalFuture.then() chains from false branch."""

        @task
        def double(x: int) -> int:
            return x * 2

        @task
        def is_positive(x: int) -> bool:
            return x > 0

        @task
        def to_string(x: int) -> str:
            return f"Result: {x}"

        value = double.bind(x=-5)  # -10
        check = is_positive.bind(x=value)
        conditional = when(
            check,
            then_branch=double.bind(x=value),  # -20 (not taken)
            else_branch=double.bind(x=value),  # -20
        )
        result = conditional.then(to_string)
        assert evaluate(result) == "Result: -20"

    def test_conditional_future_then_with_kwargs(self) -> None:
        """ConditionalFuture.then() with additional kwargs."""

        @task
        def double(x: int) -> int:
            return x * 2

        @task
        def add_ten(x: int) -> int:
            return x + 10

        @task
        def is_positive(x: int) -> bool:
            return x > 0

        @task
        def format_with_prefix(x: int, prefix: str) -> str:
            return f"{prefix}{x}"

        value = double.bind(x=5)
        check = is_positive.bind(x=value)
        conditional = when(
            check,
            then_branch=add_ten.bind(x=value),
            else_branch=double.bind(x=value),
        )
        result = conditional.then(format_with_prefix, prefix="Value: ")
        assert evaluate(result) == "Value: 20"


class TestLoopFutureThen:
    """Tests for LoopFuture.then() chaining."""

    def test_loop_future_then(self) -> None:
        """LoopFuture.then() chains from final loop state."""

        @task
        def accumulate(state: int) -> tuple[int, bool]:
            new_state = state + 1
            should_continue = new_state < 5
            return (new_state, should_continue)

        @task
        def to_string(x: int) -> str:
            return f"Result: {x}"

        result = loop(initial=0, body=accumulate, max_iterations=10).then(to_string)
        assert evaluate(result) == "Result: 5"

    def test_loop_future_then_with_kwargs(self) -> None:
        """LoopFuture.then() with additional kwargs."""

        @task
        def accumulate(state: int) -> tuple[int, bool]:
            new_state = state * 2
            should_continue = new_state < 100
            return (new_state, should_continue)

        @task
        def scale(x: int, factor: int) -> int:
            return x * factor

        result = loop(initial=1, body=accumulate, max_iterations=10).then(scale, factor=2)
        assert evaluate(result) == 256  # Final state 128 * 2


class TestMapOperations:
    """Tests for .map() fluent API."""

    def test_map_with_kwargs(self) -> None:
        """Fluent .map() accepts inline kwargs."""

        @task
        def scale(x: int, factor: int) -> int:
            return x * factor

        @task
        def identity(x: int) -> int:
            return x

        result = evaluate(identity.product(x=[1, 2, 3]).map(scale, factor=2))
        assert result == [2, 4, 6]  # Each element * 2

    def test_map_chain_with_kwargs(self) -> None:
        """Fluent .map() chains with inline kwargs."""

        @task
        def identity(x: int) -> int:
            return x

        @task
        def add(x: int, y: int) -> int:
            return x + y

        @task
        def multiply(x: int, factor: int) -> int:
            return x * factor

        # [1, 2, 3] -> add(y=10) -> [11, 12, 13] -> multiply(factor=2) -> [22, 24, 26]
        result = evaluate(identity.product(x=[1, 2, 3]).map(add, y=10).map(multiply, factor=2))
        assert result == [22, 24, 26]


class TestJoinOperations:
    """Tests for .join() fluent API."""

    def test_join_with_kwargs(self) -> None:
        """Fluent .join() accepts inline kwargs."""

        @task
        def square(x: int) -> int:
            return x**2

        @task
        def weighted_sum(values: list[int], weight: float) -> float:
            return sum(values) * weight

        result = evaluate(square.product(x=[1, 2, 3, 4]).join(weighted_sum, weight=2.0))
        assert result == 60.0  # (1 + 4 + 9 + 16) * 2.0

    def test_map_join_combined_with_kwargs(self) -> None:
        """Fluent .map() and .join() work together with kwargs."""

        @task
        def identity(x: int) -> int:
            return x

        @task
        def add(x: int, y: int) -> int:
            return x + y

        @task
        def multiply(x: int, factor: int) -> int:
            return x * factor

        @task
        def reduce_with_offset(values: list[int], offset: int) -> int:
            return sum(values) + offset

        # [1, 2, 3] -> add(y=5) -> [6, 7, 8] -> multiply(factor=2) -> [12, 14, 16] -> sum+10 -> 52
        result = evaluate(
            identity.product(x=[1, 2, 3])
            .map(add, y=5)
            .map(multiply, factor=2)
            .join(reduce_with_offset, offset=10)
        )
        assert result == 52  # (12 + 14 + 16) + 10


class TestComplexPipelines:
    """Tests for complex fluent pipeline combinations."""

    def test_complete_fluent_pipeline(self) -> None:
        """Complete fluent pipeline using .then(), .map(), and .join()."""

        @task
        def fetch_range(count: int) -> list[int]:
            return list(range(1, count + 1))

        @task
        def scale(x: int, factor: int) -> int:
            return x * factor

        @task
        def add_values(x: int, offset: int) -> int:
            return x + offset

        @task
        def compute_total(values: list[int], multiplier: int) -> int:
            return sum(values) * multiplier

        # Data: [1, 2, 3, 4]
        # scale with [2, 3]: Cartesian = [2,4,6,8,3,6,9,12]
        # add 10: [12,14,16,18,13,16,19,22]
        # sum = 130, * 2 = 260
        result = evaluate(
            scale.product(x=fetch_range.bind(count=4), factor=[2, 3])
            .map(add_values, offset=10)
            .join(compute_total, multiplier=2)
        )
        assert result == 260

    def test_fluent_with_fixed_tasks(self) -> None:
        """Fluent API works with pre-fixed tasks."""

        @task
        def scale(x: int, factor: int) -> int:
            return x * factor

        @task
        def add(x: int, offset: int) -> int:
            return x + offset

        @task
        def sum_with_multiplier(values: list[int], multiplier: int) -> int:
            return sum(values) * multiplier

        # Use zip to avoid Cartesian product, fluent chain with pre-fixed task
        # [1,2,3] zip [2,3,4] -> [2, 6, 12] -> add 10 -> [12, 16, 22] -> sum*2 -> 100
        result = evaluate(
            scale.zip(x=[1, 2, 3], factor=[2, 3, 4])
            .map(add.fix(offset=10))
            .join(sum_with_multiplier, multiplier=2)
        )
        assert result == 100  # (12 + 16 + 22) * 2

    def test_full_fluent_chain_with_extend(self) -> None:
        """Complete fluent chain: bind -> then -> product -> map -> join."""

        @task
        def start(x: int) -> int:
            return x * 2

        @task
        def increment(x: int, amount: int) -> int:
            return x + amount

        @task
        def transform_list(value: int) -> list[int]:
            return [value, value + 1, value + 2]

        @task
        def scale(x: int, factor: int) -> int:
            return x * factor

        @task
        def add_offset(x: int, offset: int) -> int:
            return x + offset

        @task
        def sum_with_bonus(values: list[int], bonus: int) -> int:
            return sum(values) + bonus

        # start(5) -> 10 -> increment(amount=3) -> 13 -> transform_list -> [13, 14, 15]
        # -> product/scale(factor=[2,3]) -> [26,28,30,39,42,45] -> add 10 -> [36,38,40,49,52,55]
        # -> sum + 100 -> 370
        list_future = start.bind(x=5).then(increment, amount=3).then(transform_list)
        result = evaluate(
            scale.product(x=list_future, factor=[2, 3])
            .map(add_offset, offset=10)
            .join(sum_with_bonus, bonus=100)
        )
        assert result == 370  # (36+38+40+49+52+55) + 100

    def test_full_fluent_chain_with_zip(self) -> None:
        """Complete fluent chain: bind -> then -> zip -> map -> join."""

        @task
        def fetch_base(count: int) -> list[int]:
            return list(range(1, count + 1))

        @task
        def fetch_multipliers(count: int) -> list[int]:
            return [10] * count

        @task
        def multiply(x: int, factor: int) -> int:
            return x * factor

        @task
        def add_offset(x: int, offset: int) -> int:
            return x + offset

        @task
        def product(values: list[int]) -> int:
            result = 1
            for v in values:
                result *= v
            return result

        # fetch [1,2,3], fetch [10,10,10] -> zip multiply -> [10, 20, 30]
        # -> add_offset(5) -> [15, 25, 35] -> product -> 13125
        base = fetch_base.bind(count=3)
        multipliers = fetch_multipliers.bind(count=3)

        result = evaluate(
            multiply.zip(x=base, factor=multipliers).map(add_offset, offset=5).join(product)
        )
        assert result == 13125  # 15 * 25 * 35

    def test_mixed_chaining(self) -> None:
        """Test chaining across different future types."""

        @task
        def double(x: int) -> int:
            return x * 2

        @task
        def sum_list(xs: list[int]) -> int:
            return sum(xs)

        @task
        def add_ten(x: int) -> int:
            return x + 10

        @task
        def is_positive(x: int) -> bool:
            return x > 0

        @task
        def to_string(x: int) -> str:
            return f"Result: {x}"

        # Start with product -> map -> then (reduces to single value)
        numbers = double.product(x=[1, 2, 3])
        total = numbers.then(sum_list)  # 12

        # Use that in a conditional
        check = is_positive.bind(x=total)
        conditional = when(
            check,
            then_branch=add_ten.bind(x=total),  # 22
            else_branch=double.bind(x=total),  # 24
        )

        # Chain from conditional
        result = conditional.then(to_string)
        assert evaluate(result) == "Result: 22"

    def test_split_then_chain(self) -> None:
        """Test chaining after split."""

        @task
        def make_pair(x: int) -> tuple[int, int]:
            return (x, x * 2)

        @task
        def double(x: int) -> int:
            return x * 2

        @task
        def add_ten(x: int) -> int:
            return x + 10

        @task
        def to_string(x: int) -> str:
            return f"Result: {x}"

        pair = make_pair.bind(x=10)
        first, second = split(pair)

        # Chain from each split element
        result1 = first.then(double).then(to_string)
        result2 = second.then(add_ten).then(to_string)

        assert evaluate(result1) == "Result: 20"
        assert evaluate(result2) == "Result: 30"


class TestErrorCases:
    """Tests for error handling in fluent API."""

    def test_then_no_unbound_params(self) -> None:
        """Test that .then() raises error when task has no unbound parameters."""

        @task
        def double(x: int) -> int:
            return x * 2

        @task
        def no_params() -> int:
            return 42

        value = double.bind(x=5)

        with pytest.raises(Exception, match="no unbound parameters"):
            value.then(no_params)

    def test_then_multiple_unbound_params(self) -> None:
        """Test that .then() raises error when task has multiple unbound parameters."""

        @task
        def double(x: int) -> int:
            return x * 2

        @task
        def two_params(x: int, y: int) -> int:
            return x + y

        value = double.bind(x=5)

        with pytest.raises(Exception, match="exactly one"):
            value.then(two_params)

    def test_then_with_fixed_param_task(self) -> None:
        """Test .then() works with FixedParamTask."""

        @task
        def double(x: int) -> int:
            return x * 2

        @task
        def scale(x: int, factor: int) -> int:
            return x * factor

        value = double.bind(x=5)
        fixed_scale = scale.fix(factor=3)
        result = value.then(fixed_scale)

        assert evaluate(result) == 30
