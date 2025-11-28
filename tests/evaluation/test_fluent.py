"""Tests for fluent API (.then(), .map(), .join() with kwargs)."""

from daglite import evaluate
from daglite import task


class TestFluentAPI:
    """Tests for fluent API (.then(), .map(), .join() with kwargs)."""

    def test_then_simple_chain(self) -> None:
        """Fluent .then() chains tasks linearly."""

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
        """Fluent .then() handles multiple inline kwargs."""

        @task
        def start() -> int:
            return 5

        @task
        def compute(x: int, factor: int, offset: int) -> int:
            return x * factor + offset

        result = evaluate(start.bind().then(compute, factor=3, offset=7))
        assert result == 22  # 5*3 + 7

    def test_then_with_fixed_task(self) -> None:
        """Fluent .then() works with pre-fixed tasks."""

        @task
        def start() -> int:
            return 10

        @task
        def scale(x: int, factor: int, offset: int) -> int:
            return x * factor + offset

        fixed_scale = scale.fix(offset=5)
        result = evaluate(start.bind().then(fixed_scale, factor=2))
        assert result == 25  # 10*2 + 5

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
