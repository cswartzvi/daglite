"""Tests for generator materialization."""

import asyncio
from typing import Iterator

from daglite import evaluate
from daglite import evaluate_async
from daglite import task


class TestGeneratorMaterialization:
    """Tests that generators are automatically materialized to lists."""

    def test_generator_is_materialized_sync(self) -> None:
        """Generators returned from tasks are materialized to lists in sync execution."""

        @task
        def generate_numbers(n: int) -> Iterator[int]:
            for i in range(n):
                yield i * 2

        result = evaluate(generate_numbers.bind(n=5))
        assert result == [0, 2, 4, 6, 8]
        assert isinstance(result, list)

    def test_generator_reusable_by_multiple_consumers(self) -> None:
        """Materialized generators can be consumed by multiple downstream tasks."""

        @task
        def generate_numbers(n: int) -> Iterator[int]:
            for i in range(n):
                yield i

        @task
        def sum_values(values: list[int]) -> int:
            return sum(values)

        @task
        def count_values(values: list[int]) -> int:
            return len(values)

        @task
        def combine(total: int, count: int) -> tuple[int, int]:
            return (total, count)

        numbers = generate_numbers.bind(n=5)
        total = sum_values.bind(values=numbers)
        count = count_values.bind(values=numbers)
        result_future = combine.bind(total=total, count=count)

        result = evaluate(result_future)
        assert result == (10, 5)  # sum([0,1,2,3,4]) = 10, len = 5

    def test_generator_in_map_operation(self) -> None:
        """Generators in map operations are materialized."""

        @task
        def generate_range(n: int) -> Iterator[int]:
            for i in range(n):
                yield i

        @task
        def sum_values(values: list[int]) -> int:
            return sum(values)

        # Multiple generators from product operation
        ranges = generate_range.product(n=[3, 4, 5])
        totals = ranges.map(sum_values)

        result = evaluate(totals)
        # [0,1,2] -> 3, [0,1,2,3] -> 6, [0,1,2,3,4] -> 10
        assert result == [3, 6, 10]

    def test_generator_is_materialized_async(self) -> None:
        """Generators are materialized in async execution."""

        @task
        def generate_numbers(n: int) -> Iterator[int]:
            for i in range(n):
                yield i * 3

        async def run():
            return await evaluate_async(generate_numbers.bind(n=4))

        result = asyncio.run(run())
        assert result == [0, 3, 6, 9]
        assert isinstance(result, list)

    def test_non_generator_iterables_unchanged(self) -> None:
        """Non-generator iterables like lists are not affected."""

        @task
        def return_list(n: int) -> list[int]:
            return [i * 2 for i in range(n)]

        @task
        def return_tuple(n: int) -> tuple[int, ...]:
            return tuple(i * 2 for i in range(n))

        list_result = evaluate(return_list.bind(n=3))
        assert list_result == [0, 2, 4]
        assert isinstance(list_result, list)

        tuple_result = evaluate(return_tuple.bind(n=3))
        assert tuple_result == (0, 2, 4)
        assert isinstance(tuple_result, tuple)

    def test_strings_not_materialized(self) -> None:
        """Strings (which are iterable) are not converted to lists."""

        @task
        def return_string() -> str:
            return "hello"

        result = evaluate(return_string.bind())
        assert result == "hello"
        assert isinstance(result, str)

    def test_generator_with_then_chain(self) -> None:
        """Generator materialization works with .then() chains."""

        @task
        def generate_numbers(n: int) -> Iterator[int]:
            for i in range(n):
                yield i

        @task
        def double_all(values: list[int]) -> list[int]:
            return [v * 2 for v in values]

        result = evaluate(generate_numbers.bind(n=4).then(double_all))
        assert result == [0, 2, 4, 6]

    def test_generator_type_inference(self) -> None:
        """Type system correctly infers list[T] from Iterator[T] return."""
        from typing import Generator

        @task
        def return_iterator(n: int) -> Iterator[int]:
            for i in range(n):
                yield i

        @task
        def return_generator(n: int) -> Generator[str, None, None]:
            for i in range(n):
                yield str(i)

        # Type checker should infer list[int] and list[str]
        iter_result = evaluate(return_iterator.bind(n=3))
        gen_result = evaluate(return_generator.bind(n=3))

        # These should work at runtime (list operations)
        assert isinstance(iter_result, list)
        assert isinstance(gen_result, list)
        assert iter_result == [0, 1, 2]
        assert gen_result == ["0", "1", "2"]

        # Verify list operations work
        iter_result.append(99)
        gen_result.append("99")
        assert len(iter_result) == 4
        assert len(gen_result) == 4
