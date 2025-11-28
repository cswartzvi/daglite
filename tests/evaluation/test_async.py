"""Asynchronous evaluation tests using evaluate_async()."""

import asyncio
import threading
import time

from daglite import evaluate_async
from daglite import task


class TestAsyncExecution:
    """Tests async evaluation with evaluate_async()."""

    def test_single_task_async(self) -> None:
        """Async evaluation succeeds for single task."""
        import asyncio

        @task
        def add(x: int, y: int) -> int:
            return x + y

        async def run():
            return await evaluate_async(add.bind(x=10, y=20))

        result = asyncio.run(run())
        assert result == 30

    def test_chain_async(self) -> None:
        """Async evaluation preserves dependency order in chains."""

        @task
        def add(x: int, y: int) -> int:
            return x + y

        @task
        def multiply(z: int, factor: int) -> int:
            return z * factor

        @task
        def subtract(value: int, decrement: int) -> int:
            return value - decrement

        added = add.bind(x=3, y=7)  # 10
        multiplied = multiply.bind(z=added, factor=4)  # 40
        subtracted = subtract.bind(value=multiplied, decrement=15)  # 25

        async def run():
            return await evaluate_async(subtracted)

        result = asyncio.run(run())
        assert result == 25

    def test_sibling_tasks_async(self) -> None:
        """Async evaluation handles multiple sibling tasks correctly."""

        threads = set()

        @task(backend="threading")
        def left() -> int:
            time.sleep(0.1)
            threads.add(threading.get_ident())
            return 5

        @task(backend="threading")
        def right() -> int:
            threads.add(threading.get_ident())
            return 10

        @task
        def combine(a: int, b: int) -> int:
            return a + b

        left_future = left.bind()
        right_future = right.bind()
        combined = combine.bind(a=left_future, b=right_future)

        async def run():
            return await evaluate_async(combined)

        result = asyncio.run(run())
        assert result == 15
        assert len(threads) == 2  # Both tasks ran in parallel threads

    def test_product_async(self) -> None:
        """Async evaluation handles parallel product operations."""

        @task
        def square(x: int) -> int:
            return x**2

        @task
        def sum_all(values: list[int]) -> int:
            return sum(values)

        squared_seq = square.product(x=[1, 2, 3, 4])  # [1, 4, 9, 16]
        total = squared_seq.join(sum_all)

        async def run():
            return await evaluate_async(total)

        result = asyncio.run(run())
        assert result == 30

    def test_zip_async(self) -> None:
        """Async evaluation handles zip operations."""

        @task
        def add(x: int, y: int) -> int:
            return x + y

        @task
        def product(values: list[int]) -> int:
            result = 1
            for v in values:
                result *= v
            return result

        added_seq = add.zip(x=[1, 2, 3], y=[10, 20, 30])  # [11, 22, 33]
        prod = added_seq.join(product)

        async def run():
            return await evaluate_async(prod)

        result = asyncio.run(run())
        assert result == 11 * 22 * 33

    def test_error_propagation_async(self) -> None:
        """Async evaluation propagates exceptions correctly."""

        @task
        def divide(x: int, y: int) -> float:
            return x / y

        divided = divide.bind(x=10, y=0)

        async def run():
            return await evaluate_async(divided)

        try:
            asyncio.run(run())
            assert False, "Should have raised ZeroDivisionError"  # pragma: no cover
        except ZeroDivisionError:
            pass  # Expected

    def test_complex_graph_async(self) -> None:
        """Async evaluation handles complex graphs with multiple paths."""

        @task
        def add(x: int, y: int) -> int:
            return x + y

        @task
        def multiply(x: int, y: int) -> int:
            return x * y

        # Diamond pattern: two paths that merge
        a = add.bind(x=1, y=2)  # 3
        b = multiply.bind(x=2, y=3)  # 6
        c = add.bind(x=a, y=b)  # 9

        async def run():
            return await evaluate_async(c)

        result = asyncio.run(run())
        assert result == 9

    def test_map_async(self) -> None:
        """Async evaluation handles map operations."""

        @task
        def double(x: int) -> int:
            return x * 2

        @task
        def square(z: int) -> int:
            return z**2

        doubled = double.product(x=[1, 2, 3])
        squared = doubled.map(square)

        async def run():
            return await evaluate_async(squared)

        result = asyncio.run(run())
        assert result == [4, 16, 36]  # [2, 4, 6] squared

    def test_join_async(self) -> None:
        """Async evaluation handles join operations."""

        @task
        def triple(x: int) -> int:
            return x * 3

        @task
        def sum_all(values: list[int]) -> int:
            return sum(values)

        tripled = triple.product(x=[1, 2, 3, 4])
        total = tripled.join(sum_all)

        async def run():
            return await evaluate_async(total)

        result = asyncio.run(run())
        assert result == 30  # (3 + 6 + 9 + 12)

    def test_map_async_with_thread_backend(self) -> None:
        """Async evaluation handles map with ThreadBackend (tests asyncio.gather path)."""

        @task(backend="threading")
        def double(x: int) -> int:
            return x * 2

        @task(backend="threading")
        def square(z: int) -> int:
            return z**2

        doubled = double.product(x=[1, 2, 3])
        squared = doubled.map(square)

        async def run():
            return await evaluate_async(squared)

        result = asyncio.run(run())
        assert result == [4, 16, 36]  # [2, 4, 6] squared
