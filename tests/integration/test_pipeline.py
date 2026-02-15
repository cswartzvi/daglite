"""Integration tests for pipeline evaluation with the @pipeline decorator."""

import asyncio

from daglite import Dag
from daglite import pipeline
from daglite import task


class TestPipelineEvaluation:
    """Tests for pipeline evaluation."""

    def test_simple_pipeline_evaluation(self) -> None:
        """Pipeline evaluation works with basic TaskFuture."""

        @task
        def add(x: int, y: int) -> int:
            return x + y

        @pipeline
        def simple_pipeline(x: int, y: int) -> Dag[int]:
            return add(x=x, y=y)

        graph = simple_pipeline(5, 10)
        result = graph.run()
        assert result == 15

    def test_pipeline_with_chained_tasks(self) -> None:
        """Pipeline evaluation works with chained tasks."""

        @task
        def add(x: int, y: int) -> int:
            return x + y

        @task
        def multiply(x: int, factor: int) -> int:
            return x * factor

        @pipeline
        def chained_pipeline(x: int, y: int, factor: int) -> Dag[int]:
            sum_result = add(x=x, y=y)
            return multiply(x=sum_result, factor=factor)

        graph = chained_pipeline(5, 10, 3)
        result = graph.run()
        assert result == 45

    def test_pipeline_with_map_task_future(self) -> None:
        """Pipeline evaluation works with MapTaskFuture."""

        @task
        def square(x: int) -> int:
            return x * x

        @pipeline
        def map_pipeline(values: list[int]) -> Dag[int]:
            return square.map(x=values)

        graph = map_pipeline([1, 2, 3, 4])
        result = graph.run()
        assert result == [1, 4, 9, 16]

    def test_pipeline_with_map_and_reduce(self) -> None:
        """Pipeline evaluation works with map and reduce pattern."""

        @task
        def double(x: int) -> int:
            return x * 2

        @task
        def sum_all(values: list[int]) -> int:
            return sum(values)

        @pipeline
        def map_reduce_pipeline(values: list[int]) -> Dag[int]:
            doubled = double.map(x=values)
            return doubled.join(sum_all)

        graph = map_reduce_pipeline([1, 2, 3, 4])
        result = graph.run()
        assert result == 20

    def test_pipeline_with_default_parameters(self) -> None:
        """Pipeline evaluation works with default parameters."""

        @task
        def multiply(x: int, factor: int) -> int:
            return x * factor

        @pipeline
        def pipeline_with_defaults(x: int, factor: int = 2) -> Dag[int]:
            return multiply(x=x, factor=factor)

        # Use default
        graph1 = pipeline_with_defaults(10)
        result1 = graph1.run()
        assert result1 == 20

        # Override default
        graph2 = pipeline_with_defaults(10, factor=3)
        result2 = graph2.run()
        assert result2 == 30

    def test_pipeline_async_evaluation(self) -> None:
        """Pipeline evaluation works with async execution."""

        @task
        def add(x: int, y: int) -> int:
            return x + y

        @task
        def multiply(x: int, y: int) -> int:
            return x * y

        @pipeline
        def async_pipeline(a: int, b: int, c: int) -> Dag[int]:
            # Create parallel branches
            sum_result = add(x=a, y=b)
            prod_result = multiply(x=b, y=c)
            # Merge
            return add(x=sum_result, y=prod_result)

        graph = async_pipeline(1, 2, 3)

        async def run():
            return await graph.run_async()

        result = asyncio.run(run())
        assert result == 9  # (1+2) + (2*3) = 3 + 6 = 9


class TestPipelineRunMethod:
    """Tests for Pipeline.run() and Pipeline.run_async() methods."""

    def test_pipeline_run_basic(self) -> None:
        """Pipeline.run() evaluates in a single step."""

        @task
        def add(x: int, y: int) -> int:
            return x + y

        @pipeline
        def simple_pipeline(x: int, y: int) -> Dag[int]:
            return add(x=x, y=y)

        result = simple_pipeline.run(5, 10)
        assert result == 15

    def test_pipeline_run_with_chained_tasks(self) -> None:
        """Pipeline.run() works with chained tasks."""

        @task
        def add(x: int, y: int) -> int:
            return x + y

        @task
        def multiply(x: int, factor: int) -> int:
            return x * factor

        @pipeline
        def chained_pipeline(x: int, y: int, factor: int) -> Dag[int]:
            sum_result = add(x=x, y=y)
            return multiply(x=sum_result, factor=factor)

        result = chained_pipeline.run(5, 10, 3)
        assert result == 45

    def test_pipeline_run_with_map(self) -> None:
        """Pipeline.run() works with MapTaskFuture."""

        @task
        def square(x: int) -> int:
            return x * x

        @pipeline
        def map_pipeline(values: list[int]) -> Dag[int]:
            return square.map(x=values)

        result = map_pipeline.run([1, 2, 3, 4])
        assert result == [1, 4, 9, 16]

    def test_pipeline_run_with_map_reduce(self) -> None:
        """Pipeline.run() works with map-reduce pattern."""

        @task
        def double(x: int) -> int:
            return x * 2

        @task
        def sum_all(values: list[int]) -> int:
            return sum(values)

        @pipeline
        def map_reduce_pipeline(values: list[int]) -> Dag[int]:
            doubled = double.map(x=values)
            return doubled.join(sum_all)

        result = map_reduce_pipeline.run([1, 2, 3, 4])
        assert result == 20

    def test_pipeline_run_with_defaults(self) -> None:
        """Pipeline.run() works with default parameters."""

        @task
        def multiply(x: int, factor: int) -> int:
            return x * factor

        @pipeline
        def pipeline_with_defaults(x: int, factor: int = 2) -> Dag[int]:
            return multiply(x=x, factor=factor)

        assert pipeline_with_defaults.run(10) == 20
        assert pipeline_with_defaults.run(10, factor=3) == 30

    def test_pipeline_run_async(self) -> None:
        """Pipeline.run_async() evaluates asynchronously."""

        @task
        def add(x: int, y: int) -> int:
            return x + y

        @task
        def multiply(x: int, y: int) -> int:
            return x * y

        @pipeline
        def async_pipeline(a: int, b: int, c: int) -> Dag[int]:
            sum_result = add(x=a, y=b)
            prod_result = multiply(x=b, y=c)
            return add(x=sum_result, y=prod_result)

        async def run():
            return await async_pipeline.run_async(1, 2, 3)

        result = asyncio.run(run())
        assert result == 9  # (1+2) + (2*3) = 3 + 6 = 9

    def test_pipeline_run_with_kwargs(self) -> None:
        """Pipeline.run() works with keyword arguments."""

        @task
        def add(x: int, y: int) -> int:
            return x + y

        @pipeline
        def simple_pipeline(x: int, y: int) -> Dag[int]:
            return add(x=x, y=y)

        result = simple_pipeline.run(x=5, y=10)
        assert result == 15
