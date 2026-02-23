"""Unit tests for IterTaskFuture and IterNode."""

from typing import AsyncIterator, Iterator

import pytest

from daglite import task
from daglite.exceptions import ParameterError
from daglite.exceptions import TaskError
from daglite.futures.iter_future import IterTaskFuture
from daglite.graph.nodes.iter_node import IterNode


class TestIterTaskFutureBuildNode:
    """Tests for IterTaskFuture.build_node()."""

    def test_build_node_creates_iter_node(self) -> None:
        """IterTaskFuture.build_node() produces an IterNode with correct fields."""

        @task
        def generate(n: int) -> Iterator[int]:
            yield from range(n)

        future = generate.iter(n=5)
        assert isinstance(future, IterTaskFuture)

        node = future.build_node()
        assert isinstance(node, IterNode)
        assert node.kind == "iter"
        assert node.name == "generate"
        assert node.func is generate.func
        assert node.id == future.id

    def test_build_node_preserves_task_options(self) -> None:
        """IterNode inherits retries, timeout, cache from the task."""

        @task(retries=3, timeout=10.0, cache=True, cache_ttl=60)
        def generate(n: int) -> Iterator[int]:
            yield from range(n)

        node = generate.iter(n=5).build_node()
        assert node.retries == 3
        assert node.timeout == 10.0
        assert node.cache is True
        assert node.cache_ttl == 60

    def test_build_node_kwargs_resolved(self) -> None:
        """IterNode kwargs are NodeInput instances (values)."""

        @task
        def generate(n: int, prefix: str) -> Iterator[str]:
            for i in range(n):
                yield f"{prefix}_{i}"

        node = generate.iter(n=3, prefix="item").build_node()
        assert "n" in node.kwargs
        assert "prefix" in node.kwargs

    def test_iter_with_upstream_future(self) -> None:
        """IterTaskFuture can accept upstream TaskFutures as arguments."""

        @task
        def get_count() -> int:
            return 10

        @task
        def generate(n: int) -> Iterator[int]:
            yield from range(n)

        count = get_count()
        future = generate.iter(n=count)
        node = future.build_node()

        # Should have a reference dependency on get_count
        assert len(node.get_dependencies()) == 1


class TestIterValidation:
    """Tests for .iter() parameter validation."""

    def test_iter_validates_non_sync_generator_function(self) -> None:
        """iter() rejects tasks whose functions don't return sync generators."""

        @task
        def not_a_generator(n: int) -> int:
            return n * 2

        with pytest.raises(TaskError, match="does not return a generator"):
            not_a_generator.iter(n=5)

    def test_iter_validates_async_generator_function(self) -> None:
        """iter() rejects tasks whose functions return async generators."""

        @task
        async def async_generator(n: int) -> AsyncIterator[int]:
            for i in range(n):
                yield i

        with pytest.raises(TaskError, match="does not return a generator"):
            async_generator.iter(n=5)

    def test_iter_validates_missing_params(self) -> None:
        """iter() rejects calls with missing parameters."""

        @task
        def generate(n: int) -> Iterator[int]:
            yield from range(n)

        with pytest.raises(ParameterError):
            generate.iter()

    def test_iter_validates_invalid_params(self) -> None:
        """iter() rejects unknown parameters."""

        @task
        def generate(n: int) -> Iterator[int]:
            yield from range(n)

        with pytest.raises(ParameterError):
            generate.iter(n=5, unknown=10)

    def test_iter_partial_then_iter(self) -> None:
        """PartialTask.iter() works with pre-fixed parameters."""

        @task
        def generate(n: int, prefix: str) -> Iterator[str]:
            for i in range(n):
                yield f"{prefix}_{i}"

        partial = generate.partial(prefix="item")
        future = partial.iter(n=3)
        assert isinstance(future, IterTaskFuture)

    def test_mixed_iter_and_non_iter_in_map_raises(self) -> None:
        """Mixing .iter() with other mapped arguments raises ParameterError."""

        @task
        def generate(n: int) -> Iterator[int]:
            yield from range(n)

        @task
        def combine(x: int, y: int) -> int:
            return x + y

        with pytest.raises(ParameterError, match="iter.*must be the only mapped argument"):
            combine.map(x=generate.iter(n=5), y=[1, 2, 3])


class TestIterTaskFutureFluentAPI:
    """Tests for .then(), .join(), .reduce() on IterTaskFuture."""

    def test_then_returns_map_task_future(self) -> None:
        """IterTaskFuture.then() returns a MapTaskFuture."""
        from daglite.futures.map_future import MapTaskFuture

        @task
        def generate(n: int) -> Iterator[int]:
            yield from range(n)

        @task
        def double(x: int) -> int:
            return x * 2

        result = generate.iter(n=5).then(double)
        assert isinstance(result, MapTaskFuture)
        assert result.task is double
        assert result.mode == "product"

    def test_then_with_partial_task(self) -> None:
        """IterTaskFuture.then() works with a PartialTask."""
        from daglite.futures.map_future import MapTaskFuture

        @task
        def generate(n: int) -> Iterator[int]:
            yield from range(n)

        @task
        def add(x: int, offset: int) -> int:
            return x + offset

        result = generate.iter(n=5).then(add.partial(offset=10))
        assert isinstance(result, MapTaskFuture)
        assert "offset" in result.fixed_kwargs

    def test_then_mapped_kwarg_references_iter_future(self) -> None:
        """The MapTaskFuture returned by .then() has iter future as upstream."""

        @task
        def generate(n: int) -> Iterator[int]:
            yield from range(n)

        @task
        def double(x: int) -> int:
            return x * 2

        iter_future = generate.iter(n=5)
        map_future = iter_future.then(double)
        assert iter_future in map_future.mapped_kwargs.values()

    def test_join_returns_task_future(self) -> None:
        """IterTaskFuture.join() returns a TaskFuture."""
        from daglite.futures.task_future import TaskFuture

        @task
        def generate(n: int) -> Iterator[int]:
            yield from range(n)

        @task
        def sum_values(values: list[int]) -> int:
            return sum(values)

        result = generate.iter(n=5).join(sum_values)
        assert isinstance(result, TaskFuture)

    def test_join_with_partial_task(self) -> None:
        """IterTaskFuture.join() works with a PartialTask."""
        from daglite.futures.task_future import TaskFuture

        @task
        def generate(n: int) -> Iterator[int]:
            yield from range(n)

        @task
        def sum_offset(values: list[int], offset: int) -> int:
            return sum(values) + offset

        result = generate.iter(n=5).join(sum_offset.partial(offset=10))
        assert isinstance(result, TaskFuture)
        assert result.kwargs["offset"] == 10

    def test_reduce_returns_reduce_future(self) -> None:
        """IterTaskFuture.reduce() returns a ReduceFuture."""
        from daglite.futures.reduce_future import ReduceFuture

        @task
        def generate(n: int) -> Iterator[int]:
            yield from range(n)

        @task
        def add_acc(acc: int, item: int) -> int:
            return acc + item

        result = generate.iter(n=5).reduce(add_acc, initial=0)
        assert isinstance(result, ReduceFuture)
        assert result.initial == 0

    def test_reduce_identity_node_is_hidden(self) -> None:
        """IterTaskFuture.reduce() marks the internal identity MapTaskFuture as hidden."""
        from daglite.graph.builder import build_graph

        @task
        def generate(n: int) -> Iterator[int]:
            yield from range(n)

        @task
        def add_acc(acc: int, item: int) -> int:
            return acc + item

        result = generate.iter(n=5).reduce(add_acc, initial=0)
        graph = build_graph(result)
        hidden_nodes = [n for n in graph.values() if n.hidden]
        assert len(hidden_nodes) == 1
        assert hidden_nodes[0].name == "_identity"

    def test_reduce_with_partial_task(self) -> None:
        """IterTaskFuture.reduce() works with a PartialTask."""
        from daglite.futures.reduce_future import ReduceFuture

        @task
        def generate(n: int) -> Iterator[int]:
            yield from range(n)

        @task
        def add_acc_scaled(acc: int, item: int, scale: int) -> int:
            return acc + item * scale

        result = generate.iter(n=5).reduce(add_acc_scaled.partial(scale=2), initial=0)
        assert isinstance(result, ReduceFuture)
        assert result.initial == 0

    def test_then_chain_to_reduce(self) -> None:
        """IterTaskFuture.then().reduce() chains correctly."""
        from daglite.futures.reduce_future import ReduceFuture

        @task
        def generate(n: int) -> Iterator[int]:
            yield from range(n)

        @task
        def double(x: int) -> int:
            return x * 2

        @task
        def add_acc(acc: int, item: int) -> int:
            return acc + item

        result = generate.iter(n=5).then(double).reduce(add_acc, initial=0)
        assert isinstance(result, ReduceFuture)

    def test_then_chain_to_join(self) -> None:
        """IterTaskFuture.then().join() chains correctly."""
        from daglite.futures.task_future import TaskFuture

        @task
        def generate(n: int) -> Iterator[int]:
            yield from range(n)

        @task
        def double(x: int) -> int:
            return x * 2

        @task
        def sum_values(values: list[int]) -> int:
            return sum(values)

        result = generate.iter(n=5).then(double).join(sum_values)
        assert isinstance(result, TaskFuture)


class TestIterNodeProperties:
    """Tests for IterNode graph IR properties."""

    def test_iter_node_kind(self) -> None:
        """IterNode.kind returns 'iter'."""

        @task
        def generate(n: int) -> Iterator[int]:
            yield from range(n)

        node = generate.iter(n=5).build_node()
        assert node.kind == "iter"

    def test_iter_node_remap_references(self) -> None:
        """IterNode.remap_references updates kwargs refs correctly."""
        from uuid import uuid4

        @task
        def generate(n: int) -> Iterator[int]:
            yield from range(n)

        node = generate.iter(n=5).build_node()
        old_id = uuid4()
        new_id = uuid4()

        # With a value-typed kwarg, remap should be a no-op
        result = node.remap_references({old_id: new_id})
        assert result is node  # No change expected

    def test_iter_node_get_dependencies_with_ref(self) -> None:
        """IterNode.get_dependencies returns upstream refs."""

        @task
        def get_count() -> int:
            return 10

        @task
        def generate(n: int) -> Iterator[int]:
            yield from range(n)

        count = get_count()
        node = generate.iter(n=count).build_node()
        deps = node.get_dependencies()
        assert count.id in deps
