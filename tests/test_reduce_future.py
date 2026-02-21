"""
Unit tests for ReduceFuture and the reduce() API on MapTaskFuture.

Tests validation, build_node(), get_upstream_builders(), and retries
wiring. Execution tests (run() / run_async()) live in
tests/integration/test_composite.py.
"""

from __future__ import annotations

import pytest

from daglite import task
from daglite.exceptions import ParameterError
from daglite.graph.nodes.reduce_node import ReduceNode


class TestReducePartialTaskValidation:
    """reduce() should count only unbound parameters when validating PartialTask."""

    def test_partial_with_extra_bound_param_passes(self) -> None:
        """3-param task with 1 bound → 2 unbound → passes validation."""

        @task
        def three_param_reducer(acc: int, item: int, extra: int) -> int:  # pragma: no cover
            return acc + item + extra

        @task
        def double(x: int) -> int:  # pragma: no cover
            return x * 2

        partial = three_param_reducer.partial(extra=99)
        _ = double.map(x=[1, 2]).reduce(partial, initial=0)  # should not raise

    def test_partial_with_wrong_unbound_count_fails(self) -> None:
        """4-param task with 1 bound → 3 unbound → raises."""

        @task
        def four_param_reducer(acc: int, item: int, x: int, y: int) -> int:  # pragma: no cover
            return acc + item + x + y

        @task
        def double(x: int) -> int:  # pragma: no cover
            return x * 2

        partial = four_param_reducer.partial(y=10)
        with pytest.raises(ParameterError, match="must have exactly 2 unbound parameter"):
            double.map(x=[1, 2]).reduce(partial, initial=0)

    def test_wrong_param_count_raises(self) -> None:
        """reduce() rejects a 3-param task with no partial binding."""

        @task
        def triple(a: int, b: int, c: int) -> int:  # pragma: no cover
            return a + b + c

        @task
        def double(x: int) -> int:  # pragma: no cover
            return x * 2

        with pytest.raises(ParameterError, match="must have exactly 2 unbound parameter"):
            double.map(x=[1, 2]).reduce(triple, initial=0)


class TestReduceFutureOutputConfigs:
    """ReduceFuture.build_node() should wire up output configs and timeout."""

    def test_reduce_node_has_correct_timeout(self) -> None:
        @task
        def double(x: int) -> int:  # pragma: no cover
            return x * 2

        @task
        def add(acc: int, item: int) -> int:  # pragma: no cover
            return acc + item

        reduce_future = double.map(x=[1, 2]).reduce(add, initial=0)
        node = reduce_future.build_node()
        assert isinstance(node, ReduceNode)
        assert node.timeout == add.timeout

    def test_reduce_initial_as_future_build_node(self) -> None:
        """build_node() creates NodeInput.from_ref when initial is a future."""

        @task
        def make_initial(x: int) -> int:  # pragma: no cover
            return x

        @task
        def double(x: int) -> int:  # pragma: no cover
            return x * 2

        @task
        def add(acc: int, item: int) -> int:  # pragma: no cover
            return acc + item

        initial_future = make_initial(x=100)
        reduce_future = double.map(x=[1, 2]).reduce(add, initial=initial_future)
        node = reduce_future.build_node()
        assert node.initial_input.reference == initial_future.id

    def test_reduce_initial_as_future_upstream_builders(self) -> None:
        """get_upstream_builders() includes the initial future."""

        @task
        def make_initial(x: int) -> int:  # pragma: no cover
            return x

        @task
        def double(x: int) -> int:  # pragma: no cover
            return x * 2

        @task
        def add(acc: int, item: int) -> int:  # pragma: no cover
            return acc + item

        initial_future = make_initial(x=100)
        reduce_future = double.map(x=[1, 2]).reduce(add, initial=initial_future)
        builders = reduce_future.get_upstream_builders()
        builder_ids = {b.id for b in builders}
        assert initial_future.id in builder_ids


class TestReduceRetries:
    """ReduceConfig.retries should be wired from the reduce task."""

    def test_reduce_retries_in_config(self) -> None:
        @task(retries=3)
        def add(acc: int, item: int) -> int:  # pragma: no cover
            return acc + item

        @task
        def double(x: int) -> int:  # pragma: no cover
            return x * 2

        reduce_future = double.map(x=[1, 2]).reduce(add, initial=0)
        node = reduce_future.build_node()
        assert node.reduce_config.retries == 3
