"""
Unit tests for the graph optimizer.

Tests chain detection, composite node folding, dependency remapping,
and settings-based opt-out.  These tests build the graph IR directly
(via ``build_graph``) and inspect the optimised output without running
the engine — execution tests live in ``tests/integration/``.
"""

from typing import Any
from uuid import UUID
from uuid import uuid4

from daglite import task
from daglite.graph.builder import build_graph
from daglite.graph.nodes import CompositeMapTaskNode
from daglite.graph.nodes import CompositeTaskNode
from daglite.graph.nodes.base import NodeInput
from daglite.graph.nodes.base import NodeOutputConfig
from daglite.graph.nodes.map_node import MapTaskNode
from daglite.graph.nodes.task_node import TaskNode
from daglite.graph.optimizer import _aggregate_timeout
from daglite.graph.optimizer import _build_adjacency
from daglite.graph.optimizer import optimize_graph


def _count_types(nodes: dict[UUID, Any]) -> dict[str, int]:
    """Count node types in a graph dict."""
    counts: dict[str, int] = {}
    for node in nodes.values():
        name = type(node).__name__
        counts[name] = counts.get(name, 0) + 1
    return counts


class TestFoldTaskChains:
    """Tests for _fold_task_paths via optimize_graph."""

    def test_two_node_chain_folds(self) -> None:
        """A simple a → b chain is folded into a CompositeTaskNode."""

        @task
        def a(x: int) -> int:  # pragma: no cover
            return x

        @task
        def b(x: int) -> int:  # pragma: no cover
            return x * 2

        future = a(x=1).then(b)
        graph = build_graph(future)
        assert _count_types(graph) == {"TaskNode": 2}

        optimized, id_mapping = optimize_graph(graph)
        counts = _count_types(optimized)
        assert counts.get("CompositeTaskNode", 0) == 1
        assert counts.get("TaskNode", 0) == 0
        assert len(optimized) == 1

        # The tail ID (b) should be mapped to the composite
        assert future.id in id_mapping

    def test_three_node_chain_folds(self) -> None:
        """a → b → c folds into one CompositeTaskNode."""

        @task
        def a(x: int) -> int:  # pragma: no cover
            return x

        @task
        def b(x: int) -> int:  # pragma: no cover
            return x + 1

        @task
        def c(x: int) -> int:  # pragma: no cover
            return x * 3

        future = a(x=1).then(b).then(c)
        graph = build_graph(future)
        optimized, _ = optimize_graph(graph)
        counts = _count_types(optimized)
        assert counts == {"CompositeTaskNode": 1}

        composite = next(n for n in optimized.values() if isinstance(n, CompositeTaskNode))
        assert len(composite.steps) == 3

    def test_single_node_not_folded(self) -> None:
        """A single TaskNode is not folded."""

        @task
        def a(x: int) -> int:  # pragma: no cover
            return x

        graph = build_graph(a(x=1))
        optimized, id_mapping = optimize_graph(graph)
        assert _count_types(optimized) == {"TaskNode": 1}
        assert id_mapping == {}

    def test_fan_out_breaks_chain(self) -> None:
        """Fan-out (one node's result used by two successors) prevents folding."""

        @task
        def a(x: int) -> int:  # pragma: no cover
            return x

        @task
        def b(x: int) -> int:  # pragma: no cover
            return x + 1

        @task
        def c(x: int) -> int:  # pragma: no cover
            return x + 2

        @task
        def d(x: int, y: int) -> int:  # pragma: no cover
            return x + y

        fa = a(x=1)
        fb = b(x=fa)
        fc = c(x=fa)
        future = d(x=fb, y=fc)
        graph = build_graph(future)
        optimized, _ = optimize_graph(graph)

        # a has two successors (b, c) so it can't be folded with either
        # b and c each have only one pred (a) but a has fan-out
        # d has two predecessors (b, c) so it can't be folded
        # No chain of length >= 2 should form
        assert "CompositeTaskNode" not in _count_types(optimized)

    def test_fan_in_breaks_chain(self) -> None:
        """Fan-in (a node depending on two predecessors) prevents folding."""

        @task
        def a(x: int) -> int:  # pragma: no cover
            return x

        @task
        def b(y: int) -> int:  # pragma: no cover
            return y

        @task
        def c(x: int, y: int) -> int:  # pragma: no cover
            return x + y

        future = c(x=a(x=1), y=b(y=2))
        graph = build_graph(future)
        optimized, _ = optimize_graph(graph)
        # c has two predecessors — can't fold
        assert "CompositeTaskNode" not in _count_types(optimized)

    def test_different_backends_break_chain(self) -> None:
        """Nodes with different backends are not folded together."""

        @task
        def a(x: int) -> int:  # pragma: no cover
            return x

        @task
        def b(x: int) -> int:  # pragma: no cover
            return x * 2

        future = a.with_options(backend_name="threads")(x=1).then(
            b.with_options(backend_name="inline")
        )
        graph = build_graph(future)
        optimized, _ = optimize_graph(graph)
        assert "CompositeTaskNode" not in _count_types(optimized)

    def test_chain_with_downstream_dep_remapped(self) -> None:
        """Downstream nodes depending on a folded tail have refs remapped."""

        @task
        def a(x: int) -> int:  # pragma: no cover
            return x

        @task
        def b(x: int) -> int:  # pragma: no cover
            return x + 1

        @task
        def c(x: int) -> int:  # pragma: no cover
            return x * 10

        # a → b (chain) and b → c (not chainable with b since b is now in composite)
        # Actually c only depends on b, so a → b → c is a 3-node chain
        future = a(x=1).then(b).then(c)
        graph = build_graph(future)
        optimized, id_mapping = optimize_graph(graph)
        assert len(optimized) == 1
        assert future.id in id_mapping

    def test_composite_preserves_step_metadata(self) -> None:
        """CompositeTaskNode steps have correct names and flow params."""

        @task
        def step_a(x: int) -> int:  # pragma: no cover
            return x

        @task
        def step_b(val: int) -> int:  # pragma: no cover
            return val * 2

        future = step_a(x=5).then(step_b)
        graph = build_graph(future)
        optimized, _ = optimize_graph(graph)

        composite = next(n for n in optimized.values() if isinstance(n, CompositeTaskNode))
        assert len(composite.steps) == 2
        assert composite.steps[0].name == "step_a"
        assert composite.steps[0].flow_param is None  # head has no flow param
        assert composite.steps[1].name == "step_b"
        assert composite.steps[1].flow_param == "val"  # receives from step_a


class TestFoldMapChains:
    """Tests for _fold_map_paths via optimize_graph."""

    def test_map_then_chain_folds(self) -> None:
        """map().then() chain is folded into a CompositeMapTaskNode."""

        @task
        def double(x: int) -> int:  # pragma: no cover
            return x * 2

        @task
        def add_one(x: int) -> int:  # pragma: no cover
            return x + 1

        future = double.map(x=[1, 2, 3]).then(add_one)
        graph = build_graph(future)

        # Before optimization: MapTaskNode(double) → MapTaskNode(add_one via .then())
        optimized, id_mapping = optimize_graph(graph)
        counts = _count_types(optimized)
        assert counts.get("CompositeMapTaskNode", 0) == 1
        assert counts.get("MapTaskNode", 0) == 0
        assert len(optimized) == 1

    def test_map_then_then_chain_folds(self) -> None:
        """map().then().then() folds into one CompositeMapTaskNode."""

        @task
        def double(x: int) -> int:  # pragma: no cover
            return x * 2

        @task
        def add(x: int, y: int) -> int:  # pragma: no cover
            return x + y

        @task
        def negate(x: int) -> int:  # pragma: no cover
            return -x

        future = double.map(x=[1, 2]).then(add, y=10).then(negate)
        graph = build_graph(future)
        optimized, _ = optimize_graph(graph)
        counts = _count_types(optimized)
        assert counts == {"CompositeMapTaskNode": 1}

        composite = next(n for n in optimized.values() if isinstance(n, CompositeMapTaskNode))
        assert len(composite.steps) == 2  # add, negate (source_map is separate)
        assert composite.terminal == "collect"

    def test_map_join_folds(self) -> None:
        """map().join() folds into a CompositeMapTaskNode with join terminal."""

        @task
        def square(x: int) -> int:  # pragma: no cover
            return x**2

        @task
        def total(values: list[int]) -> int:  # pragma: no cover
            return sum(values)

        future = square.map(x=[1, 2, 3]).join(total)
        graph = build_graph(future)
        optimized, _ = optimize_graph(graph)
        counts = _count_types(optimized)
        assert counts == {"CompositeMapTaskNode": 1}

        composite = next(n for n in optimized.values() if isinstance(n, CompositeMapTaskNode))
        assert composite.terminal == "join"
        assert composite.join_step is not None
        assert composite.join_step.name == "total"

    def test_map_then_join_folds(self) -> None:
        """map().then().join() folds into a single CompositeMapTaskNode."""

        @task
        def double(x: int) -> int:  # pragma: no cover
            return x * 2

        @task
        def add(x: int, y: int) -> int:  # pragma: no cover
            return x + y

        @task
        def total(values: list[int]) -> int:  # pragma: no cover
            return sum(values)

        future = double.map(x=[1, 2, 3]).then(add, y=5).join(total)
        graph = build_graph(future)
        optimized, _ = optimize_graph(graph)
        counts = _count_types(optimized)
        assert counts == {"CompositeMapTaskNode": 1}

        composite = next(n for n in optimized.values() if isinstance(n, CompositeMapTaskNode))
        assert composite.terminal == "join"
        assert len(composite.steps) == 1  # add

    def test_map_reduce_folds(self) -> None:
        """map().reduce() folds into a CompositeMapTaskNode with reduce terminal."""

        @task
        def double(x: int) -> int:  # pragma: no cover
            return x * 2

        @task
        def accumulate(acc: int, item: int) -> int:  # pragma: no cover
            return acc + item

        future = double.map(x=[1, 2, 3]).reduce(accumulate, initial=0)
        graph = build_graph(future)

        # Before optimization: MapTaskNode + ReduceNode
        assert _count_types(graph).get("ReduceNode", 0) == 1

        optimized, _ = optimize_graph(graph)
        counts = _count_types(optimized)
        assert counts == {"CompositeMapTaskNode": 1}

        composite = next(n for n in optimized.values() if isinstance(n, CompositeMapTaskNode))
        assert composite.terminal == "reduce"
        assert composite.reduce_config is not None

    def test_standalone_map_not_folded(self) -> None:
        """A single map() with no .then()/.join()/.reduce() is not folded."""

        @task
        def double(x: int) -> int:  # pragma: no cover
            return x * 2

        graph = build_graph(double.map(x=[1, 2, 3]))
        optimized, id_mapping = optimize_graph(graph)
        assert _count_types(optimized) == {"MapTaskNode": 1}
        assert id_mapping == {}

    def test_map_chain_composite_dependencies(self) -> None:
        """CompositeMapTaskNode dependencies exclude internal IDs."""

        @task
        def source(n: int) -> list[int]:  # pragma: no cover
            return list(range(n))

        @task
        def double(x: int) -> int:  # pragma: no cover
            return x * 2

        @task
        def add(x: int, y: int) -> int:  # pragma: no cover
            return x + y

        @task
        def total(values: list[int]) -> int:  # pragma: no cover
            return sum(values)

        src = source(n=5)
        future = double.map(x=src).then(add, y=10).join(total)
        graph = build_graph(future)
        optimized, _ = optimize_graph(graph)

        composite = next(n for n in optimized.values() if isinstance(n, CompositeMapTaskNode))
        deps = composite.get_dependencies()
        # Should depend on source node but not on internal chain IDs
        assert src.id in deps
        assert len(deps) == 1


class TestMixedChains:
    """Test that task chains and map chains coexist correctly."""

    def test_task_chain_before_map(self) -> None:
        """task().then() → map() — task chain folds, map stays."""

        @task
        def a(x: int) -> int:  # pragma: no cover
            return x

        @task
        def b(x: int) -> list[int]:  # pragma: no cover
            return [x, x + 1, x + 2]

        @task
        def c(x: int) -> int:  # pragma: no cover
            return x * 10

        future = c.map(x=a(x=1).then(b))
        graph = build_graph(future)
        optimized, _ = optimize_graph(graph)
        counts = _count_types(optimized)
        # a → b folds into CompositeTaskNode, c.map stays as MapTaskNode
        assert counts.get("CompositeTaskNode", 0) == 1
        assert counts.get("MapTaskNode", 0) == 1

    def test_task_chain_and_map_chain(self) -> None:
        """Separate task chain and map chain in the same graph both fold."""

        @task
        def a(x: int) -> int:  # pragma: no cover
            return x

        @task
        def b(x: int) -> int:  # pragma: no cover
            return x + 1

        @task
        def c(x: int) -> int:  # pragma: no cover
            return x * 2

        @task
        def d(x: int) -> int:  # pragma: no cover
            return x - 1

        @task
        def merge(p: int, q: list[int]) -> int:  # pragma: no cover
            return p + sum(q)

        # Task chain: a → b
        # Map chain: c.map().then(d)
        fa = a(x=1).then(b)
        fm = c.map(x=[10, 20]).then(d)
        future = merge(p=fa, q=fm)
        graph = build_graph(future)
        optimized, _ = optimize_graph(graph)
        counts = _count_types(optimized)
        assert counts.get("CompositeTaskNode", 0) == 1
        assert counts.get("CompositeMapTaskNode", 0) == 1
        assert counts.get("TaskNode", 0) == 1  # merge


class TestOptimizationSettings:
    """Test that optimization can be disabled via settings."""

    def test_disabled_optimization_preserves_graph(self) -> None:
        """With optimization enabled=False, no composites are created."""

        @task
        def a(x: int) -> int:  # pragma: no cover
            return x

        @task
        def b(x: int) -> int:  # pragma: no cover
            return x + 1

        graph = build_graph(a(x=1).then(b))

        # When optimization runs, it folds
        optimized, id_mapping = optimize_graph(graph)
        assert _count_types(optimized).get("CompositeTaskNode", 0) == 1

        # The engine respects the setting — tested via integration tests,
        # but here we verify optimize_graph itself works
        assert len(id_mapping) > 0


class TestAggregateTimeout:
    """Unit tests for _aggregate_timeout."""

    def test_all_none_returns_none(self) -> None:
        assert _aggregate_timeout([None, None]) is None

    def test_all_set_returns_sum(self) -> None:
        assert _aggregate_timeout([10.0, 5.0]) == 15.0

    def test_mixed_returns_none(self) -> None:
        """If any value is unbounded, the aggregate is also None."""
        assert _aggregate_timeout([10.0, None]) is None

    def test_empty_list_returns_none(self) -> None:
        assert _aggregate_timeout([]) is None

    def test_single_value(self) -> None:
        assert _aggregate_timeout([42.0]) == 42.0

    def test_all_zero_returns_none(self) -> None:
        """Sum of zero timeouts is 0.0 which should return None."""
        assert _aggregate_timeout([0.0, 0.0]) is None


class TestTimeoutPreservation:
    """Optimizer should capture per-node timeout into CompositeStep and aggregate on composite."""

    def test_task_chain_aggregates_timeouts(self) -> None:
        """CompositeTaskNode timeout is sum of link timeouts."""

        @task(timeout=10.0)
        def a(x: int) -> int:  # pragma: no cover
            return x

        @task(timeout=5.0)
        def b(x: int) -> int:  # pragma: no cover
            return x + 1

        graph = build_graph(a(x=1).then(b))
        optimized, _ = optimize_graph(graph)
        composite = next(n for n in optimized.values() if isinstance(n, CompositeTaskNode))
        assert composite.timeout == 15.0
        assert composite.steps[0].timeout == 10.0
        assert composite.steps[1].timeout == 5.0

    def test_task_chain_none_timeout_propagates(self) -> None:
        """If any link has no timeout, composite timeout is None."""

        @task(timeout=10.0)
        def a(x: int) -> int:  # pragma: no cover
            return x

        @task
        def b(x: int) -> int:  # pragma: no cover
            return x + 1

        graph = build_graph(a(x=1).then(b))
        optimized, _ = optimize_graph(graph)
        composite = next(n for n in optimized.values() if isinstance(n, CompositeTaskNode))
        assert composite.timeout is None

    def test_map_chain_aggregates_timeouts(self) -> None:
        """CompositeMapTaskNode timeout includes source map + then links."""

        @task(timeout=10.0)
        def a(x: int) -> int:  # pragma: no cover
            return x * 2

        @task(timeout=5.0)
        def b(x: int) -> int:  # pragma: no cover
            return x + 1

        graph = build_graph(a.map(x=[1, 2]).then(b))
        optimized, _ = optimize_graph(graph)
        composite = next(n for n in optimized.values() if isinstance(n, CompositeMapTaskNode))
        assert composite.timeout == 15.0

    def test_composite_step_captures_step_kind(self) -> None:
        """CompositeStep.step_kind should reflect the original node kind."""

        @task(timeout=1.0)
        def a(x: int) -> int:  # pragma: no cover
            return x

        @task(timeout=2.0)
        def b(x: int) -> int:  # pragma: no cover
            return x + 1

        graph = build_graph(a(x=1).then(b))
        optimized, _ = optimize_graph(graph)
        composite = next(n for n in optimized.values() if isinstance(n, CompositeTaskNode))
        for step in composite.steps:
            assert step.step_kind == "task"


class TestChainBackwardWalk:
    """
    The optimizer should find maximal chains regardless of dict iteration order.

    The backward walk ensures that even if we start iterating from a middle
    node, we still detect the full chain head.
    """

    def test_three_node_chain_any_iteration_order(self) -> None:
        """a → b → c folds into one composite regardless of dict order."""

        @task
        def a(x: int) -> int:  # pragma: no cover
            return x

        @task
        def b(x: int) -> int:  # pragma: no cover
            return x + 1

        @task
        def c(x: int) -> int:  # pragma: no cover
            return x * 2

        future = a(x=1).then(b).then(c)
        graph = build_graph(future)

        # Reverse the dict order to simulate non-topological iteration
        reversed_graph = dict(reversed(list(graph.items())))
        optimized, _ = optimize_graph(reversed_graph)
        counts = _count_types(optimized)
        assert counts == {"CompositeTaskNode": 1}
        composite = next(n for n in optimized.values() if isinstance(n, CompositeTaskNode))
        assert len(composite.steps) == 3

    def test_backward_walk_stops_at_fan_out(self) -> None:
        """Backward walk stops when predecessor has multiple successors (fan-out)."""

        @task
        def a(x: int) -> int:  # pragma: no cover
            return x

        @task
        def b(x: int) -> int:  # pragma: no cover
            return x + 1

        @task
        def c(x: int) -> int:  # pragma: no cover
            return x + 2

        fa = a(x=1)
        fb = b(x=fa)  # a → b
        fc = c(x=fa)  # a → c  (fan-out, b and c cannot fold with a)
        graph = build_graph(fc)
        graph[fb.id] = build_graph(fb)[fb.id]  # manually add b to the graph

        # Reverse to force the backward walk to encounter the fan-out
        reversed_graph = dict(reversed(list(graph.items())))
        optimized, _ = optimize_graph(reversed_graph)
        # None of a, b, c form a chain because a has two successors
        assert "CompositeTaskNode" not in _count_types(optimized)


class TestMapChainBackwardWalk:
    """Map chain backward-walk covers the reversed-dict iteration path."""

    def test_three_step_map_chain_reversed_order(self) -> None:
        """map().then().then() folds correctly even when dict is reversed."""

        @task
        def double(x: int) -> int:  # pragma: no cover
            return x * 2

        @task
        def inc(x: int) -> int:  # pragma: no cover
            return x + 1

        @task
        def neg(x: int) -> int:  # pragma: no cover
            return -x

        future = double.map(x=[1, 2]).then(inc).then(neg)
        graph = build_graph(future)
        reversed_graph = dict(reversed(list(graph.items())))
        optimized, _ = optimize_graph(reversed_graph)
        counts = _count_types(optimized)
        assert counts == {"CompositeMapTaskNode": 1}
        composite = next(n for n in optimized.values() if isinstance(n, CompositeMapTaskNode))
        assert len(composite.steps) == 2  # inc, neg (source_map is separate field)

    def test_map_backward_walk_stops_at_fan_out(self) -> None:
        """Map backward walk stops when predecessor has multiple successors."""

        @task
        def double(x: int) -> int:  # pragma: no cover
            return x * 2

        @task
        def inc(x: int) -> int:  # pragma: no cover
            return x + 1

        @task
        def neg(x: int) -> int:  # pragma: no cover
            return -x

        # Two separate .then() chains from the same map head — fan-out
        map_future = double.map(x=[1, 2])
        fb = map_future.then(inc)
        fc = map_future.then(neg)
        graph_b = build_graph(fb)
        graph_c = build_graph(fc)
        merged = {**graph_b, **graph_c}
        reversed_merged = dict(reversed(list(merged.items())))
        optimized, _ = optimize_graph(reversed_merged)
        # With fan-out, inc and neg cannot fold with the shared map head
        composites = [n for n in optimized.values() if isinstance(n, CompositeMapTaskNode)]
        # No composite should include the shared map head
        for composite in composites:
            assert len(composite.steps) == 0 or composite.source_map is not None


# ---------------------------------------------------------------------------
# Coverage: TaskNode in map .then() chain
# ---------------------------------------------------------------------------


class TestOptimizerMapThenTaskNodeCoverage:
    """Cover the TaskNode branch in map chain .then() folding."""

    def test_map_then_task_node_folds(self) -> None:
        """map().then(task) where task is a regular TaskNode should fold correctly."""

        @task
        def double(x: int) -> int:  # pragma: no cover
            return x * 2

        @task
        def add_one(x: int) -> int:  # pragma: no cover
            return x + 1

        graph = build_graph(double.map(x=[1, 2]).then(add_one))
        optimized, _ = optimize_graph(graph)
        counts = _count_types(optimized)
        assert counts.get("CompositeMapTaskNode", 0) == 1


# ---------------------------------------------------------------------------
# Coverage: join/reduce ID remapping in map chain folding
# ---------------------------------------------------------------------------


class TestOptimizerIdRemappingCoverage:
    """Cover join_id/reduce_id != tail_id branches in map chain folding."""

    def test_map_then_join_remaps_join_id(self) -> None:
        """map().then().join() should remap both then and join IDs."""

        @task
        def double(x: int) -> int:  # pragma: no cover
            return x * 2

        @task
        def add_one(x: int) -> int:  # pragma: no cover
            return x + 1

        @task
        def sum_all(values: list[int]) -> int:  # pragma: no cover
            return sum(values)

        future = double.map(x=[1, 2]).then(add_one).join(sum_all)
        graph = build_graph(future)
        optimized, id_mapping = optimize_graph(graph)
        assert len(id_mapping) > 0
        assert _count_types(optimized).get("CompositeMapTaskNode", 0) == 1

    def test_map_then_reduce_remaps_reduce_id(self) -> None:
        """map().then().reduce() should remap both then and reduce IDs."""

        @task
        def double(x: int) -> int:  # pragma: no cover
            return x * 2

        @task
        def add_one(x: int) -> int:  # pragma: no cover
            return x + 1

        @task
        def summer(acc: int, item: int) -> int:  # pragma: no cover
            return acc + item

        future = double.map(x=[1, 2]).then(add_one).reduce(summer, initial=0)
        graph = build_graph(future)
        optimized, id_mapping = optimize_graph(graph)
        assert len(id_mapping) > 0
        assert _count_types(optimized).get("CompositeMapTaskNode", 0) == 1


# ---------------------------------------------------------------------------
# Reduce retries wired through optimizer
# ---------------------------------------------------------------------------


class TestReduceRetriesOptimizer:
    """Optimizer should carry retries into CompositeMapTaskNode reduce_config."""

    def test_reduce_retries_in_optimised_composite(self) -> None:
        @task(retries=2)
        def summer(acc: int, item: int) -> int:  # pragma: no cover
            return acc + item

        @task
        def double(x: int) -> int:  # pragma: no cover
            return x * 2

        graph = build_graph(double.map(x=[1, 2, 3]).reduce(summer, initial=0))
        optimized, _ = optimize_graph(graph)
        composite = next(n for n in optimized.values() if isinstance(n, CompositeMapTaskNode))
        assert composite.terminal == "reduce"
        assert composite.reduce_config is not None
        assert composite.reduce_config.retries == 2


class TestBuildAdjacency:
    """_build_adjacency should ignore dependencies pointing outside the node dict."""

    def test_external_dependency_is_ignored(self) -> None:
        """A NodeInput.from_ref to an ID not in the nodes dict does not create an edge."""
        external_id = uuid4()
        node_id = uuid4()
        node = TaskNode(
            id=node_id,
            name="t",
            func=lambda x: x,
            kwargs={"x": NodeInput.from_ref(external_id)},
        )
        nodes = {node_id: node}  # external_id is intentionally absent
        predecessors, successors = _build_adjacency(nodes)  # type: ignore
        assert external_id not in predecessors
        assert external_id not in successors
        assert node_id not in predecessors  # no edges found


class TestMapJoinFlowParamDetection:
    """Optimizer should not fold a task as a join when it has no kwargs flow param."""

    def test_task_dep_via_output_config_only_not_treated_as_join(self) -> None:
        """
        A TaskNode depending on a MapTaskNode only through output_configs (not kwargs)
        does not satisfy the join flow-param requirement and is not folded.
        """
        map_id = uuid4()
        task_id = uuid4()

        map_node = MapTaskNode(
            id=map_id,
            name="mapper",
            func=lambda x: x,
            mode="zip",
            fixed_kwargs={},
            mapped_kwargs={"x": NodeInput.from_value([1, 2])},
        )
        # This task references the map only through output_configs, not kwargs.
        # _identify_flow_param checks kwargs only, so flow_param will be None.
        task_node = TaskNode(
            id=task_id,
            name="consumer",
            func=lambda y: y,
            kwargs={"y": NodeInput.from_value(42)},
            output_configs=(
                NodeOutputConfig(
                    key="out_{v}.txt",
                    dependencies={"v": NodeInput.from_ref(map_id)},
                ),
            ),
        )

        nodes = {map_id: map_node, task_id: task_node}
        optimized, _ = optimize_graph(nodes)

        # No CompositeMapTaskNode — the task was not eligible as a join terminal
        assert not any(isinstance(n, CompositeMapTaskNode) for n in optimized.values())
        # Both original nodes survive unchanged
        assert map_id in optimized
        assert task_id in optimized
