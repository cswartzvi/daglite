"""Tests for build_graph functionality."""

from functools import cached_property
from uuid import UUID
from uuid import uuid4

import pytest

from daglite.exceptions import GraphConstructionError
from daglite.graph.builder import build_graph
from daglite.tasks import task


class TestBuildGraph:
    """
    Test build_graph function with various graph structures.

    NOTE: Tests focus on graph construction, not evaluation.
    """

    def test_build_graph_single_node(self) -> None:
        """build_graph handles single node graph."""

        @task
        def simple() -> int:  # pragma: no cover
            return 42

        bound = simple.bind()
        graph = build_graph(bound)

        assert len(graph) == 1
        assert bound.id in graph

    def test_build_graph_linear_chain(self) -> None:
        """build_graph handles linear dependency chain."""

        @task
        def step1() -> int:  # pragma: no cover
            return 10

        @task
        def step2(x: int) -> int:  # pragma: no cover
            return x * 2

        @task
        def step3(x: int) -> int:  # pragma: no cover
            return x + 5

        s1 = step1.bind()
        s2 = step2.bind(x=s1)
        s3 = step3.bind(x=s2)

        graph = build_graph(s3)

        assert len(graph) == 3
        assert s1.id in graph
        assert s2.id in graph
        assert s3.id in graph

    def test_build_graph_dag_with_multiple_deps(self) -> None:
        """build_graph handles DAG with multiple dependencies."""

        @task
        def source1() -> int:  # pragma: no cover
            return 5

        @task
        def source2() -> int:  # pragma: no cover
            return 10

        @task
        def combine(a: int, b: int) -> int:  # pragma: no cover
            return a + b

        s1 = source1.bind()
        s2 = source2.bind()
        result = combine.bind(a=s1, b=s2)

        graph = build_graph(result)

        assert len(graph) == 3
        # Verify combine node depends on both sources
        combine_node = graph[result.id]
        deps = combine_node.dependencies()
        assert s1.id in deps
        assert s2.id in deps

    def test_build_graph_diamond_dependency(self) -> None:
        """build_graph handles diamond-shaped dependencies."""

        @task
        def start() -> int:  # pragma: no cover
            return 1

        @task
        def branch1(x: int) -> int:  # pragma: no cover
            return x * 2

        @task
        def branch2(x: int) -> int:  # pragma: no cover
            return x * 3

        @task
        def merge(a: int, b: int) -> int:  # pragma: no cover
            return a + b

        root = start.bind()
        b1 = branch1.bind(x=root)
        b2 = branch2.bind(x=root)
        final = merge.bind(a=b1, b=b2)

        graph = build_graph(final)

        assert len(graph) == 4
        assert all(node_id in graph for node_id in [root.id, b1.id, b2.id, final.id])

    def test_build_graph_shared_dependency(self) -> None:
        """build_graph handles shared dependencies correctly (skips already processed nodes)."""

        @task
        def shared() -> int:  # pragma: no cover
            return 10

        @task
        def use1(x: int) -> int:  # pragma: no cover
            return x * 2

        @task
        def use2(x: int) -> int:  # pragma: no cover
            return x * 3

        @task
        def combine(a: int, b: int) -> int:  # pragma: no cover
            return a + b

        # Both use1 and use2 depend on shared
        s = shared.bind()
        u1 = use1.bind(x=s)
        u2 = use2.bind(x=s)
        result = combine.bind(a=u1, b=u2)

        graph = build_graph(result)

        # Should only have 4 nodes (shared is not duplicated)
        assert len(graph) == 4
        assert s.id in graph
        assert u1.id in graph
        assert u2.id in graph
        assert result.id in graph

    def test_build_graph_skips_already_processed_nodes(self) -> None:
        """build_graph skips nodes already in the graph (covers early exit)."""

        @task
        def leaf1() -> int:  # pragma: no cover
            return 1

        @task
        def leaf2() -> int:  # pragma: no cover
            return 2

        @task
        def middle1(a: int, b: int) -> int:  # pragma: no cover
            return a + b

        @task
        def middle2(a: int, b: int) -> int:  # pragma: no cover
            return a * b

        @task
        def root(x: int, y: int) -> int:  # pragma: no cover
            return x - y

        # Create structure where leaf1 is shared across multiple paths:
        #       root
        #      /    \
        #   middle1  middle2
        #    /  \     /  \
        # leaf1 leaf2 leaf1 leaf2
        #
        # Both middle1 and middle2 depend on leaf1 and leaf2
        l1 = leaf1.bind()
        l2 = leaf2.bind()
        m1 = middle1.bind(a=l1, b=l2)
        m2 = middle2.bind(a=l1, b=l2)
        r = root.bind(x=m1, y=m2)

        graph = build_graph(r)

        # Should have 5 unique nodes (leaves should not be duplicated)
        assert len(graph) == 5
        # Verify each node appears exactly once
        assert sum(1 for nid in graph if nid == l1.id) == 1
        assert sum(1 for nid in graph if nid == l2.id) == 1

    def test_build_graph_detects_circular_dependency(self) -> None:
        """build_graph detects circular dependencies."""

        # Create mock builders that form a circular dependency
        class CircularBuilder:
            def __init__(self, node_id: UUID, other_builder) -> None:
                self._id = node_id
                self._other = other_builder

            @cached_property
            def id(self) -> UUID:
                return self._id

            def get_dependencies(self) -> list:
                return [self._other] if self._other else []

            def to_graph(self):  # pragma: no cover
                from daglite.graph.nodes import TaskNode

                return TaskNode(
                    id=self._id,
                    name="test",
                    description=None,
                    backend=None,
                    func=lambda: None,  # pragma: no cover
                    kwargs={},
                )

        # Create A -> B -> A circular dependency
        id_a = uuid4()
        id_b = uuid4()
        builder_b = CircularBuilder(id_b, None)
        builder_a = CircularBuilder(id_a, builder_b)
        builder_b._other = builder_a  # Create the cycle

        with pytest.raises(GraphConstructionError, match="Circular dependency detected"):
            build_graph(builder_a)  # pyright: ignore

    def test_build_graph_detects_self_reference(self) -> None:
        """build_graph detects nodes that reference themselves."""

        # Create a builder that references itself
        class SelfRefBuilder:
            def __init__(self) -> None:
                self._id = uuid4()

            @cached_property
            def id(self):
                return self._id

            def get_dependencies(self) -> list:
                return [self]

            def to_graph(self):  # pragma: no cover
                from daglite.graph.nodes import TaskNode

                return TaskNode(
                    id=self._id,
                    name="self_ref",
                    description=None,
                    backend=None,
                    func=lambda: None,  # pragma: no cover
                    kwargs={},
                )

        builder = SelfRefBuilder()

        with pytest.raises(GraphConstructionError, match="Circular dependency detected"):
            build_graph(builder)  # pyright: ignore
