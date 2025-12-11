from functools import cached_property
from uuid import UUID
from uuid import uuid4

import pytest

from daglite.exceptions import ExecutionError
from daglite.exceptions import GraphConstructionError
from daglite.exceptions import ParameterError
from daglite.graph.base import GraphNode
from daglite.graph.base import ParamInput
from daglite.graph.builder import build_graph
from daglite.graph.nodes import MapTaskNode
from daglite.graph.nodes import TaskNode
from daglite.tasks import task


class TestParamInput:
    """
    Test ParamInput creation and resolution.

    NOTE: Tests focus on initialization and core functionality, not evaluation.
    """

    def test_from_value(self) -> None:
        """ParamInput.from_value creates a value-type input."""
        param = ParamInput.from_value(42)
        assert param.kind == "value"
        assert param.value == 42
        assert not param.is_ref

    def test_from_ref(self) -> None:
        """ParamInput.from_ref creates a ref-type input."""
        node_id = uuid4()
        param = ParamInput.from_ref(node_id)
        assert param.kind == "ref"
        assert param.ref == node_id
        assert param.is_ref

    def test_from_sequence(self) -> None:
        """ParamInput.from_sequence creates a sequence-type input."""
        param = ParamInput.from_sequence([1, 2, 3])
        assert param.kind == "sequence"
        assert param.value == [1, 2, 3]
        assert not param.is_ref

    def test_from_sequence_ref(self) -> None:
        """ParamInput.from_sequence_ref creates a sequence_ref-type input."""
        node_id = uuid4()
        param = ParamInput.from_sequence_ref(node_id)
        assert param.kind == "sequence_ref"
        assert param.ref == node_id
        assert param.is_ref

    def test_resolve_value(self) -> None:
        """ParamInput resolves value inputs correctly."""
        param = ParamInput.from_value(100)
        assert param.resolve({}) == 100

    def test_resolve_ref(self) -> None:
        """ParamInput resolves ref inputs from values dict."""
        node_id = uuid4()
        param = ParamInput.from_ref(node_id)
        values = {node_id: "result"}
        assert param.resolve(values) == "result"

    def test_resolve_sequence_from_sequence(self) -> None:
        """ParamInput resolves sequence inputs correctly."""
        param = ParamInput.from_sequence([10, 20, 30])
        assert param.resolve_sequence({}) == [10, 20, 30]

    def test_resolve_sequence_from_ref(self) -> None:
        """ParamInput resolves sequence_ref inputs from values dict."""
        node_id = uuid4()
        param = ParamInput.from_sequence_ref(node_id)
        values = {node_id: [1, 2, 3]}
        assert param.resolve_sequence(values) == [1, 2, 3]

    def test_resolve_sequence_as_scalar_fails(self) -> None:
        """Cannot resolve sequence input as scalar value."""
        param = ParamInput.from_sequence([1, 2, 3])
        with pytest.raises(ExecutionError, match="Cannot resolve parameter of kind 'sequence'"):
            param.resolve({})

    def test_resolve_sequence_ref_as_scalar_fails(self) -> None:
        """Cannot resolve sequence_ref input as scalar value."""
        node_id = uuid4()
        param = ParamInput.from_sequence_ref(node_id)
        values = {node_id: [1, 2, 3]}
        with pytest.raises(ExecutionError, match="Cannot resolve parameter of kind 'sequence_ref'"):
            param.resolve(values)

    def test_resolve_value_as_sequence_fails(self) -> None:
        """Cannot resolve value input as sequence."""
        param = ParamInput.from_value(42)
        with pytest.raises(ExecutionError, match="Cannot resolve parameter of kind 'value'"):
            param.resolve_sequence({})

    def test_resolve_ref_as_sequence_fails(self) -> None:
        """Cannot resolve ref input as sequence."""
        node_id = uuid4()
        param = ParamInput.from_ref(node_id)
        values = {node_id: "scalar"}
        with pytest.raises(ExecutionError, match="Cannot resolve parameter of kind 'ref'"):
            param.resolve_sequence(values)


class TestGraphNodes:
    """
    Test graph node initialization and properties.

    NOTE: Tests focus on structure, not execution/submission.
    """

    def test_task_node_properties(self) -> None:
        """TaskNode initializes with correct properties and kind."""

        def add(x: int, y: int) -> int:  # pragma: no cover
            return x + y

        node = TaskNode(
            id=uuid4(),
            name="add_task",
            description="Addition",
            backend=None,
            func=add,
            kwargs={
                "x": ParamInput.from_value(1),
                "y": ParamInput.from_value(2),
            },
        )

        assert node.kind == "task"
        assert node.name == "add_task"
        assert len(node.inputs()) == 2

    def test_task_node_dependencies_with_refs(self) -> None:
        """TaskNode.dependencies() extracts refs from parameters."""
        dep_id = uuid4()

        def process(x: int) -> int:  # pragma: no cover
            return x * 2

        node = TaskNode(
            id=uuid4(),
            name="process",
            description=None,
            backend=None,
            func=process,
            kwargs={"x": ParamInput.from_ref(dep_id)},
        )

        deps = node.dependencies()
        assert len(deps) == 1
        assert dep_id in deps

    def test_task_node_dependencies_without_refs(self) -> None:
        """TaskNode.dependencies() returns empty set for value-only params."""

        def process(x: int) -> int:  # pragma: no cover
            return x * 2

        node = TaskNode(
            id=uuid4(),
            name="process",
            description=None,
            backend=None,
            func=process,
            kwargs={"x": ParamInput.from_value(10)},
        )

        deps = node.dependencies()
        assert len(deps) == 0

    def test_map_task_node_extend_mode(self) -> None:
        """MapTaskNode initializes with extend mode."""

        def process(x: int) -> int:  # pragma: no cover
            return x**2

        node = MapTaskNode(
            id=uuid4(),
            name="process_many",
            description=None,
            backend=None,
            func=process,
            mode="extend",
            fixed_kwargs={},
            mapped_kwargs={"x": ParamInput.from_sequence([1, 2, 3])},
        )

        assert node.kind == "map"
        assert node.mode == "extend"

    def test_map_task_node_zip_mode(self) -> None:
        """MapTaskNode initializes with zip mode."""

        def process(x: int) -> int:  # pragma: no cover
            return x**2

        node = MapTaskNode(
            id=uuid4(),
            name="process_many",
            description=None,
            backend=None,
            func=process,
            mode="zip",
            fixed_kwargs={},
            mapped_kwargs={"x": ParamInput.from_sequence([1, 2, 3])},
        )

        assert node.kind == "map"
        assert node.mode == "zip"

    def test_map_task_node_dependencies_from_fixed(self) -> None:
        """MapTaskNode.dependencies() extracts refs from fixed kwargs."""
        dep_id = uuid4()

        def add(x: int, offset: int) -> int:  # pragma: no cover
            return x + offset

        node = MapTaskNode(
            id=uuid4(),
            name="add_offset",
            description=None,
            backend=None,
            func=add,
            mode="extend",
            fixed_kwargs={"offset": ParamInput.from_ref(dep_id)},
            mapped_kwargs={"x": ParamInput.from_sequence([1, 2, 3])},
        )

        deps = node.dependencies()
        assert dep_id in deps

    def test_map_task_node_dependencies_from_mapped(self) -> None:
        """MapTaskNode.dependencies() extracts refs from mapped kwargs."""
        dep_id = uuid4()

        def add(x: int, offset: int) -> int:  # pragma: no cover
            return x + offset

        node = MapTaskNode(
            id=uuid4(),
            name="add_offset",
            description=None,
            backend=None,
            func=add,
            mode="extend",
            fixed_kwargs={"offset": ParamInput.from_value(10)},
            mapped_kwargs={"x": ParamInput.from_sequence_ref(dep_id)},
        )

        deps = node.dependencies()
        assert dep_id in deps

    def test_map_task_node_inputs(self) -> None:
        """MapTaskNode.inputs() returns both fixed and mapped kwargs."""

        def add(x: int, offset: int) -> int:  # pragma: no cover
            return x + offset

        node = MapTaskNode(
            id=uuid4(),
            name="add_offset",
            description=None,
            backend=None,
            func=add,
            mode="extend",
            fixed_kwargs={"offset": ParamInput.from_value(10)},
            mapped_kwargs={"x": ParamInput.from_sequence([1, 2, 3])},
        )

        inputs = node.inputs()
        assert len(inputs) == 2
        assert ("offset", ParamInput.from_value(10)) in inputs
        assert ("x", ParamInput.from_sequence([1, 2, 3])) in inputs

    def test_map_task_node_zip_mode_length_mismatch(self) -> None:
        """MapTaskNode submission fails with mismatched sequence lengths in zip mode."""

        def add(x: int, y: int) -> int:  # pragma: no cover
            return x + y

        node = MapTaskNode(
            id=uuid4(),
            name="add_pairs",
            description=None,
            backend=None,
            func=add,
            mode="zip",
            fixed_kwargs={},
            mapped_kwargs={
                "x": ParamInput.from_sequence([1, 2, 3]),
                "y": ParamInput.from_sequence([10, 20]),  # Different length
            },
        )

        # This error happens during submit, not initialization
        from daglite.backends.local import SequentialBackend

        backend = SequentialBackend()
        resolved_inputs = node.resolve_inputs({})
        with pytest.raises(
            ParameterError, match="Map task .* with `\\.zip\\(\\)` requires all sequences"
        ):
            node.submit(backend, resolved_inputs)

    def test_map_task_node_invalid_mode(self) -> None:
        """MapTaskNode submission fails with invalid mode."""

        def process(x: int) -> int:  # pragma: no cover
            return x * 2

        node = MapTaskNode(
            id=uuid4(),
            name="process_many",
            description=None,
            backend=None,
            func=process,
            mode="invalid",  # Invalid mode
            fixed_kwargs={},
            mapped_kwargs={"x": ParamInput.from_sequence([1, 2, 3])},
        )

        from daglite.backends.local import SequentialBackend

        backend = SequentialBackend()
        with pytest.raises(ExecutionError, match="Unknown map mode 'invalid'"):
            node.submit(backend, {})


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
        from uuid import UUID

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


class TestGraphOptimizer:
    """Test graph optimization and chain detection."""

    def test_simple_task_chain(self) -> None:
        """Linear TaskNode chain is combined into CompositeTaskNode."""
        from daglite.backends import SequentialBackend
        from daglite.graph.optimizer import optimize_graph

        backend = SequentialBackend()

        # Create chain: A -> B -> C
        node_a = TaskNode(
            id=uuid4(),
            name="a",
            description=None,
            backend=backend,
            func=lambda x: x + 1,
            kwargs={"x": ParamInput.from_value(1)},
        )
        node_b = TaskNode(
            id=uuid4(),
            name="b",
            description=None,
            backend=backend,
            func=lambda x: x * 2,
            kwargs={"x": ParamInput.from_ref(node_a.id)},
        )
        node_c = TaskNode(
            id=uuid4(),
            name="c",
            description=None,
            backend=backend,
            func=lambda x: x + 10,
            kwargs={"x": ParamInput.from_ref(node_b.id)},
        )

        nodes: dict[UUID, GraphNode] = {node_a.id: node_a, node_b.id: node_b, node_c.id: node_c}
        optimized = optimize_graph(nodes, node_c.id)

        # Should have 1 composite node (A, B, C grouped)
        assert len(optimized) == 1
        composite = list(optimized.values())[0]
        assert composite.name == "a→…→c"
        from daglite.graph.composite import CompositeTaskNode

        assert isinstance(composite, CompositeTaskNode)
        assert len(composite.chain) == 3

    def test_task_chain_with_external_params(self) -> None:
        """Chain with external parameters is still grouped."""
        from daglite.backends import SequentialBackend
        from daglite.graph.optimizer import optimize_graph

        backend = SequentialBackend()

        # Create chain: A -> B (with external param) -> C
        node_a = TaskNode(
            id=uuid4(),
            name="a",
            description=None,
            backend=backend,
            func=lambda: 5,
            kwargs={},
        )
        node_b = TaskNode(
            id=uuid4(),
            name="b",
            description=None,
            backend=backend,
            func=lambda x, multiplier: x * multiplier,
            kwargs={"x": ParamInput.from_ref(node_a.id), "multiplier": ParamInput.from_value(3)},
        )
        node_c = TaskNode(
            id=uuid4(),
            name="c",
            description=None,
            backend=backend,
            func=lambda x: x + 1,
            kwargs={"x": ParamInput.from_ref(node_b.id)},
        )

        nodes: dict[UUID, GraphNode] = {node_a.id: node_a, node_b.id: node_b, node_c.id: node_c}
        optimized = optimize_graph(nodes, node_c.id)

        assert len(optimized) == 1
        composite = list(optimized.values())[0]
        from daglite.graph.composite import CompositeTaskNode

        assert isinstance(composite, CompositeTaskNode)
        assert len(composite.chain) == 3
        # Check that external param is preserved
        assert "multiplier" in composite.chain[1].external_params

    def test_chain_breaks_at_backend_boundary(self) -> None:
        """Chain detection stops when backend changes."""
        from daglite.backends import SequentialBackend
        from daglite.backends import ThreadBackend
        from daglite.graph.optimizer import optimize_graph

        backend1 = SequentialBackend()
        backend2 = ThreadBackend()

        # Create: A (backend1) -> B (backend2) -> C (backend2)
        node_a = TaskNode(
            id=uuid4(),
            name="a",
            description=None,
            backend=backend1,
            func=lambda: 1,
            kwargs={},
        )
        node_b = TaskNode(
            id=uuid4(),
            name="b",
            description=None,
            backend=backend2,
            func=lambda x: x + 1,
            kwargs={"x": ParamInput.from_ref(node_a.id)},
        )
        node_c = TaskNode(
            id=uuid4(),
            name="c",
            description=None,
            backend=backend2,
            func=lambda x: x * 2,
            kwargs={"x": ParamInput.from_ref(node_b.id)},
        )

        nodes: dict[UUID, GraphNode] = {node_a.id: node_a, node_b.id: node_b, node_c.id: node_c}
        optimized = optimize_graph(nodes, node_c.id)

        # Should have 2 nodes: A (ungrouped) and composite(B, C)
        assert len(optimized) == 2

    def test_chain_breaks_at_diamond_pattern(self) -> None:
        """Chain detection stops at diamond dependencies."""
        from daglite.backends import SequentialBackend
        from daglite.graph.optimizer import optimize_graph

        backend = SequentialBackend()

        # Create diamond: A -> B, A -> C, B -> D, C -> D
        node_a = TaskNode(
            id=uuid4(),
            name="a",
            description=None,
            backend=backend,
            func=lambda: 1,
            kwargs={},
        )
        node_b = TaskNode(
            id=uuid4(),
            name="b",
            description=None,
            backend=backend,
            func=lambda x: x + 1,
            kwargs={"x": ParamInput.from_ref(node_a.id)},
        )
        node_c = TaskNode(
            id=uuid4(),
            name="c",
            description=None,
            backend=backend,
            func=lambda x: x * 2,
            kwargs={"x": ParamInput.from_ref(node_a.id)},
        )
        node_d = TaskNode(
            id=uuid4(),
            name="d",
            description=None,
            backend=backend,
            func=lambda x, y: x + y,
            kwargs={"x": ParamInput.from_ref(node_b.id), "y": ParamInput.from_ref(node_c.id)},
        )

        nodes: dict[UUID, GraphNode] = {
            node_a.id: node_a,
            node_b.id: node_b,
            node_c.id: node_c,
            node_d.id: node_d,
        }
        optimized = optimize_graph(nodes, node_d.id)

        # Should have 3 nodes: composite(A,B), composite(A,C), D
        # Or possibly: A, composite(B), composite(C), D depending on detection order
        # At minimum, D should not be in any composite due to multiple dependencies
        assert len(optimized) >= 3

    def test_single_nodes_not_grouped(self) -> None:
        """Single nodes without chain partners stay ungrouped."""
        from daglite.backends import SequentialBackend
        from daglite.graph.optimizer import optimize_graph

        backend = SequentialBackend()

        node_a = TaskNode(
            id=uuid4(),
            name="a",
            description=None,
            backend=backend,
            func=lambda: 1,
            kwargs={},
        )

        nodes: dict[UUID, GraphNode] = {node_a.id: node_a}
        optimized = optimize_graph(nodes, node_a.id)

        # Should remain as single node
        assert len(optimized) == 1
        assert optimized[node_a.id] == node_a

    def test_simple_map_chain(self) -> None:
        """Linear MapTaskNode chain is combined into CompositeMapTaskNode."""
        from daglite.backends import SequentialBackend
        from daglite.graph.optimizer import optimize_graph

        backend = SequentialBackend()

        # Create chain: map1 -> map2
        map1 = MapTaskNode(
            id=uuid4(),
            name="map1",
            description=None,
            backend=backend,
            func=lambda x: x * 2,
            mode="product",
            mapped_kwargs={"x": ParamInput.from_sequence([1, 2, 3])},
            fixed_kwargs={},
        )
        map2 = MapTaskNode(
            id=uuid4(),
            name="map2",
            description=None,
            backend=backend,
            func=lambda x: x + 1,
            mode="product",
            mapped_kwargs={"x": ParamInput.from_sequence_ref(map1.id)},
            fixed_kwargs={},
        )

        nodes: dict[UUID, GraphNode] = {map1.id: map1, map2.id: map2}
        optimized = optimize_graph(nodes, map2.id)

        # Should have 1 composite map node
        assert len(optimized) == 1
        composite = list(optimized.values())[0]
        assert composite.name == "map1→…→map2"
        from daglite.graph.composite import CompositeMapTaskNode

        assert isinstance(composite, CompositeMapTaskNode)
        assert len(composite.chain) == 2


class TestCompositeExecution:
    """Test execution behavior of composite nodes."""

    def test_composite_task_node_executes_chain(self) -> None:
        """CompositeTaskNode executes all nodes in sequence."""
        from daglite.backends import SequentialBackend
        from daglite.graph.composite import ChainLink
        from daglite.graph.composite import CompositeTaskNode

        backend = SequentialBackend()

        # Create simple chain: x -> x+1 -> x*2
        node_a = TaskNode(
            id=uuid4(),
            name="a",
            description=None,
            backend=backend,
            func=lambda x: x + 1,
            kwargs={"x": ParamInput.from_value(5)},
        )
        node_b = TaskNode(
            id=uuid4(),
            name="b",
            description=None,
            backend=backend,
            func=lambda x: x * 2,
            kwargs={"x": ParamInput.from_ref(node_a.id)},
        )

        chain = (
            ChainLink(node=node_a, position=0, flow_param=None, external_params={}),
            ChainLink(node=node_b, position=1, flow_param="x", external_params={}),
        )

        composite = CompositeTaskNode(
            id=uuid4(),
            name="composite",
            description="test",
            backend=backend,
            chain=chain,
        )

        # Execute composite
        resolved_inputs = {"x": 5}
        future = composite.submit(backend, resolved_inputs)
        result = future.result()

        # Should compute (5 + 1) * 2 = 12
        assert result == 12

    def test_composite_task_node_with_external_params(self) -> None:
        """CompositeTaskNode resolves external parameters correctly."""
        from daglite.backends import SequentialBackend
        from daglite.graph.composite import ChainLink
        from daglite.graph.composite import CompositeTaskNode

        backend = SequentialBackend()

        # Chain with external param: (x+1) -> (result*multiplier)
        node_a = TaskNode(
            id=uuid4(),
            name="a",
            description=None,
            backend=backend,
            func=lambda x: x + 1,
            kwargs={"x": ParamInput.from_value(5)},
        )
        node_b = TaskNode(
            id=uuid4(),
            name="b",
            description=None,
            backend=backend,
            func=lambda x, multiplier: x * multiplier,
            kwargs={
                "x": ParamInput.from_ref(node_a.id),
                "multiplier": ParamInput.from_value(10),
            },
        )

        chain = (
            ChainLink(node=node_a, position=0, flow_param=None, external_params={}),
            ChainLink(
                node=node_b,
                position=1,
                flow_param="x",
                external_params={"multiplier": ParamInput.from_value(10)},
            ),
        )

        composite = CompositeTaskNode(
            id=uuid4(),
            name="composite",
            description="test",
            backend=backend,
            chain=chain,
        )

        resolved_inputs = {"x": 5}
        future = composite.submit(backend, resolved_inputs)
        result = future.result()

        # Should compute (5 + 1) * 10 = 60
        assert result == 60

    def test_composite_map_node_executes_per_iteration(self) -> None:
        """CompositeMapTaskNode executes full chain per iteration."""
        from daglite.backends import SequentialBackend
        from daglite.graph.composite import ChainLink
        from daglite.graph.composite import CompositeMapTaskNode

        backend = SequentialBackend()

        # Create map chain: [1,2,3] -> x*2 -> x+10
        map1 = MapTaskNode(
            id=uuid4(),
            name="map1",
            description=None,
            backend=backend,
            func=lambda x: x * 2,
            mode="product",
            mapped_kwargs={"x": ParamInput.from_sequence([1, 2, 3])},
            fixed_kwargs={},
        )
        map2 = MapTaskNode(
            id=uuid4(),
            name="map2",
            description=None,
            backend=backend,
            func=lambda x: x + 10,
            mode="product",
            mapped_kwargs={"x": ParamInput.from_sequence_ref(map1.id)},
            fixed_kwargs={},
        )

        chain = (
            ChainLink(node=map1, position=0, flow_param=None, external_params={}),
            ChainLink(node=map2, position=1, flow_param="x", external_params={}),
        )

        composite = CompositeMapTaskNode(
            id=uuid4(),
            name="composite_map",
            description="test",
            backend=backend,
            source_map=map1,
            chain=chain,
        )

        # Execute composite
        resolved_inputs = {"x": [1, 2, 3]}
        futures = composite.submit(backend, resolved_inputs)
        results = [f.result() for f in futures]

        # Each iteration: (x * 2) + 10
        # [1*2+10, 2*2+10, 3*2+10] = [12, 14, 16]
        assert results == [12, 14, 16]

    def test_composite_task_node_inputs(self) -> None:
        """CompositeTaskNode.inputs() returns all initial and external inputs."""
        from daglite.backends import SequentialBackend
        from daglite.graph.composite import ChainLink
        from daglite.graph.composite import CompositeTaskNode

        backend = SequentialBackend()

        node_a = TaskNode(
            id=uuid4(),
            name="a",
            description=None,
            backend=backend,
            func=lambda x, y: x + y,
            kwargs={"x": ParamInput.from_value(5), "y": ParamInput.from_value(10)},
        )
        node_b = TaskNode(
            id=uuid4(),
            name="b",
            description=None,
            backend=backend,
            func=lambda x, multiplier: x * multiplier,
            kwargs={
                "x": ParamInput.from_ref(node_a.id),
                "multiplier": ParamInput.from_value(2),
            },
        )

        chain = (
            ChainLink(node=node_a, position=0, flow_param=None, external_params={}),
            ChainLink(
                node=node_b,
                position=1,
                flow_param="x",
                external_params={"multiplier": ParamInput.from_value(2)},
            ),
        )

        composite = CompositeTaskNode(
            id=uuid4(),
            name="composite",
            description="test",
            backend=backend,
            chain=chain,
        )

        # Should return initial inputs from node_a plus external params from node_b
        inputs = dict(composite.inputs())
        assert "x" in inputs
        assert "y" in inputs
        assert "multiplier" in inputs
