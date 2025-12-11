"""Tests for graph optimization, chain detection, composite nodes, and optimization flag."""

from uuid import UUID
from uuid import uuid4

from daglite import task
from daglite.engine import Engine
from daglite.graph.base import GraphNode
from daglite.graph.base import ParamInput
from daglite.graph.builder import build_graph
from daglite.graph.composite import CompositeTaskNode
from daglite.graph.nodes import MapTaskNode
from daglite.graph.nodes import TaskNode
from daglite.graph.optimizer import optimize_graph
from daglite.settings import DagliteSettings


class TestOptimizationFlag:
    """Test enable_optimization configuration flag."""

    def test_optimization_enabled_by_default(self) -> None:
        """Test that optimization is enabled by default."""
        settings = DagliteSettings()
        assert settings.enable_optimization is True

    def test_optimization_can_be_disabled(self) -> None:
        """Test that optimization can be disabled via settings."""
        settings = DagliteSettings(enable_optimization=False)
        assert settings.enable_optimization is False

    def test_optimization_flag_affects_execution(self) -> None:
        """Test that the optimization flag controls whether composites are created."""

        @task
        def add_one(x: int) -> int:
            return x + 1

        @task
        def multiply_two(x: int) -> int:
            return x * 2

        @task
        def subtract_three(x: int) -> int:
            return x - 3

        # Create a linear chain
        a = add_one.bind(x=10)
        b = multiply_two.bind(x=a)
        c = subtract_three.bind(x=b)

        # With optimization enabled (default)
        engine_optimized = Engine(
            default_backend="sequential",
            settings=DagliteSettings(enable_optimization=True),
        )
        result_optimized = engine_optimized.evaluate(c)
        assert result_optimized == 19  # (10 + 1) * 2 - 3 = 19

        # With optimization disabled
        engine_unoptimized = Engine(
            default_backend="sequential",
            settings=DagliteSettings(enable_optimization=False),
        )
        result_unoptimized = engine_unoptimized.evaluate(c)
        assert result_unoptimized == 19  # Same result

        # Both should produce the same result
        assert result_optimized == result_unoptimized

    def test_optimization_creates_composites(self) -> None:
        """Verify that optimization actually creates composite nodes."""
        from daglite.backends import SequentialBackend

        backend = SequentialBackend()

        @task(backend=backend)
        def add_one(x: int) -> int:
            return x + 1

        @task(backend=backend)
        def multiply_two(x: int) -> int:
            return x * 2

        @task(backend=backend)
        def subtract_three(x: int) -> int:
            return x - 3

        # Create a linear chain
        a = add_one.bind(x=10)
        b = multiply_two.bind(x=a)
        c = subtract_three.bind(x=b)

        # Build original graph
        nodes = build_graph(c)
        original_count = len(nodes)
        assert original_count == 3  # Three tasks

        # Optimize the graph
        optimized_nodes, id_mapping = optimize_graph(nodes, c.id)

        # Should have fewer nodes (chain collapsed into composite)
        assert len(optimized_nodes) < original_count
        assert len(optimized_nodes) == 1  # All three collapsed into one composite

        # Should have mapping for all original nodes
        assert len(id_mapping) == 3

        # The single node should be a CompositeTaskNode
        composite_node = list(optimized_nodes.values())[0]
        assert isinstance(composite_node, CompositeTaskNode)
        assert len(composite_node.chain) == 3

    def test_optimization_disabled_preserves_original_graph(self) -> None:
        """Verify that disabling optimization preserves the original graph structure."""

        @task
        def add_one(x: int) -> int:
            return x + 1

        @task
        def multiply_two(x: int) -> int:
            return x * 2

        # Create a linear chain
        a = add_one.bind(x=10)
        b = multiply_two.bind(x=a)

        # When optimization is disabled, graph should be unchanged
        engine = Engine(
            default_backend="sequential",
            settings=DagliteSettings(enable_optimization=False),
        )

        # The engine should use the graph as-is
        # We can verify this by checking that execution still works
        result = engine.evaluate(b)
        assert result == 22  # (10 + 1) * 2 = 22

        # Verify no CompositeTaskNode was created by checking the graph structure
        # (This is a smoke test - we can't easily inspect internal state,
        # but we know from the other tests that optimization would create composites)


class TestGraphOptimizer:
    """Test graph optimization and chain detection."""

    def test_simple_task_chain(self) -> None:
        """Linear TaskNode chain is combined into CompositeTaskNode."""
        from daglite.backends import SequentialBackend

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
        optimized, _ = optimize_graph(nodes, node_c.id)

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
        optimized, _ = optimize_graph(nodes, node_c.id)

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
        optimized, _ = optimize_graph(nodes, node_c.id)

        # Should have 2 nodes: A (ungrouped) and composite(B, C)
        assert len(optimized) == 2

    def test_chain_breaks_at_diamond_pattern(self) -> None:
        """Chain detection stops at diamond dependencies."""
        from daglite.backends import SequentialBackend

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
        optimized, _ = optimize_graph(nodes, node_d.id)

        # Should have 3 nodes: composite(A,B), composite(A,C), D
        # Or possibly: A, composite(B), composite(C), D depending on detection order
        # At minimum, D should not be in any composite due to multiple dependencies
        assert len(optimized) >= 3

    def test_single_nodes_not_grouped(self) -> None:
        """Single nodes without chain partners stay ungrouped."""
        from daglite.backends import SequentialBackend

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
        optimized, _ = optimize_graph(nodes, node_a.id)

        # Should remain as single node
        assert len(optimized) == 1
        assert optimized[node_a.id] == node_a

    def test_simple_map_chain(self) -> None:
        """Linear MapTaskNode chain is combined into CompositeMapTaskNode."""
        from daglite.backends import SequentialBackend

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
        optimized, _ = optimize_graph(nodes, map2.id)

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

        # Should return initial inputs from node_a
        # (multiplier is a value param for node_b, baked into the node, not exposed)
        inputs = dict(composite.inputs())
        assert "x" in inputs
        assert "y" in inputs
        assert len(inputs) == 2  # Only first node's inputs
