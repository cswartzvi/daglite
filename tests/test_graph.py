"""
Unit Tests for graph construction and nodes in daglite.graph.

Tests in this file should NOT focus on evaluation. Evaluation tests are in tests/evaluation/.
"""

from functools import cached_property
from uuid import uuid4

import pytest

from daglite.exceptions import ExecutionError
from daglite.exceptions import GraphError
from daglite.exceptions import ParameterError
from daglite.graph.base import InputParam
from daglite.graph.builder import build_graph
from daglite.graph.nodes import MapTaskNode
from daglite.graph.nodes import TaskNode
from daglite.tasks import task


class TestInputParam:
    """Test InputParam creation and resolution."""

    def test_from_value(self) -> None:
        """InputParam.from_value creates a value-type input."""
        param = InputParam.from_value(42)
        assert param._kind == "value"
        assert param.value == 42
        assert param.reference is None

    def test_from_ref(self) -> None:
        """InputParam.from_ref creates a ref-type input."""
        node_id = uuid4()
        param = InputParam.from_ref(node_id)
        assert param._kind == "ref"
        assert param.reference == node_id
        assert param.reference is not None

    def test_from_sequence(self) -> None:
        """InputParam.from_sequence creates a sequence-type input."""
        param = InputParam.from_sequence([1, 2, 3])
        assert param._kind == "sequence"
        assert param.value == [1, 2, 3]
        assert param.reference is None

    def test_from_sequence_ref(self) -> None:
        """InputParam.from_sequence_ref creates a sequence_ref-type input."""
        node_id = uuid4()
        param = InputParam.from_sequence_ref(node_id)
        assert param._kind == "sequence_ref"
        assert param.reference == node_id
        assert param.reference is not None

    def test_resolve_value(self) -> None:
        """InputParam resolves value inputs correctly."""
        param = InputParam.from_value(100)
        assert param.resolve({}) == 100

    def test_resolve_ref(self) -> None:
        """InputParam resolves ref inputs from values dict."""
        node_id = uuid4()
        param = InputParam.from_ref(node_id)
        values = {node_id: "result"}
        assert param.resolve(values) == "result"

    def test_resolve_sequence_from_sequence(self) -> None:
        """InputParam resolves sequence inputs correctly."""
        param = InputParam.from_sequence([10, 20, 30])
        assert param.resolve({}) == [10, 20, 30]

    def test_resolve_sequence_from_ref(self) -> None:
        """InputParam resolves sequence_ref inputs from values dict."""
        node_id = uuid4()
        param = InputParam.from_sequence_ref(node_id)
        values = {node_id: [1, 2, 3]}
        assert param.resolve(values) == [1, 2, 3]

    def test_invalid_ref_with_value_kind(self) -> None:
        """InputParam with value kind must have a value, not a ref."""
        with pytest.raises(GraphError, match="InputParam kind 'value' must not have a ref."):
            InputParam(_kind="value", reference=uuid4())

    def test_invalid_value_with_ref_kind(self) -> None:
        """InputParam with ref kind must have a ref, not a value."""
        with pytest.raises(GraphError, match="InputParam kind 'ref' must not have a value."):
            InputParam(_kind="ref", value=42, reference=uuid4())

    def test_invalid_scalar_with_sequence_kind(self) -> None:
        """InputParam with sequence kind must have a value, not a ref."""
        with pytest.raises(GraphError, match="InputParam kind 'sequence' must not have a ref."):
            InputParam(_kind="sequence", reference=uuid4())

    def test_invalid_scalar_with_ref_kind(self) -> None:
        """InputParam with ref kind must have a ref, not a value."""
        with pytest.raises(GraphError, match="InputParam kind 'ref' requires a ref."):
            InputParam(_kind="ref", value=42)

    def test_invalid_sequence_with_ref_kind(self) -> None:
        """InputParam with sequence_ref kind must have a ref, not a value."""
        with pytest.raises(GraphError, match="InputParam kind 'sequence_ref' requires a ref."):
            InputParam(_kind="sequence_ref", value=[1, 2, 3])


class TestTaskNodes:
    """Test TaskNode initialization and properties."""

    def test_properties(self) -> None:
        """TaskNode initializes with correct properties and kind."""

        def add(x: int, y: int) -> int:  # pragma: no cover
            return x + y

        node = TaskNode(
            id=uuid4(),
            name="add_task",
            description="Addition",
            backend_name=None,
            func=add,
            kwargs={
                "x": InputParam.from_value(1),
                "y": InputParam.from_value(2),
            },
        )

        assert node.name == "add_task"
        assert len(node.kwargs) == 2

    def test_dependencies_with_refs(self) -> None:
        """TaskNode.dependencies() extracts refs from parameters."""
        dep_id = uuid4()

        def process(x: int) -> int:  # pragma: no cover
            return x * 2

        node = TaskNode(
            id=uuid4(),
            name="process",
            description=None,
            backend_name=None,
            func=process,
            kwargs={"x": InputParam.from_ref(dep_id)},
        )

        deps = node.dependencies()
        assert len(deps) == 1
        assert dep_id in deps

    def test_dependencies_without_refs(self) -> None:
        """TaskNode.dependencies() returns empty set for value-only params."""

        def process(x: int) -> int:  # pragma: no cover
            return x * 2

        node = TaskNode(
            id=uuid4(),
            name="process",
            description=None,
            backend_name=None,
            func=process,
            kwargs={"x": InputParam.from_value(10)},
        )

        deps = node.dependencies()
        assert len(deps) == 0


class TestMapTaskNodes:
    """Test MapTaskNode initialization and properties."""

    def test_extend_mode(self) -> None:
        """MapTaskNode initializes with extend mode."""

        def process(x: int) -> int:  # pragma: no cover
            return x**2

        node = MapTaskNode(
            id=uuid4(),
            name="process_many",
            description=None,
            backend_name=None,
            func=process,
            mode="extend",
            fixed_kwargs={},
            mapped_kwargs={"x": InputParam.from_sequence([1, 2, 3])},
        )

        assert node.mode == "extend"

    def test_zip_mode(self) -> None:
        """MapTaskNode initializes with zip mode."""

        def process(x: int) -> int:  # pragma: no cover
            return x**2

        node = MapTaskNode(
            id=uuid4(),
            name="process_many",
            description=None,
            backend_name=None,
            func=process,
            mode="zip",
            fixed_kwargs={},
            mapped_kwargs={"x": InputParam.from_sequence([1, 2, 3])},
        )

        assert node.mode == "zip"

    def test_dependencies_from_fixed(self) -> None:
        """MapTaskNode.dependencies() extracts refs from fixed kwargs."""
        dep_id = uuid4()

        def add(x: int, offset: int) -> int:  # pragma: no cover
            return x + offset

        node = MapTaskNode(
            id=uuid4(),
            name="add_offset",
            description=None,
            backend_name=None,
            func=add,
            mode="extend",
            fixed_kwargs={"offset": InputParam.from_ref(dep_id)},
            mapped_kwargs={"x": InputParam.from_sequence([1, 2, 3])},
        )

        deps = node.dependencies()
        assert dep_id in deps

    def test_dependencies_from_mapped(self) -> None:
        """MapTaskNode.dependencies() extracts refs from mapped kwargs."""
        dep_id = uuid4()

        def add(x: int, offset: int) -> int:  # pragma: no cover
            return x + offset

        node = MapTaskNode(
            id=uuid4(),
            name="add_offset",
            description=None,
            backend_name=None,
            func=add,
            mode="extend",
            fixed_kwargs={"offset": InputParam.from_value(10)},
            mapped_kwargs={"x": InputParam.from_sequence_ref(dep_id)},
        )

        deps = node.dependencies()
        assert dep_id in deps

    def test_inputs(self) -> None:
        """MapTaskNode.inputs() returns both fixed and mapped kwargs."""

        def add(x: int, offset: int) -> int:  # pragma: no cover
            return x + offset

        node = MapTaskNode(
            id=uuid4(),
            name="add_offset",
            description=None,
            backend_name=None,
            func=add,
            mode="extend",
            fixed_kwargs={"offset": InputParam.from_value(10)},
            mapped_kwargs={"x": InputParam.from_sequence([1, 2, 3])},
        )

        # Check kwargs are stored correctly
        assert len(node.fixed_kwargs) == 1
        assert len(node.mapped_kwargs) == 1
        assert "offset" in node.fixed_kwargs
        assert "x" in node.mapped_kwargs

    def test_zip_mode_length_mismatch(self) -> None:
        """MapTaskNode submission fails with mismatched sequence lengths in zip mode."""

        def add(x: int, y: int) -> int:  # pragma: no cover
            return x + y

        node = MapTaskNode(
            id=uuid4(),
            name="add_pairs",
            description=None,
            backend_name=None,
            func=add,
            mode="zip",
            fixed_kwargs={},
            mapped_kwargs={
                "x": InputParam.from_sequence([1, 2, 3]),
                "y": InputParam.from_sequence([10, 20]),  # Different length
            },
        )

        resolved_inputs = node.resolve_inputs({})
        with pytest.raises(
            ParameterError, match="Map task .* in 'zip' mode requires all sequences"
        ):
            node.build_iteration_calls(resolved_inputs)

    def test_invalid_mode(self) -> None:
        """MapTaskNode build_iteration_calls fails with invalid mode."""

        def process(x: int) -> int:  # pragma: no cover
            return x * 2

        node = MapTaskNode(
            id=uuid4(),
            name="process_many",
            description=None,
            backend_name=None,
            func=process,
            mode="invalid",  # Invalid mode
            fixed_kwargs={},
            mapped_kwargs={"x": InputParam.from_sequence([1, 2, 3])},
        )

        resolved_inputs = node.resolve_inputs({})
        with pytest.raises(ExecutionError, match="Unknown map mode 'invalid'"):
            node.build_iteration_calls(resolved_inputs)


class TestBuildGraph:
    """
    Test build_graph function with various graph structures.

    NOTE: Tests focus on graph construction, not evaluation.
    """

    def test_single_node(self) -> None:
        """build_graph handles single node graph."""

        @task
        def simple() -> int:  # pragma: no cover
            return 42

        bound = simple()
        graph = build_graph(bound)

        assert len(graph) == 1
        assert bound.id in graph

    def test_linear_chain(self) -> None:
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

        s1 = step1()
        s2 = step2(x=s1)
        s3 = step3(x=s2)

        graph = build_graph(s3)

        assert len(graph) == 3
        assert s1.id in graph
        assert s2.id in graph
        assert s3.id in graph

    def test_dag_with_multiple_deps(self) -> None:
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

        s1 = source1()
        s2 = source2()
        result = combine(a=s1, b=s2)

        graph = build_graph(result)

        assert len(graph) == 3
        # Verify combine node depends on both sources
        combine_node = graph[result.id]
        deps = combine_node.dependencies()
        assert s1.id in deps
        assert s2.id in deps

    def test_diamond_dependency(self) -> None:
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

        root = start()
        b1 = branch1(x=root)
        b2 = branch2(x=root)
        final = merge(a=b1, b=b2)

        graph = build_graph(final)

        assert len(graph) == 4
        assert all(node_id in graph for node_id in [root.id, b1.id, b2.id, final.id])

    def test_shared_dependency(self) -> None:
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
        s = shared()
        u1 = use1(x=s)
        u2 = use2(x=s)
        result = combine(a=u1, b=u2)

        graph = build_graph(result)

        # Should only have 4 nodes (shared is not duplicated)
        assert len(graph) == 4
        assert s.id in graph
        assert u1.id in graph
        assert u2.id in graph
        assert result.id in graph

    def test_skips_already_processed_nodes(self) -> None:
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
        l1 = leaf1()
        l2 = leaf2()
        m1 = middle1(a=l1, b=l2)
        m2 = middle2(a=l1, b=l2)
        r = root(x=m1, y=m2)

        graph = build_graph(r)

        # Should have 5 unique nodes (leaves should not be duplicated)
        assert len(graph) == 5
        # Verify each node appears exactly once
        assert sum(1 for nid in graph if nid == l1.id) == 1
        assert sum(1 for nid in graph if nid == l2.id) == 1

    def test_detects_circular_dependency(self) -> None:
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
                    backend_name=None,
                    func=lambda: None,  # pragma: no cover
                    kwargs={},
                )

        # Create A -> B -> A circular dependency
        id_a = uuid4()
        id_b = uuid4()
        builder_b = CircularBuilder(id_b, None)
        builder_a = CircularBuilder(id_a, builder_b)
        builder_b._other = builder_a  # Create the cycle

        with pytest.raises(GraphError, match="Circular dependency detected"):
            build_graph(builder_a)  # pyright: ignore

    def test_detects_self_reference(self) -> None:
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
                    backend_name=None,
                    func=lambda: None,  # pragma: no cover
                    kwargs={},
                )

        builder = SelfRefBuilder()

        with pytest.raises(GraphError, match="Circular dependency detected"):
            build_graph(builder)  # pyright: ignore
