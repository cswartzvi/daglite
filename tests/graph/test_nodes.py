"""Tests for graph nodes (ParamInput, TaskNode, MapTaskNode)."""

from uuid import uuid4

import pytest

from daglite.exceptions import ExecutionError
from daglite.exceptions import ParameterError
from daglite.graph.base import ParamInput
from daglite.graph.nodes import MapTaskNode
from daglite.graph.nodes import TaskNode


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
