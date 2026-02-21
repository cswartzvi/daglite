"""
Unit tests for ReduceNode IR type.

Tests kind, remap_references, and get_dependencies. Execution tests
(including the fallback path when optimization is disabled) live in
tests/integration/test_composite.py.
"""

from __future__ import annotations

from uuid import uuid4

from daglite.graph.nodes.base import NodeInput
from daglite.graph.nodes.base import NodeOutputConfig
from daglite.graph.nodes.reduce_node import ReduceConfig
from daglite.graph.nodes.reduce_node import ReduceNode


def _make_reduce_node(**kwargs) -> ReduceNode:
    defaults = dict(
        id=uuid4(),
        name="reduce",
        description=None,
        backend_name=None,
        source_id=uuid4(),
        reduce_config=ReduceConfig(func=lambda acc, item: acc + item),
    )
    return ReduceNode(**{**defaults, **kwargs})


class TestReduceNodeKind:
    """ReduceNode.kind should return 'reduce'."""

    def test_kind_is_reduce(self) -> None:
        node = _make_reduce_node()
        assert node.kind == "reduce"


class TestReduceNodeRemapReferences:
    """ReduceNode.remap_references remaps source_id and initial_input."""

    def test_remap_source_id(self) -> None:
        old_id = uuid4()
        new_id = uuid4()
        node = _make_reduce_node(source_id=old_id)
        remapped = node.remap_references({old_id: new_id})
        assert remapped.source_id == new_id

    def test_remap_initial_input_ref(self) -> None:
        old_id = uuid4()
        new_id = uuid4()
        node = _make_reduce_node(initial_input=NodeInput.from_ref(old_id))
        remapped = node.remap_references({old_id: new_id})
        assert remapped.initial_input.reference == new_id

    def test_remap_no_change_returns_self(self) -> None:
        node = _make_reduce_node()
        result = node.remap_references({uuid4(): uuid4()})
        assert result is node

    def test_remap_output_configs(self) -> None:
        dep_id = uuid4()
        new_id = uuid4()
        oc = NodeOutputConfig(key="out_{v}.pkl", dependencies={"v": NodeInput.from_ref(dep_id)})
        node = _make_reduce_node(output_configs=(oc,))
        remapped = node.remap_references({dep_id: new_id})
        assert remapped.output_configs[0].dependencies["v"].reference == new_id


class TestReduceNodeDependencyCoverage:
    """get_dependencies includes source_id and optional initial_input ref."""

    def test_deps_include_source(self) -> None:
        source_id = uuid4()
        node = _make_reduce_node(source_id=source_id)
        assert source_id in node.get_dependencies()

    def test_deps_include_initial_ref(self) -> None:
        source_id = uuid4()
        initial_ref = uuid4()
        node = _make_reduce_node(
            source_id=source_id,
            initial_input=NodeInput.from_ref(initial_ref),
        )
        deps = node.get_dependencies()
        assert source_id in deps
        assert initial_ref in deps
