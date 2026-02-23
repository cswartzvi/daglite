"""
Unit tests for composite node IR types.

Tests CompositeStep, CompositeTaskNode, and CompositeMapTaskNode in isolation —
remap references, get_dependencies, and metadata correctness.
Execution tests live in tests/integration/test_composite.py.
"""

from __future__ import annotations

import tempfile
from uuid import UUID
from uuid import uuid4

from daglite._typing import NodeKind
from daglite.datasets.store import DatasetStore
from daglite.graph.nodes import CompositeMapTaskNode
from daglite.graph.nodes import CompositeTaskNode
from daglite.graph.nodes import MapTaskNode
from daglite.graph.nodes.base import NodeInput
from daglite.graph.nodes.base import NodeOutputConfig
from daglite.graph.nodes.composite_node import CompositeStep
from daglite.graph.nodes.composite_node import IterSourceConfig
from daglite.graph.nodes.composite_node import _remap_composite_step
from daglite.graph.nodes.composite_node import _remap_composite_steps
from daglite.graph.nodes.dataset_node import DatasetNode
from daglite.graph.nodes.reduce_node import ReduceConfig
from daglite.graph.nodes.reduce_node import ReduceNode
from daglite.graph.nodes.task_node import TaskNode


def _dummy_func(x: int = 0) -> int:
    return x


def _make_composite_step(
    *,
    timeout: float | None = None,
    step_kind: NodeKind = "task",
    ref: UUID | None = None,
) -> CompositeStep:
    external: dict[str, NodeInput] = {}
    if ref is not None:
        external["x"] = NodeInput.from_ref(ref)
    return CompositeStep(
        id=uuid4(),
        name="step",
        description=None,
        func=lambda x: x,
        flow_param="x",
        external_params=external,
        output_configs=(),
        retries=0,
        cache=False,
        cache_ttl=None,
        timeout=timeout,
        step_kind=step_kind,
    )


def _make_output_config(dep_id: UUID) -> tuple[NodeOutputConfig, ...]:
    return (
        NodeOutputConfig(
            key="out_{v}.pkl",
            dependencies={"v": NodeInput.from_ref(dep_id)},
        ),
    )


class TestCompositeStepMetadata:
    """CompositeStep.metadata should use the step_kind field."""

    def test_metadata_kind_defaults_to_task(self) -> None:
        step = _make_composite_step()
        assert step.metadata.kind == "task"

    def test_metadata_kind_uses_step_kind(self) -> None:
        step = _make_composite_step(step_kind="map")
        assert step.metadata.kind == "map"


class TestRemapStepHelpers:
    """Unit tests for composite step remap helper functions."""

    def test_remap_composite_step_changes_reference(self) -> None:
        old_id = uuid4()
        new_id = uuid4()
        step = _make_composite_step(ref=old_id)
        remapped = _remap_composite_step(step, {old_id: new_id})
        assert remapped.external_params["x"].reference == new_id
        assert remapped is not step

    def test_remap_composite_step_no_change_returns_same(self) -> None:
        old_id = uuid4()
        step = _make_composite_step(ref=old_id)
        remapped = _remap_composite_step(step, {uuid4(): uuid4()})
        assert remapped is step

    def test_remap_steps_returns_none_if_unchanged(self) -> None:
        steps = (_make_composite_step(), _make_composite_step())
        result = _remap_composite_steps(steps, {uuid4(): uuid4()})
        assert result is None

    def test_remap_steps_returns_new_tuple_if_changed(self) -> None:
        old_id = uuid4()
        new_id = uuid4()
        steps = (_make_composite_step(), _make_composite_step(ref=old_id))
        result = _remap_composite_steps(steps, {old_id: new_id})
        assert result is not None
        assert result[1].external_params["x"].reference == new_id


class TestCompositeTaskNodeRemapReferences:
    """remap_references on CompositeTaskNode."""

    def test_remap_updates_chain_refs(self) -> None:
        old_id = uuid4()
        new_id = uuid4()
        step = _make_composite_step(ref=old_id)
        composite = CompositeTaskNode(id=uuid4(), name="c", steps=(step,))
        remapped = composite.remap_references({old_id: new_id})
        assert remapped is not composite
        assert remapped.steps[0].external_params["x"].reference == new_id

    def test_remap_no_change_returns_self(self) -> None:
        step = _make_composite_step()
        composite = CompositeTaskNode(id=uuid4(), name="c", steps=(step,))
        result = composite.remap_references({uuid4(): uuid4()})
        assert result is composite

    def test_remap_only_output_configs_changes(self) -> None:
        """When only the node's own output_configs dep changes (no step ref), chain preserved."""
        dep_id = uuid4()
        new_id = uuid4()
        step = _make_composite_step()  # no external refs
        oc = NodeOutputConfig(key="out_{v}.pkl", dependencies={"v": NodeInput.from_ref(dep_id)})
        composite = CompositeTaskNode(id=uuid4(), name="c", steps=(step,), output_configs=(oc,))
        remapped = composite.remap_references({dep_id: new_id})
        assert remapped is not composite
        assert remapped.steps is composite.steps  # chain tuple identity preserved
        assert remapped.output_configs[0].dependencies["v"].reference == new_id


class TestCompositeMapTaskNodeRemapReferences:
    """remap_references on CompositeMapTaskNode."""

    def _make_source_map(self) -> MapTaskNode:
        return MapTaskNode(
            id=uuid4(),
            name="src_map",
            func=lambda x: x,
            mode="zip",
            fixed_kwargs={},
            mapped_kwargs={"x": NodeInput.from_value([1])},
        )

    def test_remap_chain_refs(self) -> None:
        old_id = uuid4()
        new_id = uuid4()
        source = self._make_source_map()
        step = _make_composite_step(ref=old_id)
        composite = CompositeMapTaskNode(id=uuid4(), name="cmap", source_map=source, steps=(step,))
        remapped = composite.remap_references({old_id: new_id})
        assert remapped is not composite
        assert remapped.steps[0].external_params["x"].reference == new_id

    def test_remap_join_step_refs(self) -> None:
        old_id = uuid4()
        new_id = uuid4()
        source = self._make_source_map()
        join = _make_composite_step(ref=old_id)
        composite = CompositeMapTaskNode(
            id=uuid4(), name="cmap", source_map=source, steps=(), terminal="join", join_step=join
        )
        remapped = composite.remap_references({old_id: new_id})
        assert remapped is not composite
        assert remapped.join_step is not None
        assert remapped.join_step.external_params["x"].reference == new_id

    def test_remap_initial_input_ref(self) -> None:
        old_id = uuid4()
        new_id = uuid4()
        source = self._make_source_map()
        composite = CompositeMapTaskNode(
            id=uuid4(),
            name="cmap",
            source_map=source,
            steps=(),
            terminal="reduce",
            reduce_config=ReduceConfig(func=lambda acc, item: acc + item),
            initial_accumulator=NodeInput.from_ref(old_id),
        )
        remapped = composite.remap_references({old_id: new_id})
        assert remapped is not composite
        assert remapped.initial_accumulator is not None
        assert remapped.initial_accumulator.reference == new_id

    def test_remap_iter_source_kwargs_ref(self) -> None:
        """When iter_source kwargs reference a remapped node, the ref is updated."""
        old_id = uuid4()
        new_id = uuid4()
        source = self._make_source_map()
        iter_src = IterSourceConfig(
            id=uuid4(),
            func=lambda: iter([]),
            kwargs={"data": NodeInput.from_ref(old_id)},
        )
        composite = CompositeMapTaskNode(
            id=uuid4(), name="cmap", source_map=source, steps=(), iter_source=iter_src
        )
        remapped = composite.remap_references({old_id: new_id})
        assert remapped is not composite
        assert remapped.iter_source is not None
        assert remapped.iter_source.kwargs["data"].reference == new_id

    def test_remap_no_change_returns_self(self) -> None:
        source = self._make_source_map()
        composite = CompositeMapTaskNode(id=uuid4(), name="cmap", source_map=source, steps=())
        result = composite.remap_references({uuid4(): uuid4()})
        assert result is composite

    def test_remap_only_output_configs_changes(self) -> None:
        """When only the node's own output_configs dep changes, chain/join preserved."""
        dep_id = uuid4()
        new_id = uuid4()
        source = self._make_source_map()
        oc = NodeOutputConfig(key="out_{v}.pkl", dependencies={"v": NodeInput.from_ref(dep_id)})
        composite = CompositeMapTaskNode(
            id=uuid4(), name="cmap", source_map=source, steps=(), output_configs=(oc,)
        )
        remapped = composite.remap_references({dep_id: new_id})
        assert remapped is not composite
        assert remapped.steps == composite.steps
        assert remapped.output_configs[0].dependencies["v"].reference == new_id


class TestCompositeMapDependencyCoverage:
    """get_dependencies with initial_input.reference."""

    def test_composite_map_deps_include_initial_ref(self) -> None:
        initial_ref = uuid4()
        source = MapTaskNode(
            id=uuid4(),
            name="src",
            func=lambda x: x,
            mode="zip",
            fixed_kwargs={},
            mapped_kwargs={"x": NodeInput.from_value([1])},
        )
        composite = CompositeMapTaskNode(
            id=uuid4(),
            name="cmap",
            source_map=source,
            steps=(),
            terminal="reduce",
            reduce_config=ReduceConfig(func=lambda acc, item: acc + item),
            initial_accumulator=NodeInput.from_ref(initial_ref),
        )
        deps = composite.get_dependencies()
        assert initial_ref in deps


class TestRemapReferencesOutputConfigs:
    """Every dep UUID from get_dependencies() must survive remap_references()."""

    def _assert_remap_covers_deps(self, node: object) -> None:
        dep_ids = node.get_dependencies()  # type: ignore[union-attr]
        id_mapping = {uid: uuid4() for uid in dep_ids}
        remapped = node.remap_references(id_mapping)  # type: ignore[union-attr]
        new_deps = remapped.get_dependencies()
        for old_id in dep_ids:
            assert old_id not in new_deps, (
                f"{type(node).__name__}: old dep {old_id} survived remap_references"
            )
            assert id_mapping[old_id] in new_deps, (
                f"{type(node).__name__}: new dep {id_mapping[old_id]} not found"
            )

    def test_task_node(self) -> None:
        dep_id = uuid4()
        node = TaskNode(
            id=uuid4(),
            name="t",
            func=_dummy_func,
            kwargs={"x": NodeInput.from_value(1)},
            output_configs=_make_output_config(dep_id),
        )
        self._assert_remap_covers_deps(node)

    def test_map_task_node(self) -> None:
        dep_id = uuid4()
        node = MapTaskNode(
            id=uuid4(),
            name="m",
            func=_dummy_func,
            mode="zip",
            fixed_kwargs={},
            mapped_kwargs={"x": NodeInput.from_sequence([1, 2])},
            output_configs=_make_output_config(dep_id),
        )
        self._assert_remap_covers_deps(node)

    def test_dataset_node(self) -> None:
        dep_id = uuid4()
        with tempfile.TemporaryDirectory() as tmpdir:
            node = DatasetNode(
                id=uuid4(),
                name="d",
                store=DatasetStore(tmpdir),
                load_key="data.pkl",
                kwargs={},
                output_configs=_make_output_config(dep_id),
            )
            self._assert_remap_covers_deps(node)

    def test_composite_task_node_composite_step(self) -> None:
        dep_id = uuid4()
        step = CompositeStep(
            id=uuid4(),
            name="step",
            description=None,
            func=_dummy_func,
            flow_param="x",
            external_params={"x": NodeInput.from_ref(dep_id)},
            output_configs=_make_output_config(dep_id),
            retries=0,
            cache=False,
            cache_ttl=None,
            timeout=None,
            step_kind="task",
        )
        node = CompositeTaskNode(id=uuid4(), name="c", steps=(step,))
        self._assert_remap_covers_deps(node)

    def test_composite_map_task_node_join(self) -> None:
        dep_id = uuid4()
        source = MapTaskNode(
            id=uuid4(),
            name="src",
            func=_dummy_func,
            mode="zip",
            fixed_kwargs={},
            mapped_kwargs={"x": NodeInput.from_sequence([1, 2])},
        )
        join_step = CompositeStep(
            id=uuid4(),
            name="join",
            description=None,
            func=_dummy_func,
            flow_param="x",
            external_params={"x": NodeInput.from_ref(dep_id)},
            output_configs=_make_output_config(dep_id),
            retries=0,
            cache=False,
            cache_ttl=None,
            timeout=None,
            step_kind="task",
        )
        node = CompositeMapTaskNode(
            id=uuid4(),
            name="cm",
            steps=(),
            source_map=source,
            terminal="join",
            join_step=join_step,
        )
        self._assert_remap_covers_deps(node)

    def test_reduce_node(self) -> None:
        dep_id = uuid4()
        source_id = uuid4()
        node = ReduceNode(
            id=uuid4(),
            name="r",
            source_id=source_id,
            reduce_config=ReduceConfig(func=lambda acc, item: acc + item),
            output_configs=_make_output_config(dep_id),
        )
        self._assert_remap_covers_deps(node)
