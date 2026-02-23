"""
Unit tests for the @workflow decorator, Workflow, and WorkflowResult.

Tests in this file should NOT focus on evaluation. Evaluation behavior tests are in tests/behavior/
and cross-subsystem scenarios are in tests/integration/.
"""

from __future__ import annotations

from uuid import UUID
from uuid import uuid4

import pytest

from daglite import task
from daglite import workflow
from daglite.exceptions import AmbiguousResultError
from daglite.exceptions import GraphError
from daglite.graph.builder import build_graph_multi
from daglite.workflows import Workflow
from daglite.workflows import WorkflowResult


@task
def add(x: int, y: int) -> int:
    return x + y


@task
def mul(x: int, y: int) -> int:
    return x * y


@task
def double(x: int) -> int:
    return x * 2


@task
def negate(x: int) -> int:
    return -x


class TestWorkflowDecorator:
    def test_bare_decorator(self):
        @workflow
        def wf():  # pragma: no cover
            return add(x=1, y=2)

        assert isinstance(wf, Workflow)
        assert wf.name == "wf"
        assert wf.description == ""

    def test_decorator_with_parentheses(self):
        @workflow()
        def wf():  # pragma: no cover
            return add(x=1, y=2)

        assert isinstance(wf, Workflow)
        assert wf.name == "wf"

    def test_decorator_with_name_and_description(self):
        @workflow(name="custom", description="my workflow")
        def wf():  # pragma: no cover
            return add(x=1, y=2)

        assert wf.name == "custom"
        assert wf.description == "my workflow"

    def test_decorator_preserves_docstring(self):
        @workflow
        def wf():  # pragma: no cover
            """This is my workflow."""
            return add(x=1, y=2)

        assert wf.description == "This is my workflow."

    def test_call_returns_raw_future(self):
        @workflow
        def wf(x: int, y: int):
            return add(x=x, y=y)

        future = wf(x=3, y=4)
        assert not isinstance(future, WorkflowResult)

    def test_cannot_decorate_class(self):
        with pytest.raises(TypeError, match="callable functions"):

            @workflow
            class NotAFunction:  # pragma: no cover
                pass


class TestCollectFutures:
    def setup_method(self):
        @workflow
        def wf():  # pragma: no cover
            pass

        self.wf = wf

    def test_single_future(self):
        future = add(x=1, y=2)
        assert self.wf._collect_futures(future) == [future]

    def test_tuple_of_futures(self):
        f1 = add(x=1, y=2)
        f2 = mul(x=3, y=4)
        assert self.wf._collect_futures((f1, f2)) == [f1, f2]

    def test_list_of_futures(self):
        f1 = add(x=1, y=2)
        f2 = mul(x=3, y=4)
        assert self.wf._collect_futures([f1, f2]) == [f1, f2]

    def test_invalid_return_raises_type_error(self):
        with pytest.raises(TypeError, match="@workflow function must return"):
            self.wf._collect_futures(42)

    def test_invalid_string_raises_type_error(self):
        with pytest.raises(TypeError, match="str"):
            self.wf._collect_futures("not_a_future")

    def test_empty_list_raises(self):
        with pytest.raises(TypeError, match="empty"):
            self.wf._collect_futures([])

    def test_empty_tuple_raises(self):
        with pytest.raises(TypeError, match="empty"):
            self.wf._collect_futures(())

    def test_list_with_non_future_element_raises(self):
        future = add(x=1, y=2)
        with pytest.raises(TypeError, match="index 1"):
            self.wf._collect_futures([future, 42])


class TestWorkflowResult:
    def _make_result(self, data: dict[str, int]) -> WorkflowResult:
        results = {}
        name_for = {}
        for name, value in data.items():
            uid = uuid4()
            results[uid] = value
            name_for[uid] = name
        return WorkflowResult.build(results, name_for)

    def test_getitem_by_name(self):
        result = self._make_result({"add": 5, "mul": 6})
        assert result["add"] == 5
        assert result["mul"] == 6

    def test_getitem_by_uuid(self):
        uid = uuid4()
        result = WorkflowResult.build({uid: 99}, {uid: "task"})
        assert result[uid] == 99

    def test_getitem_missing_name_raises_key_error(self):
        result = self._make_result({"add": 5})
        with pytest.raises(KeyError, match="no_such"):
            _ = result["no_such"]

    def test_getitem_missing_uuid_raises_key_error(self):
        result = self._make_result({"add": 5})
        with pytest.raises(KeyError):
            _ = result[uuid4()]

    def test_ambiguous_name_raises_error(self):
        uid1, uid2 = uuid4(), uuid4()
        result = WorkflowResult.build(
            {uid1: 10, uid2: 20},
            {uid1: "task", uid2: "task"},
        )
        with pytest.raises(AmbiguousResultError, match="task"):
            _ = result["task"]

    def test_uuid_lookup_works_on_name_collision(self):
        uid1, uid2 = uuid4(), uuid4()
        result = WorkflowResult.build(
            {uid1: 10, uid2: 20},
            {uid1: "task", uid2: "task"},
        )
        assert result[uid1] == 10
        assert result[uid2] == 20

    def test_keys(self):
        result = self._make_result({"a": 1, "b": 2})
        assert set(result.keys()) == {"a", "b"}

    def test_values(self):
        result = self._make_result({"a": 1, "b": 2})
        assert set(result.values()) == {1, 2}

    def test_len(self):
        result = self._make_result({"a": 1, "b": 2})
        assert len(result) == 2

    def test_values_len(self):
        result = self._make_result({"a": 1, "b": 2})
        assert len(result.values()) == 2

    def test_items(self):
        result = self._make_result({"a": 1, "b": 2})
        assert dict(result.items()) == {"a": 1, "b": 2}

    def test_items_len(self):
        result = self._make_result({"a": 1, "b": 2})
        assert len(result.items()) == 2

    def test_items_expands_duplicates(self):
        uid1, uid2 = uuid4(), uuid4()
        result = WorkflowResult.build(
            {uid1: 10, uid2: 20},
            {uid1: "task", uid2: "task"},
        )
        pairs = list(result.items())
        assert len(pairs) == 2
        assert all(name == "task" for name, _ in pairs)
        assert {v for _, v in pairs} == {10, 20}

    def test_repr(self):
        result = self._make_result({"alpha": 1})
        assert "WorkflowResult" in repr(result)
        assert "alpha" in repr(result)


class TestWorkflowResultSingleAndAll:
    def _make_result(self, data: dict[str, int]) -> WorkflowResult:
        results = {}
        name_for = {}
        for name, value in data.items():
            uid = uuid4()
            results[uid] = value
            name_for[uid] = name
        return WorkflowResult.build(results, name_for)

    def test_single_returns_value(self):
        result = self._make_result({"add": 5})
        assert result.single("add") == 5

    def test_single_raises_on_missing(self):
        result = self._make_result({"add": 5})
        with pytest.raises(KeyError):
            result.single("missing")

    def test_single_raises_on_ambiguous(self):
        uid1, uid2 = uuid4(), uuid4()
        result = WorkflowResult.build(
            {uid1: 10, uid2: 20},
            {uid1: "task", uid2: "task"},
        )
        from daglite.exceptions import AmbiguousResultError

        with pytest.raises(AmbiguousResultError):
            result.single("task")

    def test_all_returns_list_for_single(self):
        result = self._make_result({"add": 5})
        assert result.all("add") == [5]

    def test_all_returns_list_for_multiple(self):
        uid1, uid2 = uuid4(), uuid4()
        result = WorkflowResult.build(
            {uid1: 10, uid2: 20},
            {uid1: "task", uid2: "task"},
        )
        assert sorted(result.all("task")) == [10, 20]

    def test_all_returns_empty_list_for_missing(self):
        result = self._make_result({"add": 5})
        assert result.all("missing") == []


class TestFutureAlias:
    def test_alias_sets_name(self):
        future = add(x=1, y=2)
        aliased = future.alias("my_add")
        assert aliased._alias == "my_add"

    def test_alias_preserves_id(self):
        future = add(x=1, y=2)
        aliased = future.alias("my_add")
        assert aliased.id == future.id

    def test_original_unaffected(self):
        future = add(x=1, y=2)
        _ = future.alias("my_add")
        assert future._alias is None

    def test_alias_survives_save_chain(self):
        # alias should be carried forward when .save() is called after .alias()
        future = add(x=1, y=2).alias("first").save("some_key")
        assert future._alias == "first"

    def test_alias_chaining(self):
        """Last alias wins when chained."""
        future = add(x=1, y=2).alias("first").alias("second")
        assert future._alias == "second"

    def test_default_alias_is_none(self):
        future = add(x=1, y=2)
        assert future._alias is None


class TestBuildGraphMulti:
    def test_shared_node_included_once(self):
        shared = double(x=10)
        sink_a = negate(x=shared)
        sink_b = add(x=shared, y=1)

        nodes = build_graph_multi([sink_a, sink_b])

        assert len(nodes) == 3
        assert shared.id in nodes
        assert sink_a.id in nodes
        assert sink_b.id in nodes

    def test_duplicate_root_does_not_raise(self):
        """Passing the same future twice must not trigger a false cycle error."""
        future = add(x=1, y=2)
        nodes = build_graph_multi([future, future])
        assert len(nodes) == 1
        assert future.id in nodes

    def test_circular_dependency_raises(self):
        """A genuine cycle in the builder graph must raise GraphError."""

        class FakeNode:
            def __init__(self, node_id: UUID, upstreams: list) -> None:
                self.id = node_id
                self._upstreams = upstreams

            def get_upstream_builders(self) -> list:
                return self._upstreams

            def build_node(self) -> None:  # pragma: no cover
                raise NotImplementedError

        a = FakeNode(uuid4(), [])
        b = FakeNode(uuid4(), [a])
        a._upstreams = [b]  # a -> b -> a

        with pytest.raises(GraphError, match="Circular dependency"):
            build_graph_multi([a])  # type: ignore

    def test_shared_ancestor_visited_once(self):
        """A diamond graph (b→a→shared, b→shared) must not duplicate nodes."""
        # b's upstreams are [a, shared]; a's upstream is [shared].
        # This forces 'shared' onto the DFS stack twice; the guard on
        # line 68 (``continue``) discards the redundant pop.
        shared = double(x=5)
        a = negate(x=shared)
        b = add(x=a, y=shared)

        nodes = build_graph_multi([b])

        assert len(nodes) == 3
        assert shared.id in nodes
        assert a.id in nodes
        assert b.id in nodes


class TestGetTypedParams:
    """Tests for Workflow.get_typed_params() and Workflow.has_typed_params()."""

    def test_typed_params_resolved(self):
        """Parameters with standard type annotations are returned as real types."""

        @workflow
        def wf(x: int, y: str, z: float) -> None:  # pragma: no cover
            pass

        assert wf.get_typed_params() == {"x": int, "y": str, "z": float}

    def test_untyped_params_return_none(self):
        """Parameters without annotations map to None."""

        @workflow
        def wf(x, y):  # noqa: ANN001  # pragma: no cover
            pass

        assert wf.get_typed_params() == {"x": None, "y": None}

    def test_get_type_hints_failure_falls_back_to_none(self):
        """
        When get_type_hints raises (e.g. an unresolvable forward reference) every
        parameter falls back to None so the CLI can still run with string values.
        """

        def _bad_func(x: NonExistentType) -> None:  # type: ignore[name-defined]  # pragma: no cover  # noqa: F821
            pass

        # Directly construct a Workflow with the broken function so we can call
        # get_typed_params() without going through the decorator magic.
        wf = Workflow(func=_bad_func, name="bad", description="")
        result = wf.get_typed_params()
        assert result == {"x": None}

    def test_has_typed_params_true(self):
        @workflow
        def wf(x: int) -> None:  # pragma: no cover
            pass

        assert wf.has_typed_params() is True

    def test_has_typed_params_false(self):
        @workflow
        def wf(x) -> None:  # noqa: ANN001  # pragma: no cover
            pass

        assert wf.has_typed_params() is False

    def test_has_typed_params_mixed_returns_false(self):
        @workflow
        def wf(x: int, y) -> None:  # noqa: ANN001  # pragma: no cover
            pass

        assert wf.has_typed_params() is False
