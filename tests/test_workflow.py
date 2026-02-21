"""
Unit tests for the @workflow decorator, Workflow, and WorkflowResult.

Tests in this file should NOT focus on evaluation. Evaluation tests are in tests/integration/.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from daglite import task
from daglite import workflow
from daglite.exceptions import AmbiguousResultError
from daglite.graph.builder import build_graph_multi
from daglite.workflow_result import WorkflowResult
from daglite.workflows import Workflow


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


class TestWorkflowResult:
    def _make_result(self, data: dict[str, int]) -> WorkflowResult:
        results = {}
        name_for = {}
        for name, value in data.items():
            uid = uuid4()
            results[uid] = value
            name_for[uid] = name
        return WorkflowResult._build(results, name_for)

    def test_getitem_by_name(self):
        result = self._make_result({"add": 5, "mul": 6})
        assert result["add"] == 5
        assert result["mul"] == 6

    def test_getitem_by_uuid(self):
        uid = uuid4()
        result = WorkflowResult._build({uid: 99}, {uid: "task"})
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
        result = WorkflowResult._build(
            {uid1: 10, uid2: 20},
            {uid1: "task", uid2: "task"},
        )
        with pytest.raises(AmbiguousResultError, match="task"):
            _ = result["task"]

    def test_uuid_lookup_works_on_name_collision(self):
        uid1, uid2 = uuid4(), uuid4()
        result = WorkflowResult._build(
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

    def test_items(self):
        result = self._make_result({"a": 1, "b": 2})
        assert dict(result.items()) == {"a": 1, "b": 2}

    def test_items_raises_on_collision(self):
        uid1, uid2 = uuid4(), uuid4()
        result = WorkflowResult._build(
            {uid1: 10, uid2: 20},
            {uid1: "task", uid2: "task"},
        )
        with pytest.raises(AmbiguousResultError):
            list(result.items())

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
        return WorkflowResult._build(results, name_for)

    def test_single_returns_value(self):
        result = self._make_result({"add": 5})
        assert result.single("add") == 5

    def test_single_raises_on_missing(self):
        result = self._make_result({"add": 5})
        with pytest.raises(KeyError):
            result.single("missing")

    def test_single_raises_on_ambiguous(self):
        uid1, uid2 = uuid4(), uuid4()
        result = WorkflowResult._build(
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
        result = WorkflowResult._build(
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
