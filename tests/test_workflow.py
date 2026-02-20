"""
Unit and integration tests for the @workflow decorator and WorkflowResult.
"""

from __future__ import annotations

import asyncio
from uuid import UUID

import pytest

from daglite import task, workflow
from daglite.exceptions import AmbiguousResultError
from daglite.workflow_result import WorkflowResult
from daglite.workflows import Workflow


# ---------------------------------------------------------------------------
# Shared task fixtures
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Tests: @workflow decorator
# ---------------------------------------------------------------------------


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
        # Calling the workflow returns the underlying future, not a WorkflowResult
        assert not isinstance(future, WorkflowResult)

    def test_cannot_decorate_class(self):
        with pytest.raises(TypeError, match="callable functions"):

            @workflow
            class NotAFunction:  # pragma: no cover
                pass


# ---------------------------------------------------------------------------
# Tests: _collect_futures
# ---------------------------------------------------------------------------


class TestCollectFutures:
    def setup_method(self):
        @workflow
        def wf():  # pragma: no cover
            pass

        self.wf = wf

    def test_single_future(self):
        future = add(x=1, y=2)
        collected = self.wf._collect_futures(future)
        assert collected == [future]

    def test_tuple_of_futures(self):
        f1 = add(x=1, y=2)
        f2 = mul(x=3, y=4)
        collected = self.wf._collect_futures((f1, f2))
        assert collected == [f1, f2]

    def test_list_of_futures(self):
        f1 = add(x=1, y=2)
        f2 = mul(x=3, y=4)
        collected = self.wf._collect_futures([f1, f2])
        assert collected == [f1, f2]

    def test_invalid_return_raises_type_error(self):
        with pytest.raises(TypeError, match="@workflow function must return"):
            self.wf._collect_futures(42)

    def test_invalid_string_raises_type_error(self):
        with pytest.raises(TypeError, match="str"):
            self.wf._collect_futures("not_a_future")


# ---------------------------------------------------------------------------
# Tests: WorkflowResult
# ---------------------------------------------------------------------------


class TestWorkflowResult:
    def _make_result(self, data: dict[str, int]) -> WorkflowResult:
        """Helper to build a WorkflowResult from {name: value} pairs."""
        from uuid import uuid4

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
        from uuid import uuid4

        uid = uuid4()
        result = WorkflowResult._build({uid: 99}, {uid: "task"})
        assert result[uid] == 99

    def test_getitem_missing_name_raises_key_error(self):
        result = self._make_result({"add": 5})
        with pytest.raises(KeyError, match="no_such"):
            _ = result["no_such"]

    def test_getitem_missing_uuid_raises_key_error(self):
        from uuid import uuid4

        result = self._make_result({"add": 5})
        with pytest.raises(KeyError):
            _ = result[uuid4()]

    def test_ambiguous_name_raises_error(self):
        from uuid import uuid4

        uid1, uid2 = uuid4(), uuid4()
        result = WorkflowResult._build(
            {uid1: 10, uid2: 20},
            {uid1: "task", uid2: "task"},
        )
        with pytest.raises(AmbiguousResultError, match="task"):
            _ = result["task"]

    def test_uuid_lookup_works_on_name_collision(self):
        from uuid import uuid4

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
        from uuid import uuid4

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


# ---------------------------------------------------------------------------
# Integration tests: workflow.run()
# ---------------------------------------------------------------------------


class TestWorkflowRun:
    def test_single_future_workflow(self):
        @workflow
        def wf(x: int, y: int):
            return add(x=x, y=y)

        result = wf.run(x=2, y=3)
        assert isinstance(result, WorkflowResult)
        assert result["add"] == 5

    def test_multi_future_disjoint_branches(self):
        @workflow
        def wf(x: int, y: int):
            return add(x=x, y=y), mul(x=x, y=y)

        result = wf.run(x=3, y=4)
        assert result["add"] == 7
        assert result["mul"] == 12

    def test_shared_upstream_diamond(self):
        """Common upstream node should run exactly once."""
        call_count = 0

        @task
        def base(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 10

        @task
        def branch_a(val: int) -> int:
            return val + 1

        @task
        def branch_b(val: int) -> int:
            return val + 2

        @workflow
        def wf(x: int):
            shared = base(x=x)
            return branch_a(val=shared), branch_b(val=shared)

        result = wf.run(x=5)
        assert result["branch_a"] == 51
        assert result["branch_b"] == 52
        assert call_count == 1  # shared node ran only once

    def test_list_return(self):
        @workflow
        def wf(x: int, y: int):
            return [add(x=x, y=y), mul(x=x, y=y)]

        result = wf.run(x=2, y=5)
        assert result["add"] == 7
        assert result["mul"] == 10

    def test_invalid_return_type_raises(self):
        @workflow
        def wf():
            return 42  # not a future

        with pytest.raises(TypeError, match="@workflow function must return"):
            wf.run()

    def test_workflow_run_async(self):
        @workflow
        def wf(x: int, y: int):
            return add(x=x, y=y), mul(x=x, y=y)

        result = asyncio.run(wf.run_async(x=4, y=5))
        assert isinstance(result, WorkflowResult)
        assert result["add"] == 9
        assert result["mul"] == 20


# ---------------------------------------------------------------------------
# Integration tests: build_graph_multi (via workflow)
# ---------------------------------------------------------------------------


class TestBuildGraphMulti:
    def test_shared_node_included_once(self):
        """build_graph_multi should not duplicate shared ancestors."""
        from daglite.graph.builder import build_graph_multi

        shared = double(x=10)
        sink_a = negate(x=shared)
        sink_b = add(x=shared, y=1)

        nodes = build_graph_multi([sink_a, sink_b])

        # shared, sink_a, sink_b — 3 distinct nodes
        assert len(nodes) == 3
        assert shared.id in nodes
        assert sink_a.id in nodes
        assert sink_b.id in nodes
