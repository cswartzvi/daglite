"""Integration tests for workflow evaluation with the @workflow decorator."""

from __future__ import annotations

import asyncio

import pytest

from daglite import task, workflow
from daglite.workflow_result import WorkflowResult


class TestWorkflowEvaluation:
    def test_single_future_workflow(self):
        @task
        def add(x: int, y: int) -> int:
            return x + y

        @workflow
        def wf(x: int, y: int):
            return add(x=x, y=y)

        result = wf.run(x=2, y=3)
        assert isinstance(result, WorkflowResult)
        assert result["add"] == 5

    def test_multi_future_disjoint_branches(self):
        @task
        def add(x: int, y: int) -> int:
            return x + y

        @task
        def mul(x: int, y: int) -> int:
            return x * y

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
        assert call_count == 1

    def test_list_return(self):
        @task
        def add(x: int, y: int) -> int:
            return x + y

        @task
        def mul(x: int, y: int) -> int:
            return x * y

        @workflow
        def wf(x: int, y: int):
            return [add(x=x, y=y), mul(x=x, y=y)]

        result = wf.run(x=2, y=5)
        assert result["add"] == 7
        assert result["mul"] == 10

    def test_invalid_return_type_raises(self):
        @workflow
        def wf():
            return 42

        with pytest.raises(TypeError, match="@workflow function must return"):
            wf.run()

    def test_run_async(self):
        @task
        def add(x: int, y: int) -> int:
            return x + y

        @task
        def mul(x: int, y: int) -> int:
            return x * y

        @workflow
        def wf(x: int, y: int):
            return add(x=x, y=y), mul(x=x, y=y)

        result = asyncio.run(wf.run_async(x=4, y=5))
        assert isinstance(result, WorkflowResult)
        assert result["add"] == 9
        assert result["mul"] == 20
