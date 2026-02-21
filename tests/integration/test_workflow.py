"""Integration tests for workflow evaluation with the @workflow decorator."""

from __future__ import annotations

import asyncio

import pytest

from daglite import task
from daglite import workflow
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

    def test_custom_task_name_used_as_result_key(self):
        @task(name="my_sum")
        def add(x: int, y: int) -> int:
            return x + y

        @task(name="my_product")
        def mul(x: int, y: int) -> int:
            return x * y

        @workflow
        def wf(x: int, y: int):
            return add(x=x, y=y), mul(x=x, y=y)

        result = wf.run(x=3, y=4)
        assert result["my_sum"] == 7
        assert result["my_product"] == 12
        with pytest.raises(KeyError):
            _ = result["add"]
        with pytest.raises(KeyError):
            _ = result["mul"]

    def test_alias_disambiguates_same_named_sinks(self):
        @task
        def add(x: int, y: int) -> int:
            return x + y

        @workflow
        def wf(x: int):
            return add(x=x, y=1).alias("small"), add(x=x, y=100).alias("large")

        result = wf.run(x=5)
        assert result["small"] == 6
        assert result["large"] == 105

    def test_alias_overrides_task_name(self):
        @task(name="my_task")
        def compute(x: int) -> int:
            return x * 2

        @workflow
        def wf(x: int):
            return compute(x=x).alias("result")

        result = wf.run(x=7)
        assert result["result"] == 14
        with pytest.raises(KeyError):
            _ = result["my_task"]

    def test_all_collects_same_named_sinks(self):
        @task
        def add(x: int, y: int) -> int:
            return x + y

        @workflow
        def wf(x: int):
            return add(x=x, y=1), add(x=x, y=2)

        result = wf.run(x=10)
        values = result.all("add")
        assert sorted(values) == [11, 12]

    def test_single_on_alias(self):
        @task
        def add(x: int, y: int) -> int:
            return x + y

        @workflow
        def wf(x: int):
            return add(x=x, y=1).alias("my_add")

        result = wf.run(x=3)
        assert result.single("my_add") == 4

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

    def test_workflow_with_optimization_disabled(self):
        """Running with optimization disabled must still produce correct results."""
        from daglite.settings import DagliteSettings
        from daglite.settings import get_global_settings
        from daglite.settings import set_global_settings

        original = get_global_settings()
        try:
            set_global_settings(DagliteSettings(enable_graph_optimization=False))

            @task
            def add(x: int, y: int) -> int:
                return x + y

            @workflow
            def wf(x: int, y: int):
                return add(x=x, y=y)

            result = wf.run(x=2, y=3)
            assert result["add"] == 5
        finally:
            set_global_settings(original)

    def test_failing_task_propagates_error(self):
        """An exception raised inside a task must propagate out of wf.run()."""

        @task
        def bad() -> int:
            raise ValueError("intentional failure")

        @workflow
        def wf():
            return bad()

        with pytest.raises(ValueError, match="intentional failure"):
            wf.run()

    def test_evaluate_workflow_in_async_context_raises(self):
        """evaluate_workflow() must raise when called from inside a running event loop."""
        from daglite.engine import evaluate_workflow

        @task
        def add(x: int, y: int) -> int:
            return x + y

        async def call_from_async() -> None:
            evaluate_workflow([add(x=1, y=2)])

        with pytest.raises(RuntimeError, match="async context"):
            asyncio.run(call_from_async())

    def test_evaluate_workflow_async_within_task_raises(self, mocker):
        """evaluate_workflow_async() must raise when called from within a task."""
        from daglite.engine import evaluate_workflow_async

        mocker.patch("daglite.backends.context.get_current_task", return_value=object())

        @task
        def add(x: int, y: int) -> int:
            return x + y

        with pytest.raises(RuntimeError, match="within another task"):
            asyncio.run(evaluate_workflow_async([add(x=1, y=2)]))
