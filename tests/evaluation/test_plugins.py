"""
Integration tests for the plugin system with evaluate/evaluate_async.

These tests verify that plugins work correctly during actual DAG execution.
They use evaluate() and evaluate_async() to test plugin behavior with real
task execution.

For unit tests of the plugin system (without evaluation), see tests/plugins/.
"""

import asyncio

import pytest

from daglite import evaluate
from daglite import task
from daglite.engine import evaluate_async
from daglite.graph.base import GraphMetadata
from daglite.plugins.hooks.markers import hook_impl
from daglite.plugins.manager import _get_global_plugin_manager
from daglite.plugins.manager import register_plugins
from tests.examples.plugins import CounterPlugin
from tests.examples.plugins import ParameterCapturePlugin


class TestPerExecutionHooks:
    """
    Test per-execution hooks functionality.

    Per-execution hooks are plugins passed to evaluate()/evaluate_async()
    via the plugins parameter. They should:
    1. Only affect that specific execution
    2. Not pollute the global plugin manager
    3. Work alongside globally registered plugins
    """

    def test_per_execution_hooks_basic(self) -> None:
        """Per-execution hooks are called during evaluation."""
        counter = CounterPlugin()

        @task
        def add(x: int, y: int) -> int:
            return x + y

        result = evaluate(add(x=2, y=3), plugins=[counter])

        assert result == 5
        assert counter.before_node_count == 1
        assert counter.after_node_count == 1
        assert counter.before_graph_count == 1
        assert counter.after_graph_count == 1

    def test_per_execution_hooks_no_global_pollution(self) -> None:
        """Per-execution hooks don't affect other evaluations."""
        counter1 = CounterPlugin()
        counter2 = CounterPlugin()

        @task
        def add(x: int, y: int) -> int:
            return x + y

        # First execution with counter1
        result1 = evaluate(add(x=2, y=3), plugins=[counter1])
        assert result1 == 5
        assert counter1.before_node_count == 1

        # Second execution with counter2 (counter1 should be unchanged)
        result2 = evaluate(add(x=5, y=7), plugins=[counter2])
        assert result2 == 12
        assert counter1.before_node_count == 1  # Still 1
        assert counter2.before_node_count == 1

    def test_per_execution_hooks_combine_with_global(self) -> None:
        """Per-execution hooks work alongside globally registered plugins."""
        global_counter = CounterPlugin()
        local_counter = CounterPlugin()

        # Register global plugin
        register_plugins(global_counter)

        @task
        def add(x: int, y: int) -> int:
            return x + y

        # Execute with local plugin - both should be called
        result = evaluate(add(x=2, y=3), plugins=[local_counter])
        assert result == 5
        assert global_counter.before_node_count == 1
        assert local_counter.before_node_count == 1

        # Execute without local plugin - only global should be called
        result2 = evaluate(add(x=5, y=7))
        assert result2 == 12
        assert global_counter.before_node_count == 2  # Incremented
        assert local_counter.before_node_count == 1  # Unchanged

        # Cleanup
        hook_manager = _get_global_plugin_manager()
        hook_manager.unregister(global_counter)

    def test_per_execution_hooks_multiple(self) -> None:
        """Multiple per-execution plugins can be registered simultaneously."""
        counter1 = CounterPlugin()
        counter2 = CounterPlugin()

        @task
        def add(x: int, y: int) -> int:
            return x + y

        result = evaluate(add(x=2, y=3), plugins=[counter1, counter2])

        assert result == 5
        assert counter1.before_node_count == 1
        assert counter2.before_node_count == 1

    def test_per_execution_hooks_async(self) -> None:
        """Per-execution hooks work with evaluate_async."""
        counter = CounterPlugin()

        @task
        def add(x: int, y: int) -> int:
            return x + y

        async def run():
            return await evaluate_async(add(x=2, y=3), plugins=[counter])

        result = asyncio.run(run())

        assert result == 5
        assert counter.before_node_count == 1
        assert counter.after_node_count == 1

    def test_per_execution_hooks_error_propagates(self) -> None:
        """Errors in tasks propagate correctly with per-execution hooks."""
        counter = CounterPlugin()

        @task
        def failing_task() -> int:
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            evaluate(failing_task(), plugins=[counter])

        # Hook should have been called before the error
        assert counter.before_node_count == 1
        # Error hook should have been called
        assert counter.on_error_count == 1

    def test_per_execution_hooks_validates_instances(self) -> None:
        """Plugins must be instances, not classes."""

        class MyHook:
            @hook_impl
            def before_node_execute(self, metadata, inputs):  # pragma: no cover
                pass

        @task
        def add(x: int, y: int) -> int:  # pragma: no cover
            return x + y

        # Passing class instead of instance should raise TypeError
        with pytest.raises(
            TypeError, match="daglite expects plugins to be registered as instances"
        ):
            evaluate(add(x=2, y=3), plugins=[MyHook])  # Missing ()


class TestHookParameters:
    """Test that hooks receive correct parameters during execution."""

    def test_hook_receives_correct_parameters(self) -> None:
        """Hooks receive correct parameters with proper types."""
        capture = ParameterCapturePlugin()

        @task
        def add(x: int, y: int) -> int:
            return x + y

        result = evaluate(add(x=2, y=3), plugins=[capture])
        assert result == 5

        # Verify graph-level hooks
        assert len(capture.graph_starts) == 1
        assert capture.graph_starts[0]["node_count"] == 1

        assert len(capture.graph_ends) == 1
        assert capture.graph_ends[0]["result"] == 5
        assert capture.graph_ends[0]["duration"] >= 0

        # Verify node-level hooks
        assert len(capture.node_executions) == 1
        assert isinstance(capture.node_executions[0]["key"], str)
        assert isinstance(capture.node_executions[0]["metadata"], GraphMetadata)
        assert capture.node_executions[0]["inputs"] == {"x": 2, "y": 3}

        assert len(capture.node_results) == 1
        assert capture.node_results[0]["result"] == 5
        assert capture.node_results[0]["duration"] >= 0

    def test_hooks_with_map_tasks(self) -> None:
        """Hooks correctly handle map tasks (product/zip operations)."""
        capture = ParameterCapturePlugin()

        @task
        def process(x: int) -> int:
            return x * 2

        @task
        def sum_all(vals: list[int]) -> int:
            return sum(vals)

        result = evaluate(process.product(x=[1, 2, 3]).join(sum_all), plugins=[capture])
        assert result == 12

        # Map task calls hook for each iteration (3) + join task (1) = 4 total
        assert len(capture.node_executions) == 4

        # First three are map iterations
        assert capture.node_executions[0]["inputs"] == {"x": 1}
        assert capture.node_executions[1]["inputs"] == {"x": 2}
        assert capture.node_executions[2]["inputs"] == {"x": 3}

        # Last is the join
        assert capture.node_executions[3]["inputs"] == {"vals": [2, 4, 6]}

        # Verify results
        assert len(capture.node_results) == 4
        assert capture.node_results[0]["result"] == 2
        assert capture.node_results[1]["result"] == 4
        assert capture.node_results[2]["result"] == 6
        assert capture.node_results[3]["result"] == 12


class TestGlobalPluginRegistration:
    """Test global plugin registration with evaluate."""

    def test_register_custom_hooks(self) -> None:
        """Globally registered plugins are called during evaluation."""

        class CustomHook:
            def __init__(self):
                self.called = False

            @hook_impl
            def before_graph_execute(self, root_id, node_count):
                self.called = True

        custom_hook = CustomHook()
        register_plugins(custom_hook)

        @task
        def simple() -> int:
            return 42

        result = evaluate(simple())

        assert result == 42
        assert custom_hook.called

        # Cleanup
        hook_manager = _get_global_plugin_manager()
        hook_manager.unregister(custom_hook)
