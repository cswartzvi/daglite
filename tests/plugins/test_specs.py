"""Tests for the pluggy hooks system."""

from daglite import evaluate
from daglite import task
from daglite.plugins import hook_impl
from daglite.plugins.manager import _get_global_plugin_manager
from daglite.plugins.manager import register_hooks


class TestHooksBasic:
    """Test basic hook functionality."""

    def test_hooks_initialize_on_import(self) -> None:
        """Hook manager is initialized when daglite is imported."""
        hook_manager = _get_global_plugin_manager()
        assert hook_manager is not None
        assert hook_manager.project_name == "daglite"

    def test_register_custom_hooks(self) -> None:
        """Can register custom hook implementations."""

        class CustomHook:
            def __init__(self):
                self.called = False

            @hook_impl
            def before_graph_execute(self, root_id, node_count, is_async):
                self.called = True

        custom_hook = CustomHook()
        register_hooks(custom_hook)

        @task
        def simple() -> int:
            return 42

        result = evaluate(simple.bind())
        assert result == 42
        assert custom_hook.called

        # Clean up
        hook_manager = _get_global_plugin_manager()
        hook_manager.unregister(custom_hook)

    def test_hook_receives_correct_parameters(self) -> None:
        """Hooks receive correct parameters during execution."""
        from uuid import UUID

        from daglite.backends.base import Backend
        from daglite.graph.base import GraphNode

        class ParameterCapture:
            def __init__(self):
                self.node_executions = []
                self.graph_start = None
                self.graph_end = None

            @hook_impl
            def before_graph_execute(self, root_id, node_count, is_async):
                self.graph_start = {
                    "root_id": root_id,
                    "node_count": node_count,
                    "is_async": is_async,
                }

            @hook_impl
            def before_node_execute(self, node_id, node, backend, inputs):
                assert isinstance(node_id, UUID)
                assert isinstance(node, GraphNode)
                assert isinstance(backend, Backend)
                assert isinstance(inputs, dict)
                self.node_executions.append({"node_id": node_id, "inputs": inputs})

            @hook_impl
            def after_graph_execute(self, root_id, result, duration, is_async):
                self.graph_end = {
                    "root_id": root_id,
                    "result": result,
                    "duration": duration,
                    "is_async": is_async,
                }

        capture = ParameterCapture()
        register_hooks(capture)

        @task
        def add(x: int, y: int) -> int:
            return x + y

        result = evaluate(add.bind(x=2, y=3))
        assert result == 5

        # Verify captured data
        assert capture.graph_start is not None
        assert capture.graph_start["node_count"] == 1
        assert capture.graph_start["is_async"] is False

        assert len(capture.node_executions) == 1
        assert capture.node_executions[0]["inputs"] == {"x": 2, "y": 3}

        assert capture.graph_end is not None
        assert capture.graph_end["result"] == 5
        assert capture.graph_end["duration"] >= 0
        assert capture.graph_end["is_async"] is False

        # Clean up
        hook_manager = _get_global_plugin_manager()
        hook_manager.unregister(capture)


class TestHooksMapTasks:
    """Test hooks with map tasks (product/zip operations)."""

    def test_hooks_with_sequence_inputs(self) -> None:
        """Hooks correctly handle sequence inputs in map tasks."""

        class InputCapture:
            def __init__(self):
                self.inputs = []

            @hook_impl
            def before_node_execute(self, node_id, node, backend, inputs):
                self.inputs.append(inputs)

        capture = InputCapture()
        register_hooks(capture)

        @task
        def process(x: int) -> int:
            return x * 2

        @task
        def sum_all(vals: list[int]) -> int:
            return sum(vals)

        result = evaluate(process.product(x=[1, 2, 3]).join(sum_all))
        assert result == 12

        # First node should have sequence input
        assert len(capture.inputs) == 2
        assert "x" in capture.inputs[0]
        assert capture.inputs[0]["x"] == [1, 2, 3]

        # Clean up
        hook_manager = _get_global_plugin_manager()
        hook_manager.unregister(capture)
