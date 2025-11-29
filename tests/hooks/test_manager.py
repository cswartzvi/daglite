"""Tests for per-execution hooks and manager functionality."""

import pytest

from daglite import evaluate
from daglite import hooks
from daglite import task
from daglite.engine import evaluate_async


class TestPerExecutionHooks:
    """Test per-execution hooks functionality."""

    def test_per_execution_hooks_basic(self) -> None:
        """Per-execution hooks work correctly."""

        class Counter:
            def __init__(self):
                self.count = 0

            @hooks.hook_impl
            def before_node_execute(self, node_id, node, backend, inputs):
                self.count += 1

        counter = Counter()

        @task
        def add(x: int, y: int) -> int:
            return x + y

        result = evaluate(add.bind(x=2, y=3), hooks=[counter])
        assert result == 5
        assert counter.count == 1

    def test_per_execution_hooks_no_global_pollution(self) -> None:
        """Per-execution hooks don't affect global hook manager."""

        class Counter:
            def __init__(self):
                self.count = 0

            @hooks.hook_impl
            def before_node_execute(self, node_id, node, backend, inputs):
                self.count += 1

        @task
        def add(x: int, y: int) -> int:
            return x + y

        # First execution with counter1
        counter1 = Counter()
        result1 = evaluate(add.bind(x=2, y=3), hooks=[counter1])
        assert result1 == 5
        assert counter1.count == 1

        # Second execution with counter2
        counter2 = Counter()
        result2 = evaluate(add.bind(x=5, y=7), hooks=[counter2])
        assert result2 == 12
        assert counter1.count == 1  # Unchanged
        assert counter2.count == 1

    def test_per_execution_hooks_combine_with_global(self) -> None:
        """Per-execution hooks combine with globally registered hooks."""

        class GlobalCounter:
            def __init__(self):
                self.count = 0

            @hooks.hook_impl
            def before_node_execute(self, node_id, node, backend, inputs):
                self.count += 1

        class LocalCounter:
            def __init__(self):
                self.count = 0

            @hooks.hook_impl
            def before_node_execute(self, node_id, node, backend, inputs):
                self.count += 1

        global_counter = GlobalCounter()
        local_counter = LocalCounter()

        # Register global hook
        hooks.register_hooks(global_counter)

        @task
        def add(x: int, y: int) -> int:
            return x + y

        # Execute with local hook - both should fire
        result = evaluate(add.bind(x=2, y=3), hooks=[local_counter])
        assert result == 5
        assert global_counter.count == 1
        assert local_counter.count == 1

        # Execute without local hook - only global should fire
        result2 = evaluate(add.bind(x=5, y=7))
        assert result2 == 12
        assert global_counter.count == 2  # Incremented
        assert local_counter.count == 1  # Unchanged

        # Cleanup
        hook_manager = hooks.get_hook_manager()
        hook_manager.unregister(global_counter)

    def test_per_execution_hooks_multiple(self) -> None:
        """Multiple per-execution hooks can be registered."""

        class Counter1:
            def __init__(self):
                self.count = 0

            @hooks.hook_impl
            def before_node_execute(self, node_id, node, backend, inputs):
                self.count += 1

        class Counter2:
            def __init__(self):
                self.count = 0

            @hooks.hook_impl
            def before_node_execute(self, node_id, node, backend, inputs):
                self.count += 1

        counter1 = Counter1()
        counter2 = Counter2()

        @task
        def add(x: int, y: int) -> int:
            return x + y

        result = evaluate(add.bind(x=2, y=3), hooks=[counter1, counter2])
        assert result == 5
        assert counter1.count == 1
        assert counter2.count == 1

    def test_per_execution_hooks_async(self) -> None:
        """Per-execution hooks work with evaluate_async."""
        import asyncio

        class Counter:
            def __init__(self):
                self.count = 0

            @hooks.hook_impl
            def before_node_execute(self, node_id, node, backend, inputs):
                self.count += 1

        counter = Counter()

        @task
        def add(x: int, y: int) -> int:
            return x + y

        async def run():
            return await evaluate_async(add.bind(x=2, y=3), hooks=[counter])

        result = asyncio.run(run())
        assert result == 5
        assert counter.count == 1

    def test_per_execution_hooks_error_propagates(self) -> None:
        """Errors in user code still propagate correctly with per-execution hooks."""

        class Counter:
            def __init__(self):
                self.count = 0

            @hooks.hook_impl
            def before_node_execute(self, node_id, node, backend, inputs):
                self.count += 1

        counter = Counter()

        @task
        def failing_task() -> int:
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            evaluate(failing_task.bind(), hooks=[counter])

        # Hook should have been called before the error
        assert counter.count == 1

    def test_per_execution_hooks_validates_instances(self) -> None:
        """Per-execution hooks must be instances, not classes."""

        class MyHook:
            @hooks.hook_impl
            def before_node_execute(self, node_id, node, backend, inputs):  # pragma: no cover
                pass

        @task
        def add(x: int, y: int) -> int:  # pragma: no cover
            return x + y

        # Passing class instead of instance should raise TypeError
        with pytest.raises(TypeError, match="daglite expects hooks to be registered as instances"):
            evaluate(add.bind(x=2, y=3), hooks=[MyHook])  # Missing ()


class TestRegisterHooksEntryPoints:
    """Test entry point loading functionality."""

    def test_register_hooks_entry_points_exists(self) -> None:
        """register_hooks_entry_points function exists and is callable."""
        # This just verifies the function exists and can be called
        # without error (even if no entry points are defined)
        hooks.register_hooks_entry_points()

    def test_register_hooks_entry_points_no_crash_on_missing(self) -> None:
        """Entry point loading doesn't crash when no plugins are installed."""
        # Should gracefully handle no entry points being defined
        hooks.register_hooks_entry_points()

        # Should still be able to evaluate normally
        @task
        def add(x: int, y: int) -> int:
            return x + y

        result = evaluate(add.bind(x=2, y=3))
        assert result == 5


class TestHookManagerFunctions:
    """Test hook manager utility functions."""

    def test_initialize_hooks_idempotent(self) -> None:
        """initialize_hooks can be called multiple times safely."""
        # Already called on import, but should be safe to call again
        hooks.initialize_hooks()
        hooks.initialize_hooks()

        # Should still work
        @task
        def add(x: int, y: int) -> int:
            return x + y

        result = evaluate(add.bind(x=2, y=3))
        assert result == 5

    def test_get_hook_manager_returns_same_instance(self) -> None:
        """get_hook_manager returns the same global instance."""
        manager1 = hooks.get_hook_manager()
        manager2 = hooks.get_hook_manager()
        assert manager1 is manager2

    def test_register_hooks_validation(self) -> None:
        """register_hooks validates that hooks are instances."""

        class MyHook:
            @hooks.hook_impl
            def before_node_execute(self, node_id, node, backend, inputs):  # pragma: no cover
                pass

        # Should raise TypeError when passing class instead of instance
        with pytest.raises(TypeError, match="daglite expects hooks to be registered as instances"):
            hooks.register_hooks(MyHook)  # Missing ()

    def test_register_hooks_prevents_duplicates(self) -> None:
        """register_hooks doesn't register the same instance twice."""

        class Counter:
            def __init__(self):
                self.count = 0

            @hooks.hook_impl
            def before_node_execute(self, node_id, node, backend, inputs):
                self.count += 1

        counter = Counter()

        # Register twice
        hooks.register_hooks(counter)
        hooks.register_hooks(counter)  # Should be no-op

        @task
        def add(x: int, y: int) -> int:
            return x + y

        result = evaluate(add.bind(x=2, y=3))
        assert result == 5
        # Should only increment once (not registered twice)
        assert counter.count == 1

        # Cleanup
        hook_manager = hooks.get_hook_manager()
        hook_manager.unregister(counter)
