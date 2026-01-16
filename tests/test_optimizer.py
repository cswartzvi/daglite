"""
Unit tests for graph optimization and composite nodes.
"""

from daglite import evaluate
from daglite import task
from daglite.graph.builder import build_graph
from daglite.graph.nodes import CompositeMapTaskNode
from daglite.graph.nodes import CompositeTaskNode
from daglite.graph.nodes import TaskNode
from daglite.graph.optimizer import optimize_graph
from daglite.settings import DagliteSettings
from daglite.settings import set_global_settings


class TestGraphOptimizer:
    """Test the graph optimizer that creates composite nodes."""

    def test_optimize_linear_chain(self) -> None:
        """Optimizer groups a simple linear chain into a composite node."""

        @task
        def add_one(x: int) -> int:
            return x + 1

        @task
        def multiply_two(x: int) -> int:
            return x * 2

        @task
        def subtract_three(x: int) -> int:
            return x - 3

        # Create a linear chain: a -> b -> c
        a = add_one(x=1)
        b = multiply_two(x=a)
        c = subtract_three(x=b)

        # Build graph
        graph = build_graph(c)

        # Should have 3 nodes before optimization
        assert len(graph) == 3

        # Optimize
        optimized = optimize_graph(graph, c.id, enable_optimization=True)

        # Should have 1 composite node after optimization
        assert len(optimized) == 1

        # The single node should be a CompositeTaskNode
        composite = list(optimized.values())[0]
        assert isinstance(composite, CompositeTaskNode)
        assert len(composite.nodes) == 3

    def test_no_optimization_when_disabled(self) -> None:
        """Optimizer returns original graph when disabled."""

        @task
        def add_one(x: int) -> int:
            return x + 1

        @task
        def multiply_two(x: int) -> int:
            return x * 2

        # Create a chain
        a = add_one(x=1)
        b = multiply_two(x=a)

        # Build graph
        graph = build_graph(b)

        # Optimize with disabled flag
        optimized = optimize_graph(graph, b.id, enable_optimization=False)

        # Should be unchanged
        assert len(optimized) == len(graph)
        assert optimized == graph

    def test_no_optimization_for_branching(self) -> None:
        """Optimizer detects and preserves branching when both branches are in the graph."""

        @task
        def shared(x: int) -> int:
            return x + 1

        @task
        def branch_a(x: int) -> int:
            return x * 2

        @task
        def branch_b(x: int) -> int:
            return x * 3

        @task
        def combine(a: int, b: int) -> int:
            return a + b

        # Create branching: shared -> [branch_a, branch_b] -> combine
        s = shared(x=1)
        a = branch_a(x=s)
        b = branch_b(x=s)
        c = combine(a=a, b=b)

        # Build graph with all branches
        graph = build_graph(c)

        # Optimize
        optimized = optimize_graph(graph, c.id, enable_optimization=True)

        # Should not create composites because shared has multiple successors
        # (both branches are visible in the graph)
        assert (
            len(optimized) >= 3
        )  # At minimum: shared, branch_a, branch_b (combine might be grouped)

    def test_different_backends_prevent_grouping(self) -> None:
        """Optimizer does not group nodes with different backends."""

        @task(backend_name="sequential")
        def task_seq(x: int) -> int:
            return x + 1

        @task(backend_name="threading")
        def task_thread(x: int) -> int:
            return x * 2

        # Create a chain with different backends
        a = task_seq(x=1)
        b = task_thread(x=a)

        # Build graph
        graph = build_graph(b)

        # Optimize
        optimized = optimize_graph(graph, b.id, enable_optimization=True)

        # Should not create composite because backends differ
        assert len(optimized) == 2
        assert all(isinstance(node, TaskNode) for node in optimized.values())

    def test_map_chains_are_optimized(self) -> None:
        """Optimizer groups chained map operations created with .then()."""

        @task
        def square(x: int) -> int:
            return x * x

        @task
        def add_ten(x: int) -> int:
            return x + 10

        # Create a map chain: MapTaskNode -> MapTaskNode (via .then())
        m1 = square.product(x=[1, 2, 3])
        m2 = m1.then(add_ten)

        # Build graph
        graph = build_graph(m2)

        # Optimize
        optimized = optimize_graph(graph, m2.id, enable_optimization=True)

        # Should create a composite map node
        assert len(optimized) == 1
        composite = list(optimized.values())[0]
        assert isinstance(composite, CompositeMapTaskNode)
        assert len(composite.nodes) == 2

    def test_single_node_no_optimization(self) -> None:
        """Optimizer leaves single nodes unchanged."""

        @task
        def solo(x: int) -> int:
            return x + 1

        f = solo(x=1)
        graph = build_graph(f)

        optimized = optimize_graph(graph, f.id, enable_optimization=True)

        assert len(optimized) == 1
        assert list(optimized.values())[0] == list(graph.values())[0]


class TestCompositeNodeExecution:
    """Test execution of composite nodes."""

    def test_composite_task_node_execution(self) -> None:
        """CompositeTaskNode executes all child nodes in sequence."""

        @task
        def add_one(x: int) -> int:
            return x + 1

        @task
        def multiply_two(x: int) -> int:
            return x * 2

        @task
        def subtract_three(x: int) -> int:
            return x - 3

        # Create and evaluate a chain
        a = add_one(x=1)
        b = multiply_two(x=a)
        c = subtract_three(x=b)

        result = evaluate(c)

        # (1 + 1) * 2 - 3 = 4 - 3 = 1
        assert result == 1

    def test_map_chain_with_multiple_then_operations(self) -> None:
        """Map chains with multiple .then() operations are optimized and execute correctly."""

        @task
        def square(x: int) -> int:
            return x * x

        @task
        def add_ten(x: int) -> int:
            return x + 10

        @task
        def multiply_two(x: int) -> int:
            return x * 2

        # Create the pattern: .product() -> .then() -> .then()
        m1 = square.product(x=[1, 2, 3])
        m2 = m1.then(add_ten)
        m3 = m2.then(multiply_two)

        result = evaluate(m3)

        # [1, 2, 3] -> square -> [1, 4, 9]
        #           -> add_ten -> [11, 14, 19]
        #           -> multiply_two -> [22, 28, 38]
        assert result == [22, 28, 38]

        # Verify optimization occurred
        graph = build_graph(m3)
        optimized = optimize_graph(graph, m3.id, enable_optimization=True)
        assert len(optimized) == 1
        composite = list(optimized.values())[0]
        assert isinstance(composite, CompositeMapTaskNode)
        assert len(composite.nodes) == 3

    def test_optimization_produces_correct_results(self) -> None:
        """Optimized graphs produce the same results as unoptimized."""

        @task
        def step1(x: int) -> int:
            return x + 10

        @task
        def step2(x: int) -> int:
            return x * 3

        @task
        def step3(x: int) -> int:
            return x - 5

        # Create chain
        a = step1(x=5)
        b = step2(x=a)
        c = step3(x=b)

        # Test with optimization enabled
        set_global_settings(DagliteSettings(enable_optimization=True))
        result_optimized = evaluate(c)

        # Test with optimization disabled
        set_global_settings(DagliteSettings(enable_optimization=False))
        result_unoptimized = evaluate(c)

        # Both should produce the same result
        # (5 + 10) * 3 - 5 = 45 - 5 = 40
        assert result_optimized == result_unoptimized == 40

        # Reset to default
        set_global_settings(DagliteSettings())

    def test_long_chain_optimization(self) -> None:
        """Optimizer handles long chains efficiently."""

        @task
        def add_one(x: int) -> int:
            return x + 1

        # Create a long chain
        result = add_one(x=0)
        for _ in range(10):
            result = add_one(x=result)

        # Build and optimize graph
        graph = build_graph(result)
        optimized = optimize_graph(graph, result.id, enable_optimization=True)

        # Should create a single composite node
        assert len(optimized) == 1
        composite = list(optimized.values())[0]
        assert isinstance(composite, CompositeTaskNode)
        assert len(composite.nodes) == 11

        # Evaluate and check result
        final_result = evaluate(result)
        assert final_result == 11

    def test_optimization_with_retries(self) -> None:
        """Optimization works correctly with task retries."""

        attempt_count = {"count": 0}

        @task(retries=2)
        def flaky_task(x: int) -> int:
            attempt_count["count"] += 1
            if attempt_count["count"] < 2:
                raise ValueError("Flaky!")
            return x + 1

        @task
        def stable_task(x: int) -> int:
            return x * 2

        # Create chain
        a = flaky_task(x=1)
        b = stable_task(x=a)

        # Evaluate (should succeed with retries)
        result = evaluate(b)

        # (1 + 1) * 2 = 4
        assert result == 4
        assert attempt_count["count"] == 2  # Flaky task was retried


class TestCompositeNodeHooks:
    """Test that composite nodes properly fire hooks."""

    def test_composite_node_fires_group_hooks(self) -> None:
        """CompositeTaskNode fires before_group_execute and after_group_execute hooks."""
        from daglite.plugins.hooks.markers import hook_impl

        hook_calls = []

        class GroupHookPlugin:
            @hook_impl
            def before_group_execute(self, group_metadata, initial_inputs, reporter):
                hook_calls.append(("before_group", len(group_metadata)))

            @hook_impl
            def after_group_execute(
                self, group_metadata, initial_inputs, final_result, duration, reporter
            ):
                hook_calls.append(("after_group", final_result))

        @task
        def add(x: int) -> int:
            return x + 1

        # Create chain
        a = add(x=1)
        b = add(x=a)
        c = add(x=b)

        _ = evaluate(c, plugins=[GroupHookPlugin()])

        assert len(hook_calls) == 2
        assert hook_calls[0][0] == "before_group"
        assert hook_calls[0][1] == 3
        assert hook_calls[1] == ("after_group", 4)


class TestOptimizationSettings:
    """Test optimization settings and environment variables."""

    def test_optimization_enabled_by_default(self) -> None:
        """Optimization is enabled by default."""
        settings = DagliteSettings()
        assert settings.enable_optimization is True

    def test_optimization_can_be_disabled(self) -> None:
        """Optimization can be disabled via settings."""
        settings = DagliteSettings(enable_optimization=False)
        assert settings.enable_optimization is False

    def test_optimization_respects_global_settings(self) -> None:
        """Graph optimization respects global settings."""

        @task
        def inc(x: int) -> int:
            return x + 1

        # Create chain
        a = inc(x=1)
        b = inc(x=a)

        # Disable optimization
        set_global_settings(DagliteSettings(enable_optimization=False))

        # Build and check that optimization doesn't occur
        graph = build_graph(b)
        assert len(graph) == 2  # Two separate nodes

        # Enable optimization
        set_global_settings(DagliteSettings(enable_optimization=True))

        # Evaluate - should optimize internally
        result = evaluate(b)
        assert result == 3

        # Reset
        set_global_settings(DagliteSettings())
