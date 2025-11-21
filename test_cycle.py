"""Test cycle detection in graph construction."""

from functools import cached_property
from uuid import UUID
from uuid import uuid4

from daglite import task
from daglite.exceptions import GraphConstructionError
from daglite.graph.base import ParamInput
from daglite.graph.builder import build_graph
from daglite.graph.nodes import TaskNode


@task
def identity(x: int) -> int:
    """Return the input unchanged."""
    return x


def test_cycle_detection():
    """Test that circular dependencies are detected."""

    class CyclicGraphBuilder:
        """A mock GraphBuilder that creates a cycle."""

        def __init__(self):
            self._id = uuid4()
            self.visited = False

        @cached_property
        def id(self) -> UUID:
            return self._id

        def to_graph(self, ctx, visit):
            """Create a node that depends on itself."""
            if not self.visited:
                self.visited = True
                # This will cause the cycle: visiting ourselves
                visit(self)
            return TaskNode(
                id=self.id,
                task=identity,
                params={"x": ParamInput(kind="value", value=42)},
                backend=None,
            )

    cyclic = CyclicGraphBuilder()

    try:
        build_graph(cyclic)
        print("ERROR: Cycle was not detected!")
        return False
    except GraphConstructionError as e:
        print(f"✓ Cycle detected correctly: {e}")
        return True


if __name__ == "__main__":
    success = test_cycle_detection()
    if success:
        print("\nAll cycle detection tests passed!")
    else:
        print("\nCycle detection tests FAILED!")
        exit(1)
