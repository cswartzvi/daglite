"""Tests for daglite-viz visualization functions."""

import pytest

from daglite import task
from daglite_viz import to_graphviz
from daglite_viz import to_mermaid
from daglite_viz import visualize_future


@task
def add(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y


@task
def multiply(x: int, y: int) -> int:
    """Multiply two numbers."""
    return x * y


def test_to_mermaid_single_node():
    """Test Mermaid generation for a single node."""
    future = add(x=1, y=2)
    diagram = to_mermaid([future])

    assert "flowchart TB" in diagram
    assert "add" in diagram


def test_to_mermaid_multiple_nodes():
    """Test Mermaid generation for multiple connected nodes."""
    a = add(x=1, y=2)
    b = add(x=3, y=4)
    c = multiply(x=a, y=b)

    diagram = to_mermaid([c])

    assert "flowchart TB" in diagram
    assert "add" in diagram
    assert "multiply" in diagram
    assert "-->" in diagram


def test_to_mermaid_direction():
    """Test Mermaid generation with different directions."""
    future = add(x=1, y=2)

    for direction in ["TB", "BT", "LR", "RL"]:
        diagram = to_mermaid([future], direction=direction)
        assert f"flowchart {direction}" in diagram


def test_to_graphviz_single_node():
    """Test Graphviz generation for a single node."""
    future = add(x=1, y=2)
    dot = to_graphviz([future])

    assert "digraph {" in dot
    assert "rankdir=TB" in dot
    assert "add" in dot
    assert "shape=box" in dot


def test_to_graphviz_multiple_nodes():
    """Test Graphviz generation for multiple connected nodes."""
    a = add(x=1, y=2)
    b = add(x=3, y=4)
    c = multiply(x=a, y=b)

    dot = to_graphviz([c])

    assert "digraph {" in dot
    assert "add" in dot
    assert "multiply" in dot
    assert "->" in dot


def test_to_graphviz_rankdir():
    """Test Graphviz generation with different rankdirs."""
    future = add(x=1, y=2)

    for rankdir in ["TB", "BT", "LR", "RL"]:
        dot = to_graphviz([future], rankdir=rankdir)
        assert f"rankdir={rankdir}" in dot


def test_visualize_future_mermaid():
    """Test visualize_future convenience function with Mermaid."""
    a = add(x=1, y=2)
    b = multiply(x=a, y=5)

    diagram = visualize_future(b, format="mermaid")

    assert "flowchart TB" in diagram
    assert "add" in diagram
    assert "multiply" in diagram


def test_visualize_future_graphviz():
    """Test visualize_future convenience function with Graphviz."""
    a = add(x=1, y=2)
    b = multiply(x=a, y=5)

    dot = visualize_future(b, format="graphviz")

    assert "digraph {" in dot
    assert "add" in dot
    assert "multiply" in dot


def test_visualize_future_invalid_format():
    """Test visualize_future with invalid format."""
    future = add(x=1, y=2)

    with pytest.raises(ValueError, match="Unknown format"):
        visualize_future(future, format="invalid")  # type: ignore


def test_map_task_visualization():
    """Test visualization of map tasks."""

    @task
    def process(x: int, scale: int) -> int:
        return x * scale

    future = process.product(x=[1, 2, 3], scale=[2])
    diagram = to_mermaid([future])

    assert "flowchart TB" in diagram
    assert "process" in diagram
    # Map nodes use parallelogram shape in Mermaid
    assert "/]" in diagram or "[/" in diagram


def test_empty_nodes():
    """Test handling of empty node collections."""
    mermaid = to_mermaid([])
    assert mermaid == "flowchart TB\n"

    graphviz = to_graphviz([])
    assert "digraph {" in graphviz
    assert "rankdir=TB" in graphviz
