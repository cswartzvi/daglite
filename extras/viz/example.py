"""Example demonstrating daglite-viz graph visualization."""

from daglite import evaluate
from daglite import task
from daglite_viz import to_graphviz
from daglite_viz import to_mermaid
from daglite_viz import visualize_future


@task
def load_data(source: str) -> dict:
    """Load data from a source."""
    return {"source": source, "records": 100}


@task
def transform(data: dict, operation: str) -> dict:
    """Transform data with an operation."""
    return {**data, "transformed": operation}


@task
def aggregate(data1: dict, data2: dict) -> dict:
    """Aggregate two data sources."""
    return {
        "sources": [data1["source"], data2["source"]],
        "total_records": data1["records"] + data2["records"],
    }


def main():
    """Demonstrate graph visualization."""
    # Create a task graph
    source1 = load_data(source="database")
    source2 = load_data(source="api")

    transformed1 = transform(data=source1, operation="filter")
    transformed2 = transform(data=source2, operation="normalize")

    result = aggregate(data1=transformed1, data2=transformed2)

    # Generate Mermaid diagram
    print("=== Mermaid Diagram ===")
    mermaid = visualize_future(result, format="mermaid", direction="LR")
    print(mermaid)
    print()

    # Generate Graphviz diagram
    print("=== Graphviz DOT ===")
    graphviz = visualize_future(result, format="graphviz", rankdir="LR")
    print(graphviz)
    print()

    # Evaluate the graph
    print("=== Execution Result ===")
    output = evaluate(result)
    print(output)

    # You can also generate visualization from a collection of nodes
    print("\n=== Visualize multiple futures ===")
    mermaid_multi = to_mermaid([transformed1, transformed2], direction="TB")
    print(mermaid_multi)


if __name__ == "__main__":
    main()
