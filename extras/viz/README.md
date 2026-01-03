# daglite-viz

Graph visualization for daglite using Mermaid and Graphviz.

## Installation

```bash
pip install daglite-viz
```

For Graphviz support (requires graphviz to be installed on your system):

```bash
pip install daglite-viz[graphviz]
```

Or install all optional dependencies:

```bash
pip install daglite-viz[all]
```

## Usage

### Visualizing a TaskFuture

```python
from daglite import task
from daglite_viz import visualize_future, to_mermaid, to_graphviz

@task
def add(x: int, y: int) -> int:
    return x + y

@task
def multiply(x: int, y: int) -> int:
    return x * y

# Create a task graph
a = add(1, 2)
b = add(3, 4)
c = multiply(a, b)

# Generate Mermaid diagram
mermaid_diagram = visualize_future(c, format="mermaid")
print(mermaid_diagram)

# Or use the direct function
mermaid_diagram = to_mermaid([a, b, c])
print(mermaid_diagram)
```

### Visualizing with Graphviz

```python
from daglite_viz import to_graphviz

# Generate Graphviz diagram
dot_source = to_graphviz([a, b, c])
print(dot_source)

# Or render to file (requires graphviz package)
from graphviz import Source
graph = Source(dot_source)
graph.render('graph', format='png')
```

## Features

- **Mermaid diagrams**: Generate Mermaid flowchart syntax for web-based visualization
- **Graphviz**: Generate DOT format for rendering with Graphviz
- **Automatic graph extraction**: Convenience functions to extract graphs from TaskFutures
- **Customizable**: Simple API for generating visualizations from collections of nodes
