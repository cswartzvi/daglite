# Creating Custom Plugins

Daglite uses [pluggy](https://pluggy.readthedocs.io/) for its plugin system, allowing you to extend functionality through hooks. This guide shows you how to create custom plugins that integrate with Daglite's execution lifecycle.

## Plugin System Overview

Daglite's plugin system is based on **hook specifications** that define extension points in the execution lifecycle:

- **Node-level hooks** - Called before/after individual task execution
- **Graph-level hooks** - Called before/after entire DAG execution
- **Event system** - Bidirectional communication between plugins and Daglite

## Quick Start

### Simple Plugin

Create a plugin that logs task execution:

```python
from daglite.plugins.hooks import hook_impl
from daglite.graph.base import GraphMetadata
from typing import Any

class LoggingPlugin:
    """Plugin that logs task execution events."""

    @hook_impl
    def before_node_execute(
        self,
        key: str,
        metadata: GraphMetadata,
        inputs: dict[str, Any],
        reporter=None,
    ) -> None:
        print(f"Starting task: {metadata.name}")

    @hook_impl
    def after_node_execute(
        self,
        key: str,
        metadata: GraphMetadata,
        inputs: dict[str, Any],
        result: Any,
        duration: float,
        reporter=None,
    ) -> None:
        print(f"Completed task: {metadata.name} in {duration:.2f}s")
```

### Register and Use

```python
from daglite import task, evaluate
from daglite.plugins.manager import register_plugins

# Register plugin globally
register_plugins(LoggingPlugin())

@task
def add(x: int, y: int) -> int:
    return x + y

# Plugin hooks are called during evaluation
result = evaluate(add(x=1, y=2))
# Output:
# Starting task: add
# Completed task: add in 0.00s
```

---

## Available Hooks

### Node-Level Hooks

Called for each task execution:

#### `before_node_execute`

Called before a task begins execution.

```python
@hook_impl
def before_node_execute(
    self,
    key: str,              # Unique key for this node
    metadata: GraphMetadata,  # Task metadata (name, description)
    inputs: dict[str, Any],   # Resolved input parameters
    reporter=None,            # Event reporter (optional)
) -> None:
    ...
```

#### `after_node_execute`

Called after a task completes successfully.

```python
@hook_impl
def after_node_execute(
    self,
    key: str,
    metadata: GraphMetadata,
    inputs: dict[str, Any],
    result: Any,           # Task result
    duration: float,       # Execution time in seconds
    reporter=None,
) -> None:
    ...
```

#### `on_node_error`

Called when a task execution fails.

```python
@hook_impl
def on_node_error(
    self,
    key: str,
    metadata: GraphMetadata,
    inputs: dict[str, Any],
    error: Exception,      # The exception that was raised
    duration: float,
    reporter=None,
) -> None:
    ...
```

### Graph-Level Hooks

Called for entire DAG execution:

#### `before_graph_execute`

```python
from uuid import UUID

@hook_impl
def before_graph_execute(
    self,
    root_id: UUID,         # UUID of root node
    node_count: int,       # Total nodes in graph
    is_async: bool,        # True for async execution
) -> None:
    ...
```

#### `after_graph_execute`

```python
@hook_impl
def after_graph_execute(
    self,
    root_id: UUID,
    result: Any,           # Final result
    duration: float,       # Total execution time
    is_async: bool,
) -> None:
    ...
```

#### `on_graph_error`

```python
@hook_impl
def on_graph_error(
    self,
    root_id: UUID,
    error: Exception,
    duration: float,
    is_async: bool,
) -> None:
    ...
```

---

## Example Plugins

### Timing Plugin

Track execution time for performance analysis:

```python
from daglite.plugins.hooks import hook_impl
import time

class TimingPlugin:
    """Track execution times for all tasks."""

    def __init__(self):
        self.timings = {}

    @hook_impl
    def after_node_execute(self, key, metadata, inputs, result, duration, reporter=None):
        self.timings[metadata.name] = duration

    @hook_impl
    def after_graph_execute(self, root_id, result, duration, is_async):
        print("\nExecution Times:")
        for task_name, task_duration in sorted(
            self.timings.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            print(f"  {task_name}: {task_duration:.4f}s")
        print(f"\nTotal: {duration:.4f}s")
```

### Progress Plugin

Show progress for long-running workflows:

```python
from daglite.plugins.hooks import hook_impl

class ProgressPlugin:
    """Display execution progress."""

    def __init__(self):
        self.total = 0
        self.completed = 0

    @hook_impl
    def before_graph_execute(self, root_id, node_count, is_async):
        self.total = node_count
        self.completed = 0
        print(f"Starting execution: 0/{self.total} tasks")

    @hook_impl
    def after_node_execute(self, key, metadata, inputs, result, duration, reporter=None):
        self.completed += 1
        pct = (self.completed / self.total) * 100
        print(f"Progress: {self.completed}/{self.total} ({pct:.0f}%)")

    @hook_impl
    def after_graph_execute(self, root_id, result, duration, is_async):
        print(f"Completed: {self.total}/{self.total} tasks in {duration:.2f}s")
```

### Error Tracking Plugin

Collect errors for debugging:

```python
from daglite.plugins.hooks import hook_impl

class ErrorTrackingPlugin:
    """Track all errors during execution."""

    def __init__(self):
        self.errors = []

    @hook_impl
    def on_node_error(self, key, metadata, inputs, error, duration, reporter=None):
        self.errors.append({
            'task': metadata.name,
            'error': str(error),
            'error_type': type(error).__name__,
            'inputs': inputs,
            'duration': duration
        })

    @hook_impl
    def on_graph_error(self, root_id, error, duration, is_async):
        print(f"\nExecution failed after {duration:.2f}s")
        print(f"Total errors: {len(self.errors)}")
        for err in self.errors:
            print(f"  - {err['task']}: {err['error_type']}: {err['error']}")
```

### Caching Plugin

Simple result caching:

```python
from daglite.plugins.hooks import hook_impl
import hashlib
import json

class CachingPlugin:
    """Cache task results based on inputs."""

    def __init__(self):
        self.cache = {}

    def _hash_inputs(self, inputs):
        """Create hash from inputs."""
        return hashlib.sha256(
            json.dumps(inputs, sort_keys=True).encode()
        ).hexdigest()

    @hook_impl
    def before_node_execute(self, key, metadata, inputs, reporter=None):
        cache_key = (metadata.name, self._hash_inputs(inputs))
        if cache_key in self.cache:
            print(f"Cache hit for {metadata.name}")
            # Note: Currently can't skip execution, just track hits
            self.cache_hits = self.cache.get('_hits', 0) + 1

    @hook_impl
    def after_node_execute(self, key, metadata, inputs, result, duration, reporter=None):
        cache_key = (metadata.name, self._hash_inputs(inputs))
        self.cache[cache_key] = result
```

---

## Plugin Registration

### Global Registration

Register plugins that apply to all evaluations:

```python
from daglite.plugins.manager import register_plugins

# Register multiple plugins at once
register_plugins(
    TimingPlugin(),
    ProgressPlugin(),
    ErrorTrackingPlugin()
)
```

### Per-Execution Registration

Register plugins for specific evaluations only:

```python
from daglite import evaluate

timing = TimingPlugin()
result = evaluate(
    my_dag,
    plugins=[timing]  # Only active for this evaluation
)

print(timing.timings)
```

### Entry Point Registration

For distributable plugins, use setuptools entry points:

```python
# setup.py or pyproject.toml
entry_points={
    'daglite.hooks': [
        'myplugin = mypackage.plugin:MyPlugin'
    ]
}
```

Plugins registered via entry points are automatically loaded.

---

## Best Practices

### 1. Use Descriptive Names

```python
# Good
class PostgresResultWriterPlugin:
    """Write task results to PostgreSQL database."""
    ...

# Less clear
class DBPlugin:
    ...
```

### 2. Handle Errors Gracefully

Plugins shouldn't crash the DAG execution:

```python
@hook_impl
def after_node_execute(self, key, metadata, inputs, result, duration, reporter=None):
    try:
        self.save_to_database(metadata.name, result)
    except Exception as e:
        # Log but don't crash
        logger.warning(f"Failed to save result: {e}")
```

### 3. Make Plugins Configurable

```python
class LoggingPlugin:
    def __init__(self, verbose: bool = False, log_file: str | None = None):
        self.verbose = verbose
        self.log_file = log_file

    @hook_impl
    def after_node_execute(self, key, metadata, inputs, result, duration, reporter=None):
        message = f"{metadata.name} completed in {duration:.2f}s"
        if self.verbose:
            message += f" with inputs: {inputs}"

        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(message + '\n')
        else:
            print(message)
```

### 4. Document Your Hooks

```python
class MetricsPlugin:
    """
    Collect execution metrics and export to monitoring system.

    This plugin tracks:
    - Task execution times
    - Task success/failure rates
    - Total graph execution time

    Metrics are exported via Prometheus client.
    """

    @hook_impl
    def after_node_execute(self, key, metadata, inputs, result, duration, reporter=None):
        """Record task execution time metric."""
        ...
```

### 5. Test Your Plugins

```python
from daglite import task, evaluate

def test_timing_plugin():
    plugin = TimingPlugin()

    @task
    def slow_task(x: int) -> int:
        import time
        time.sleep(0.1)
        return x * 2

    result = evaluate(slow_task(x=5), plugins=[plugin])

    assert 'slow_task' in plugin.timings
    assert plugin.timings['slow_task'] >= 0.1
    assert result == 10
```

---

## Advanced Topics

### Stateful Plugins

Maintain state across executions:

```python
class StatefulPlugin:
    def __init__(self):
        self.execution_count = 0
        self.total_duration = 0

    @hook_impl
    def after_graph_execute(self, root_id, result, duration, is_async):
        self.execution_count += 1
        self.total_duration += duration

    def get_average_duration(self):
        if self.execution_count == 0:
            return 0
        return self.total_duration / self.execution_count
```

### Plugin Dependencies

One plugin can depend on another:

```python
class MetricsCollector:
    def __init__(self):
        self.metrics = []

    @hook_impl
    def after_node_execute(self, key, metadata, inputs, result, duration, reporter=None):
        self.metrics.append({
            'task': metadata.name,
            'duration': duration
        })

class MetricsExporter:
    def __init__(self, collector: MetricsCollector):
        self.collector = collector

    @hook_impl
    def after_graph_execute(self, root_id, result, duration, is_async):
        # Export metrics collected by MetricsCollector
        self.export_to_prometheus(self.collector.metrics)
```

---

## See Also

- [pluggy documentation](https://pluggy.readthedocs.io/) - Plugin framework used by Daglite
- [Hook Specifications Source](https://github.com/cswartzvi/daglite/blob/main/src/daglite/plugins/hooks/specs.py) - Complete hook specifications
- [CLI Plugin Source](https://github.com/cswartzvi/daglite/tree/main/extras/cli) - Example plugin implementation
