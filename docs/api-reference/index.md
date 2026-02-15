# API Reference

Complete API documentation for Daglite's core modules and functions.

## Core Modules

### [Tasks](tasks.md)

Task definition and composition.

- `@task` decorator
- `Task` class — `.map()`, `.partial()`, `.with_options()`
- `PartialTask` class

### [Futures](futures.md)

Task futures and fluent operations.

- `TaskFuture` class — `.then()`, `.then_map()`, `.split()`, `.save()`
- `MapTaskFuture` class — `.then()`, `.join()`, `.save()`
- `DatasetFuture` class

### [Engine](engine.md)

DAG evaluation and execution.

- `evaluate()` - Synchronous evaluation
- `evaluate_async()` - Asynchronous evaluation

### [Pipelines](pipelines.md)

CLI pipeline definition.

- `@pipeline` decorator
- `load_pipeline()` function

### [Settings](settings.md)

Global configuration.

- `DagliteSettings` class
- `get_global_settings()` / `set_global_settings()` functions

### [Exceptions](exceptions.md)

Error types raised by Daglite.

- `DagliteError` base exception
- `TaskError`, `GraphError`, `ParameterError`, `BackendError`, `DatasetError`, `ExecutionError`

---

## Quick Links

- [User Guide](../user-guide/tasks.md) - Learn how to use Daglite
- [Examples](../examples/index.md) - Real-world usage examples
- [Source Code](https://github.com/cswartzvi/daglite) - View on GitHub
