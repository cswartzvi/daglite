# API Reference

Complete API documentation for Daglite's core modules and functions.

## Core Modules

### [Tasks](tasks.md)

Task definition and composition.

- `@task` decorator
- `Task` class
- `PartialTask` class
- `.product()`, `.zip()`, `.partial()` methods

### [Futures](futures.md)

Task futures and fluent operations.

- `TaskFuture` class
- `MapTaskFuture` class
- `.then()`, `.map()`, `.join()` methods

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
- `set_global_settings()` function

---

## Quick Links

- [User Guide](../user-guide/tasks.md) - Learn how to use Daglite
- [Examples](../examples/) - Real-world usage examples
- [Source Code](https://github.com/cswartzvi/daglite) - View on GitHub
