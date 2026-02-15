# Evaluation

!!! note "Coming Soon"
    This page is under construction. A detailed guide is planned covering the topics outlined below.

Once you have composed a DAG of task futures, Daglite's evaluation engine executes the graph in topological order, resolving dependencies and dispatching work to the configured backends. This page will cover the `evaluate()` and `evaluate_async()` functions, backend selection, the plugin system, error handling, and performance tuning.

---

## Planned Content

### `evaluate()` for Synchronous Execution

- Passing a `TaskFuture` or `MapTaskFuture` to `evaluate(future)`
- Creates an event loop internally via `asyncio.run()`
- Cannot be called from within an existing async context (raises `RuntimeError`)
- Returns the concrete result of the root future

### `evaluate_async()` for Async Contexts

- Awaiting `evaluate_async(future)` inside `async def` functions
- Required when running inside Jupyter notebooks, async frameworks, or nested pipelines
- Same evaluation semantics as `evaluate()`, but non-blocking

### `.run()` and `.run_async()` Convenience Methods

- `future.run()` as shorthand for `evaluate(future)`
- `future.run_async()` as shorthand for `evaluate_async(future)`
- Available on both `TaskFuture` and `MapTaskFuture`
- Accepting an optional `plugins` parameter for per-execution hooks

### Execution Backends

- **Inline** (default) -- tasks run sequentially in the main event loop thread
- **Threading** -- tasks run in a thread pool, suited for I/O-bound work
- **Multiprocessing** -- tasks run in a process pool, suited for CPU-bound work
- Platform-specific multiprocessing context selection (fork, spawn, forkserver)

### Backend Selection: Per-Task vs Global

- Setting a per-task backend with `@task(backend_name="threading")`
- Setting the global default backend via `DagliteSettings.default_backend`
- Overriding the backend from the CLI with `--backend`
- Mixing backends within a single DAG

### The `plugins` Parameter

- Passing a list of plugin instances to `evaluate()`, `.run()`, or `.run_async()`
- Plugins are combined with any globally registered plugins
- Use cases: logging, metrics, custom hooks per execution
- Built-in plugins: `CentralizedLoggingPlugin`

### Error Handling During Evaluation

- Fail-fast behavior: pending sibling tasks are cancelled on first exception
- `ExecutionError` for cycle detection in the task graph
- `RuntimeError` when calling `evaluate()` from an async context
- `RuntimeError` when calling `evaluate()` from inside a running task
- Task-level retries with `@task(retries=N)`
- Task-level timeouts with `@task(timeout=seconds)`

### Performance Considerations

- Sibling parallelism: independent tasks at the same depth run concurrently
- Backend thread/process pool sizing via `DagliteSettings`
- Overhead of process serialization vs thread shared-memory
- When inline execution is faster than threaded execution

---

## Quick Example

```python
from daglite import task, evaluate

@task
def add(x: int, y: int) -> int:
    return x + y

@task(backend_name="threading")
def multiply(a: int, b: int) -> int:
    return a * b

# Build the DAG
sum_future = add(x=3, y=4)
product_future = multiply(a=sum_future, b=10)

# Synchronous evaluation
result = evaluate(product_future)  # 70

# Or use the convenience method
result = product_future.run()       # 70
```

---

## See Also

- [Tasks](tasks.md) - Backend and retry options on the `@task` decorator
- [Composition Patterns](composition.md) - Building DAGs to evaluate
- [Fluent API](fluent-api.md) - `.run()` and `.run_async()` on futures
- [Pipelines & CLI](pipelines.md) - `pipeline.run()` for end-to-end execution
- [CLI Reference](cli.md) - `daglite run` with `--backend` and `--parallel`
