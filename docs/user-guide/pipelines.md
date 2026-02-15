# Pipelines & CLI

!!! note "Coming Soon"
    This page is under construction. A detailed guide is planned covering the topics outlined below.

Pipelines are the entry points for Daglite DAGs. The `@pipeline` decorator wraps a function that builds a task graph and provides convenient methods to execute it -- from Python with `.run()` / `.run_async()`, or from the command line with `daglite run`. This page will cover pipeline definition, parameterization, execution, CLI integration, and discovery.

---

## Planned Content

### The `@pipeline` Decorator

- Wrapping a graph-building function with `@pipeline`
- Using `@pipeline` without parentheses vs `@pipeline(name=..., description=...)`
- Return type annotation with `Dag[T]` (alias for `BaseTaskFuture[T]`)
- Pipeline objects are callable: `my_pipeline(...)` returns the underlying future

### Pipeline Parameters and Type Annotations

- Defining typed parameters for automatic CLI type conversion
- Supported types: `int`, `float`, `bool`, `str`, `list`, and more
- Default parameter values for optional arguments
- `pipeline.get_typed_params()` and `pipeline.has_typed_params()` introspection
- `pipeline.signature` for accessing the underlying function signature

### `pipeline.run()` and `pipeline.run_async()`

- `pipeline.run(*args, **kwargs)` builds the DAG and evaluates synchronously
- `pipeline.run_async(*args, **kwargs)` for async contexts
- Cannot call `.run()` from within a running event loop (use `.run_async()`)
- Positional and keyword argument forwarding to the pipeline function

### CLI Invocation with `daglite run`

- `daglite run <module.path.pipeline_name>` to execute a pipeline
- `--param name=value` / `-p name=value` for passing parameters
- `--backend` / `-b` for selecting the execution backend
- `--parallel` flag for enabling sibling parallelism via async evaluation
- `--settings` / `-s` for overriding global `DagliteSettings` values

### Parameter Passing from CLI

- All parameters passed as `--param key=value` strings
- Automatic type conversion based on pipeline function's type annotations
- JSON parsing for complex types like `list[str]`
- Warning when passing parameters to untyped pipelines
- Validation of required vs optional parameters

### Pipeline Discovery via Module Paths

- Dotted module paths: `myproject.pipelines.my_pipeline`
- `load_pipeline()` function for programmatic loading
- Current directory is automatically added to `sys.path`
- Error messages for invalid paths, missing modules, and non-pipeline objects

### Integration with the Full CLI Reference Page

- Cross-reference to the [CLI Reference](cli.md) for full option details
- CI/CD integration patterns (GitHub Actions, GitLab CI)
- Settings overrides: `--settings max_backend_threads=16`

---

## Quick Example

```python
from daglite import Dag, pipeline, task

@task
def double(x: int) -> int:
    return x * 2

@task
def total(values: list[int]) -> int:
    return sum(values)

@pipeline
def my_sweep(n: int) -> Dag[int]:
    """Double each number from 0 to n-1 and sum the results."""
    return double.map(x=list(range(n))).join(total)

# Run from Python
result = my_sweep.run(n=5)  # 0+2+4+6+8 = 20
```

```bash
# Run from the command line
daglite run myproject.my_sweep --param n=5
```

---

## See Also

- [CLI Reference](cli.md) - Full `daglite run` command reference, options, and CI/CD examples
- [Evaluation](evaluation.md) - `evaluate()` and backend details
- [Composition Patterns](composition.md) - Building the DAG that a pipeline returns
- [Tasks](tasks.md) - Defining the tasks used inside pipelines
