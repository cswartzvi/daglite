# Composition Patterns

!!! note "Coming Soon"
    This page is under construction. A detailed guide is planned covering the topics outlined below.

Daglite provides several patterns for composing tasks into DAGs: direct invocation for single calls, `.map()` for fan-out over sequences, and `.partial()` for fixing parameters. These patterns can be freely combined within a single DAG to express complex data-parallel workflows.

---

## Planned Content

### Direct Task Invocation with `task(**kwargs)`

- Calling a task returns a `TaskFuture`, not the result itself
- Positional and keyword argument support
- Passing `TaskFuture` objects as arguments to create dependencies
- Chaining multiple tasks into a linear pipeline

### Fan-Out with `.map(map_mode="product")` for Cartesian Product

- Using `.map()` with `map_mode="product"` to generate every combination of input sequences
- Parameter sweep use cases (e.g., grid search over hyperparameters)
- Return type is `MapTaskFuture[R]` containing one result per combination
- Ordering of results in the output list

### Pairwise Fan-Out with `.map()` (Default Zip Mode)

- Default `map_mode="zip"` pairs elements by index
- All sequences must have the same length
- Use cases: batch processing parallel lists of inputs
- Single-sequence mapping for simple transforms

### Partial Application with `.partial()`

- Fixing one or more parameters with `.partial(**kwargs)`
- Returns a `PartialTask` that can be called or mapped
- Analogy with `functools.partial`
- Reusable task templates for repeated invocations

### Combining Fixed and Mapped Parameters

- Using `.partial()` to fix scalar parameters, then `.map()` over remaining ones
- Broadcasting a single upstream result across a fan-out
- Mixing `TaskFuture` values with literal values in `.partial()`

### Mixing Patterns in a Single DAG

- Combining direct calls, fan-out, and partial application in one graph
- When to use `.map()` vs. calling a task in a loop
- Performance implications of different composition strategies

---

## Quick Example

```python
from daglite import task

@task
def score(model: str, dataset: str, metric: str) -> float:
    """Score a model on a dataset with a given metric."""
    ...  # scoring logic
    return 0.95

# Direct invocation
single = score(model="bert", dataset="squad", metric="f1")

# Pairwise fan-out (zip mode, default)
pairwise = score.partial(metric="f1").map(
    model=["bert", "gpt2"],
    dataset=["squad", "mnli"],
)
# Evaluates: score("bert","squad","f1"), score("gpt2","mnli","f1")

# Cartesian product fan-out
sweep = score.partial(metric="f1").map(
    model=["bert", "gpt2"],
    dataset=["squad", "mnli"],
    map_mode="product",
)
# Evaluates all 4 combinations: bert/squad, bert/mnli, gpt2/squad, gpt2/mnli
```

---

## See Also

- [Tasks](tasks.md) - Defining tasks and the `@task` decorator
- [Fluent API](fluent-api.md) - `.then()`, `.join()`, and chaining patterns
- [Evaluation](evaluation.md) - Running composed DAGs with `evaluate()`
