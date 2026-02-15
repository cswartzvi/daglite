# Fluent API

!!! note "Coming Soon"
    This page is under construction. A detailed guide is planned covering the topics outlined below.

The fluent API lets you build DAGs by chaining method calls on futures, rather than assigning intermediate variables. Methods like `.then()`, `.then_map()`, `.join()`, and `.split()` turn multi-step graphs into readable, pipeline-style expressions.

---

## Planned Content

### `.then()` for Inline Chaining (TaskFuture)

- Piping a `TaskFuture` into the next task's single unbound parameter
- Passing additional fixed keyword arguments alongside the chained value
- Using `.then()` with a `PartialTask` that has one remaining parameter
- How the unbound parameter is automatically resolved

### `.then_map()` for Fan-Out Chaining

- Expanding a single `TaskFuture` value across mapped parameter sequences
- The current future becomes a fixed (broadcast) argument while `**mapped_kwargs` are iterated
- Choosing `map_mode="zip"` (default) vs `map_mode="product"` for the mapped arguments
- Requirement for at least one mapped parameter (use `.then()` for 1-to-1 chaining)

### `.then()` on MapTaskFuture for Element-Wise Transforms

- Chaining `.then()` on a `MapTaskFuture` to apply a task to every element
- Preserving the fan-out shape through successive `.then()` calls
- Passing extra fixed keyword arguments to the element-wise task

### `.join()` for Reducing Sequences

- Collapsing a `MapTaskFuture` back into a single `TaskFuture`
- The reducer task receives the full list of results via its unbound parameter
- Common reducers: sum, concatenate, pick-best
- Passing additional fixed parameters to the reducer

### `.split()` for Tuple Destructuring

- Breaking a tuple-producing `TaskFuture` into individual `TaskFuture` objects
- Automatic size inference from `tuple[T1, T2, ...]` type annotations
- Explicit `size` parameter when type hints are unavailable
- Enabling parallel downstream processing of each tuple element

### Chaining Patterns and Best Practices

- Full map-then-join pattern: `.map() -> .then() -> .join()`
- Splitting a result for parallel branches that later merge
- When to prefer the fluent style vs. explicit variable assignment
- Readability tips for long chains

### `.run()` and `.run_async()` for Execution

- Calling `.run()` on any `TaskFuture` or `MapTaskFuture` to evaluate the graph
- Using `.run_async()` inside existing async contexts
- Passing `plugins` to `.run()` for per-execution hooks

---

## Quick Example

```python
from daglite import task

@task
def generate(n: int) -> int:
    return n

@task
def square(x: int) -> int:
    return x * x

@task
def total(values: list[int]) -> int:
    return sum(values)

# Fluent chain: fan-out, transform each element, then reduce
result = (
    generate
    .map(n=[1, 2, 3, 4])
    .then(square)
    .join(total)
    .run()
)
# result == 30  (1 + 4 + 9 + 16)
```

---

## See Also

- [Composition Patterns](composition.md) - `.map()`, `.partial()`, and direct invocation
- [Evaluation](evaluation.md) - `evaluate()`, `evaluate_async()`, and backend details
- [Tasks](tasks.md) - Defining tasks and the `@task` decorator
