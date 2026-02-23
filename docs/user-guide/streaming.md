# Streaming with `.iter()`

Daglite supports **lazy streaming** of generator-based tasks via `.iter()`.
Instead of materialising an entire sequence into memory before dispatching work,
`.iter()` lets the coordinator iterate the generator on-the-fly and dispatch each
yielded item to a backend worker as it arrives.

---

## Why `.iter()`?

Consider a task that produces a large sequence of items:

```python
from daglite import task

@task
def generate_rows(path: str) -> Iterator[dict]:
    """Stream rows from a large CSV — millions of rows."""
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row
```

### Without `.iter()` — eager materialisation

```python
# ❌ Materialises all rows into a list BEFORE any processing starts
result = process_row.map(x=generate_rows(path="data.csv")).run()
```

Calling the task directly (`generate_rows(path=...)`) produces a `TaskFuture`
that will materialise the generator into a `list` at execution time. *Every* row
is held in memory simultaneously before `.map()` even begins dispatching work.

### With `.iter()` — lazy streaming

```python
# ✅ Rows flow lazily — O(1) coordinator memory
result = generate_rows.iter(path="data.csv").then(process_row).run()
```

With `.iter()`, the generator runs on the coordinator and each yielded item is
dispatched to a worker immediately. The coordinator never holds more than one
item at a time.

---

## The Fluent API (recommended)

The preferred way to use `.iter()` is the fluent pipeline style:

```python
generate_rows.iter(path="data.csv").then(process_row).reduce(accumulate, initial=0).run()
```

This reads **left-to-right** as a data flow:

1. **`.iter()`** — lazily yield items from the generator
2. **`.then()`** — apply a task to each item (fan-out)
3. **`.reduce()`** — fold results into a single value (fan-in)
4. **`.run()`** — execute the pipeline

### Available chaining methods

| Method | Returns | Description |
|--------|---------|-------------|
| `.then(task)` | `MapTaskFuture` | Apply *task* to each yielded item |
| `.join(task)` | `TaskFuture` | Collect all results, pass the list to *task* |
| `.reduce(task, initial=...)` | `ReduceFuture` | Streaming fold with O(1) memory |
| `.save(key)` | `Self` | Save each yielded item (see [Saving items](#saving-yielded-items)) |
| `.run()` | `list[R]` | Execute and return materialized results |

### Chaining `.then()` calls

Multiple `.then()` calls chain naturally:

```python
@task
def double(x: int) -> int:
    return x * 2

@task
def add_ten(x: int) -> int:
    return x + 10

# generate → double → add_ten → collect
result = generate_numbers.iter(n=5).then(double).then(add_ten).run()
# [10, 12, 14, 16, 18]
```

### Using `.reduce()` for streaming aggregation

For commutative/associative operations, use `ordered=False` for maximum
throughput — items are reduced as soon as they complete rather than waiting for
iteration order:

```python
@task
def accumulate(acc: int, item: int) -> int:
    return acc + item

result = (
    generate_numbers.iter(n=1_000_000)
    .then(double)
    .reduce(accumulate, initial=0, ordered=False)
    .run()
)
```

### Using `.join()` for collection

When you need the full list (e.g. for sorting or aggregation that needs all
values at once):

```python
@task
def summarise(values: list[int]) -> dict:
    return {"count": len(values), "total": sum(values)}

result = generate_numbers.iter(n=100).then(double).join(summarise).run()
```

---

## Alternative: `.map()` with `.iter()`

You can also pass an `.iter()` future as a mapped argument to `.map()`:

```python
result = double.map(x=generate_numbers.iter(n=100)).run()
```

This produces the **same optimised graph** as `.iter().then()`. However, the
fluent form is preferred because:

- It reads as a natural left-to-right pipeline
- It avoids the restriction that `.iter()` must be the **only** mapped argument
  (mixing `.iter()` with other mapped kwargs like `y=[1, 2, 3]` raises a
  `ParameterError`)

The `.map()` form is useful for **programmatic graph construction** where the
mapping target is determined dynamically.

---

## Saving yielded items

Use `.save()` on the iter future to persist each yielded item with an
`{iteration_index}` key:

```python
from daglite.datasets.store import DatasetStore

store = DatasetStore("/tmp/pipeline_output")

result = (
    generate_rows.iter(path="data.csv")
    .save("row_{iteration_index}", save_store=store)
    .then(process_row)
    .reduce(accumulate, initial=0)
    .run()
)

# Saved: row_0, row_1, row_2, ... (one per yielded item)
```

Each yielded item is saved **before** it is dispatched to a worker, so saves
happen on the coordinator as the generator produces items. This works
identically whether graph optimisation is enabled or disabled.

### Key template variables

The following variables are available in save key templates:

| Variable | Description |
|----------|-------------|
| `{iteration_index}` | Zero-based index of the yielded item |
| `{param_name}` | Any parameter passed to `.iter()` (e.g. `{path}`) |

---

## Partial tasks with `.iter()`

`.iter()` works with `PartialTask` — pre-fix some parameters and iterate the
rest:

```python
@task
def generate_range(start: int, n: int) -> Iterator[int]:
    for i in range(start, start + n):
        yield i

partial_gen = generate_range.partial(start=100)
result = partial_gen.iter(n=50).then(double).run()
```

---

## How it works

Under the hood, `.iter()` creates an `IterNode` in the graph IR. When the graph
optimizer runs (enabled by default), it detects `IterNode → MapTaskNode` chains
and folds them into a single `CompositeMapTaskNode`. The composite executes the
generator on the coordinator and dispatches each yielded item to a backend
worker, avoiding the overhead of materialising the full sequence.

When optimisation is disabled, the `IterNode` falls back to running the generator
on the backend where it is materialised into a list — functionally identical to
calling the task directly, but `.save()` still persists items individually.

```
┌─────────────────── Optimised (default) ───────────────────┐
│                                                           │
│  Coordinator          Workers                             │
│  ┌──────────┐                                             │
│  │ IterNode │─ yield 0 ──▶ [worker: double(0)] ──▶ 0     │
│  │ (gen fn) │─ yield 1 ──▶ [worker: double(1)] ──▶ 2     │
│  │          │─ yield 2 ──▶ [worker: double(2)] ──▶ 4     │
│  └──────────┘                                    ▼        │
│                                         reduce(acc, item) │
│                                                           │
└───────────────────────────────────────────────────────────┘
```
