# Welcome to Daglite

**Lightweight, type-safe Python framework for building DAGs**

[Getting Started](getting-started.md){ .md-button .md-button--primary }
[User Guide](user-guide/tasks.md){ .md-button }
[Examples](examples.md){ .md-button }

---

## What is Daglite?

Daglite is a lightweight Python framework for building **Directed Acyclic Graphs (DAGs)** with explicit data flow, full type safety, and composable operations. It provides a simple, intuitive API for defining and executing complex computational pipelines.

### Key Principles

- **🎯 Explicit is better than implicit** - All data dependencies are clearly defined
- **🔒 Type safety first** - Full support for mypy, pyright, and pyrefly
- **🧩 Composition over configuration** - Build complex DAGs from simple building blocks
- **⚡ Lazy by default** - Execution only happens when you explicitly evaluate
- **🔍 Testable** - Pure functions make testing straightforward

---

## Quick Example

```python
from daglite import task, evaluate

@task
def fetch_data(url: str) -> dict:
    return requests.get(url).json()

@task
def transform(data: dict, multiplier: int) -> list:
    return [item * multiplier for item in data["values"]]

@task
def save(items: list, path: str) -> None:
    with open(path, "w") as f:
        json.dump(items, f)

# Build the DAG (lazy - doesn't execute yet)
result = (
    fetch_data.bind(url="https://api.example.com/data")
    .then(transform, multiplier=10)
    .then(save, path="output.json")
)

# Execute the DAG
evaluate(result)
```

---

## Why Daglite?

### Compared to other DAG frameworks

| Feature | Daglite | Airflow | Prefect | Dask |
|---------|---------|---------|---------|------|
| **Dependencies** | 0 (core) | 50+ | 30+ | 20+ |
| **Type Safety** | ✅ Full | ⚠️ Partial | ⚠️ Partial | ⚠️ Partial |
| **Learning Curve** | Low | High | Medium | Medium |
| **DAG Definition** | Pure Python | YAML + Python | Python | Python |
| **Execution** | Sync/Async | Scheduled | Flows | Distributed |
| **Best For** | ETL, Scripts | Production Scheduling | Workflows | Big Data |

### When to use Daglite

✅ **Good fit:**

- Data transformation pipelines
- ETL workflows
- Local computation graphs
- Type-safe data processing
- Testing and prototyping
- CLI tools

❌ **Not ideal for:**

- Production job scheduling (use Airflow)
- Real-time streaming (use Kafka, Flink)
- Big data processing (use Spark, Dask)
- Multi-tenant orchestration (use Prefect, Dagster)

---

## Core Features

### Fluent API

Chain operations naturally with method chaining:

```python
result = (
    fetch.bind(url="https://example.com")
    .then(parse, format="json")
    .then(validate, strict=True)
    .then(save, path="output.json")
)
```

### Map-Reduce Patterns

Built-in support for fan-out/fan-in operations:

```python
# Cartesian product
results = task.product(x=[1, 2, 3], y=[10, 20])
# Result: [11, 21, 12, 22, 13, 23]

# Pairwise zip
results = task.zip(x=[1, 2, 3], y=[10, 20, 30])
# Result: [11, 22, 33]

# Map and reduce
total = (
    square.product(x=[1, 2, 3, 4])
    .map(double)
    .join(sum_all)
)
```

### Full Type Safety

Works seamlessly with type checkers:

```python
@task
def process(data: pd.DataFrame) -> dict[str, float]:
    return {"mean": data["value"].mean()}

result = process.bind(data=df)
reveal_type(result)  # TaskFuture[dict[str, float]]
```

### Async Execution

Run tasks in parallel with threading or multiprocessing:

```python
result = evaluate(my_dag, use_async=True)

# Per-task backends
@task(backend="threading")
def io_task(url: str) -> bytes:
    return requests.get(url).content

@task(backend="multiprocessing")
def cpu_task(data: np.ndarray) -> np.ndarray:
    return expensive_computation(data)
```

---

## Next Steps

- **[Getting Started](getting-started.md)** - Install Daglite and learn the basics in 5 minutes
- **[User Guide](user-guide/tasks.md)** - In-depth guides on tasks, composition, and evaluation
- **[CLI & Pipelines](user-guide/pipelines.md)** - Command-line interface and pipeline definitions
- **[Examples](examples.md)** - Real-world examples and patterns
