---
hide:
  - navigation
---

# Welcome to Daglite

**Lightweight, type-safe Python framework for building DAGs**

---

## What is Daglite?

Daglite is a lightweight Python framework for building **Directed Acyclic Graphs (DAGs)** with explicit data flow, complete type safety, and composable operations. It provides a simple, intuitive API for defining and executing computational pipelines—no infrastructure required.

**Built for computational work in restricted environments.** Originally designed for operations research analysts working on air-gapped, Windows-only systems, Daglite solves a specific problem: building workflows that are easy to analyze, share with colleagues, and re-run—even after returning to a project months later.

---

## Core Principles

**No infrastructure required**
Daglite runs anywhere Python runs—no databases, no containers, no cloud services, no servers. When you need more capabilities, plugins extend functionality without adding mandatory dependencies.

**Explicit over implicit**
All data dependencies are clearly defined. The DAG structure is static and analyzable before execution. Type checkers catch errors before runtime.

**Type-safe and modular**
Full support for `mypy`, `pyright`, `pyrefly`, and `ty`. Your IDE provides autocomplete and catches type mismatches.

**Lazy by default**
Execution only happens when you explicitly evaluate. Build your entire pipeline, then execute it once.

**Testable**
Pure functions make testing straightforward—no mocking infrastructure or database connections.

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
result = evaluate(
    fetch_data(url="https://api.example.com/data")
    .then(transform, multiplier=10)
    .then(save, path="output.json")
)
```

---

## When to Use Daglite

### Perfect for:

- Data transformation pipelines and ETL workflows
- Machine learning pipelines (feature engineering, training, evaluation)
- Computational science workflows
- Analysts and data scientists who need reproducible workflows
- Air-gapped or restricted environments
- CLI tools with workflow orchestration
- Local development and prototyping
- Projects where simplicity and type safety matter

### Not ideal for:

- Production job scheduling with cron-like triggers → Use [Airflow](https://airflow.apache.org/), [Prefect](https://www.prefect.io/)
- Real-time streaming data → Use Kafka, Flink
- Distributed computing at massive scale → Use Spark, Dask
- Multi-tenant orchestration platforms → Use [Dagster](https://dagster.io/)

Daglite complements these excellent tools by providing a lightweight alternative for local, type-safe workflows.

---

## Key Features

### Fluent API

Chain operations naturally with method chaining:

```python
result = (
    fetch(url="https://example.com")
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

# Pairwise zip
results = task.zip(x=[1, 2, 3], y=[10, 20, 30])

# Map and reduce
total = (
    square.product(x=[1, 2, 3, 4])
    .map(double)
    .join(sum_all)
)
```

### Complete Type Safety

Works seamlessly with type checkers:

```python
@task
def process(data: pd.DataFrame) -> dict[str, float]:
    return {"mean": data["value"].mean()}

result = process(data=df)
reveal_type(result)  # TaskFuture[dict[str, float]]
```

### Async Execution

Run tasks in parallel with threading or multiprocessing:

```python
result = evaluate(my_dag, use_async=True)

# Per-task backends
@task(backend_name="threading")
def io_task(url: str) -> bytes:
    return requests.get(url).content

@task(backend_name="multiprocessing")
def cpu_task(data: np.ndarray) -> np.ndarray:
    return expensive_computation(data)
```

---

## Next Steps

- **[Getting Started](getting-started.md)** - Install Daglite and learn the basics in 5 minutes
- **[User Guide](user-guide/tasks.md)** - In-depth guides on tasks, composition, and evaluation
- **[CLI & Pipelines](user-guide/pipelines.md)** - Command-line interface and pipeline definitions
- **[Plugins](plugins/)** - Extend Daglite with CLI, serialization, and more
- **[Examples](examples/)** - Real-world examples and patterns
- **[API Reference](api-reference/)** - Complete API documentation
