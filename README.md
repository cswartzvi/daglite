# Daglite

[![PyPI](https://img.shields.io/pypi/v/daglite?label=PyPI)](https://pypi.org/project/daglite/)
[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![type checked](https://img.shields.io/badge/type%20checked-mypy%2C%20pyright%2C%20pyrefly%2C%20ty-blue)](https://github.com/cswartzvi/daglite)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![tests](https://img.shields.io/github/actions/workflow/status/cswartzvi/daglite/testing.yaml?branch=main&label=tests&logo=github)](https://github.com/cswartzvi/daglite/actions/workflows/testing.yaml)
[![codecov](https://codecov.io/github/cswartzvi/daglite/graph/badge.svg?token=1o01x0xk7i)](https://codecov.io/github/cswartzvi/daglite)

A lightweight, type-safe Python framework for building and executing DAGs (Directed Acyclic Graphs) with explicit data flow and composable operations.

**[📚 Documentation](https://cswartzvi.github.io/daglite/)** | **[🚀 Getting Started](https://cswartzvi.github.io/daglite/getting-started/)** | **[💡 Examples](https://cswartzvi.github.io/daglite/examples/)**

---

> [!WARNING]
> This project is in early development. The API may change in future releases. Feedback and contributions are welcome!

## Quick Start

### Installation

```bash
uv pip install daglite

# With CLI support
uv pip install daglite[cli]
```

### Your First DAG

```python
from daglite import task, evaluate

@task
def fetch_data(url: str) -> dict:
    """Fetch data from an API."""
    return {"url": url, "data": [...]}

@task
def process(data: dict) -> list:
    """Process the fetched data."""
    return [item.upper() for item in data["data"]]

@task
def save(items: list, path: str) -> None:
    """Save results to a file."""
    with open(path, "w") as f:
        f.write("\n".join(items))

# Build and execute the DAG
result = evaluate(
    fetch_data(url="https://api.example.com")
    .then(process)
    .then(save, path="output.txt")
)
```

---

## Why Daglite?

**Daglite is built for computational work in restricted environments.**

Originally designed for operations research analysts working on air-gapped, Windows-only systems, Daglite solves a specific problem: building workflows that are easy to analyze, share with colleagues, and re-run—even after returning to a project months later.

### The Core Philosophy

**No infrastructure required.** Daglite runs anywhere Python runs—no databases, no containers, no cloud services, no servers. Install it with `uv pip`, define your tasks, and execute them. When you need more (like distributed execution or advanced serialization), plugins extend functionality without adding mandatory dependencies.

**Explicit over implicit.** Every data dependency is visible in your code. The DAG structure is static and analyzable before execution. Type checkers catch errors before runtime. This makes workflows self-documenting and maintainable.

**Type-safe and modular.** Full support for `mypy`, `pyright`, and other type checkers means your IDE provides autocomplete and catches type mismatches. Compose simple functions into complex pipelines using familiar Python patterns.

### When to Use Daglite

**Perfect for:**
- ETL scripts and data transformations
- Machine learning pipelines (feature engineering, training, evaluation)
- Computational science workflows
- Analysts and data scientists who need reproducible workflows
- Air-gapped or restricted environments
- CLI tools with workflow orchestration
- Local development and prototyping
- Projects where simplicity and type safety matter

**Not ideal for:**
- Production job scheduling with cron-like triggers → Use [Airflow](https://airflow.apache.org/), [Prefect](https://www.prefect.io/)
- Real-time streaming data → Use Kafka, Flink
- Distributed computing at massive scale → Use Spark, Dask
- Multi-tenant orchestration platforms → Use [Dagster](https://dagster.io/)

Daglite complements these excellent tools. Think of it like Flask vs Django—we give you ownership of the toolchain for local, explicit workflows, while respecting the power and sophistication of infrastructure-heavy frameworks for production orchestration.

---

## Key Features

- **Type-Safe Task Composition** - Complete type checking support with `mypy`, `pyright`, `pyrefly`, and `ty`. Your IDE catches errors before runtime.

- **Fluent API** - Chain operations naturally with `.then()`, `.map()`, `.join()`. Build complex pipelines with readable code.

- **Lightweight Core** - No mandatory infrastructure—runs anywhere Python runs. Optional plugins add capabilities when you need them.

- **Async Execution** - Built-in support for threading and multiprocessing backends. Run tasks in parallel without changing your code structure.

- **Composable Patterns** - Mix and match patterns: Inline pipelines, fan-out/fan-in, map-reduce, parameter sweeps, pairwise operations.

- **Testable** - Pure functions make DAGs easy to test and debug. No mocking infrastructure or database connections.

- **CLI Support** - Define pipelines once, run them from the command line with argument parsing included.

---

## Core Concepts

### Tasks

Functions decorated with `@task` become composable DAG nodes:

```python
@task
def process_data(input: str, param: int = 10) -> dict:
    """Tasks are just functions with explicit inputs/outputs."""
    return {"result": input * param}
```

### Lazy Evaluation

Tasks return futures—they don't execute until you call `evaluate()`:

```python
# Create a future (lazy evaluation)
future = process_data(input="hello", param=5)

# Execute when ready
result = evaluate(future)
```

### Composition Patterns

| Pattern | Method | Use Case |
|---------|--------|----------|
| Inline | `()` + `.then()` | Chain dependent operations |
| Cartesian | `.product()` | Parameter sweeps, all combinations |
| Pairwise | `.zip()` | Element-wise operations |
| Transform | `.map()` | Apply function to each element |
| Reduce | `.join()` | Aggregate sequence to single value |
| Partial | `.partial()` | Fix parameters, reuse tasks |

---

## Common Patterns

### Inline Pipeline

```python
@task
def load_config(path: str) -> dict:
    return json.load(open(path))

@task
def init_model(config: dict) -> Model:
    return Model(**config)

@task
def train(model: Model, data: pd.DataFrame) -> Model:
    model.fit(data)
    return model

# Chain operations
result = evaluate(
    load_config(path="config.json")
    .then(init_model)
    .then(train, data=training_data)
)
```

### Parallel Fan-Out

```python
@task
def fetch_user(user_id: int) -> dict:
    return api.get(f"/users/{user_id}")

@task
def save_all(users: list[dict]) -> None:
    db.bulk_insert(users)

# Process multiple users in parallel
result = evaluate(
    fetch_user.product(user_id=[1, 2, 3, 4, 5])
    .join(save_all)
)
```

### Map-Reduce

```python
@task
def square(x: int) -> int:
    return x ** 2

@task
def double(x: int) -> int:
    return x * 2

@task
def sum_all(values: list[int]) -> int:
    return sum(values)

# Fan-out, transform, reduce
result = evaluate(
    square.product(x=[1, 2, 3, 4])
    .map(double)
    .join(sum_all)
)
# Result: 60 = (2 + 8 + 18 + 32)
```

### Async Execution

```python
# Run DAG with threading backend
result = evaluate(my_dag, use_async=True)

# Per-task backends
@task(backend_name="threading")
def io_bound_task(url: str) -> bytes:
    return requests.get(url).content

@task(backend_name="multiprocessing")
def cpu_bound_task(data: np.ndarray) -> np.ndarray:
    return expensive_computation(data)
```

### CLI Pipelines

```python
from daglite import pipeline

@pipeline
def ml_pipeline(model_path: str, data_path: str, epochs: int = 10):
    """Train a machine learning model."""
    data = load_data(path=data_path)
    model = train_model(data=data, epochs=epochs)
    return save_model(model=model, path=model_path)
```

Run from command line:

```bash
daglite run ml_pipeline --model-path model.pkl --data-path data.csv --epochs 20
```

---

## Documentation

Full documentation is available at **[cswartzvi.github.io/daglite](https://cswartzvi.github.io/daglite/)**

- [Getting Started Guide](https://cswartzvi.github.io/daglite/getting-started/)
- [User Guide](https://cswartzvi.github.io/daglite/user-guide/tasks/)
- [Plugins](https://cswartzvi.github.io/daglite/plugins/)
- [API Reference](https://cswartzvi.github.io/daglite/api-reference/)
- [Examples](https://cswartzvi.github.io/daglite/examples/)

---

## Community

### 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### 💬 Discussions

Join the conversation on [GitHub Discussions](https://github.com/cswartzvi/daglite/discussions).

### 🐛 Issues

Found a bug or have a feature request? [Open an issue](https://github.com/cswartzvi/daglite/issues).

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

Inspired by the design patterns and philosophies of:

- [Apache Airflow](https://airflow.apache.org/) - DAG orchestration at scale
- [Prefect](https://www.prefect.io/) - Modern workflow design
- [Dagster](https://dagster.io/) - Data pipeline architecture
- [Dask](https://dask.org/) - Lazy evaluation and graph execution
- [itertools](https://docs.python.org/3/library/itertools.html) - Composable Python operations

Each of these projects excels in their domain. Daglite aims to complement them by providing a lightweight alternative for local, type-safe workflows.
