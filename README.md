# Daglite

[![PyPI](https://img.shields.io/pypi/v/daglite?label=PyPI)](https://pypi.org/project/daglite/)
[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![type checked](https://img.shields.io/badge/type%20checked-mypy%2C%20pyright%2C%20pyrefly%2C%20ty-blue)](https://github.com/cswartzvi/daglite/tree/main/tests/typing)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![tests](https://img.shields.io/github/actions/workflow/status/cswartzvi/daglite/testing.yaml?branch=main&label=tests&logo=github)](https://github.com/cswartzvi/daglite/actions/workflows/testing.yaml)
[![codecov](https://codecov.io/github/cswartzvi/daglite/graph/badge.svg?token=1o01x0xk7i)](https://codecov.io/github/cswartzvi/daglite)

A lightweight, type-safe Python framework for building and executing task-based workflows with explicit data flow and composable operations.


---

> [!WARNING]
> This project is in early development. The API may change in future releases. Feedback and contributions are welcome!

## Quick Start

### Installation

```bash
uv pip install daglite  # or pip install daglite
```

### A Simple Workflow

```python
from daglite import task, workflow, save_dataset

@task
def load_data(path: str) -> list[int]:
    with open(path) as f:
        return [int(line) for line in f]

@task(cache=True)
def square_all(numbers: list[int]) -> list[int]:
    return [x ** 2 for x in numbers]

@task
def total(numbers: list[int]) -> int:
    return sum(numbers)

@workflow
def sum_of_squares(path: str) -> int:
    data = load_data(path=path)
    squared = square_all(numbers=data)
    save_dataset("squared_values", squared)
    return total(numbers=squared)

result = sum_of_squares(path="numbers.txt")
```

Tasks execute eagerly — they return real values, not futures. The `@workflow` decorator
manages the execution context automatically. `cache=True` skips recomputation when
inputs haven't changed, and `save_dataset` persists intermediate results for later
inspection or downstream use.

---

## Why Daglite?

Daglite was originally designed for operations research analysts working on air-gapped,
Windows-only systems. In these environments, heavyweight orchestration frameworks are
impractical — yet the need for reproducible, shareable workflows is just as real.
Daglite fills that gap with a simple core that stays out of your way, backed by a
plugin system that scales with your needs. The result: workflows that are easy to
understand, share with colleagues, and re-run — even after returning to a project
months later no matter what environment you're in.

### Design Principles

**No infrastructure required.** Daglite runs anywhere Python runs — no databases, containers,
cloud services, or servers are required. Install it with `uv pip`, define your tasks, and execute
them. When you need more (distributed execution, advanced serialization), plugins extend
functionality without adding mandatory dependencies.

**Explicit over implicit.** Every data dependency is visible in your code. Type checkers
catch errors before runtime. Workflows are self-documenting and maintainable.

**Type-safe.** Full support for `mypy`, `pyright`, `pyrefly`, and `ty`. Your IDE provides
autocomplete, catches type mismatches, and validates task composition at edit time.

### Use Cases

Daglite is well-suited for data transformation pipelines, machine learning workflows,
computational science, and reproducible analysis — particularly in environments where
infrastructure is limited or unavailable.

For production job scheduling, real-time streaming, or distributed computing at scale,
consider purpose-built tools like [Airflow](https://airflow.apache.org/),
[Prefect](https://www.prefect.io/), [Dagster](https://dagster.io/),
[Spark](https://spark.apache.org/), or [Dask](https://dask.org/).

---

## Key Features

- **Eager Execution:** Tasks return real values immediately. No graph construction step, no deferred evaluation.
- **Lightweight Core:** No mandatory infrastructure. Optional plugins add capabilities when needed.
- **Type-Safe Composition:** Complete type checking with `mypy`, `pyright`, `pyrefly`, and `ty`. Errors are caught before runtime.
- **Concurrent Execution:** Built-in threading and multiprocessing backends. Run tasks in parallel without changing code structure.
- **Fan-Out with `map_tasks`:** Distribute work across items with `map_tasks` and `gather_tasks` for sync and async fan-out patterns.
- **Caching:** Skip recomputation when inputs haven't changed. Configure per-task with TTL and custom hash functions.
- **Dataset Storage:** Save and load intermediate results with pluggable serialization formats.
- **Plugin System:** Lifecycle hooks are available for task execution. Backend workers can pass events to the coordinator process.
- **Testable:** Tasks are plain functions. Test them directly without mocking infrastructure.
- **CLI Integration:** Discover and run workflows from the command line.

---

## Core Concepts

### Tasks

The `@task` decorator marks a function as a composable unit of work:

```python
from daglite import task

@task
def process_data(input: str, param: int = 10) -> dict:
    return {"result": input * param}

# Tasks execute eagerly — this returns the result directly
result = process_data(input="hello", param=5)
```

### Sessions

A `session` provides a managed execution context with backend selection, caching,
dataset storage, and plugin support:

```python
from daglite import task, session, map_tasks

@task
def double(x: int) -> int:
    return x * 2

with session(backend="thread"):
    results = map_tasks(double, [1, 2, 3, 4])
    # [2, 4, 6, 8]
```

### Workflows

The `@workflow` decorator creates a named entry point that automatically manages
its own session. Workflows are also discoverable by the CLI:

```python
from daglite import task, workflow

@task
def add(x: int, y: int) -> int:
    return x + y

@task
def multiply(x: int, factor: int) -> int:
    return x * factor

@workflow
def compute(x: int, y: int, factor: int = 10) -> int:
    s = add(x=x, y=y)
    return multiply(x=s, factor=factor)

compute(2, 3)  # 50
```

### Fan-Out

Use `map_tasks` to distribute work across a backend, or `gather_tasks` for
async fan-out:

```python
from daglite import task, session, map_tasks

@task
def fetch_user(user_id: int) -> dict:
    return api.get(f"/users/{user_id}")

with session(backend="thread"):
    users = map_tasks(fetch_user, [1, 2, 3, 4, 5])
```

### CLI

Discover, describe, and run workflows from the command line:

```bash
# List available workflows in a module
daglite list myproject.workflows

# Describe a workflow's parameters
daglite describe myproject.workflows:compute

# Run a workflow with arguments
daglite run myproject.workflows:compute --x 2 --y 3

# Filesystem paths are also supported (directory must be importable from CWD)
daglite run path/to/workflows.py:compute --x 2 --y 3
```

---

## Documentation

Full documentation is available at **[cswartzvi.github.io/daglite](https://cswartzvi.github.io/daglite/)**

- [Getting Started](https://cswartzvi.github.io/daglite/getting-started/)
- [User Guide](https://cswartzvi.github.io/daglite/user-guide/tasks/)
- [Plugins](https://cswartzvi.github.io/daglite/plugins/)
- [API Reference](https://cswartzvi.github.io/daglite/api-reference/)
- [Examples](https://cswartzvi.github.io/daglite/examples/)

---

## Contributing

Contributions are welcome. Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

- [GitHub Discussions](https://github.com/cswartzvi/daglite/discussions)
- [Issue Tracker](https://github.com/cswartzvi/daglite/issues)

---

## License

MIT License: see [LICENSE](LICENSE) for details.

---

## Acknowledgments

Inspired by the design patterns of:

- [Apache Airflow](https://airflow.apache.org/): DAG orchestration at scale
- [Prefect](https://www.prefect.io/): Modern workflow automation
- [Dagster](https://dagster.io/): Data pipeline architecture
- [Dask](https://dask.org/): Parallel and distributed computing
- [itertools](https://docs.python.org/3/library/itertools.html): Composable Python operations
