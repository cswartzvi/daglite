# Daglite

[![PyPI](https://img.shields.io/pypi/v/daglite?label=PyPI)](https://pypi.org/project/daglite/)
[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![mypy](https://img.shields.io/badge/mypy-checked-blue)](http://mypy-lang.org/)
[![Pyright](https://img.shields.io/badge/pyright-checked-blue)](https://github.com/microsoft/pyright)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![tests](https://img.shields.io/github/actions/workflow/status/cswartzvi/daglite/testing.yaml?branch=main&label=tests&logo=github)](https://github.com/cswartzvi/daglite/actions/workflows/testing.yaml)
[![codecov](https://codecov.io/github/cswartzvi/daglite/graph/badge.svg?token=1o01x0xk7i)](https://codecov.io/github/cswartzvi/daglite)

A lightweight, type-safe Python framework for building and executing DAGs (Directed Acyclic Graphs) with explicit data flow and composable operations.

**[📚 Documentation](https://cswartzvi.github.io/daglite/)** | **[🚀 Getting Started](#quick-start)** | **[💡 Examples](#examples)**

---

> [!WARNING]
> This project is in early development. The API may change in future releases. Feedback and contributions are welcome!

## ✨ Key Features

- **🎯 Explicit & Type-Safe**: Complete type checking support with `mypy`, `pyright`, and `pyrefly`
- **🔗 Fluent API**: Chain operations naturally with `.then()`, `.map()`, `.join()`
- **📦 Zero Dependencies**: Core library has no external dependencies
- **⚡ Async Support**: Built-in async execution with threading/multiprocessing
- **🧩 Composable**: Mix and match patterns - sequential, fan-out, map-reduce
- **🔍 Testable**: Pure functions make DAGs easy to test and debug
- **📋 CLI Support**: Define pipelines and run them from the command line

---

## 🎬 Quick Start

### Installation

```bash
pip install daglite

# With CLI support
pip install daglite[cli]
```

### Basic Example

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

# Build the DAG
fetched = fetch_data.bind(url="https://api.example.com")
processed = process.bind(data=fetched)
saved = save.bind(items=processed, path="output.txt")

# Execute the DAG
evaluate(saved)
```

---

## 🌟 The Fluent API

Daglite provides two ways to compose tasks: **explicit** (shown above) and **fluent**:

### Fluent Chaining

```python
from daglite import task, evaluate

@task
def fetch(url: str) -> str:
    return requests.get(url).text

@task
def parse(html: str, selector: str) -> dict:
    return extract(html, selector)

@task
def transform(data: dict, format: str) -> str:
    return convert(data, format)

# Fluent style - chain operations naturally
result = evaluate(
    fetch.bind(url="https://example.com")
    .then(parse, selector=".content")
    .then(transform, format="json")
)
```

### Map-Reduce Patterns

```python
@task
def square(x: int) -> int:
    return x ** 2

@task
def sum_all(values: list[int]) -> int:
    return sum(values)

# Fan-out with .product() (Cartesian product)
result = evaluate(
    square.product(x=[1, 2, 3, 4])
    .map(lambda x: x * 2)  # Transform each
    .join(sum_all)          # Reduce to single value
)
# Result: 60 = (2 + 8 + 18 + 32)
```

### Pairwise Operations

```python
@task
def multiply(x: int, y: int) -> int:
    return x * y

# Zip sequences element-wise
numbers = multiply.zip(x=[1, 2, 3], y=[10, 20, 30])
result = evaluate(numbers)
# Result: [10, 40, 90]
```

---

## 💡 Examples

### Sequential Pipeline

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

# Build pipeline
model = (
    load_config.bind(path="config.json")
    .then(init_model)
    .then(train, data=training_data)
)

result = evaluate(model)
```

### Parallel Fan-Out

```python
@task
def fetch_user(user_id: int) -> dict:
    return api.get(f"/users/{user_id}")

@task
def enrich(user: dict, with_avatar: bool) -> dict:
    if with_avatar:
        user["avatar"] = fetch_avatar(user["id"])
    return user

@task
def save_all(users: list[dict]) -> None:
    db.bulk_insert(users)

# Process multiple users in parallel
result = evaluate(
    fetch_user.product(user_id=[1, 2, 3, 4, 5])
    .map(enrich, with_avatar=True)
    .join(save_all)
)
```

### Complex DAG

```python
@task
def fetch_prices(symbols: list[str]) -> pd.DataFrame:
    return yfinance.download(symbols)

@task
def calculate_returns(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    return prices.pct_change(window)

@task
def compute_correlation(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.corr()

@task
def find_pairs(corr: pd.DataFrame, threshold: float) -> list[tuple]:
    return [(i, j) for i, j in high_correlation_pairs(corr, threshold)]

# Build analytics pipeline
result = evaluate(
    fetch_prices.bind(symbols=["AAPL", "GOOGL", "MSFT"])
    .then(calculate_returns, window=20)
    .then(compute_correlation)
    .then(find_pairs, threshold=0.8)
)
```

---

## 🎯 Why Daglite?

Modern data workflows need structure, but most DAG frameworks are overkill. Airflow requires Docker and Kubernetes. Prefect needs a server. Dagster has dozens of dependencies. What if you just want to build a clean, type-safe pipeline that runs locally?

**Daglite fills this gap.**

### The Problem

Building data pipelines typically involves:
- **Manual dependency tracking** - Functions scattered across files, implicit data flow
- **No type safety** - Runtime errors from type mismatches, no IDE help
- **Heavy infrastructure** - Can't run without databases, containers, or cloud services
- **Complex APIs** - Steep learning curve just to chain a few functions

### The Solution

Daglite provides:
- **Explicit dependencies** - Clear data flow graph visible in your code
- **Complete type checking** - Catch errors before runtime, autocomplete everywhere
- **Zero infrastructure** - Pure Python, runs anywhere Python runs
- **Simple API** - If you know Python functions, you know Daglite

### When to Use Daglite

✅ **Perfect for:**
- ETL scripts and data transformations
- ML pipelines (feature engineering, training, evaluation)
- CLI tools that need workflow orchestration
- Local development and testing
- Air-gapped or restricted environments
- Projects where you want type safety and simplicity

❌ **Not ideal for:**
- Production job scheduling (use Airflow, Prefect)
- Real-time streaming (use Kafka, Flink)
- Distributed computing at scale (use Spark, Dask)
- Multi-tenant workflow orchestration (use Dagster)

Daglite is the **lightweight alternative** - maximum clarity with minimal complexity.

---

## 🏗️ Core Concepts

### Tasks

Functions decorated with `@task` become composable DAG nodes:

```python
@task
def process_data(input: str, param: int = 10) -> dict:
    """Tasks are just functions with explicit inputs/outputs."""
    return {"result": input * param}
```

### Binding & Futures

Tasks don't execute immediately - they return futures:

```python
# Create a future (lazy evaluation)
future = process_data.bind(input="hello", param=5)

# Execute when ready
result = evaluate(future)  # Returns {"result": "hellohellohellohellohello"}
```

### Composition Patterns

| Pattern | Method | Use Case |
|---------|--------|----------|
| Sequential | `.bind()` + `.then()` | Chain dependent operations |
| Cartesian | `.product()` | Parameter sweeps, all combinations |
| Pairwise | `.zip()` | Element-wise operations |
| Transform | `.map()` | Apply function to each element |
| Reduce | `.join()` | Aggregate sequence to single value |

---

## 🔧 Advanced Features

### Async Execution

```python
# Run DAG with threading backend
result = evaluate(my_dag, use_async=True)

# Custom backends
@task(backend="threading")
def io_bound_task(url: str) -> str:
    return requests.get(url).text

@task(backend="multiprocessing")
def cpu_bound_task(data: np.ndarray) -> np.ndarray:
    return expensive_computation(data)
```

### Fixed Parameters

```python
# Partially apply parameters
normalize = scale.fix(factor=100, offset=10)

# Use in different contexts
result1 = normalize.bind(x=5)           # Single value
result2 = normalize.product(x=[1,2,3])  # Multiple values
```

### CLI Pipelines

```python
from daglite import pipeline

@pipeline
def ml_pipeline(model_path: str, data_path: str, epochs: int = 10):
    """Train a machine learning model."""
    data = load_data.bind(path=data_path)
    model = train_model.bind(data=data, epochs=epochs)
    return save_model.bind(model=model, path=model_path)
```

Run from command line:

```bash
daglite run ml_pipeline --model-path model.pkl --data-path data.csv --epochs 20
```

---

## 📊 Comparison

| Feature | Daglite | Airflow | Prefect | Dagster |
|---------|---------|---------|---------|---------|
| **Lightweight** | ✅ 0 deps | ❌ Heavy | ❌ Heavy | ❌ Heavy |
| **Type Safety** | ✅ Full | ⚠️ Partial | ⚠️ Partial | ✅ Full |
| **Pure Python** | ✅ Yes | ❌ YAML/Config | ⚠️ Decorators | ⚠️ Config |
| **Static DAGs** | ✅ Yes | ❌ Dynamic | ❌ Dynamic | ✅ Yes |
| **Fluent API** | ✅ Yes | ❌ No | ⚠️ Limited | ❌ No |
| **Testing** | ✅ Simple | ⚠️ Complex | ⚠️ Complex | ⚠️ Complex |
| **Use Case** | Local/ETL | Production Orchestration | Workflows | Data Pipelines |

---

## 📚 Documentation

Full documentation is available at **[cswartzvi.github.io/daglite](https://cswartzvi.github.io/daglite/)**

- [Getting Started Guide](https://cswartzvi.github.io/daglite/getting-started/)
- [User Guide](https://cswartzvi.github.io/daglite/user-guide/tasks/)
- [API Reference](https://cswartzvi.github.io/daglite/api/)
- [Examples](https://cswartzvi.github.io/daglite/examples/)

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

Inspired by:
- [Apache Airflow](https://airflow.apache.org/) - DAG orchestration patterns
- [Prefect](https://www.prefect.io/) - Modern workflow design
- [Dask](https://dask.org/) - Lazy evaluation and graph execution
- [itertools](https://docs.python.org/3/library/itertools.html) - Composable operations

---

**Built with ❤️ for simplicity and type safety**
