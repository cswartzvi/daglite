# Tasks

Tasks are the fundamental building blocks of Daglite DAGs. This guide covers everything you need to know about defining and using tasks.

---

## Defining Tasks

### Basic Task

Use the `@task` decorator to convert a function into a task:

```python
from daglite import task

@task
def add(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y
```

### Task with Options

Customize task behavior with decorator parameters:

```python
@task(
    name="custom_add",
    description="Adds two numbers with custom config",
    backend_name="threading"
)
def add(x: int, y: int) -> int:
    return x + y
```

**Options:**

- `name` - Custom task name (default: function name)
- `description` - Task description (default: docstring)
- `backend_name` - Execution backend: `"inline"`, `"threading"`, or `"multiprocessing"`
- `retries` - Number of times to retry on failure (default: 0)
- `timeout` - Maximum execution time in seconds
- `cache` - Enable result caching (default: `False`)
- `cache_ttl` - Time-to-live for cached results in seconds
- `store` - Storage backend for task results

---

## Task Signatures

### Type Annotations

Tasks should have full type annotations for best IDE support:

```python
@task
def process(data: pd.DataFrame, threshold: float = 0.5) -> dict[str, float]:
    """Process a dataframe and return statistics."""
    return {
        "mean": data["value"].mean(),
        "count": len(data[data["value"] > threshold])
    }
```

### Default Parameters

Tasks can have default values:

```python
@task
def scale(x: float, factor: float = 1.0) -> float:
    return x * factor

# Use default
result = scale(x=5.0)  # factor=1.0

# Override default
result = scale(x=5.0, factor=2.0)  # factor=2.0
```

---

## Task Futures

When you `()` a task, you get a `TaskFuture` - a lazy representation of the task's future result:

```python
@task
def compute() -> int:
    return 42

future = compute()  # Returns TaskFuture[int]
print(type(future))      # <class 'daglite.futures.TaskFuture'>

result = evaluate(future)  # Executes and returns 42
```

!!! warning "Futures are lazy"
    Calling the task does **not** execute it. It only builds the DAG.
    Execution happens when you call `evaluate()`.

---

## Partial Application with `.partial()`

Fix some parameters while leaving others unbound, creating a reusable task template:

```python
@task
def power(base: int, exponent: int) -> int:
    return base ** exponent

# Create a "square" function
square = power.partial(exponent=2)

# Use it in different contexts
result1 = square(base=5)                # 25
result2 = square.map(base=[1,2,3], map_mode="product")  # [1, 4, 9]
result3 = square(base=7)                # 49
```

This pattern is especially useful for creating reusable task templates:

```python
# Create a scorer with partially-applied parameters
scorer = score.partial(threshold=0.5, metric="accuracy")

# Reuse with different models
future1 = scorer(model=model_a)
future2 = scorer(model=model_b)
future3 = scorer(model=model_c)
```

---

## Task Composition

Tasks can depend on other tasks:

```python
@task
def fetch() -> dict:
    return {"value": 42}

@task
def extract(data: dict) -> int:
    return data["value"]

@task
def double(x: int) -> int:
    return x * 2

# Connect tasks
fetched = fetch()
extracted = extract(data=fetched)  # depends on fetched
doubled = double(x=extracted)      # depends on extracted

result = evaluate(doubled)  # 84
```

---

## Execution Backends

### Inline (Default)

Tasks run one at a time in the main thread:

```python
@task  # backend_name="inline" by default
def task1() -> int:
    return 42
```

**Use when:** Tasks are fast or order matters

### Threading

Tasks run in separate threads (good for I/O-bound work):

```python
@task(backend_name="threading")
def fetch_url(url: str) -> bytes:
    return requests.get(url).content
```

**Use when:** I/O-bound operations (network, files)

### Multiprocessing

Tasks run in separate processes (good for CPU-bound work):

```python
@task(backend_name="multiprocessing")
def compute_fft(data: np.ndarray) -> np.ndarray:
    return np.fft.fft(data)
```

**Use when:** CPU-bound operations (computation, data processing)

!!! tip "Mixing Backends"
    You can mix backends in the same DAG. Daglite will handle the coordination.

---

## Task Options

### Modifying Task Options

Change task configuration after creation:

```python
@task
def process(data: list) -> list:
    return [x * 2 for x in data]

# Create variant with different backend
process_parallel = process.with_options(
    name="process_parallel",
    backend_name="threading"
)
```

---

## Best Practices

### ✅ Do

- **Use type annotations** - Enables IDE support and type checking
- **Keep tasks pure** - Avoid side effects when possible
- **Use descriptive names** - Makes DAGs easier to understand
- **Document with docstrings** - Helps with generated documentation

```python
@task
def calculate_discount(
    price: float,
    discount_rate: float,
    minimum_price: float = 0.0
) -> float:
    """
    Calculate discounted price.

    Args:
        price: Original price
        discount_rate: Discount percentage (0.0 to 1.0)
        minimum_price: Floor price

    Returns:
        Discounted price, clamped to minimum
    """
    discounted = price * (1 - discount_rate)
    return max(discounted, minimum_price)
```

### ❌ Don't

- **Avoid global state** - Makes tasks hard to test and reason about
- **Don't use mutable defaults** - Can cause unexpected behavior
- **Avoid heavy computation in task definition** - Keep it in the task body

```python
# ❌ Bad: Global state
counter = 0

@task
def increment() -> int:
    global counter
    counter += 1
    return counter

# ✅ Good: Explicit state
@task
def increment(counter: int) -> int:
    return counter + 1
```

---

## Testing Tasks

Tasks are just functions, so they're easy to test:

```python
@task
def add(x: int, y: int) -> int:
    return x + y

def test_add():
    # Test the underlying function
    assert add.func(2, 3) == 5

    # Or evaluate the task
    result = evaluate(add(x=2, y=3))
    assert result == 5
```

---

## Next Steps

- **[Learn about composition patterns](composition.md)** - `()`, `.map()`, `.partial()`
- **[Explore the fluent API](fluent-api.md)** - `.then()`, `.then_map()`, `.join()`
- **[See real examples](../examples/index.md)** - ETL pipelines and more
