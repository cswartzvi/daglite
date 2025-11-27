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
    backend="threading"
)
def add(x: int, y: int) -> int:
    return x + y
```

**Options:**

- `name` - Custom task name (default: function name)
- `description` - Task description (default: docstring)
- `backend` - Execution backend: `"sequential"`, `"threading"`, or `"multiprocessing"`

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

###

 Default Parameters

Tasks can have default values:

```python
@task
def scale(x: float, factor: float = 1.0) -> float:
    return x * factor

# Use default
result = scale.bind(x=5.0)  # factor=1.0

# Override default
result = scale.bind(x=5.0, factor=2.0)  # factor=2.0
```

---

## Task Futures

When you `.bind()` a task, you get a `TaskFuture` - a lazy representation of the task's future result:

```python
@task
def compute() -> int:
    return 42

future = compute.bind()  # Returns TaskFuture[int]
print(type(future))      # <class 'daglite.futures.TaskFuture'>

result = evaluate(future)  # Executes and returns 42
```

!!! warning "Futures are lazy"
    Calling `.bind()` does **not** execute the task. It only builds the DAG.
    Execution happens when you call `evaluate()`.

---

## Partial Application with `.fix()`

Fix some parameters while leaving others unbound:

```python
@task
def power(base: int, exponent: int) -> int:
    return base ** exponent

# Create a "square" function
square = power.fix(exponent=2)

# Use it in different contexts
result1 = square.bind(base=5)           # 25
result2 = square.product(base=[1,2,3])  # [1, 4, 9]
```

This is useful for creating reusable task templates.

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
fetched = fetch.bind()
extracted = extract.bind(data=fetched)  # depends on fetched
doubled = double.bind(x=extracted)      # depends on extracted

result = evaluate(doubled)  # 84
```

---

## Execution Backends

### Sequential (Default)

Tasks run one at a time in the main thread:

```python
@task  # backend="sequential" by default
def task1() -> int:
    return 42
```

**Use when:** Tasks are fast or order matters

### Threading

Tasks run in separate threads (good for I/O-bound work):

```python
@task(backend="threading")
def fetch_url(url: str) -> bytes:
    return requests.get(url).content
```

**Use when:** I/O-bound operations (network, files)

### Multiprocessing

Tasks run in separate processes (good for CPU-bound work):

```python
@task(backend="multiprocessing")
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
    backend="threading"
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
    result = evaluate(add.bind(x=2, y=3))
    assert result == 5
```

---

## Next Steps

- **[Learn about composition patterns](composition.md)** - `.bind()`, `.product()`, `.zip()`
- **[Explore the fluent API](fluent-api.md)** - `.then()`, `.map()`, `.join()`
- **[See real examples](../examples.md)** - ETL pipelines and more
