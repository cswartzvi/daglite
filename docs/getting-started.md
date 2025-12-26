---
hide:
  - navigation
---

# Getting Started

This guide will help you install Daglite and build your first DAG in just a few minutes.

---

## Installation

Daglite requires Python 3.10 or higher.

### Basic Installation

```bash
pip install daglite
```

This installs the core library with **zero dependencies**.

### With CLI Support

To use the `daglite` command-line tool:

```bash
pip install daglite[cli]
```

### Development Installation

To contribute or run tests:

```bash
git clone https://github.com/cswartzvi/daglite.git
cd daglite
uv sync --all-groups
```

---

## Your First DAG

Let's build a simple data processing pipeline:

```python
from daglite import task, evaluate

# Step 1: Define tasks
@task
def load_data(path: str) -> list[int]:
    """Load numbers from a file."""
    with open(path) as f:
        return [int(line) for line in f]

@task
def square_all(numbers: list[int]) -> list[int]:
    """Square each number."""
    return [x ** 2 for x in numbers]

@task
def sum_values(numbers: list[int]) -> int:
    """Sum all numbers."""
    return sum(numbers)

# Step 2: Build the DAG
data = load_data(path="numbers.txt")
squared = square_all(numbers=data)
total = sum_values(numbers=squared)

# Step 3: Execute
result = evaluate(total)
print(f"Sum of squares: {result}")
```

### What's Happening?

1. **`@task`** - Decorates functions to make them composable
2. **Calling tasks** - Creates a "future" representing the task's eventual result
3. **`evaluate()`** - Executes the DAG and returns the final result

!!! tip "Lazy Evaluation"
    Calling a task doesn't execute it - it just builds the DAG.
    Execution only happens when you call `evaluate()`.

---

## The Fluent API

The same DAG can be written more concisely using method chaining:

```python
result = evaluate(
    load_data(path="numbers.txt")
    .then(square_all)
    .then(sum_values)
)
```

The `.then()` method automatically connects the output of one task to the input of the next.

---

## Fan-Out with `.product()`

Process multiple inputs using Cartesian product:

```python
@task
def add(x: int, y: int) -> int:
    return x + y

# Create all combinations of x and y
results = evaluate(
    add.product(x=[1, 2, 3], y=[10, 20])
)
# Result: [11, 21, 12, 22, 13, 23]
#         (1+10, 1+20, 2+10, 2+20, 3+10, 3+20)
```

---

## Map-Reduce Pattern

Transform and aggregate sequences:

```python
@task
def square(x: int) -> int:
    return x ** 2

@task
def sum_all(values: list[int]) -> int:
    return sum(values)

# Square each number, then sum them
result = evaluate(
    square.product(x=[1, 2, 3, 4])
    .join(sum_all)
)
# Result: 30 (1² + 2² + 3² + 4² = 1 + 4 + 9 + 16 = 30)
```

### With `.map()` for transformations

```python
@task
def double(x: int) -> int:
    return x * 2

result = evaluate(
    square.product(x=[1, 2, 3, 4])  # [1, 4, 9, 16]
    .map(double)                     # [2, 8, 18, 32]
    .join(sum_all)                   # 60
)
```

---

## Pairwise Operations with `.zip()`

Process sequences element-by-element:

```python
@task
def multiply(x: int, y: int) -> int:
    return x * y

# Zip sequences together (must be same length)
results = evaluate(
    multiply.zip(x=[1, 2, 3], y=[10, 20, 30])
)
# Result: [10, 40, 90]
#         (1×10, 2×20, 3×30)
```

---

## Passing Additional Parameters

You can pass extra parameters inline:

```python
@task
def scale(x: int, factor: int) -> int:
    return x * factor

# Inline parameter with .then()
result = evaluate(
    square.product(x=[1, 2, 3])
    .map(scale, factor=10)  # Pass factor inline
)
# Result: [10, 40, 90]
```

---

## Type Safety

Daglite works seamlessly with type checkers:

```python
@task
def get_number() -> int:
    return 42

@task
def double(x: int) -> int:
    return x * 2

@task
def to_string(x: int) -> str:
    return f"Result: {x}"

# Type checker knows the result is str
result: str = evaluate(
    get_number()
    .then(double)
    .then(to_string)
)
```

Your IDE will provide autocomplete and catch type errors:

```python
# ❌ Type error - expects int, got str
evaluate(double(x="hello"))
```

---

## Async Execution

Run tasks in parallel using threading or multiprocessing:

```python
import time

@task(backend="threading")
def slow_io(url: str) -> bytes:
    time.sleep(1)  # Simulate network call
    return b"data"

# Execute with async support
result = evaluate(
    slow_io.product(url=["url1", "url2", "url3"]),
    use_async=True  # Runs in parallel!
)
```

---

## Next Steps

Now that you understand the basics:

- **[Learn more about tasks](user-guide/tasks.md)** - Task decorators, options, and patterns
- **[Explore composition patterns](user-guide/composition.md)** - Calling tasks, `()`, `.product()`, `.zip()`, `.partial()`
- **[Master the fluent API](user-guide/fluent-api.md)** - `.then()`, `.map()`, `.join()`
- **[See real examples](examples.md)** - ETL pipelines, ML workflows, and more

---

## Common Patterns

### Sequential Pipeline

```python
result = (
    fetch(url=url)
    .then(parse, format="json")
    .then(validate, strict=True)
    .then(save, path="output.json")
)
```

### Parallel Processing

```python
results = (
    fetch_user.product(user_id=[1, 2, 3, 4, 5])
    .map(enrich, include_avatar=True)
    .join(save_all)
)
```

### Parameter Sweeps

```python
results = (
    train_model.product(
        lr=[0.001, 0.01, 0.1],
        batch_size=[32, 64, 128]
    )
    .map(evaluate_model, test_data=test_set)
    .join(find_best)
)
```

!!! success "You're Ready!"
    You now know enough to build useful DAGs with Daglite. Check out the [User Guide](user-guide/tasks.md) for in-depth documentation.
