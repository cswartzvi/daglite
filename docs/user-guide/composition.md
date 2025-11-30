# Composition Patterns

Daglite provides powerful composition patterns to build complex workflows from simple tasks. This page documents all the available composition methods.

## Basic Composition

### `.bind()` - Basic Task Binding

Bind parameters to create a task future:

```python
from daglite import task, evaluate

@task
def add(x: int, y: int) -> int:
    return x + y

result = evaluate(add.bind(x=1, y=2))
# Result: 3
```

### `.then()` - Sequential Chaining

Chain tasks sequentially, passing the output of one as input to the next:

```python
@task
def double(x: int) -> int:
    return x * 2

@task
def add_ten(x: int) -> int:
    return x + 10

result = evaluate(
    double.bind(x=5).then(add_ten)
)
# Result: 20 (5 * 2 + 10)
```

### `.fix()` - Partial Parameter Application

Fix some parameters of a task for reuse:

```python
@task
def scale(x: int, factor: int, offset: int) -> int:
    return x * factor + offset

# Create a specialized version with fixed parameters
normalize = scale.fix(factor=2, offset=10)

result1 = evaluate(normalize.bind(x=5))  # 5 * 2 + 10 = 20
result2 = evaluate(normalize.bind(x=10))  # 10 * 2 + 10 = 30
```

## Fan-Out Patterns

### `.product()` - Cartesian Product

Execute a task for all combinations of parameter values:

```python
@task
def multiply(x: int, y: int) -> int:
    return x * y

result = evaluate(
    multiply.product(x=[1, 2, 3], y=[10, 20])
)
# Result: [10, 20, 20, 40, 30, 60]
# All combinations: (1,10), (1,20), (2,10), (2,20), (3,10), (3,20)
```

### `.zip()` - Pairwise Operations

Execute a task for paired elements from sequences:

```python
@task
def multiply(x: int, y: int) -> int:
    return x * y

result = evaluate(
    multiply.zip(x=[1, 2, 3], y=[10, 20, 30])
)
# Result: [10, 40, 90]
# Pairs: (1,10), (2,20), (3,30)
```

## Map-Reduce Patterns

### `.map()` - Transform Sequences

Apply a transformation to each element of a sequence:

```python
@task
def identity(x: int) -> int:
    return x

@task
def square(x: int) -> int:
    return x ** 2

result = evaluate(
    identity.product(x=[1, 2, 3, 4]).map(square)
)
# Result: [1, 4, 9, 16]
```

### `.join()` - Reduce to Single Value

Aggregate a sequence into a single value:

```python
@task
def sum_all(values: list[int]) -> int:
    return sum(values)

result = evaluate(
    identity.product(x=[1, 2, 3, 4])
        .map(square)
        .join(sum_all)
)
# Result: 30 (1 + 4 + 9 + 16)
```

## Conditional Composition

### `.choose()` - If/Else Branching

Execute different tasks based on a condition:

```python
@task
def get_value(x: int) -> int:
    return x

@task
def positive_handler(x: int) -> str:
    return f"positive: {x}"

@task
def negative_handler(x: int) -> str:
    return f"negative: {x}"

result = evaluate(
    get_value.bind(x=5).choose(
        condition=lambda x: x > 0,
        if_true=positive_handler,
        if_false=negative_handler
    )
)
# Result: "positive: 5"
```

### `.switch()` - Multi-Way Branching

Select from multiple tasks based on a key:

```python
@task
def get_status_code(response: dict) -> int:
    return response["code"]

@task
def handle_success(code: int) -> str:
    return f"Success: {code}"

@task
def handle_error(code: int) -> str:
    return f"Error: {code}"

@task
def handle_redirect(code: int) -> str:
    return f"Redirect: {code}"

result = evaluate(
    get_status_code.bind(response={"code": 200}).switch(
        {200: handle_success, 404: handle_error, 301: handle_redirect},
        key=lambda code: code  # Optional: transform the input value
    )
)
# Result: "Success: 200"
```

With a key function to classify inputs:

```python
@task
def get_value(x: int) -> int:
    return x

@task
def handle_small(x: int) -> str:
    return f"small: {x}"

@task
def handle_large(x: int) -> str:
    return f"large: {x}"

result = evaluate(
    get_value.bind(x=50).switch(
        {"small": handle_small, "large": handle_large},
        key=lambda x: "small" if x < 10 else "large"
    )
)
# Result: "large: 50"
```

## Loop Patterns

### `.while_loop()` - Conditional Iteration

Repeatedly execute a task while a condition is true:

```python
@task
def start(x: int) -> int:
    return x

@task
def decrement(x: int) -> int:
    return x - 1

result = evaluate(
    start.bind(x=10).while_loop(
        condition=lambda x: x > 0,
        body=decrement,
        max_iterations=1000  # Safety limit
    )
)
# Result: 0 (counts down from 10 to 0)
```

Accumulator pattern:

```python
@task
def init_state() -> dict:
    return {"sum": 0, "count": 0}

@task
def accumulate(state: dict, increment: int) -> dict:
    return {
        "sum": state["sum"] + increment,
        "count": state["count"] + 1
    }

result = evaluate(
    init_state.bind().while_loop(
        condition=lambda s: s["count"] < 5,
        body=accumulate,
        body_kwargs={"increment": 10}
    )
)
# Result: {"sum": 50, "count": 5}
```

## Combining Patterns

All composition patterns can be combined to create complex workflows:

```python
@task
def fetch_numbers(count: int) -> list[int]:
    return list(range(1, count + 1))

@task
def square(x: int) -> int:
    return x ** 2

@task
def classify(x: int) -> str:
    if x < 10:
        return "small"
    elif x < 100:
        return "medium"
    else:
        return "large"

@task
def handle_small(x: int) -> str:
    return f"Small value: {x}"

@task
def handle_medium(x: int) -> str:
    return f"Medium value: {x}"

@task
def handle_large(x: int) -> str:
    return f"Large value: {x}"

# Complex pipeline: fetch -> square each -> classify -> handle each type
result = evaluate(
    fetch_numbers.bind(count=5)
        .then(lambda nums: square.product(x=nums))
        .map(lambda x: x)  # Identity to get each squared number
        .map(lambda x: x)  # Can chain operations
        .choose(
            condition=lambda x: x > 50,
            if_true=handle_large,
            if_false=handle_small
        )
)
```

## Best Practices

1. **Keep tasks pure**: Tasks should not have side effects unless necessary
2. **Use descriptive names**: Task names help understand the workflow
3. **Validate early**: Use conditions and checks at the start of workflows
4. **Set safety limits**: Always set `max_iterations` for while loops
5. **Compose incrementally**: Build complex workflows from simple pieces
6. **Test individual tasks**: Test tasks independently before composing
