"""Simple test of global thread pool."""

import time

from daglite import evaluate
from daglite import task


@task(backend="threading")
def square(x: int) -> int:
    """Square a number."""
    time.sleep(0.05)
    return x * x


# Test that it works
future = square.bind(x=5)
result = evaluate(future)
print(f"Result: {result}")
print(f"Expected: 25")
assert result == 25

print("\n✓ Basic test passed!")
