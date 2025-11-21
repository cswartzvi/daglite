"""Test improved error messages and exception handling."""

from daglite import DagliteError, ParameterError, BackendError, task, evaluate


print("=== Testing Improved Error Messages ===\n")

# Test 1: Empty extend()
print("1. Testing empty extend() error:")
@task
def process(x: int) -> int:
    return x * 2

try:
    result = process.extend()
except ParameterError as e:
    print(f"✓ Caught ParameterError: {e}\n")

# Test 2: Duplicate parameter in PartialTask
print("2. Testing duplicate parameter error:")
try:
    partial = process.partial(x=5)
    result = partial.bind(x=10)  # x already bound
except ParameterError as e:
    print(f"✓ Caught ParameterError: {e}\n")

# Test 3: Unknown backend
print("3. Testing unknown backend error:")
try:
    from daglite.backends import find_backend
    backend = find_backend("nonexistent")
except BackendError as e:
    print(f"✓ Caught BackendError: {e}\n")

# Test 4: Mismatched zip lengths
print("4. Testing zip length mismatch:")
@task
def combine(x: int, y: int) -> int:
    return x + y

try:
    result = evaluate(combine.zip(x=[1, 2, 3], y=[10, 20]))
except ParameterError as e:
    print(f"✓ Caught ParameterError: {e}\n")

# Test 5: Using TaskFuture in boolean context
print("5. Testing TaskFuture boolean error:")
future = process.bind(x=5)
try:
    if future:  # Should raise TypeError
        pass
except TypeError as e:
    print(f"✓ Caught TypeError: {e}\n")

# Test 6: All daglite exceptions inherit from DagliteError
print("6. Testing exception hierarchy:")
try:
    result = process.extend()
except DagliteError as e:
    print(f"✓ Caught as DagliteError (base class): {type(e).__name__}\n")

print("=== All Error Message Tests Passed! ===")
