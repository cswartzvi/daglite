import time

from daglite import evaluate
from daglite import task


@task(backend="threading")
def left() -> str:
    print("Starting left task...")
    time.sleep(1)
    return "left"


@task(backend="threading")
def right() -> str:
    print("Starting right task...")
    time.sleep(1)
    return "right"


@task
def combine(a: str, b: str) -> str:
    print("Combining results...")
    return f"combine({a}, {b})"


future = combine.bind(a=left.bind(), b=right.bind())

print("=== Sequential Backend (should take ~2s) ===")
start = time.time()
result = evaluate(future)
elapsed = time.time() - start
print("Result:", result)
print(f"Elapsed time: {elapsed:.2f}s")


print("\n=== Async with Threading Backend (should take ~1s) ===")
start = time.time()
result = evaluate(future, use_async=True)
elapsed = time.time() - start
print("Result:", result)
print(f"Elapsed time: {elapsed:.2f}s")
