import time
from threading import current_thread

from daglite import evaluate
from daglite import task
from daglite.engine import Backend

# --- Tasks --------------------------------------------------------------


@task
def inner(x: int) -> int:
    print(f"[inner]   x={x}  thread={current_thread().name}")
    time.sleep(0.5)
    return x * 10


@task
def outer(y: int, z: int) -> int:
    print(f"[outer]   y={y}, z={z}  thread={current_thread().name}")
    time.sleep(0.5)
    return y + z


@task
def sum_list(xs: list[int]) -> int:
    print(f"[sum_list] xs={xs}  thread={current_thread().name}")
    return sum(xs)


# --- Pipeline using nested extend --------------------------------------


def nested_extend_flow(backend: str | Backend) -> None:
    start_time = time.time()

    # Level 1 fan-out
    inner_seq = inner.extend(x=[1, 2, 3, 4, 5], backend=backend)

    # Level 2 fan-out over the results of Level 1 (nested extend)
    outer_seq = outer.extend(y=inner_seq, z=[100, 200, 300], backend=backend)

    # Collapse the final results
    total = sum_list.bind(xs=outer_seq)

    result = evaluate(total)

    elapsed = time.time() - start_time
    print(f"Final result {result}")
    print(f"Elapsed time: {elapsed:.2f}s")


def nested_zip_flow(backend: str | Backend) -> None:
    start_time = time.time()

    # Level 1 fan-out
    inner_seq = inner.zip(x=[1, 2, 3, 4, 5], backend=backend)

    # Level 2 fan-out over the results of Level 1 (nested zip)
    outer_seq = outer.zip(y=inner_seq, z=[100, 200, 300, 400, 500], backend=backend)

    # Collapse the final results
    total = sum_list.bind(xs=outer_seq)

    result = evaluate(total)

    elapsed = time.time() - start_time
    print(f"Final result {result}")
    print(f"Elapsed time: {elapsed:.2f}s")


# --- Run with LocalBackend ----------------------------------------------

print("=== Sequential Backend ===")
print("\n--- Nested extend flow ---")
nested_extend_flow(backend="sequential")
print("\n--- Nested zip flow ---")
nested_zip_flow(backend="sequential")


# --- Run with ThreadBackend ---------------------------------------------

print("\n=== ThreadBackend (threads) ===")
print("\n--- Nested extend flow ---")
nested_extend_flow(backend="threading")
print("\n--- Nested zip flow ---")
nested_zip_flow(backend="threading")
