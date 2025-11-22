import time
from threading import current_thread

from daglite import evaluate
from daglite import task
from daglite.engine import Backend


@task
def score(x: int, y: int) -> int:
    print(f"[score] x={x}, y={y}")
    return x + 2 * y


@task
def sum_list(xs: list[int]) -> int:
    print(f"[sum_list] xs={xs}")
    return sum(xs)


@task
def combine(a: int, b: int) -> int:
    print(f"[combine] a={a}, b={b}")
    return a + b


@task
def prepare(n: int) -> int:
    print(f"[prepare] n={n}")
    return n


@task
def double(x: int) -> int:
    print(f"[double] thread={current_thread().native_id} x={x}")
    time.sleep(0.5)  # simulate work
    return 2 * x


def flow(backend: str | Backend) -> None:
    initial = prepare.with_options(backend=backend).extend(n=[3, 4])  # pure fan-out
    doubled = initial.map(double)  # map over sequence
    total = doubled.join(sum_list)  # join with default param
    result = evaluate(total)
    print(f"Final result2: {result}")


print("=== Local evaluate (Engine + LocalBackend) ===")
flow(backend="sequential")

print("\n=== Threaded Engine (ThreadBackend) ===")
flow(backend="threading")
