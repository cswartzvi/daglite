"""
Example: lazy generator pipeline with .iter().map().reduce().

This script demonstrates the iter() feature, which allows a generator-producing task
to be consumed lazily — one item at a time — without materialising the full sequence.

Comparison with the standard approach
--------------------------------------
Standard map().reduce() path:
  1. generate_numbers runs and ALL values are materialised to a list.
  2. N separate double() submissions are made to the thread pool.
  3. reduce_sum runs on the coordinator (main event-loop thread).

Lazy iter() path:
  1. A SINGLE submission is made to the thread pool.
  2. Inside that submission: generate_numbers yields one item at a time, each is
     immediately passed to double() on the same thread, then reduce_sum folds it in.
  3. No intermediate list is ever created; all three functions run in one worker thread.
"""

from threading import current_thread
from typing import Iterator

from daglite import task


@task(backend_name="threading")
def generate_numbers(n: int) -> Iterator[int]:
    """Yield integers 0 .. n-1 lazily."""
    for i in range(n):
        print(f"[{current_thread().name}] generating: {i}")
        yield i


@task(backend_name="threading")
def double(x: int) -> int:
    result = x * 2
    print(f"[{current_thread().name}] double({x}) = {result}")
    return result


@task
def reduce_sum(acc: int, item: int) -> int:
    result = acc + item
    print(f"[{current_thread().name}] reduce_sum({acc}, {item}) = {result}")
    return result


if __name__ == "__main__":
    print("=== Standard map().reduce() (generator materialised, reduce on main thread) ===")
    standard_future = double.map(x=generate_numbers(n=5)).reduce(reduce_sum, initial=0)
    standard_result = standard_future.run()
    print(f"Standard result: {standard_result}\n")

    print("=== Lazy iter().map().reduce() (single worker submission, reduce off main thread) ===")
    lazy_future = generate_numbers(n=5).iter().map(double).reduce(reduce_sum, initial=0)
    lazy_result = lazy_future.run()
    print(f"Lazy result: {lazy_result}\n")

    assert standard_result == lazy_result == 0 + 2 + 4 + 6 + 8, "Results must match!"
    print("Both approaches produce the same result.")
