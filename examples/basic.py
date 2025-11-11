"""Example pipeline demonstrating graphlite usage."""
from __future__ import annotations

from statistics import mean as stats_mean

from graphlite import Executor, conditional, fanout, task


@task
def prepare(n: int) -> list[int]:
    return list(range(1, n + 1))


@task
def square(value: int) -> int:
    return value * value


@task
def mean(values: list[int]) -> float:
    return stats_mean(values)


@task
def choose_total(values: list[int]) -> float:
    total = sum(values)
    return float(total)


@task
def take_mean(n: int) -> bool:
    return n % 2 == 0


@task
def pipeline(n: int) -> float:
    numbers = prepare(n)
    squares = fanout(square, numbers)
    return conditional(take_mean(n), mean(squares), choose_total(numbers))


def main() -> None:
    expr = pipeline(10)
    with Executor() as executor:
        result = executor.evaluate(expr)
    print(f"mean of squares 1..10 = {result}")


if __name__ == "__main__":
    main()
