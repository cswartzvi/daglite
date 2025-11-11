from __future__ import annotations

from time import sleep
from typing import Any

import pytest

from graphlite import Executor, conditional, fanout, task


@task
def prepare(n: int) -> list[int]:
    return list(range(1, n + 1))


@task
def sum_ints(values: list[int]) -> int:
    return sum(values)


@task
def square(value: int) -> int:
    sleep(0.01)
    return value * value


@task
def mean(values: list[int]) -> float:
    return sum(values) / len(values)


@task
def is_even(n: int) -> bool:
    return n % 2 == 0


@task
def choose(values: list[int]) -> int:
    return len(values)


def evaluate(expr: Any) -> Any:
    with Executor() as executor:
        return executor.evaluate(expr)


def test_simple_chain() -> None:
    result = evaluate(sum_ints(prepare(5)))
    assert result == 15


def test_fanout_mean() -> None:
    result = evaluate(mean(fanout(square, prepare(10))))
    assert result == pytest.approx(38.5)


def test_conditional() -> None:
    expr = conditional(is_even(4), sum_ints(prepare(2)), choose(prepare(3)))
    assert evaluate(expr) == 3
    expr = conditional(is_even(3), sum_ints(prepare(2)), choose(prepare(3)))
    assert evaluate(expr) == 3


def test_typing_signature_preserved() -> None:
    from typing import get_type_hints

    hints = get_type_hints(prepare)
    assert hints["n"] is int
    assert hints["return"] == list[int]


def test_parallel_fanout_runs() -> None:
    result = evaluate(fanout(square, prepare(5)))
    assert result == [n * n for n in range(1, 6)]
