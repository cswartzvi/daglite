"""Unit tests for TaskFuture.split() method construction and validation only.
Evaluation tests for split() are located in tests/evaluation/test_fluent_api.py.
"""

import pytest

from daglite import task
from daglite.exceptions import DagliteError
from daglite.tasks import TaskFuture


class TestSplitMethod:
    """Tests for TaskFuture.split() method construction."""

    def test_split_method_with_annotations(self) -> None:
        """TaskFuture.split() method should work with type annotations."""

        @task
        def make_pair() -> tuple[int, str]:
            return (1, "a")

        futures = make_pair.bind().split()

        assert len(futures) == 2
        assert all(isinstance(f, TaskFuture) for f in futures)

    def test_split_method_with_size_parameter(self) -> None:
        """TaskFuture.split() method should accept explicit size."""

        @task
        def make_triple():
            return (1, 2, 3)

        futures = make_triple.bind().split(size=3)

        assert len(futures) == 3
        assert all(isinstance(f, TaskFuture) for f in futures)

    def test_split_method_raises_without_size(self) -> None:
        """TaskFuture.split() method should raise when size cannot be inferred."""

        @task
        def make_untyped():
            return (1, 2, 3)

        with pytest.raises(DagliteError, match="Cannot infer tuple size"):
            make_untyped.bind().split()

    def test_split_method_with_large_tuple(self) -> None:
        """TaskFuture.split() should handle larger tuples."""

        @task
        def make_five() -> tuple[int, int, int, int, int]:
            return (1, 2, 3, 4, 5)

        futures = make_five.bind().split()

        assert len(futures) == 5
        assert all(isinstance(f, TaskFuture) for f in futures)
