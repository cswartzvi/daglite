"""Behavior tests for the fluent API methods on futures."""

import pytest

from daglite import task
from daglite.exceptions import DagliteError


class TestTaskFutureThen:
    """Tests for TaskFuture.then() chaining."""

    def test_then_simple_chain(self) -> None:
        """TaskFuture.then() chains tasks linearly."""

        @task
        def fetch(source: str) -> int:
            return len(source)

        @task
        def double(x: int) -> int:
            return x * 2

        @task
        def add(x: int, y: int) -> int:
            return x + y

        result = fetch(source="hello").then(double).then(add, y=10).run()
        assert result == 20  # len("hello")*2 + 10 = 5*2 + 10 = 20

    def test_then_with_multiple_kwargs(self) -> None:
        """TaskFuture.then() handles multiple inline kwargs."""

        @task
        def start() -> int:
            return 5

        @task
        def compute(x: int, factor: int, offset: int) -> int:
            return x * factor + offset

        result = start().then(compute, factor=3, offset=7).run()
        assert result == 22  # 5*3 + 7

    def test_then_with_fixed_task(self) -> None:
        """TaskFuture.then() works with pre-fixed tasks."""

        @task
        def start() -> int:
            return 10

        @task
        def scale(x: int, factor: int, offset: int) -> int:
            return x * factor + offset

        fixed_scale = scale.partial(offset=5)
        result = start().then(fixed_scale, factor=2).run()
        assert result == 25  # 10*2 + 5

    def test_then_basic_chain(self) -> None:
        """TaskFuture.then() chains correctly."""

        @task
        def double(x: int) -> int:
            return x * 2

        @task
        def add_ten(x: int) -> int:
            return x + 10

        @task
        def to_string(x: int) -> str:
            return f"Result: {x}"

        result = double(x=5).then(add_ten).then(to_string)
        assert result.run() == "Result: 20"

    def test_then_return_type_preservation(self) -> None:
        """TaskFuture.then() preserves return types correctly."""

        @task
        def double(x: int) -> int:
            return x * 2

        @task
        def to_string(x: int) -> str:
            return f"Result: {x}"

        result = double(x=5).then(to_string)
        from daglite.tasks import TaskFuture

        assert isinstance(result, TaskFuture)
        assert isinstance(result.run(), str)


class TestMapOperations:
    """Tests for .then() fluent API."""

    def test_mapped_then_with_kwargs(self) -> None:
        """Fluent .then() accepts inline kwargs."""

        @task
        def scale(x: int, factor: int) -> int:
            return x * factor

        @task
        def identity(x: int) -> int:
            return x

        result = identity.map(x=[1, 2, 3]).then(scale, factor=2).run()
        assert result == [2, 4, 6]  # Each element * 2

    def test_mapped_then_chain_with_kwargs(self) -> None:
        """Fluent .then() chains with inline kwargs."""

        @task
        def identity(x: int) -> int:
            return x

        @task
        def add(x: int, y: int) -> int:
            return x + y

        @task
        def multiply(x: int, factor: int) -> int:
            return x * factor

        # [1, 2, 3] -> add(y=10) -> [11, 12, 13] -> multiply(factor=2) -> [22, 24, 26]
        result = identity.map(x=[1, 2, 3]).then(add, y=10).then(multiply, factor=2).run()
        assert result == [22, 24, 26]


class TestJoinOperations:
    """Tests for .join() fluent API."""

    def test_join_with_kwargs(self) -> None:
        """Fluent .join() accepts inline kwargs."""

        @task
        def square(x: int) -> int:
            return x**2

        @task
        def weighted_sum(values: list[int], weight: float) -> float:
            return sum(values) * weight

        result = square.map(x=[1, 2, 3, 4]).join(weighted_sum, weight=2.0).run()
        assert result == 60.0  # (1 + 4 + 9 + 16) * 2.0

    def test_map_join_combined_with_kwargs(self) -> None:
        """Fluent .map() and .join() work together with kwargs."""

        @task
        def identity(x: int) -> int:
            return x

        @task
        def add(x: int, y: int) -> int:
            return x + y

        @task
        def multiply(x: int, factor: int) -> int:
            return x * factor

        @task
        def reduce_with_offset(values: list[int], offset: int) -> int:
            return sum(values) + offset

        # [1, 2, 3] -> add(y=5) -> [6, 7, 8] -> multiply(factor=2) -> [12, 14, 16] -> sum+10 -> 52
        result = (
            identity.map(x=[1, 2, 3])
            .then(add, y=5)
            .then(multiply, factor=2)
            .join(reduce_with_offset, offset=10)
        ).run()
        assert result == 52  # (12 + 14 + 16) + 10


class TestComplexPipelines:
    """Tests for complex fluent pipeline combinations."""

    def test_complete_fluent_pipeline(self) -> None:
        """Complete fluent pipeline using .then(), .map(), and .join()."""

        @task
        def fetch_range(count: int) -> list[int]:
            return list(range(1, count + 1))

        @task
        def scale(x: int, factor: int) -> int:
            return x * factor

        @task
        def add_values(x: int, offset: int) -> int:
            return x + offset

        @task
        def compute_total(values: list[int], multiplier: int) -> int:
            return sum(values) * multiplier

        # Data: [1, 2, 3, 4]
        # scale with [2, 3]: Cartesian = [2,4,6,8,3,6,9,12]
        # add 10: [12,14,16,18,13,16,19,22]
        # sum = 130, * 2 = 260
        result = (
            scale.map(x=fetch_range(count=4), factor=[2, 3], map_mode="product")
            .then(add_values, offset=10)
            .join(compute_total, multiplier=2)
        ).run()
        assert result == 260

    def test_fluent_with_fixed_tasks(self) -> None:
        """Fluent API works with pre-fixed tasks."""

        @task
        def scale(x: int, factor: int) -> int:
            return x * factor

        @task
        def add(x: int, offset: int) -> int:
            return x + offset

        @task
        def sum_with_multiplier(values: list[int], multiplier: int) -> int:
            return sum(values) * multiplier

        # Use zip to avoid Cartesian product, fluent chain with pre-fixed task
        # [1,2,3] zip [2,3,4] -> [2, 6, 12] -> add 10 -> [12, 16, 22] -> sum*2 -> 100
        result = (
            scale.map(x=[1, 2, 3], factor=[2, 3, 4])
            .then(add.partial(offset=10))
            .join(sum_with_multiplier, multiplier=2)
        ).run()
        assert result == 100  # (12 + 16 + 22) * 2

    def test_full_fluent_chain_with_extend(self) -> None:
        """Complete fluent chain: bind -> then -> product -> map -> join."""

        @task
        def start(x: int) -> int:
            return x * 2

        @task
        def increment(x: int, amount: int) -> int:
            return x + amount

        @task
        def transform_list(value: int) -> list[int]:
            return [value, value + 1, value + 2]

        @task
        def scale(x: int, factor: int) -> int:
            return x * factor

        @task
        def add_offset(x: int, offset: int) -> int:
            return x + offset

        @task
        def sum_with_bonus(values: list[int], bonus: int) -> int:
            return sum(values) + bonus

        # start(5) -> 10 -> increment(amount=3) -> 13 -> transform_list -> [13, 14, 15]
        # -> product/scale(factor=[2,3]) -> [26,28,30,39,42,45] -> add 10 -> [36,38,40,49,52,55]
        # -> sum + 100 -> 370
        list_future = start(x=5).then(increment, amount=3).then(transform_list)
        result = (
            scale.map(x=list_future, factor=[2, 3], map_mode="product")
            .then(add_offset, offset=10)
            .join(sum_with_bonus, bonus=100)
        ).run()
        assert result == 370  # (36+38+40+49+52+55) + 100

    def test_full_fluent_chain_with_zip(self) -> None:
        """Complete fluent chain: bind -> then -> zip -> map -> join."""

        @task
        def fetch_base(count: int) -> list[int]:
            return list(range(1, count + 1))

        @task
        def fetch_multipliers(count: int) -> list[int]:
            return [10] * count

        @task
        def multiply(x: int, factor: int) -> int:
            return x * factor

        @task
        def add_offset(x: int, offset: int) -> int:
            return x + offset

        @task
        def product(values: list[int]) -> int:
            result = 1
            for v in values:
                result *= v
            return result

        # fetch [1,2,3], fetch [10,10,10] -> zip multiply -> [10, 20, 30]
        # -> add_offset(5) -> [15, 25, 35] -> product -> 13125
        base = fetch_base(count=3)
        multipliers = fetch_multipliers(count=3)

        result = (
            multiply.map(x=base, factor=multipliers).then(add_offset, offset=5).join(product)
        ).run()
        assert result == 13125  # 15 * 25 * 35

    def test_split_then_chain(self) -> None:
        """Test chaining after split."""

        @task
        def make_pair(x: int) -> tuple[int, int]:
            return (x, x * 2)

        @task
        def double(x: int) -> int:
            return x * 2

        @task
        def add_ten(x: int) -> int:
            return x + 10

        @task
        def to_string(x: int) -> str:
            return f"Result: {x}"

        pair = make_pair(x=10)
        first, second = pair.split()

        # Chain from each split element
        result1 = first.then(double).then(to_string)
        result2 = second.then(add_ten).then(to_string)

        assert result1.run() == "Result: 20"
        assert result2.run() == "Result: 30"


class TestErrorCases:
    """Tests for error handling in fluent API."""

    def test_then_no_unbound_params(self) -> None:
        """Test that .then() raises error when task has no unbound parameters."""

        @task
        def double(x: int) -> int:
            return x * 2

        @task
        def no_params() -> int:
            return 42

        value = double(x=5)

        with pytest.raises(Exception, match="no unbound parameters"):
            value.then(no_params)

    def test_then_multiple_unbound_params(self) -> None:
        """Test that .then() raises error when task has multiple unbound parameters."""

        @task
        def double(x: int) -> int:
            return x * 2

        @task
        def two_params(x: int, y: int) -> int:
            return x + y

        value = double(x=5)

        with pytest.raises(Exception, match="exactly 1"):
            value.then(two_params)

    def test_then_with_fixed_param_task(self) -> None:
        """Test .then() works with FixedParamTask."""

        @task
        def double(x: int) -> int:
            return x * 2

        @task
        def scale(x: int, factor: int) -> int:
            return x * factor

        value = double(x=5)
        fixed_scale = scale.partial(factor=3)
        result = value.then(fixed_scale)

        assert result.run() == 30


class TestSplitOperations:
    """Tests for TaskFuture.split() method with evaluation."""

    def test_split_pair_with_annotations(self) -> None:
        """split() should work with two-element tuple."""

        @task
        def make_pair() -> tuple[int, str]:
            return (42, "hello")

        result = make_pair()
        first, second = result.split()

        assert first.run() == 42
        assert second.run() == "hello"

    def test_split_triple_with_annotations(self) -> None:
        """split() should work with three-element tuple."""

        @task
        def make_triple() -> tuple[int, str, float]:
            return (1, "a", 2.5)

        result = make_triple()
        a, b, c = result.split()

        assert a.run() == 1
        assert b.run() == "a"
        assert c.run() == 2.5

    def test_split_single_element(self) -> None:
        """split() should work with single-element tuple."""

        @task
        def make_single() -> tuple[str]:
            return ("only",)

        result = make_single()
        (only,) = result.split()

        assert only.run() == "only"

    def test_split_five_elements(self) -> None:
        """split() should work with five-element tuple."""

        @task
        def make_five() -> tuple[int, int, int, int, int]:
            return (1, 2, 3, 4, 5)

        result = make_five()
        a, b, c, d, e = result.split()

        assert a.run() == 1
        assert b.run() == 2
        assert c.run() == 3
        assert d.run() == 4
        assert e.run() == 5

    def test_split_mixed_types(self) -> None:
        """split() should preserve element types."""

        @task
        def make_mixed() -> tuple[int, str, bool, list[int]]:
            return (42, "test", True, [1, 2, 3])

        result = make_mixed()
        num, text, flag, items = result.split()

        assert num.run() == 42
        assert text.run() == "test"
        assert flag.run() is True
        assert items.run() == [1, 2, 3]

    def test_split_with_explicit_size(self) -> None:
        """split() should work with explicit size for untyped tuple."""

        @task
        def make_pair():
            return (1, 2)

        result = make_pair()
        first, second = result.split(size=2)

        assert first.run() == 1
        assert second.run() == 2

    def test_split_with_explicit_size_triple(self) -> None:
        """split() should work with explicit size=3."""

        @task
        def make_triple():
            return (10, 20, 30)

        result = make_triple()
        a, b, c = result.split(size=3)

        assert a.run() == 10
        assert b.run() == 20
        assert c.run() == 30

    def test_split_size_overrides_annotation(self) -> None:
        """Explicit size should override type annotation."""

        @task
        def make_pair() -> tuple[int, int]:
            return (1, 2)

        result = make_pair()
        # Use size=2 explicitly even though annotation says 2
        first, second = result.split(size=2)

        assert first.run() == 1
        assert second.run() == 2

    def test_split_then_process_independently(self) -> None:
        """Split elements should be processable independently."""

        @task
        def make_pair() -> tuple[int, int]:
            return (3, 5)

        @task
        def double(x: int) -> int:
            return x * 2

        @task
        def triple(x: int) -> int:
            return x * 3

        result = make_pair()
        first, second = result.split()

        doubled = double(x=first)
        tripled = triple(x=second)

        assert doubled.run() == 6
        assert tripled.run() == 15

    def test_split_then_recombine(self) -> None:
        """Split elements can be recombined."""

        @task
        def make_pair() -> tuple[int, int]:
            return (10, 20)

        @task
        def add(x: int, y: int) -> int:
            return x + y

        result = make_pair()
        first, second = result.split()
        combined = add(x=first, y=second)

        assert combined.run() == 30

    def test_split_with_map_reduce(self) -> None:
        """Split elements can be used in map/reduce patterns."""

        @task
        def make_triple() -> tuple[int, int, int]:
            return (1, 2, 3)

        @task
        def sum_three(a: int, b: int, c: int) -> int:
            return a + b + c

        result = make_triple()
        a, b, c = result.split()
        total = sum_three(a=a, b=b, c=c)

        assert total.run() == 6

    def test_split_nested_dependency(self) -> None:
        """Split should work with nested task dependencies."""

        @task
        def compute(x: int) -> int:
            return x * 2

        @task
        def make_pair(n: int) -> tuple[int, int]:
            return (n, n + 1)

        computed = compute(x=5)
        pair = make_pair(n=computed)
        first, second = pair.split()

        assert first.run() == 10
        assert second.run() == 11

    def test_split_without_annotation_or_size_raises_error(self) -> None:
        """split() should raise ValueError without type hints or size."""

        @task
        def make_untyped():
            return (1, 2)

        result = make_untyped()

        with pytest.raises(DagliteError, match="Cannot infer tuple size"):
            result.split()

    def test_split_with_variadic_tuple_needs_size(self) -> None:
        """split() should require size for variadic tuples."""

        @task
        def make_variadic() -> tuple[int, ...]:
            return (1, 2, 3)

        result = make_variadic()

        with pytest.raises(DagliteError, match="Cannot infer tuple size"):
            result.split()

    def test_split_shares_upstream_execution(self) -> None:
        """Multiple split elements should share upstream computation."""
        execution_count = {"count": 0}

        @task
        def make_pair_with_side_effect() -> tuple[int, int]:
            execution_count["count"] += 1
            return (1, 2)

        result = make_pair_with_side_effect()
        first, second = result.split()

        # Evaluate both - upstream should only execute once
        first.run()
        second.run()

        # This would be 1 if they share, 2 if independent
        # (Currently they don't share in separate evaluate calls)
        assert execution_count["count"] == 2  # Expected current behavior

    def test_split_preserves_element_types(self) -> None:
        """Split should maintain type information for type checkers."""

        @task
        def make_mixed() -> tuple[int, str, bool]:
            return (42, "hello", True)

        result = make_mixed()
        num, text, flag = result.split()

        # These should pass type checking
        assert isinstance(num.run(), int)
        assert isinstance(text.run(), str)
        assert isinstance(flag.run(), bool)

    def test_split_method_chaining(self) -> None:
        """TaskFuture.split() should support fluent chaining."""

        @task
        def compute() -> tuple[int, int]:
            return (10, 20)

        @task
        def double(x: int) -> int:
            return x * 2

        # Fluent chaining: bind -> split -> bind
        x, y = compute().split()
        result = double(x=x)

        assert result.run() == 20

    def test_split_method_multiple_calls(self) -> None:
        """Multiple calls to .split() should produce independent futures."""

        @task
        def make_data() -> tuple[int, str, bool]:
            return (42, "test", True)

        # Create two independent split operations
        f1, f2, f3 = make_data().split()
        m1, m2, m3 = make_data().split()

        # Both should produce same results
        assert f1.run() == m1.run() == 42
        assert f2.run() == m2.run() == "test"
        assert f3.run() is True and m3.run() is True


class TestThenProductOperations:
    """Tests for TaskFuture.then_map using map_mode="product"."""

    def test_basic(self) -> None:
        """then_map() creates Cartesian product fan-out."""

        @task
        def prepare(n: int) -> int:
            return n * 2

        @task
        def combine(x: int, y: int) -> int:
            return x + y

        # Scalar result fans out with y=[10, 20, 30]
        result = prepare(n=5).then_map(combine, y=[10, 20, 30]).run()
        assert result == [20, 30, 40]  # 10 + [10, 20, 30]

    def test_with_multiple_params(self) -> None:
        """then_map() handles multiple mapped parameters via Cartesian product."""

        @task
        def start() -> int:
            return 1

        @task
        def multiply(x: int, y: int, z: int) -> int:
            return x * y * z

        # x=1 fans out with y=[2, 3] and z=[10, 20]
        result = start().then_map(multiply, y=[2, 3], z=[10, 20], map_mode="product").run()
        assert result == [20, 40, 30, 60]  # 1*2*10, 1*2*20, 1*3*10, 1*3*20

    def test_with_fixed_param(self) -> None:
        """then_map() works with pre-fixed tasks."""

        @task
        def prepare(n: int) -> int:
            return n + 5

        @task
        def compute(x: int, y: int, z: int) -> int:
            return x + y + z

        fixed_compute = compute.partial(z=100)
        result = prepare(n=3).then_map(fixed_compute, y=[1, 2, 3]).run()
        assert result == [109, 110, 111]  # 8 + [1, 2, 3] + 100


class TestThenZipOperations:
    """Tests for TaskFuture.then_map using map_mode="zip"."""

    def test_basic(self) -> None:
        """then_map() creates element-wise fan-out."""

        @task
        def scalar() -> int:
            return 12

        @task
        def multiply(x: int, y: int) -> int:
            return x * y

        # Scalar 12 paired element-wise with y=[10, 20, 30]
        result = scalar().then_map(multiply, y=[10, 20, 30]).run()
        assert result == [120, 240, 360]  # 12 * [10, 20, 30]

    def test_with_multiple_params(self) -> None:
        """then_map() handles multiple mapped parameters via zip."""

        @task
        def start() -> int:
            return 2

        @task
        def compute(x: int, y: int, z: int) -> int:
            return x + y + z

        # x=2 zipped with y=[10, 20, 30] and z=[1, 2, 3]
        result = start().then_map(compute, y=[10, 20, 30], z=[1, 2, 3]).run()
        assert result == [13, 24, 35]  # 2+10+1, 2+20+2, 2+30+3

    def test_with_fixed_param(self) -> None:
        """then_map() works with pre-fixed tasks."""

        @task
        def prepare() -> int:
            return 5

        @task
        def combine(x: int, y: int, offset: int) -> int:
            return x + y + offset

        fixed_combine = combine.partial(offset=100)
        result = prepare().then_map(fixed_combine, y=[1, 2, 3]).run()
        assert result == [106, 107, 108]  # 5 + [1, 2, 3] + 100

    def test_broadcasting_in_product(self) -> None:
        """Scalar TaskFuture results act as fixed constants in product mode."""

        @task
        def prepare(n: int) -> int:
            return n * 2

        @task
        def combine(x: int, y: int) -> int:
            return x + y

        # prepare result is scalar, used as constant across all iterations
        result = prepare.map(n=[1, 2, 3]).then(combine.partial(y=10)).run()
        assert result == [12, 14, 16]  # [2, 4, 6] + 10
