from daglite import evaluate
from daglite import task


class TestSinglePathExecution:
    """Tests engine evaluation of single path tasks."""


    def test_single_task_evaluation_empty(self) -> None:
        """Evaluation succeeds for tasks without parameters."""

        @task
        def dummy() -> None:
            return

        dummies = dummy.bind()
        result = evaluate(dummies)
        assert result is None

    def test_single_task_evaluation_without_params(self) -> None:
        """Evaluation succeeds for parameterless tasks."""

        @task
        def prepare() -> int:
            return 5

        prepared = prepare.bind()
        result = evaluate(prepared)
        assert result == 5

    def test_single_task_evaluation_with_params(self) -> None:
        """Evaluation succeeds when parameters are provided."""

        @task
        def multiply(x: int, y: int) -> int:
            return x * y

        multiplied = multiply.bind(x=4, y=6)
        result = evaluate(multiplied)
        assert result == 24

    def test_short_chain_evaluation_without_params(self) -> None:
        """Evaluation succeeds for a chain of dependent tasks."""

        @task
        def prepare() -> int:
            return 5

        @task
        def square(z: int) -> int:
            return z**2

        prepared = prepare.bind()
        squared = square.bind(z=prepared)
        result = evaluate(squared)
        assert result == 25  # = 5 ** 2

    def test_long_chain_evaluation_with_params(self) -> None:
        """Evaluation succeeds for a longer chain of tasks with parameters."""

        @task
        def add(x: int, y: int) -> int:
            return x + y

        @task
        def multiply(z: int, factor: int) -> int:
            return z * factor

        @task
        def subtract(value: int, decrement: int) -> int:
            return value - decrement

        added = add.bind(x=3, y=7)  # 10
        multiplied1 = multiply.bind(z=added, factor=4)
        multiplied2 = multiply.bind(z=multiplied1, factor=2)  # 80
        multiplied3 = multiply.bind(z=multiplied2, factor=3)  # 240
        multiplied4 = multiply.bind(z=multiplied3, factor=1)  # 240
        subtracted = subtract.bind(value=multiplied4, decrement=40)  # 200
        result = evaluate(subtracted)
        assert result == 200

    def test_fixed_task_evaluation_without_params(self) -> None:
        """Evaluation succeeds for tasks with fixed parameters."""

        @task
        def power(base: int, exponent: int) -> int:
            return base**exponent

        powered = power.fix(exponent=3).bind(base=2)
        result = evaluate(powered)
        assert result == 8  # = 2 ** 3

    def test_fixed_task_evaluation_with_params(self) -> None:
        """Evaluation succeeds for tasks with some fixed and some provided parameters."""

        @task
        def compute_area(length: float, width: float) -> float:
            return length * width

        area_task = compute_area.fix(width=5.0)
        area = area_task.bind(length=10.0)
        result = evaluate(area)
        assert result == 50.0  # = 10.0 * 5.0

    def test_mixed_chain_evaluation_without_params(self) -> None:
        """Evaluation succeeds for a chain mixing fixed and provided parameters."""

        @task
        def add(x: int, y: int) -> int:
            return x + y

        @task
        def multiply(z: int, factor: int) -> int:
            return z * factor

        fixed_multiply = multiply.fix(factor=10)

        added = add.bind(x=2, y=3)  # 5
        multiplied = fixed_multiply.bind(z=added)  # 50
        result = evaluate(multiplied)
        assert result == 50

    def test_mixed_chain_evaluation_with_params(self) -> None:
        """Evaluation succeeds for a chain with both fixed and provided parameters."""

        @task
        def add(x: int, y: int) -> int:
            return x + y

        @task
        def multiply(z: int, factor: int) -> int:
            return z * factor

        @task
        def subtract(value: int, decrement: int) -> int:
            return value - decrement

        fixed_multiply = multiply.fix(factor=3)
        fixed_subtract = subtract.fix(decrement=4)

        added = add.bind(x=7, y=8)  # 15
        multiplied = fixed_multiply.bind(z=added)  # 45
        subtracted = fixed_subtract.bind(value=multiplied)  # 41
        result = evaluate(subtracted)
        assert result == 41

    def test_deep_chain_with_task_reuse_and_independent_branches(self) -> None:
        """Evaluation succeeds for deep chains with reused tasks and independent branches."""

        @task
        def multiply(x: int, factor: int) -> int:
            return x * factor

        # Start with 2, multiply by 2 repeatedly
        chain_depth = 2000
        current = multiply.bind(x=2, factor=2)
        for i in range(chain_depth - 1):
            current = multiply.bind(x=current, factor=2)

        # Evaluate all three independently
        result_chain = evaluate(current)

        assert result_chain == 2 * (2**chain_depth)

