import subprocess
from pathlib import Path

import pytest


@pytest.mark.slow
@pytest.mark.parametrize(
    "checker,subcommand",
    [
        ("pyright", None),
        ("mypy", None),
        ("pyrefly", "check"),
        # ("ty", "check"),  # Uncomment when ty is more stable
    ],
)
def test_assert_types(checker: str, subcommand: str) -> None:
    """Tests results of the type checkers directly using `assert_type` calls."""
    file = (Path(__file__).parent / "assert_type_tests.py").as_posix()

    command = [checker]
    if subcommand:
        command.append(subcommand)
    command.append(file)

    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0
