import subprocess
import sys
from pathlib import Path

import pytest

# pyrefly has a known overload resolution bug with self-type narrowing when
# targeting Python 3.10 stubs (works fine on 3.11+).
_skip_pyrefly_310 = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="pyrefly overload resolution bug with Python 3.10 stubs",
)


@pytest.mark.slow
@pytest.mark.parametrize(
    "checker,subcommand",
    [
        ("pyright", None),
        ("mypy", None),
        pytest.param("pyrefly", "check", marks=_skip_pyrefly_310),
        ("ty", "check"),
    ],
)
def test_assert_types(checker: str, subcommand: str) -> None:
    """Tests results of the type checkers directly using `assert_type` calls."""
    # Common tests for all type checkers
    common_file = (Path(__file__).parent / "assert_type_all.py").as_posix()

    # Checker-specific tests (if they exist)
    checker_file = Path(__file__).parent / f"assert_type_{checker}.py"

    files = [common_file]
    if checker_file.exists():
        files.append(checker_file.as_posix())

    for file in files:
        command = [checker]
        if subcommand:
            command.append(subcommand)
        command.append(file)

        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, (
            f"Type checker {checker} failed on {file}:\n{result.stdout}\n{result.stderr}"
        )
