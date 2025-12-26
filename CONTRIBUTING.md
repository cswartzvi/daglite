# Contributing to Daglite

Thank you for considering contributing to Daglite! This document provides guidelines and instructions for contributing.

## 🤝 Ways to Contribute

There are many ways to contribute to Daglite beyond writing code:

- **🐛 Report bugs** - Help us identify and fix issues
- **💡 Suggest features** - Share ideas for new functionality
- **📝 Improve documentation** - Fix typos, clarify explanations, add examples
- **🧪 Write tests** - Increase code coverage and reliability
- **💬 Answer questions** - Help others in discussions
- **🎨 Share use cases** - Show how you're using Daglite
- **📦 Create plugins** - Extend Daglite's functionality

## 🚀 Getting Started

### Development Setup

1. **Fork and clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/daglite.git
cd daglite
```

2. **Install dependencies with uv** (recommended)

```bash
# Install uv if you don't have it
pip install uv

# Install all development dependencies
uv sync --all-groups
```

Or with pip:

```bash
uv pip install -e ".[dev,test,docs]"
```

3. **Install pre-commit hooks**

```bash
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov --cov-report=term-missing

# Run specific test file
pytest tests/test_tasks.py

# Run specific test
pytest tests/test_tasks.py::test_task_decorator
```

### Code Quality

```bash
# Format code
ruff format src tests

# Lint code
ruff check src tests

# Type checking
mypy src
pyright src
```

Or use the convenience commands:

```bash
# Format
poe format

# Lint
poe lint

# Test with coverage
poe test-cov
```

## 📋 Pull Request Process

1. **Create a feature branch**

```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes**

- Write clear, concise code
- Add tests for new functionality
- Update documentation as needed
- Follow the existing code style

3. **Commit your changes**

```bash
git add .
git commit -m "Add feature: description of your changes"
```

We use conventional commit messages when possible:

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `test:` - Test changes
- `refactor:` - Code refactoring
- `chore:` - Maintenance tasks

4. **Push to your fork**

```bash
git push origin feature/your-feature-name
```

5. **Open a pull request**

- Provide a clear description of the changes
- Reference any related issues
- Ensure all tests pass
- Request a review

## 🐛 Reporting Bugs

Found a bug? Help us fix it by providing:

1. **A clear description** of the bug
2. **Steps to reproduce** the issue
3. **Expected behavior** vs actual behavior
4. **Environment details** (Python version, OS, Daglite version)
5. **Code example** if applicable

[Open a bug report](https://github.com/cswartzvi/daglite/issues/new)

### Example Bug Report

```markdown
**Bug Description**
Task with async function fails with TypeError

**Steps to Reproduce**
1. Define async task: `@task async def fetch(): ...`
2. Call evaluate: `evaluate(fetch())`
3. See error

**Expected Behavior**
Async task should execute successfully

**Actual Behavior**
TypeError: object dict can't be used in 'await' expression

**Environment**
- Python 3.10.5
- Daglite 0.4.0
- Ubuntu 22.04
```

## 💡 Requesting Features

Have an idea for improving Daglite? We'd love to hear it!

1. **Search existing issues** - Your idea might already be discussed
2. **Open a discussion** - For large changes, start a discussion first
3. **Describe the use case** - Help us understand why this is valuable
4. **Propose a solution** - If you have ideas for implementation

[Start a discussion](https://github.com/cswartzvi/daglite/discussions)

## 📝 Documentation Contributions

Documentation improvements are always welcome! You can:

- Fix typos or unclear explanations
- Add examples for existing features
- Write tutorials or guides
- Improve API documentation

Documentation is in the `docs/` directory and uses [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/).

### Building Documentation Locally

```bash
# Install docs dependencies
uv pip install -e ".[docs]"

# Serve docs locally
mkdocs serve

# Open http://127.0.0.1:8000 in your browser
```

## 🧪 Writing Tests

Good tests help maintain code quality and prevent regressions. When writing tests:

- **Test behavior, not implementation** - Focus on what the code does, not how
- **Use descriptive names** - Test names should explain what they verify
- **Keep tests independent** - Each test should run in isolation
- **Test edge cases** - Cover boundary conditions and error cases

### Example Test

```python
def test_task_returns_future():
    """Task decorator should return a future when called."""
    @task
    def add(x: int, y: int) -> int:
        return x + y

    future = add(x=1, y=2)

    assert isinstance(future, TaskFuture)
    assert evaluate(future) == 3
```

## 🎯 Code Style Guidelines

- **Type annotations** - Use type hints for all public APIs
- **Docstrings** - Use Google-style docstrings for public functions/classes
- **Naming** - Use descriptive names (e.g., `process_data` not `pd`)
- **Simplicity** - Prefer simple, readable code over clever solutions
- **Comments** - Explain *why*, not *what*

### Example

```python
@task
def normalize_values(data: list[float], method: str = "minmax") -> list[float]:
    """
    Normalize a list of values using the specified method.

    Args:
        data: List of numeric values to normalize.
        method: Normalization method ('minmax' or 'zscore').

    Returns:
        Normalized values as a list.

    Raises:
        ValueError: If method is not recognized.
    """
    if method == "minmax":
        min_val, max_val = min(data), max(data)
        return [(x - min_val) / (max_val - min_val) for x in data]
    elif method == "zscore":
        mean = sum(data) / len(data)
        std = (sum((x - mean) ** 2 for x in data) / len(data)) ** 0.5
        return [(x - mean) / std for x in data]
    else:
        raise ValueError(f"Unknown method: {method}")
```

## 🔧 Creating Plugins

Want to extend Daglite with a plugin? Check out our [Plugin Development Guide](https://cswartzvi.github.io/daglite/plugins/creating/).

Plugin contributions are welcome! Consider:

- **Standalone packages** - Plugins should be separate packages
- **Optional dependencies** - Use extras for plugin-specific dependencies
- **Documentation** - Include usage examples and API docs
- **Tests** - Plugins should have comprehensive tests

## 📚 Resources

### 📖 Documentation

- [Getting Started Guide](https://cswartzvi.github.io/daglite/getting-started/)
- [User Guide](https://cswartzvi.github.io/daglite/user-guide/tasks/)
- [API Reference](https://cswartzvi.github.io/daglite/api-reference/)

### 💬 Community

- [GitHub Discussions](https://github.com/cswartzvi/daglite/discussions) - Ask questions, share ideas
- [GitHub Issues](https://github.com/cswartzvi/daglite/issues) - Report bugs, request features

### 🛠️ Development Tools

- [pluggy](https://pluggy.readthedocs.io/) - Plugin system
- [pytest](https://docs.pytest.org/) - Testing framework
- [ruff](https://docs.astral.sh/ruff/) - Linting and formatting
- [mypy](https://mypy.readthedocs.io/) / [pyright](https://github.com/microsoft/pyright) - Type checking
- [uv](https://docs.astral.sh/uv/) - Fast Python package manager

## ⚖️ Code of Conduct

### Our Standards

- **Be respectful** - Treat everyone with respect and kindness
- **Be constructive** - Offer helpful feedback and suggestions
- **Be inclusive** - Welcome contributors of all backgrounds and experience levels
- **Be patient** - Remember that everyone is learning

### Unacceptable Behavior

- Harassment or discriminatory language
- Personal attacks or insults
- Trolling or inflammatory comments
- Publishing others' private information

## 📄 License

By contributing to Daglite, you agree that your contributions will be licensed under the [MIT License](LICENSE).

---

## 🙏 Thank You!

Every contribution, no matter how small, helps make Daglite better. We appreciate your time and effort!

If you have questions about contributing, feel free to:

- 💬 [Start a discussion](https://github.com/cswartzvi/daglite/discussions)
- 📧 Contact the maintainers
- 🐛 [Open an issue](https://github.com/cswartzvi/daglite/issues)

Happy coding! 🚀
