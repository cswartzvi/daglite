# Command-Line Interface

The `daglite` CLI is built in — no extra install needed. It lets you run pipelines
defined with the `@pipeline` decorator from the terminal, CI/CD systems, or
scheduled jobs.

---

## Basic Usage

### Defining a Pipeline

Use the `@pipeline` decorator to create a runnable pipeline:

```python
# myproject/pipelines.py
from daglite import task, pipeline

@task
def load_data(path: str) -> list:
    with open(path) as f:
        return [line.strip() for line in f]

@task
def process(data: list, multiplier: int = 1) -> list:
    return [item * multiplier for item in data]

@task
def save(data: list, output_path: str) -> None:
    with open(output_path, "w") as f:
        f.write("\n".join(data))

@pipeline
def data_pipeline(input_file: str, output_file: str, multiplier: int = 2):
    """Process data with a multiplier."""
    data = load_data(path=input_file)
    processed = process(data=data, multiplier=multiplier)
    return save(data=processed, output_path=output_file)
```

### Running from Command Line

```bash
# Run with required parameters
daglite run myproject.pipelines.data_pipeline \
    --param input_file=data.txt \
    --param output_file=output.txt

# Override default parameters
daglite run myproject.pipelines.data_pipeline \
    --param input_file=data.txt \
    --param output_file=output.txt \
    --param multiplier=5
```

### Running from Python

```python
# Build and evaluate in one step
data_pipeline.run(
    input_file="data.txt",
    output_file="output.txt",
    multiplier=5,
)
```

---

## Command Reference

### `daglite run`

Execute a pipeline from the command line.

```bash
daglite run [OPTIONS] PIPELINE
```

**Arguments:**

- `PIPELINE` - Dotted path to pipeline function (e.g., `myproject.pipelines.my_pipeline`)

**Options:**

- `--param NAME=VALUE`, `-p NAME=VALUE` - Pipeline parameter (can be specified multiple times)
- `--backend NAME`, `-b NAME` - Execution backend (`inline`, `threading`, `processes`)
- `--async` - Use async evaluation (for async tasks or event loop integration)
- `--settings NAME=VALUE`, `-s NAME=VALUE` - Override global settings (can be specified multiple times)

!!! note "Parallel Execution"
    Sibling tasks run in parallel when using `--backend threading` or `--backend processes`, even without `--async`.
    Use `--async` only when you have async tasks or need async/await semantics.

---

## Examples

### Simple Pipeline

```bash
daglite run myproject.pipelines.simple_pipeline
```

### Pipeline with Parameters

```bash
daglite run myproject.pipelines.etl_pipeline \
    --param source=raw_data.csv \
    --param destination=processed_data.csv \
    --param batch_size=1000
```

### Async Execution

```bash
daglite run myproject.pipelines.parallel_pipeline \
    --param input_dir=./data \
    --async
```

### Custom Backend

```bash
daglite run myproject.pipelines.cpu_intensive_pipeline \
    --backend multiprocessing \
    --settings max_parallel_processes=8
```

### Multiple Parameters

```bash
daglite run myproject.pipelines.ml_pipeline \
    --param model_path=model.pkl \
    --param data_path=train.csv \
    --param epochs=50 \
    --param learning_rate=0.001 \
    --async
```

---

## Type Conversion

The CLI automatically converts parameter values based on type annotations:

```python
@pipeline
def typed_pipeline(
    count: int,           # Converts to int
    rate: float,          # Converts to float
    enabled: bool,        # Converts to bool
    name: str,            # Keeps as string
    items: list[str],     # Parses JSON list
):
    ...
```

```bash
daglite run myproject.pipelines.typed_pipeline \
    --param count=42 \
    --param rate=0.95 \
    --param enabled=true \
    --param name="My Pipeline" \
    --param items='["a","b","c"]'
```

!!! tip "Type Annotations Required"
    Add type annotations to your pipeline parameters for automatic type conversion. Without annotations, all parameters are passed as strings.

---

## Settings

Override global settings for a single run:

```bash
daglite run myproject.pipelines.pipeline \
    --settings max_backend_threads=16 \
    --settings max_parallel_processes=4
```

**Available Settings:**

- `max_backend_threads` - Maximum threads for threading backend
- `max_parallel_processes` - Maximum processes for multiprocessing backend

---

## CI/CD Integration

### GitHub Actions

```yaml
name: Run Pipeline

on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM

jobs:
  pipeline:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install daglite
      - run: |
          daglite run myproject.pipelines.daily_etl \
            --param date=$(date +%Y-%m-%d) \
            --async
```

### GitLab CI

```yaml
pipeline:
  script:
    - pip install daglite
    - daglite run myproject.pipelines.deployment_pipeline
        --param environment=production
        --param version=$CI_COMMIT_TAG
```

---

## Best Practices

### 1. Use Type Annotations

```python
# Good - automatic type conversion
@pipeline
def pipeline(count: int, threshold: float):
    ...

# Bad - everything is a string
@pipeline
def pipeline(count, threshold):
    ...
```

### 2. Provide Docstrings

```python
@pipeline
def data_pipeline(input_path: str, output_path: str):
    """
    Process data from input_path and save to output_path.

    Args:
        input_path: Path to input CSV file
        output_path: Path to save processed data
    """
    ...
```

### 3. Use Default Values

```python
@pipeline
def pipeline(required_param: str, optional_param: int = 10):
    """Required params must be passed, optional params have defaults."""
    ...
```

---

## Troubleshooting

### "Unknown parameter" Error

Ensure parameter names match exactly (case-sensitive):

```bash
# Wrong
daglite run myproject.pipeline --param InputFile=data.txt

# Correct
daglite run myproject.pipeline --param input_file=data.txt
```

### "Missing required parameters" Error

All parameters without default values must be provided:

```python
@pipeline
def pipeline(required: str, optional: str = "default"):
    ...
```

```bash
# Wrong - missing 'required'
daglite run myproject.pipeline --param optional=value

# Correct
daglite run myproject.pipeline --param required=value
```

### Module Not Found Error

Ensure the pipeline module is in your Python path:

```bash
# Option 1: Run from project root
cd /path/to/project
daglite run myproject.pipelines.pipeline

# Option 2: Add to PYTHONPATH
export PYTHONPATH=/path/to/project:$PYTHONPATH
daglite run myproject.pipelines.pipeline
```

---

## See Also

- [Pipelines User Guide](pipelines.md) - Learn about the `@pipeline` decorator
- [Creating Plugins](../plugins/creating.md) - Extend Daglite with custom hooks
