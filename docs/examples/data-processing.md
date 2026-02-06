# Data Processing

Parallel data processing patterns using Daglite's fan-out/fan-in operations.

## Overview

Data processing often involves:

1. **Fan-out** - Split work into parallel tasks
2. **Process** - Transform each piece independently
3. **Fan-in** - Aggregate results back together

Daglite provides `.product()`, `.map()`, and `.join()` for these patterns, with optional async execution for I/O-bound workloads.

## Basic Parallel Processing

### Process Multiple Files

```python
from daglite import task, evaluate
import pandas as pd
from pathlib import Path

@task
def process_file(path: str) -> pd.DataFrame:
    """Process a single CSV file."""
    df = pd.read_csv(path)

    # Clean data
    df = df.dropna()

    # Add source file column
    df["source_file"] = Path(path).name

    return df

@task
def combine_files(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine processed files."""
    return pd.concat(dfs, ignore_index=True)

# Process multiple files in parallel
files = ["data_2023.csv", "data_2024.csv", "data_2025.csv"]

result = evaluate(
    process_file.product(path=files)
    .join(combine_files)
)

print(f"Total rows: {len(result)}")
```

### Process with Transformation

Use `.map()` to transform each result:

```python
@task
def fetch_data(id: int) -> dict:
    """Fetch data from API."""
    import requests
    response = requests.get(f"https://api.example.com/items/{id}")
    return response.json()

@task
def enrich(data: dict, category: str) -> dict:
    """Enrich with additional info."""
    data["category"] = category
    data["processed_at"] = pd.Timestamp.now()
    return data

@task
def save_all(items: list[dict]) -> None:
    """Save all items to database."""
    df = pd.DataFrame(items)
    df.to_sql("items", connection, if_exists="append")

# Fetch, enrich, and save
evaluate(
    fetch_data.product(id=range(1, 101))
    .map(enrich, category="products")
    .join(save_all)
)
```

## Batch Processing

### Process in Batches

```python
from typing import TypeVar
T = TypeVar('T')

@task
def create_batches(items: list[T], batch_size: int) -> list[list[T]]:
    """Split items into batches."""
    return [
        items[i:i + batch_size]
        for i in range(0, len(items), batch_size)
    ]

@task
def process_batch(batch: list[int]) -> list[dict]:
    """Process a batch of IDs."""
    results = []
    for id in batch:
        # Expensive operation
        result = expensive_computation(id)
        results.append(result)
    return results

@task
def flatten(batches: list[list[dict]]) -> list[dict]:
    """Flatten list of lists."""
    return [item for batch in batches for item in batch]

# Process 1000 items in batches of 50
items = list(range(1, 1001))

result = evaluate(
    create_batches(items=items, batch_size=50)
    .then(process_batch.product)
    .join(flatten)
)
```

### Batch API Calls

```python
@task
def batch_api_call(ids: list[int]) -> list[dict]:
    """Call API with batch of IDs."""
    import requests
    response = requests.post(
        "https://api.example.com/batch",
        json={"ids": ids}
    )
    return response.json()["results"]

@task
def create_batches(items: list[int], batch_size: int) -> list[list[int]]:
    """Create batches."""
    return [items[i:i+batch_size] for i in range(0, len(items), batch_size)]

@task
def flatten_results(batches: list[list[dict]]) -> list[dict]:
    """Flatten results."""
    return [item for batch in batches for item in batch]

# Process 10,000 IDs in batches of 100
ids = range(1, 10001)

results = evaluate(
    create_batches(items=list(ids), batch_size=100)
    .then(batch_api_call.product)
    .join(flatten_results)
)
```

## Map-Reduce Patterns

### Word Count

Classic map-reduce example:

```python
@task
def read_file(path: str) -> str:
    """Read file content."""
    with open(path) as f:
        return f.read()

@task
def count_words(text: str) -> dict[str, int]:
    """Count words in text (map phase)."""
    from collections import Counter
    words = text.lower().split()
    return Counter(words)

@task
def merge_counts(counts: list[dict[str, int]]) -> dict[str, int]:
    """Merge word counts (reduce phase)."""
    from collections import Counter
    total = Counter()
    for count in counts:
        total.update(count)
    return dict(total)

# Count words across multiple files
files = ["doc1.txt", "doc2.txt", "doc3.txt"]

word_counts = evaluate(
    read_file.product(path=files)
    .map(count_words)
    .join(merge_counts)
)

# Top 10 words
top_words = sorted(
    word_counts.items(),
    key=lambda x: x[1],
    reverse=True
)[:10]
```

### Aggregate Statistics

```python
@task
def load_partition(partition: int) -> pd.DataFrame:
    """Load data partition."""
    return pd.read_parquet(f"data/partition_{partition}.parquet")

@task
def compute_stats(df: pd.DataFrame) -> dict:
    """Compute statistics for partition."""
    return {
        "count": len(df),
        "sum": df["value"].sum(),
        "min": df["value"].min(),
        "max": df["value"].max(),
        "sum_squares": (df["value"] ** 2).sum()
    }

@task
def merge_stats(stats: list[dict]) -> dict:
    """Merge statistics from all partitions."""
    total_count = sum(s["count"] for s in stats)
    total_sum = sum(s["sum"] for s in stats)

    return {
        "count": total_count,
        "mean": total_sum / total_count,
        "min": min(s["min"] for s in stats),
        "max": max(s["max"] for s in stats),
        "std": (
            sum(s["sum_squares"] for s in stats) / total_count -
            (total_sum / total_count) ** 2
        ) ** 0.5
    }

# Compute statistics across 100 partitions
stats = evaluate(
    load_partition.product(partition=range(100))
    .map(compute_stats)
    .join(merge_stats)
)
```

## Async Processing

### I/O-Bound Tasks

Use async execution for I/O-heavy workloads:

```python
@task(backend_name="threading")
def fetch_url(url: str) -> dict:
    """Fetch data from URL (I/O-bound)."""
    import requests
    response = requests.get(url, timeout=10)
    return {
        "url": url,
        "status": response.status_code,
        "size": len(response.content)
    }

@task
def summarize(results: list[dict]) -> dict:
    """Summarize fetch results."""
    return {
        "total": len(results),
        "successful": sum(1 for r in results if r["status"] == 200),
        "total_size": sum(r["size"] for r in results)
    }

# Fetch 100 URLs in parallel
urls = [f"https://example.com/page{i}" for i in range(100)]

summary = evaluate(
    fetch_url.product(url=urls)
    .join(summarize),
    use_async=True  # Enable async execution
)
```

### CPU-Bound Tasks

Use multiprocessing backend for CPU-intensive tasks:

```python
@task(backend_name="multiprocessing")
def compute_intensive(n: int) -> float:
    """CPU-intensive computation."""
    import math
    result = 0.0
    for i in range(n):
        result += math.sin(i) * math.cos(i)
    return result

@task
def sum_results(results: list[float]) -> float:
    """Sum all results."""
    return sum(results)

# Run CPU-intensive tasks in parallel
result = evaluate(
    compute_intensive.product(n=[1000000, 2000000, 3000000])
    .join(sum_results),
    use_async=True
)
```

## Data Validation Pipeline

### Validate Multiple Files

```python
from typing import Literal

@task
def validate_schema(path: str, expected_columns: list[str]) -> dict:
    """Validate file schema."""
    df = pd.read_csv(path)

    missing = set(expected_columns) - set(df.columns)
    extra = set(df.columns) - set(expected_columns)

    return {
        "path": path,
        "valid": len(missing) == 0,
        "missing_columns": list(missing),
        "extra_columns": list(extra),
        "row_count": len(df)
    }

@task
def check_data_quality(path: str) -> dict:
    """Check data quality metrics."""
    df = pd.read_csv(path)

    return {
        "path": path,
        "null_percentage": (df.isnull().sum().sum() / df.size) * 100,
        "duplicate_rows": df.duplicated().sum(),
        "has_negatives": (df.select_dtypes(include=[np.number]) < 0).any().any()
    }

@task
def generate_report(
    schema_results: list[dict],
    quality_results: list[dict]
) -> dict:
    """Generate validation report."""
    return {
        "total_files": len(schema_results),
        "valid_schemas": sum(1 for r in schema_results if r["valid"]),
        "total_rows": sum(r["row_count"] for r in schema_results),
        "quality_issues": [
            r for r in quality_results
            if r["null_percentage"] > 5 or r["duplicate_rows"] > 0
        ]
    }

# Validate multiple files
files = ["file1.csv", "file2.csv", "file3.csv"]
expected_cols = ["id", "name", "value", "date"]

schema_checks = validate_schema.product(
    path=files,
    expected_columns=[expected_cols]
)

quality_checks = check_data_quality.product(path=files)

report = evaluate(
    generate_report(
        schema_results=schema_checks,
        quality_results=quality_checks
    )
)
```

## Pairwise Operations with `.zip()`

### Process Pairs

```python
@task
def compare_files(file1: str, file2: str) -> dict:
    """Compare two files."""
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    return {
        "files": (file1, file2),
        "row_diff": len(df1) - len(df2),
        "col_diff": set(df1.columns) - set(df2.columns),
        "identical": df1.equals(df2)
    }

# Compare file pairs
current_files = ["current_1.csv", "current_2.csv", "current_3.csv"]
previous_files = ["previous_1.csv", "previous_2.csv", "previous_3.csv"]

comparisons = evaluate(
    compare_files.zip(file1=current_files, file2=previous_files)
)
```

### Merge Corresponding Items

```python
@task
def merge_data(main: pd.DataFrame, supplementary: pd.DataFrame) -> pd.DataFrame:
    """Merge main and supplementary data."""
    return main.merge(supplementary, on="id", how="left")

# Load main files
main_dfs = load_file.product(path=["main_1.csv", "main_2.csv"])

# Load supplementary files
supp_dfs = load_file.product(path=["supp_1.csv", "supp_2.csv"])

# Merge pairwise
merged = evaluate(
    merge_data.zip(main=main_dfs, supplementary=supp_dfs)
)
```

## Best Practices

### 1. Choose Appropriate Backend

```python
# I/O-bound: Use threading
@task(backend_name="threading")
def fetch_api(url: str):
    ...

# CPU-bound: Use multiprocessing
@task(backend_name="multiprocessing")
def compute_fft(data: np.ndarray):
    ...

# Mixed: Let Daglite choose based on task
@task
def process(data):
    ...
```

### 2. Handle Errors Gracefully

```python
@task
def safe_process(path: str) -> dict:
    """Process file with error handling."""
    try:
        df = pd.read_csv(path)
        result = process_dataframe(df)
        return {"path": path, "success": True, "result": result}
    except Exception as e:
        return {"path": path, "success": False, "error": str(e)}

@task
def filter_successes(results: list[dict]) -> list[dict]:
    """Extract successful results."""
    successes = [r["result"] for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]

    if failures:
        print(f"Failed files: {[r['path'] for r in failures]}")

    return successes
```

### 3. Monitor Progress

```python
from daglite.plugins.hooks import hook_impl

class ProgressTracker:
    """Track task completion."""

    def __init__(self):
        self.completed = 0
        self.total = 0

    @hook_impl
    def before_graph_execute(self, root_id, node_count):
        self.total = node_count
        self.completed = 0
        print(f"Starting: 0/{self.total}")

    @hook_impl
    def after_node_execute(self, key, metadata, inputs, result, duration, reporter=None):
        self.completed += 1
        print(f"Progress: {self.completed}/{self.total}")

# Use with plugin
from daglite.plugins.manager import register_plugins

tracker = ProgressTracker()
register_plugins(tracker)

result = evaluate(
    process_file.product(path=files)
    .join(combine_files)
)
```

### 4. Batch for Efficiency

```python
# Bad: One API call per item (100 calls)
results = evaluate(
    fetch_single_item.product(id=range(100))
)

# Good: Batch API calls (10 calls)
batches = create_batches(list(range(100)), batch_size=10)
results = evaluate(
    fetch_batch.product(batch=batches)
    .join(flatten_results)
)
```

## See Also

- [ETL Pipeline](etl-pipeline.md) - Data transformation patterns
- [ML Workflow](ml-workflow.md) - Machine learning examples
- [Evaluation Guide](../user-guide/evaluation.md) - Async execution details
