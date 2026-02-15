# ETL Pipeline

Build a complete Extract-Transform-Load (ETL) pipeline using Daglite and Pandas.

## Overview

ETL pipelines are a common pattern in data engineering:

1. **Extract** - Load data from various sources (CSV, databases, APIs)
2. **Transform** - Clean, filter, and reshape the data
3. **Load** - Save the processed data to a destination

Daglite's fluent API makes it easy to chain these operations while maintaining type safety and explicit dependencies.

## Basic ETL Pipeline

### Simple Inline Pipeline

```python
from daglite import task, evaluate
import pandas as pd

@task
def extract(source: str) -> pd.DataFrame:
    """Extract data from CSV source."""
    return pd.read_csv(source)

@task
def transform(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Transform by selecting and cleaning columns."""
    return df[columns].dropna()

@task
def load(df: pd.DataFrame, destination: str) -> None:
    """Load data to parquet destination."""
    df.to_parquet(destination)

# Build and execute pipeline
evaluate(
    extract(source="data.csv")
    .then(transform, columns=["id", "name", "value"])
    .then(load, destination="output.parquet")
)
```

### With Error Handling

```python
@task
def extract_with_validation(source: str) -> pd.DataFrame:
    """Extract and validate data."""
    try:
        df = pd.read_csv(source)
    except FileNotFoundError:
        raise ValueError(f"Source file not found: {source}")

    # Validate required columns
    required_columns = ["id", "name", "value"]
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df

@task
def transform_with_logging(
    df: pd.DataFrame,
    columns: list[str]
) -> pd.DataFrame:
    """Transform data with logging."""
    print(f"Input rows: {len(df)}")

    # Select columns
    df = df[columns]

    # Remove nulls
    df = df.dropna()

    print(f"Output rows: {len(df)} ({len(df)/len(df)*100:.1f}% of input)")

    return df

# Execute with error handling
try:
    result = evaluate(
        extract_with_validation(source="data.csv")
        .then(transform_with_logging, columns=["id", "name", "value"])
        .then(load, destination="output.parquet")
    )
    print("Pipeline completed successfully!")
except Exception as e:
    print(f"Pipeline failed: {e}")
```

## Multi-Source ETL

Combine data from multiple sources:

```python
@task
def extract_csv(path: str) -> pd.DataFrame:
    """Extract from CSV."""
    return pd.read_csv(path)

@task
def extract_json(path: str) -> pd.DataFrame:
    """Extract from JSON."""
    return pd.read_json(path)

@task
def extract_db(table: str, connection_string: str) -> pd.DataFrame:
    """Extract from database."""
    import sqlalchemy
    engine = sqlalchemy.create_engine(connection_string)
    return pd.read_sql_table(table, engine)

@task
def merge_sources(
    customers: pd.DataFrame,
    orders: pd.DataFrame,
    products: pd.DataFrame
) -> pd.DataFrame:
    """Merge multiple data sources."""
    # Join customers with orders
    customer_orders = customers.merge(
        orders,
        on="customer_id",
        how="inner"
    )

    # Join with products
    result = customer_orders.merge(
        products,
        on="product_id",
        how="inner"
    )

    return result

@task
def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate aggregates."""
    return df.groupby("customer_id").agg({
        "order_id": "count",
        "amount": "sum",
        "product_id": "nunique"
    }).reset_index()

# Extract from multiple sources
customers_df = extract_csv(path="customers.csv")
orders_df = extract_json(path="orders.json")
products_df = extract_db(
    table="products",
    connection_string="postgresql://localhost/db"
)

# Combine and process
result = evaluate(
    merge_sources(
        customers=customers_df,
        orders=orders_df,
        products=products_df
    )
    .then(aggregate)
    .then(load, destination="customer_summary.parquet")
)
```

## Incremental ETL

Process data incrementally with date partitions:

```python
from datetime import datetime, timedelta

@task
def extract_partition(
    source: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """Extract data for a date range."""
    df = pd.read_csv(source)
    df["date"] = pd.to_datetime(df["date"])

    mask = (df["date"] >= start_date) & (df["date"] <= end_date)
    return df[mask]

@task
def deduplicate(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Remove duplicates keeping latest record."""
    return df.sort_values("updated_at").drop_duplicates(
        subset=[key],
        keep="last"
    )

@task
def load_partition(
    df: pd.DataFrame,
    destination: str,
    partition_date: str
) -> None:
    """Load data to partitioned destination."""
    partition_path = f"{destination}/date={partition_date}"
    df.to_parquet(partition_path)

# Process last 7 days
today = datetime.now()
dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]

@task
def process_date(date: str) -> None:
    """Process single date partition."""
    return evaluate(
        extract_partition(
            source="raw_data.csv",
            start_date=date,
            end_date=date
        )
        .then(transform, columns=["id", "value", "date"])
        .then(deduplicate, key="id")
        .then(load_partition, destination="output", partition_date=date)
    )

# Process all dates in parallel
evaluate(
    process_date.map(date=dates, map_mode="product")
)
```

## Async ETL for I/O-Bound Tasks

Use async execution for better performance with I/O operations:

```python
@task(backend_name="threading")
def fetch_api_data(endpoint: str, api_key: str) -> pd.DataFrame:
    """Fetch data from API (I/O-bound)."""
    import requests
    response = requests.get(
        f"https://api.example.com/{endpoint}",
        headers={"Authorization": f"Bearer {api_key}"}
    )
    return pd.DataFrame(response.json())

@task
def combine_api_results(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine multiple API responses."""
    return pd.concat(dfs, ignore_index=True)

# Fetch from multiple endpoints in parallel
endpoints = ["users", "orders", "products", "reviews"]

result = evaluate(
    fetch_api_data.map(
        endpoint=endpoints,
        api_key="your_api_key",
        map_mode="product"
    )
    .join(combine_api_results)
    .then(transform, columns=["id", "name", "value"])
    .then(load, destination="api_data.parquet")
)
```

## CLI Pipeline

Make your ETL pipeline runnable from the command line:

```python
# etl_pipeline.py
from daglite import task, pipeline
import pandas as pd

@task
def extract(source: str) -> pd.DataFrame:
    return pd.read_csv(source)

@task
def transform(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df[columns].dropna()

@task
def load(df: pd.DataFrame, destination: str) -> None:
    df.to_parquet(destination)

@pipeline
def daily_etl(
    source: str,
    destination: str,
    columns: list[str] = None
):
    """Daily ETL pipeline."""
    if columns is None:
        columns = ["id", "name", "value"]

    return (
        extract(source=source)
        .then(transform, columns=columns)
        .then(load, destination=destination)
    )
```

Run from command line:

```bash
daglite run etl_pipeline.daily_etl \
    --param source=data.csv \
    --param destination=output.parquet \
    --param columns='["id","name","value"]'
```

## Best Practices

### 1. Validate Early

Validate data as soon as it's extracted:

```python
@task
def extract_and_validate(source: str) -> pd.DataFrame:
    df = pd.read_csv(source)

    # Check for empty data
    if df.empty:
        raise ValueError("Source data is empty")

    # Validate schema
    expected_columns = ["id", "name", "value"]
    if not all(col in df.columns for col in expected_columns):
        raise ValueError(f"Missing columns. Expected: {expected_columns}")

    return df
```

### 2. Make Transforms Idempotent

Ensure transforms can be re-run safely:

```python
@task
def idempotent_transform(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize column names (safe to run multiple times)
    df.columns = df.columns.str.lower().str.strip()

    # Remove duplicates (safe to run multiple times)
    df = df.drop_duplicates()

    # Fill nulls with default (safe to run multiple times)
    df = df.fillna({"value": 0})

    return df
```

### 3. Use Type Hints

Type hints help catch errors and improve IDE support:

```python
from typing import Literal

@task
def transform(
    df: pd.DataFrame,
    operation: Literal["filter", "aggregate", "pivot"]
) -> pd.DataFrame:
    if operation == "filter":
        return df[df["value"] > 0]
    elif operation == "aggregate":
        return df.groupby("category").sum()
    elif operation == "pivot":
        return df.pivot_table(index="date", columns="category", values="value")
```

### 4. Log Progress

Add logging for long-running pipelines:

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@task
def transform_with_logging(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"Starting transform on {len(df)} rows")

    df = df.dropna()
    logger.info(f"After dropna: {len(df)} rows")

    df = df[df["value"] > 0]
    logger.info(f"After filtering: {len(df)} rows")

    return df
```

## See Also

- [Pandas Documentation](https://pandas.pydata.org/docs/) - DataFrame operations
- [Data Processing Example](data-processing.md) - Parallel processing patterns
- [CLI Reference](../user-guide/cli.md) - Command-line interface for pipelines
