# Examples

Real-world examples of using Daglite for common tasks.

---

## ETL Pipeline

```python
from daglite import task, evaluate
import pandas as pd

@task
def extract(source: str) -> pd.DataFrame:
    """Extract data from source."""
    return pd.read_csv(source)

@task
def transform(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Transform by selecting and cleaning columns."""
    return df[columns].dropna()

@task
def load(df: pd.DataFrame, destination: str) -> None:
    """Load data to destination."""
    df.to_parquet(destination)

# Build and execute pipeline
evaluate(
    extract.bind(source="data.csv")
    .then(transform, columns=["id", "name", "value"])
    .then(load, destination="output.parquet")
)
```

---

## Parameter Sweep

```python
@task
def train_model(lr: float, batch_size: int, epochs: int) -> dict:
    """Train model with given hyperparameters."""
    model = Model(lr=lr)
    model.fit(data, batch_size=batch_size, epochs=epochs)
    return {"accuracy": model.evaluate(), "lr": lr, "batch_size": batch_size}

@task
def find_best(results: list[dict]) -> dict:
    """Find best hyperparameters."""
    return max(results, key=lambda r: r["accuracy"])

# Search hyperparameter space
best = evaluate(
    train_model.product(
        lr=[0.001, 0.01, 0.1],
        batch_size=[32, 64],
        epochs=[10]
    )
    .join(find_best)
)
```

---

## Parallel Data Processing

```python
@task
def fetch_user(user_id: int) -> dict:
    """Fetch user data from API."""
    return api.get(f"/users/{user_id}")

@task
def enrich_user(user: dict, include_orders: bool) -> dict:
    """Enrich user with additional data."""
    if include_orders:
        user["orders"] = api.get(f"/users/{user['id']}/orders")
    return user

@task
def save_users(users: list[dict], path: str) -> None:
    """Save users to database."""
    db.bulk_insert("users", users)

# Process users in parallel
evaluate(
    fetch_user.product(user_id=range(1, 101))
    .map(enrich_user, include_orders=True)
    .join(save_users, path="users.db")
)
```

---

## More Examples Coming Soon!

We're working on adding more examples. In the meantime, check out the [tests directory](https://github.com/cswartzvi/daglite/tree/main/tests) for additional usage patterns.
