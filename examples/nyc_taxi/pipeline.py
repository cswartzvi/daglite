"""
NYC Taxi Trip Duration — daglite example pipeline.

Predicts yellow-cab trip duration (minutes) from NYC TLC open data.

This pipeline is intentionally structured to showcase every daglite composer:

  gather_tasks  — concurrently download monthly parquet files via async HTTP
  map_tasks     — parallel per-month cleaning and feature engineering
  save_dataset  — persist intermediate artifacts between pipeline stages
  load_dataset  — reload a checkpointed artifact for the training stage

Pipeline stages
---------------
1.  gather_tasks  - async-concurrent download of N monthly parquet files
2.  map_tasks     - clean each month in parallel (drop bad rows, derive target)
3.  save_dataset  - checkpoint one cleaned DataFrame per month
4.  map_tasks     - engineer features for each month in parallel
5.  save_dataset  - checkpoint the final combined feature matrix
6.  load_dataset  - reload the feature matrix (decouples ETL from training)
7.  map_tasks     - train multiple sklearn models in parallel

Usage
-----
    # Quick run (default: 3 months, thread backend)
    python pipeline.py

    # Override months or backend via CLI (requires cyclopts to be available)
    daglite run examples.nyc_taxi.pipeline.nyc_taxi_pipeline

    # Or drive programmatically
    import asyncio
    from examples.nyc_taxi.pipeline import nyc_taxi_pipeline
    asyncio.run(nyc_taxi_pipeline(months=["2024-01"], backend="inline"))
"""

from __future__ import annotations

import asyncio
import io
import logging
from typing import Any

import httpx
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from daglite import gather_tasks
from daglite import load_dataset
from daglite import map_tasks
from daglite import save_dataset
from daglite import task
from daglite import workflow
from daglite.settings import DagliteSettings
from daglite.settings import set_global_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)

set_global_settings(
    DagliteSettings(
        backend="thread",
        dataset_store="data/nyc_taxi",
        cache_store=".cache/nyc_taxi",
    )
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TLC_BASE = "https://d37ci6vzurychx.cloudfront.net/trip-data"

# Single shared client — one connection pool reused across all concurrent fetch_month calls
_http_client = httpx.AsyncClient(timeout=300.0)

DEFAULT_MONTHS = ["2024-01", "2024-02", "2024-03"]

# Columns kept as model features
FEATURES = [
    "trip_distance",
    "passenger_count",
    "PULocationID",
    "DOLocationID",
    "pickup_hour",
    "pickup_dayofweek",
    "pickup_month",
]
TARGET = "duration_minutes"

# Sanity-filter bounds
_MIN_DURATION = 1.0  # minutes — drop sub-minute "trips"
_MAX_DURATION = 120.0  # minutes — drop suspiciously long trips
_MIN_DISTANCE = 0.1  # miles
_MAX_DISTANCE = 100.0  # miles

# One entry per model to train; all share the same X/y arrays via dict packing
MODEL_CONFIGS: list[tuple[str, Any]] = [
    ("linear_regression", LinearRegression()),
    ("ridge", Ridge(alpha=1.0)),
    ("random_forest", RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)),
    ("gradient_boosting", GradientBoostingRegressor(n_estimators=100, random_state=42)),
]


# ---------------------------------------------------------------------------
# Stage 1 — Ingest
# ---------------------------------------------------------------------------


@task(cache=True)  # cache=True: the download is skipped on subsequent runs
async def fetch_month(month: str) -> pd.DataFrame:
    """
    Async task: download one month of yellow-cab data from the NYC TLC CDN.

    The ``cache=True`` flag means the HTTP round-trip only happens once;
    subsequent calls with the same ``month`` value hit the local cache.

    This task is designed for ``gather_tasks`` — multiple months are fetched
    concurrently via ``asyncio.gather`` rather than sequentially.
    """
    url = f"{TLC_BASE}/yellow_tripdata_{month}.parquet"
    logger.info("Fetching %s …", url)
    response = await _http_client.get(url)
    response.raise_for_status()
    df = pd.read_parquet(io.BytesIO(response.content))
    logger.info("  → %s: %d rows downloaded", month, len(df))
    return df


# ---------------------------------------------------------------------------
# Stage 2 — Clean
# ---------------------------------------------------------------------------


@task
def clean_month(df: pd.DataFrame, month: str) -> pd.DataFrame:
    """
    Sync task: drop implausible trips and derive the ``duration_minutes`` target.

    Designed for ``map_tasks`` — one DataFrame per month is cleaned in parallel.
    """
    df = df.copy()

    # Derive regression target from raw timestamps
    df[TARGET] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60.0

    mask = (
        df[TARGET].between(_MIN_DURATION, _MAX_DURATION)
        & df["trip_distance"].between(_MIN_DISTANCE, _MAX_DISTANCE)
        & df["passenger_count"].between(1, 8)
    )
    df = df.loc[mask].reset_index(drop=True)
    logger.info("clean_month: %s → %d rows kept", month, len(df))
    return df


# ---------------------------------------------------------------------------
# Stage 3 — Feature engineering
# ---------------------------------------------------------------------------


@task
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sync task: extract temporal and categorical features from raw trip data.

    Designed for ``map_tasks`` — one cleaned DataFrame per month is processed
    in parallel.  Returns only the columns needed for modelling.
    """
    df = df.copy()
    pickup = pd.to_datetime(df["tpep_pickup_datetime"])
    df["pickup_hour"] = pickup.dt.hour
    df["pickup_dayofweek"] = pickup.dt.dayofweek
    df["pickup_month"] = pickup.dt.month
    return df[FEATURES + [TARGET]].dropna()


# ---------------------------------------------------------------------------
# Stage 4 — Training
# ---------------------------------------------------------------------------


@task
def train_model(config: dict[str, Any]) -> dict[str, Any]:
    """
    Sync task: scale, fit, and evaluate a single sklearn estimator.

    Accepts a configuration dict so that a list of configs can be passed
    directly to ``map_tasks(train_model, configs)`` without needing to
    repeat the shared X/y arrays across separate iterables.
    """
    name = config["name"]
    model = config["model"]
    X_train = config["X_train"]
    y_train = config["y_train"]
    X_test = config["X_test"]
    y_test = config["y_test"]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    model.fit(X_tr_s, y_train)
    preds = model.predict(X_te_s)

    mae = float(mean_absolute_error(y_test, preds))
    logger.info("  %-26s  MAE = %.3f min", name, mae)
    return {"name": name, "mae": mae}


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


@workflow
async def nyc_taxi_pipeline(
    months: list[str] = DEFAULT_MONTHS,
) -> list[dict[str, Any]]:
    """
    End-to-end NYC yellow-cab trip-duration pipeline.

    The ``@workflow`` decorator wraps everything in a managed ``async_session``
    that activates the backend, cache store, and dataset store configured via
    ``daglite.settings.set_global_settings`` (or the defaults).

    Args:
        months: ``YYYY-MM`` strings to include.  Defaults to Q1 2024.

    Returns:
        List of ``{"name": str, "mae": float}`` dicts, sorted by MAE ascending.
    """
    # ------------------------------------------------------------------
    # 1. gather_tasks — fire off all monthly downloads concurrently.
    #    Each fetch_month call is an independent async coroutine; asyncio
    #    runs them at the same time rather than sequentially.
    # ------------------------------------------------------------------
    raw_dfs: list[pd.DataFrame] = await gather_tasks(fetch_month, months)

    # ------------------------------------------------------------------
    # 2. map_tasks — clean each month in parallel using the active backend
    #    (defaults to thread pool; set backend="process" for CPU-bound work).
    # ------------------------------------------------------------------
    clean_dfs: list[pd.DataFrame] = map_tasks(clean_month, raw_dfs, months)

    # ------------------------------------------------------------------
    # 3. save_dataset — persist cleaned DataFrames so a partial re-run can
    #    skip the network stage.  Keys are relative to the dataset_store root.
    # ------------------------------------------------------------------
    for month, df in zip(months, clean_dfs):
        save_dataset(f"clean/{month}.pkl", df)

    # ------------------------------------------------------------------
    # 4. map_tasks — engineer features for all months in parallel.
    # ------------------------------------------------------------------
    feature_dfs: list[pd.DataFrame] = map_tasks(engineer_features, clean_dfs)

    # ------------------------------------------------------------------
    # 5. save_dataset — write the combined feature matrix as a single
    #    checkpoint artifact.
    # ------------------------------------------------------------------
    combined = pd.concat(feature_dfs).reset_index(drop=True)
    save_dataset("features/combined.pkl", combined)

    # ------------------------------------------------------------------
    # 6. load_dataset — reload from the store to demonstrate stage
    #    decoupling: a separate training script could start here by loading
    #    this checkpoint without re-running any ETL logic.
    # ------------------------------------------------------------------
    features: pd.DataFrame = load_dataset("features/combined.pkl", pd.DataFrame)

    X = features[FEATURES].to_numpy(dtype=float)
    y = features[TARGET].to_numpy(dtype=float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Bundle shared arrays into each model config so map_tasks only needs
    # one iterable (the list of configs).
    configs = [
        {
            "name": name,
            "model": model,
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }
        for name, model in MODEL_CONFIGS
    ]

    # ------------------------------------------------------------------
    # 7. map_tasks — train all models in parallel.
    # ------------------------------------------------------------------
    results: list[dict[str, Any]] = map_tasks(train_model, configs)

    results.sort(key=lambda r: r["mae"])
    _print_leaderboard(results)
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print_leaderboard(results: list[dict[str, Any]]) -> None:
    print(f"\n{'Model':<26}  {'MAE (min)':>10}")
    print("-" * 40)
    for r in results:
        print(f"{r['name']:<26}  {r['mae']:>10.3f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Configure daglite before the @workflow creates its async_session.
    # These settings are picked up by session() / async_session() on first call.
    asyncio.run(nyc_taxi_pipeline())
