# NYC Taxi Trip Duration — daglite example

Predicts yellow-cab **trip duration (minutes)** end-to-end using
[NYC TLC open data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
(no login required).

## What this example showcases

| Composer | Where | Why |
|---|---|---|
| `gather_tasks` | Stage 1 — Ingest | Fires off N async HTTP downloads concurrently via `asyncio.gather` |
| `map_tasks` | Stage 2 — Clean | Cleans one `DataFrame` per month in parallel (thread pool) |
| `map_tasks` | Stage 4 — Features | Engineers features for each month in parallel |
| `save_dataset` | Stages 3 & 5 | Persists cleaned data and the feature matrix to disk |
| `load_dataset` | Stage 6 — Train | Reloads the feature checkpoint — decouples ETL from training |
| `map_tasks` | Stage 7 — Train | Fits LinearRegression, Ridge, RandomForest, GradientBoosting in parallel |

The `@workflow` decorator wraps everything in a managed `async_session` that
activates the configured backend, cache store, and dataset store.

`cache=True` on `fetch_month` means the parquet files are only downloaded once
— repeat runs read from the local cache.

## Install

```bash
pip install daglite
pip install -r examples/nyc_taxi/requirements.txt
```

## Run

```bash
# From the repo root
python examples/nyc_taxi/pipeline.py
```

Expected output (timings depend on network and CPU):

```
2026-03-18 … Fetching https://…/yellow_tripdata_2024-01.parquet …
2026-03-18 … Fetching https://…/yellow_tripdata_2024-02.parquet …
2026-03-18 … Fetching https://…/yellow_tripdata_2024-03.parquet …
…
Model                       MAE (min)
----------------------------------------
gradient_boosting               3.421
random_forest                   3.589
ridge                           4.812
linear_regression               4.813
```

## Adapting it

* **Different months** — pass a `months` list:
  ```python
  asyncio.run(nyc_taxi_pipeline(months=["2023-06", "2023-07"]))
  ```

* **Different backend** — change the `DagliteSettings` at the bottom of `pipeline.py`:
  ```python
  set_global_settings(DagliteSettings(backend="process", ...))
  ```

* **Add a feature** — decorate a new function with `@task` and add a
  `map_tasks(my_task, ...)` call in the workflow body.

* **Add a model** — append to `MODEL_CONFIGS`:
  ```python
  ("xgboost", XGBRegressor(n_estimators=200))
  ```

## Data

Monthly yellow-cab parquet files (~40–100 MB each) are fetched from the
NYC TLC CDN and cached locally under `.cache/nyc_taxi/`.
Intermediate artifacts (cleaned DataFrames, feature matrix) are stored
under `data/nyc_taxi/`.
