# Datasets & Persistence

!!! note "Coming Soon"
    This page is under construction. A detailed guide is planned covering the topics outlined below.

Daglite provides a built-in dataset system for saving and loading task outputs, enabling persistence, checkpointing, and data sharing between pipeline runs.

---

## Planned Content

### Saving Task Outputs with `.save()`

- Saving a future's result to disk after evaluation
- Key templates with `{param}` placeholders
- Choosing serialization formats
- Saving to custom stores (directories, remote paths)
- Chaining multiple `.save()` calls

### Loading Datasets with `load_dataset()`

- Loading previously saved results into a new DAG
- Key templates and type hints for deserialization
- Specifying format and store overrides
- Connecting loaded datasets to downstream tasks

### Dataset Stores

- Default store configuration via `DagliteSettings.datastore_store`
- Per-task stores with `@task(store=...)`
- Per-save stores with `.save(save_store=...)`
- Using string paths vs `DatasetStore` instances

### Checkpointing

- Marking `.save()` calls as checkpoints with `save_checkpoint=True`
- Resuming pipelines from checkpoints
- Named checkpoints for multi-stage workflows

### Built-in Serialization

- Supported types: `bytes`, `str`, `int`, `float`, `bool`, `dict`, `list`, `tuple`, `set`, `frozenset`, `None`
- Extending with `daglite-serialization` for NumPy, Pandas, Pillow

### Custom Serialization

- Registering custom serializers via `SerializationRegistry`
- Custom hash strategies for caching
- Format registration and file extension mapping

---

## Quick Example

```python
from daglite import task, load_dataset

@task(store="./outputs")
def train_model(data: list, epochs: int) -> dict:
    # ... training logic ...
    return {"weights": ..., "accuracy": 0.95}

# Save output after evaluation
future = train_model(data=[1, 2, 3], epochs=10).save("model_{epochs}")

# In a later run, load the saved result
loaded = load_dataset("model_10", load_type=dict, load_store="./outputs")
```

---

## See Also

- [Serialization Plugin](../plugins/serialization.md) - NumPy, Pandas, and Pillow support
- [Settings](../api-reference/settings.md) - Default store configuration
- [Tasks](tasks.md) - Task `store` option
