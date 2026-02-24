# Caching & Checkpointing Overhaul — Implementation Plan

**Status**: Pending Approval
**Target**: daglite v0.9.0

---

## Summary

Overhaul the caching system to be a built-in feature (not a plugin) and implement
checkpoints on `.run()` / `.run_async()`. Replace the `SerializationRegistry` with
cloudpickle. The new `CacheStore` mimics the `DatasetStore` pattern (uses `Driver`
under the hood) but is its own implementation.

---

## Phase 1: Add cloudpickle & Refactor CacheStore

### 1.1 — Add `cloudpickle` dependency

**File**: `pyproject.toml`

- Add `cloudpickle>=3.0` to `dependencies` list.

### 1.2 — Rewrite `CacheStore` using `Driver`

The new `CacheStore` mimics `DatasetStore` (uses `Driver` for byte storage) but handles
cache-specific concerns: git-style sharded keys, TTL metadata, cloudpickle serialization.

**File**: `src/daglite/cache/store.py` (rewrite)

```python
class CacheStore:
    """Hash-based cache store using Driver for byte storage and cloudpickle for serialization."""

    def __init__(self, driver: Driver | str) -> None:
        """Accept a Driver instance or string path (creates FileDriver)."""

    def get(self, hash_key: str) -> Any | None:
        """Retrieve cached value. Returns None on miss or TTL expiry."""

    def put(self, hash_key: str, value: Any, ttl: int | None = None) -> None:
        """Store value with optional TTL."""

    def invalidate(self, hash_key: str) -> None:
        """Remove a cached entry (data + metadata)."""

    def clear(self) -> None:
        """Remove all cached entries."""

    def _hash_to_key(self, hash_key: str) -> str:
        """Git-style sharding: 'abcdef...' → 'ab/cdef...'"""
```

**Key implementation details**:

- Uses `FileDriver` under the hood (or any `Driver` subclass for S3/GCS).
- Cloudpickle for `serialize` / `deserialize` (replaces stdlib pickle).
- Metadata sidecar: `{data_key}.meta.json` with `{"timestamp": ..., "ttl": ...}`.
- Git-style 2-char prefix sharding for the hash key path.
- Pickle-safe `__getstate__` / `__setstate__` for process backends.

### 1.3 — Remove `CacheStore` Protocol

**File**: `src/daglite/cache/base.py` (delete or keep as reference)

- The `CacheStore` Protocol in `base.py` is replaced by the concrete class in `store.py`.
- If we want to keep a protocol for future extensibility, we can keep a slimmed-down
  version, but the user-facing API is the concrete `CacheStore` class.

**Decision**: Delete `base.py`. The concrete `CacheStore` class is the API.

### 1.4 — Update `cache/__init__.py`

**File**: `src/daglite/cache/__init__.py`

```python
from daglite.cache.store import CacheStore
from daglite.cache.core import default_cache_hash

__all__ = ["CacheStore", "default_cache_hash"]
```

Remove the `FileCacheStore` export (now just `CacheStore`).

---

## Phase 2: Remove SerializationRegistry & Update Cache Hashing

### 2.1 — Rewrite cache hash function with cloudpickle

**File**: `src/daglite/cache/core.py` (rewrite)

```python
import cloudpickle
import hashlib
import inspect

def default_cache_hash(func: Callable, bound_args: dict[str, Any]) -> str:
    """Generate cache key from function source and cloudpickle'd parameters."""
    h = hashlib.sha256()

    # Hash function source
    try:
        source = inspect.getsource(func)
        h.update(source.encode())
    except (OSError, TypeError):
        h.update(func.__qualname__.encode())

    # Hash each parameter via cloudpickle
    for name, value in sorted(bound_args.items()):
        h.update(name.encode())
        h.update(cloudpickle.dumps(value))

    return h.hexdigest()
```

- Removes dependency on `SerializationRegistry.hash_value()`.
- Uses `cloudpickle.dumps()` for all types (simpler, handles lambdas/closures).
- Trade-off: potentially slower for very large objects (numpy arrays, DataFrames).
  This is acceptable per user decision.

### 2.2 — Delete `serialization.py`

**File**: `src/daglite/serialization.py` (delete)

- Remove the entire `SerializationRegistry` class and all built-in type registrations.
- The only consumer is `cache/core.py` (which is being rewritten above).

### 2.3 — Deprecate `extras/serialization` package

**Files**: `extras/serialization/` directory

- The numpy/pandas/pillow hash strategies are no longer needed since we use cloudpickle.
- Mark the package as deprecated or remove it entirely.
- Update `pyproject.toml` to remove the `serialization` optional dependency.

### 2.4 — Remove serialization tests

**Files**:
- `tests/test_serialization.py` (delete)
- `extras/serialization/tests/test_serialization_plugin.py` (delete/deprecate)

---

## Phase 3: Make Caching a Built-in Feature

### 3.1 — Add `cache_store` to `DagliteSettings`

**File**: `src/daglite/settings.py`

```python
@dataclass(frozen=True)
class DagliteSettings:
    # ... existing fields ...

    cache_store: CacheStore | str | None = field(
        default_factory=lambda: os.getenv("DAGLITE_CACHE_STORE")
    )
    """
    Default cache store for @task(cache=True).

    Can be a CacheStore instance or a string path (creates CacheStore with FileDriver).
    If None, caching is disabled even when cache=True on tasks.
    """
```

### 3.2 — Move caching logic into worker (remove plugin indirection)

**File**: `src/daglite/graph/nodes/_workers.py`

In `_run_task_func`, replace the plugin hook calls with direct cache operations:

```python
# BEFORE (plugin-based):
cached_result = hook.check_cache(func=func, metadata=metadata, inputs=inputs, ...)
# ...
hook.update_cache(func=func, metadata=metadata, inputs=inputs, result=result, ...)

# AFTER (built-in):
if cache_enabled:
    cache_store = _get_cache_store()  # From execution context
    if cache_store is not None:
        cache_key = default_cache_hash(func, inputs)
        cached = cache_store.get(cache_key)
        if cached is not None:
            result = cached["value"]
            hook.on_cache_hit(...)  # Keep observability hook
            return result

# ... execute task ...

if cache_enabled and cache_store is not None:
    cache_store.put(cache_key, {"value": result}, ttl=cache_ttl)
```

**Key changes**:
- Caching logic is direct, not mediated through pluggy hooks.
- The `on_cache_hit` hook is preserved for observability (logging plugin can still react).
- Cache store is obtained from execution context (settings or per-run override).

### 3.3 — Thread `cache_store` through execution context

**File**: `src/daglite/backends/context.py`

Add cache store to the execution context (similar to how `plugin_manager` and
`dataset_reporter` are threaded):

```python
_cache_store_var: ContextVar[CacheStore | None] = ContextVar("cache_store", default=None)

def set_cache_store(store: CacheStore | None) -> Token:
    return _cache_store_var.set(store)

def get_cache_store() -> CacheStore | None:
    return _cache_store_var.get()
```

**File**: `src/daglite/engine.py`

In `evaluate_async()`:
```python
# Resolve cache store: explicit param > settings > None
cache_store = cache_store_param or settings.cache_store
if isinstance(cache_store, str):
    cache_store = CacheStore(cache_store)

# Set in context for workers
set_cache_store(cache_store)
```

### 3.4 — Add `cache_store` parameter to `.run()` / `.run_async()`

**File**: `src/daglite/futures/base.py`

```python
def run(self, *, plugins=None, cache_store=None) -> Any:
    from daglite.engine import evaluate
    return evaluate(self, plugins=plugins, cache_store=cache_store)

async def run_async(self, *, plugins=None, cache_store=None) -> Any:
    from daglite.engine import evaluate_async
    return await evaluate_async(self, plugins=plugins, cache_store=cache_store)
```

**File**: `src/daglite/engine.py`

Add `cache_store` parameter to `evaluate()`, `evaluate_async()`, etc.

### 3.5 — Remove `CachePlugin` and cache hook specs

**Files**:
- `src/daglite/plugins/builtin/cache.py` (delete)
- `src/daglite/plugins/hooks/specs.py` — remove `check_cache` and `update_cache` specs
  (keep `on_cache_hit` for observability)

### 3.6 — Update plugin manager

**File**: `src/daglite/plugins/manager.py`

- Remove auto-registration of `CachePlugin`.
- Update any references to cache plugin.

---

## Phase 4: Implement `.checkpoint()` Fluent Method

### 4.1 — Add `.checkpoint()` to `BaseTaskFuture`

**File**: `src/daglite/futures/base.py`

```python
def checkpoint(
    self,
    name: str,
    *,
    key: str | None = None,
    store: DatasetStore | str | None = None,
    format: str | None = None,
    **extras: Any,
) -> Self:
    """
    Mark this future's result for checkpointing (named, persistent save).

    The checkpoint is saved after task execution completes. It can be used for:
    - Pipeline resumption via .run(from_={name: key})
    - Debugging and inspection
    - Audit trails

    Args:
        name: Checkpoint name (used for resumption via from_=).
        key: Storage key template. If None, uses name as the key.
            Supports {param} format strings resolved from task params.
        store: Override dataset store for this checkpoint.
        format: Serialization format hint (e.g., "pickle").
        **extras: Additional values for key formatting. Can include
            TaskFutures whose results are resolved at execution time.

    Returns:
        Self for method chaining.

    Examples:
        Simple checkpoint:
        >>> result = expensive_task(data).checkpoint("step1").run()

        With key template and extras:
        >>> version = get_version()
        >>> result = train_model(data).checkpoint(
        ...     "trained_model",
        ...     key="models/{model_type}/{version}",
        ...     version=version,  # TaskFuture, resolved at execution time
        ... ).run()

        Chained checkpoints:
        >>> result = (
        ...     step1(data)
        ...     .checkpoint("raw_features")
        ...     .then(step2)
        ...     .checkpoint("normalized")
        ...     .then(step3)
        ...     .run()
        ... )
    """
    return self.save(
        key or name,
        save_checkpoint=name,
        save_store=store,
        save_format=format,
        **extras,
    )
```

This is a clean, focused API that delegates to the existing `.save()` infrastructure
with `save_checkpoint=name`. The user gets a nice fluent API while reusing all the
existing save/output machinery.

---

## Phase 5: Implement `checkpoint` and `from_` on `.run()` / `.run_async()`

### 5.1 — Add `checkpoint` and `from_` parameters to `.run()`

**File**: `src/daglite/futures/base.py`

```python
def run(
    self,
    *,
    plugins: list[Any] | None = None,
    cache_store: CacheStore | str | None = None,
    checkpoint: dict[str, str] | None = None,
    from_: dict[str, str] | None = None,
) -> Any:
    """
    Args:
        checkpoint: Dict of {checkpoint_name: key_template}. Saves the final
            result to each specified checkpoint after execution.
        from_: Dict of {checkpoint_name: key}. Attempts to resume from
            checkpoint instead of re-executing. Falls through to normal
            execution if checkpoint not found.
    """
```

### 5.2 — Implement `checkpoint` save in engine

**File**: `src/daglite/engine.py`

After graph execution completes, if `checkpoint` dict is provided, save the result:

```python
async def evaluate_async(future, *, plugins=None, cache_store=None, checkpoint=None, from_=None):
    # ... existing setup ...

    # Check from_ first: try to load from checkpoint
    if from_:
        loaded = _try_load_from_checkpoint(from_)
        if loaded is not None:
            return loaded

    # ... execute graph ...

    # Save checkpoints after execution
    if checkpoint:
        _save_checkpoints(result, checkpoint)

    return result
```

### 5.3 — Implement `from_` resumption in engine

**File**: `src/daglite/engine.py`

```python
def _try_load_from_checkpoint(
    from_: dict[str, str],
) -> Any | None:
    """
    Attempt to load the result from a previously saved checkpoint.

    Args:
        from_: Dict of {checkpoint_name: key} to try loading from.

    Returns:
        The loaded result, or None if no checkpoint found.
    """
    from daglite.settings import get_global_settings

    settings = get_global_settings()
    store = _resolve_dataset_store(settings.datastore_store)

    for name, key in from_.items():
        if store.exists(key):
            return store.load(key)

    return None
```

### 5.4 — Implement checkpoint save helper

```python
def _save_checkpoints(result: Any, checkpoint: dict[str, str]) -> None:
    """Save result to each checkpoint location."""
    from daglite.settings import get_global_settings

    settings = get_global_settings()
    store = _resolve_dataset_store(settings.datastore_store)

    for name, key in checkpoint.items():
        store.save(key, result)
```

---

## Phase 6: Cleanup & Tests

### 6.1 — Update existing cache tests

**File**: `tests/integration/test_cache.py`

- Remove dependency on `CachePlugin`
- Use `CacheStore` directly via settings or `.run(cache_store=...)`
- Test the new built-in caching flow

```python
def test_cache_hit():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = CacheStore(tmpdir)

        @task(cache=True)
        def expensive(x: int) -> int:
            return x * 2

        result1 = expensive(x=5).run(cache_store=store)
        result2 = expensive(x=5).run(cache_store=store)  # cache hit
        assert result1 == result2 == 10
```

### 6.2 — Add checkpoint tests

**File**: `tests/integration/test_checkpoint.py` (new)

```python
def test_checkpoint_fluent():
    """Test .checkpoint() fluent method saves result."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = DatasetStore(tmpdir)
        result = my_task(x=5).checkpoint("step1", store=store).run()
        assert store.exists("step1")

def test_checkpoint_with_extras():
    """Test .checkpoint() with TaskFuture extras for key formatting."""
    version = get_version()
    result = train(data).checkpoint(
        "model", key="models/{version}", version=version
    ).run()

def test_run_checkpoint_dict():
    """Test .run(checkpoint={...}) saves final result."""
    result = my_task(x=5).run(checkpoint={"output": "results/output"})

def test_run_from_checkpoint():
    """Test .run(from_={...}) resumes from checkpoint."""
    # First run saves
    result1 = my_task(x=5).run(checkpoint={"output": "results/output"})
    # Second run resumes
    result2 = my_task(x=5).run(from_={"output": "results/output"})
    assert result1 == result2

def test_from_fallthrough():
    """Test from_ falls through to execution if checkpoint missing."""
    result = my_task(x=5).run(from_={"missing": "nonexistent"})
    assert result == expected_value
```

### 6.3 — Update unit tests for cache plugin removal

**File**: `tests/plugins/builtin/test_cache.py`

- Remove or rewrite tests that reference `CachePlugin`.
- Add tests for the built-in cache store behavior.

### 6.4 — Remove serialization tests

**Files**:
- `tests/test_serialization.py` — delete (SerializationRegistry removed)
- `extras/serialization/tests/test_serialization_plugin.py` — delete (package deprecated)

### 6.5 — Update imports and references

Grep the codebase for all references to:
- `SerializationRegistry`, `default_registry` (from serialization)
- `FileCacheStore` (renamed to `CacheStore`)
- `CachePlugin`
- `check_cache`, `update_cache` (removed hook specs)

---

## Files Changed Summary

### New Files
- (none — we rewrite existing files)

### Modified Files
- `pyproject.toml` — add cloudpickle dependency, remove serialization optional dep
- `src/daglite/cache/__init__.py` — update exports
- `src/daglite/cache/store.py` — rewrite with Driver + cloudpickle
- `src/daglite/cache/core.py` — rewrite hash to use cloudpickle
- `src/daglite/settings.py` — add `cache_store` field
- `src/daglite/futures/base.py` — add `.checkpoint()`, update `.run()` / `.run_async()`
- `src/daglite/engine.py` — add `cache_store`/`checkpoint`/`from_` params
- `src/daglite/graph/nodes/_workers.py` — inline cache logic, remove hook calls
- `src/daglite/backends/context.py` — add cache store context var
- `src/daglite/plugins/hooks/specs.py` — remove `check_cache`/`update_cache`
- `src/daglite/workflows.py` — thread new params through workflow run
- `tests/integration/test_cache.py` — rewrite for built-in caching
- `tests/integration/test_checkpoint.py` — new checkpoint tests

### Deleted Files
- `src/daglite/cache/base.py` — CacheStore protocol (replaced by concrete class)
- `src/daglite/serialization.py` — entire SerializationRegistry
- `src/daglite/plugins/builtin/cache.py` — CachePlugin (logic moved to workers)
- `tests/test_serialization.py` — serialization tests
- `extras/serialization/` — deprecated (optionally kept for backwards compat)

---

## Dependency Changes

| Package | Action | Reason |
|---------|--------|--------|
| `cloudpickle>=3.0` | **Add** | Replaces SerializationRegistry for serialization + hashing |
| `fsspec` | Keep | Still used by FileDriver (underlying CacheStore) |
| `pluggy` | Keep | Still used for observability hooks (on_cache_hit, etc.) |

---

## Migration Notes

### For users of `CachePlugin`:

```python
# BEFORE (v0.8):
from daglite.cache.store import FileCacheStore
from daglite.plugins.builtin.cache import CachePlugin

store = FileCacheStore("/tmp/cache")
plugin = CachePlugin(store=store)
result = my_task(x=5).run(plugins=[plugin])

# AFTER (v0.9):
from daglite.cache import CacheStore

store = CacheStore("/tmp/cache")
result = my_task(x=5).run(cache_store=store)

# Or via settings (no per-run config needed):
from daglite.settings import set_global_settings, DagliteSettings
set_global_settings(DagliteSettings(cache_store="/tmp/cache"))
result = my_task(x=5).run()  # Uses global cache store
```

### For users of `SerializationRegistry`:

```python
# BEFORE (v0.8):
from daglite.serialization import default_registry
default_registry.register(MyType, serializer, deserializer, ...)
default_registry.register_hash_strategy(MyType, hasher, ...)

# AFTER (v0.9):
# No registration needed — cloudpickle handles all serializable types.
# If a type isn't cloudpickle-serializable, implement __reduce__ or __getstate__.
```

---

## Implementation Order

1. **Phase 1** (CacheStore refactor) — Foundation, no breaking changes yet
2. **Phase 2** (Remove SerializationRegistry) — Clean break from old infrastructure
3. **Phase 3** (Built-in caching) — Core feature, removes plugin indirection
4. **Phase 4** (.checkpoint() fluent method) — User-facing API addition
5. **Phase 5** (checkpoint/from_ on .run()) — Complete checkpoint system
6. **Phase 6** (Tests & cleanup) — Verify everything works end-to-end
