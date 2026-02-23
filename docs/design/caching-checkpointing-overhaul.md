# Caching & Checkpointing Overhaul — Implementation Plan

**Status**: Ready for Review
**Created**: 2026-02-23
**Target**: daglite v0.9.0
**Branch**: `claude/caching-checkpoints-overhaul-8gm39`

---

## Motivation

Caching in daglite currently works through the plugin system — users must manually
instantiate `CachePlugin(store=FileCacheStore(...))` and pass it to every `.run()`
call.  This is cumbersome and un-ergonomic compared to competitors like Prefect
(`@task(cache_key_fn=...)`) or Hamilton (`@cache`).  The goal is to make caching
**built-in**: `@task(cache=True)` should *just work* once a cache store is
configured in settings.

Additionally, the existing `SerializationRegistry` (629 lines) maintains its own
type→handler mapping, but in practice the codebase only uses `pickle` for actual
data persistence.  Replacing the serialization layer with `cloudpickle` simplifies
the code substantially while gaining support for lambdas, closures, and dynamic
classes.

Finally, we implement **checkpointing** — the ability to resume a graph from a
previously saved intermediate result — by extending the existing `.save(save_checkpoint=...)`
mechanism with a corresponding `from_checkpoint` parameter on `.run()` / `.run_async()`.

---

## Overview of Changes

| Area | Current State | Target State |
|------|--------------|--------------|
| Serialization | `SerializationRegistry` with per-type handlers | `cloudpickle` for serialization; keep hash strategies only |
| Cache wiring | Plugin-based (`CachePlugin` + hook system) | Built-in via `DagliteSettings.cache_store` + context var |
| Cache activation | `future.run(plugins=[CachePlugin(store)])` | `future.run(cache_store=...)` or settings-level |
| Checkpoint write | `.save(save_checkpoint="name")` exists | No change (already works) |
| Checkpoint resume | Not implemented | `future.run(from_checkpoint="name")` prunes DAG |
| Process backends | Cache store via plugin serialization | Cache store via context variable + `cloudpickle` |

---

## Phase 1: Replace Serialization with cloudpickle

### 1.1  Add `cloudpickle` dependency

**File**: `pyproject.toml`

Add `"cloudpickle>=3.0"` to `dependencies`.

### 1.2  Simplify `serialization.py`

**File**: `src/daglite/serialization.py` (currently 629 lines → ~200 lines)

Keep:
- `SerializationRegistry` class
- `hash_value()` method and all hash strategies (used by `default_cache_hash`)
- `register_hash_strategy()` for external types (numpy, pandas, pillow)
- `default_registry` global instance
- All built-in hash strategies (int, str, float, bytes, list, dict, tuple, set, frozenset)

Replace:
- `serialize()` → `cloudpickle.dumps(obj)`
- `deserialize()` → `cloudpickle.loads(data)`

Remove:
- Per-type serializer/deserializer registration (`register()` with serializer/deserializer pairs)
- `SerializationHandler` dataclass
- Format registration/lookup (`get_format_from_extension`, `get_formats_for_extension`, `get_extension`, `set_default_format`)
- File extension tracking

The `extras/serialization/daglite-serialization/` plugin continues to work — it
registers **hash strategies** for numpy/pandas/pillow, which are unaffected.

### 1.3  Update `FileCacheStore` to use cloudpickle

**File**: `src/daglite/cache/store.py`

Replace `import pickle` with `import cloudpickle` and update the `put()`/`get()`
methods to use `cloudpickle.dumps()`/`cloudpickle.loads()`.

### 1.4  Update dataset store fallback

**File**: `src/daglite/datasets/store.py`

The `load()` method has a pickle-based fallback path. Replace
`pickle.loads(data)` with `cloudpickle.loads(data)`.  Similarly update `save()`
if it uses `pickle.dumps`.

### 1.5  Update tests

**File**: `tests/test_serialization.py`

Adapt tests to the simplified API:
- Keep all hash-related tests (determinism, uniqueness, stability)
- Remove/update tests for per-type serializer/deserializer registration
- Add test for cloudpickle round-trip with lambdas/closures

---

## Phase 2: Make Caching Built-in

### 2.1  Add `cache_store` to `DagliteSettings`

**File**: `src/daglite/settings.py`

```python
cache_store: str | CacheStore | None = field(
    default_factory=lambda: os.getenv("DAGLITE_CACHE_STORE") or None
)
"""
Default cache store for @task(cache=True).

Can be:
- A CacheStore instance (e.g., FileCacheStore("/tmp/cache"))
- A string path (auto-creates FileCacheStore at that path)
- None (caching disabled — tasks with cache=True execute normally)

Can also be set via DAGLITE_CACHE_STORE environment variable.
"""
```

Use `TYPE_CHECKING` guard for the `CacheStore` import to avoid circular imports.

### 2.2  Add cache store to execution context

**File**: `src/daglite/backends/context.py`

Add a new context variable and accessors:

```python
_cache_store: ContextVar[CacheStore | None] = ContextVar("cache_store", default=None)

def set_execution_context(
    plugin_manager, event_reporter, dataset_reporter=None,
    cache_store=None,  # NEW
) -> tuple[...]:
    return (
        _plugin_manager.set(plugin_manager),
        _event_reporter.set(event_reporter),
        _dataset_reporter.set(dataset_reporter),
        _cache_store.set(cache_store),  # NEW
    )

def get_cache_store() -> CacheStore | None:
    return _cache_store.get()
```

Update `reset_execution_context()` to also reset `_cache_store`.

### 2.3  Resolve cache store in engine

**File**: `src/daglite/engine.py`

Add a helper and thread it through:

```python
def _resolve_cache_store(explicit: Any = None) -> CacheStore | None:
    """Resolve cache store: explicit param → settings → None."""
    from daglite.cache.store import FileCacheStore
    from daglite.settings import get_global_settings

    store = explicit
    if store is None:
        store = get_global_settings().cache_store

    if isinstance(store, str):
        return FileCacheStore(store)
    return store  # CacheStore instance or None
```

Update `evaluate_async()` and `evaluate_workflow_async()` signatures:

```python
async def evaluate_async(
    future, *, plugins=None, cache_store=None  # NEW
) -> Any:
    resolved_store = _resolve_cache_store(cache_store)
    # ... existing setup ...
    backend_manager = BackendManager(
        plugin_manager, event_processor, dataset_processor,
        cache_store=resolved_store,  # NEW
    )
```

### 2.4  Thread cache store through backends

**File**: `src/daglite/backends/manager.py`

Accept `cache_store` in `BackendManager.__init__()` and pass it through to
`backend_instance.start()`.

**File**: `src/daglite/backends/base.py`

Add `cache_store` attribute to `Backend`, accepted in `start()`:

```python
def start(self, plugin_manager, event_processor, dataset_processor,
          cache_store=None):  # NEW
    self.cache_store = cache_store
    # ... rest unchanged ...
```

**File**: `src/daglite/backends/impl/local.py`

Pass `cache_store` to `set_execution_context()` in all three backends:

- `InlineBackend.submit()` — direct call to `set_execution_context(..., cache_store=self.cache_store)`
- `_thread_initializer()` — accept `cache_store` parameter, pass to `set_execution_context()`
- `_process_initializer()` — `cloudpickle` handles `FileCacheStore` serialization; pass through queue or re-create from path

For `ProcessBackend`, the simplest approach is to pass the cache store's
configuration (e.g., base path string) and reconstruct it in the child process,
since `FileCacheStore` with fsspec may not be directly picklable. Alternative: just
cloudpickle the store object.

### 2.5  Replace hook-based cache calls with direct calls

**File**: `src/daglite/graph/nodes/_workers.py`

In `_run_task_func()`, replace the plugin hook calls with direct cache store access:

**Before** (lines 212-234):
```python
cached_result = hook.check_cache(
    func=func, metadata=metadata, inputs=inputs,
    cache_enabled=cache_enabled, cache_ttl=cache_ttl,
)
if cached_result is not None:
    result = cached_result["value"] if isinstance(cached_result, dict) ...
    hook.on_cache_hit(...)
    return result
```

**After**:
```python
if cache_enabled:
    from daglite.backends.context import get_cache_store
    from daglite.cache.core import default_cache_hash

    cache_store = get_cache_store()
    if cache_store is not None:
        cache_key = default_cache_hash(func, inputs)
        cached_wrapper = cache_store.get(cache_key, return_type=dict)
        if (cached_wrapper is not None
                and isinstance(cached_wrapper, dict)
                and "value" in cached_wrapper):
            result = cached_wrapper["value"]
            hook.on_cache_hit(
                func=func, metadata=metadata, inputs=inputs,
                result=result, reporter=reporter,
            )
            reset_current_task(token)
            return result
```

Similarly replace `hook.update_cache(...)` (lines 292-299):

```python
if cache_enabled:
    cache_store = get_cache_store()
    if cache_store is not None:
        cache_key = default_cache_hash(func, inputs)
        cache_store.put(cache_key, {"value": result}, ttl=cache_ttl)
```

**Keep** `hook.on_cache_hit()` — it remains useful for logging/observability plugins.

### 2.6  Add `cache_store` parameter to `.run()` / `.run_async()`

**File**: `src/daglite/futures/base.py`

```python
def run(self, *, plugins=None, cache_store=None) -> Any:
    from daglite.engine import evaluate
    return evaluate(self, plugins=plugins, cache_store=cache_store)

async def run_async(self, *, plugins=None, cache_store=None) -> Any:
    from daglite.engine import evaluate_async
    return await evaluate_async(self, plugins=plugins, cache_store=cache_store)
```

**File**: `src/daglite/futures/task_future.py`

Update all `run()` / `run_async()` method overloads to accept and pass through `cache_store`.

### 2.7  Deprecate `CachePlugin` and plugin hooks

**File**: `src/daglite/plugins/builtin/cache.py`

Add deprecation warning:

```python
class CachePlugin:
    """Deprecated: Caching is now built-in. Use DagliteSettings.cache_store instead."""
    def __init__(self, store):
        import warnings
        warnings.warn(
            "CachePlugin is deprecated. Configure cache_store in DagliteSettings "
            "or pass cache_store= to .run() instead.",
            DeprecationWarning, stacklevel=2,
        )
        self.store = store
    # ... keep hook_impl methods for backward compatibility ...
```

**File**: `src/daglite/plugins/hooks/specs.py`

Add deprecation notes to `check_cache` and `update_cache` docstrings. Keep `on_cache_hit`
as a supported hook.

### 2.8  Export cache types from `__init__.py`

**File**: `src/daglite/__init__.py`

```python
from daglite.cache.store import FileCacheStore

__all__ = [
    # ... existing ...
    "FileCacheStore",
]
```

---

## Phase 3: Implement Checkpoint Resumption

### 3.1  Design overview

The `.save(save_checkpoint="name")` mechanism already writes checkpoint data during
execution. What's missing is the **resume** path: loading a checkpoint and skipping
upstream nodes.

The approach:

1. When `from_checkpoint` is passed to `.run()`, the engine builds the graph normally
2. It scans all nodes for `NodeOutputConfig` entries with matching checkpoint names
3. For each match, it loads the checkpointed value from the dataset store
4. It marks the target node **and all its ancestors** as complete (DAG pruning)
5. Execution proceeds from the checkpoint forward

### 3.2  Add `CheckpointError` exception

**File**: `src/daglite/exceptions.py`

```python
class CheckpointError(DagliteError):
    """Raised when checkpoint loading or resumption fails."""
```

### 3.3  Add `from_checkpoint` parameter to `.run()` / `.run_async()`

**File**: `src/daglite/futures/base.py`

```python
def run(
    self, *,
    plugins=None,
    cache_store=None,
    from_checkpoint: str | dict[str, Any] | None = None,
) -> Any:
    """
    Args:
        from_checkpoint: Resume from a named checkpoint.
            - str: Checkpoint name — loads value from the dataset store
            - dict: Map of {checkpoint_name: pre-computed_value}
            - None: Normal execution (default)
    """
    from daglite.engine import evaluate
    return evaluate(self, plugins=plugins, cache_store=cache_store,
                    from_checkpoint=from_checkpoint)
```

Same for `run_async()`.

### 3.4  Implement DAG pruning in the engine

**File**: `src/daglite/engine.py`

Add checkpoint-related helpers:

```python
def _apply_checkpoint(
    state: _ExecutionState,
    from_checkpoint: str | dict[str, Any],
) -> None:
    """Pre-populate execution state with checkpoint data, pruning upstream nodes."""
    if isinstance(from_checkpoint, str):
        checkpoint_values = _load_checkpoint(state, from_checkpoint)
    else:
        checkpoint_values = from_checkpoint

    # Find nodes whose output_configs have matching checkpoint names
    for nid, node in state.nodes.items():
        for config in node.output_configs:
            if config.name and config.name in checkpoint_values:
                _prune_upstream(state, nid, checkpoint_values[config.name])
                break

def _prune_upstream(
    state: _ExecutionState,
    target_nid: UUID,
    value: Any,
) -> None:
    """Mark target node and all ancestors as complete, removing them from execution."""
    # BFS backward through dependencies
    ancestors = set()
    queue = collections.deque()

    # Collect all ancestors
    node = state.nodes[target_nid]
    for dep_id in node.get_dependencies():
        queue.append(dep_id)
    while queue:
        nid = queue.popleft()
        if nid in ancestors:
            continue
        ancestors.add(nid)
        dep_node = state.nodes[nid]
        for dep_id in dep_node.get_dependencies():
            queue.append(dep_id)

    _PRUNED = object()  # sentinel for ancestor results

    # Mark ancestors as complete with sentinel values
    for nid in ancestors:
        state.completed_nodes[nid] = _PRUNED
        if nid in state.indegree:
            del state.indegree[nid]

    # Mark target node with actual value
    state.completed_nodes[target_nid] = value
    if target_nid in state.indegree:
        del state.indegree[target_nid]

    # Update indegrees: decrement for each pruned node's successors
    all_pruned = ancestors | {target_nid}
    for pruned_nid in all_pruned:
        for succ_nid in state.successors.get(pruned_nid, ()):
            if succ_nid in state.indegree:
                state.indegree[succ_nid] -= 1


def _load_checkpoint(
    state: _ExecutionState,
    checkpoint_name: str,
) -> dict[str, Any]:
    """Load checkpoint value from the dataset store."""
    from daglite.datasets.store import DatasetStore
    from daglite.settings import get_global_settings

    # Find the node with this checkpoint name to get its store config
    for node in state.nodes.values():
        for config in node.output_configs:
            if config.name == checkpoint_name:
                store = config.store
                if store is None:
                    store_or_path = get_global_settings().datastore_store
                    if isinstance(store_or_path, str):
                        store = DatasetStore(store_or_path)
                    else:
                        store = store_or_path
                value = store.load(config.key, format=config.format)
                return {checkpoint_name: value}

    raise CheckpointError(
        f"No checkpoint named '{checkpoint_name}' found in the graph. "
        f"Ensure a task in the graph uses .save(save_checkpoint='{checkpoint_name}')."
    )
```

Wire into `evaluate_async()`:

```python
async def evaluate_async(
    future, *, plugins=None, cache_store=None, from_checkpoint=None,
) -> Any:
    state = _setup_graph_execution_state(future)

    if from_checkpoint is not None:
        _apply_checkpoint(state, from_checkpoint)

    # ... rest of execution unchanged ...
```

### 3.5  Handle checkpoint key template resolution

The `NodeOutputConfig.key` may contain `{param}` placeholders (e.g., `"model_{model_type}"`).
When loading a checkpoint by name, we need the resolved key.

Two approaches:
1. **Simple**: Require the user to pass the resolved key when loading by name (or store the
   resolved key in a manifest file during save)
2. **Manifest-based**: Write a small JSON manifest alongside the checkpoint data recording
   the mapping from checkpoint name → resolved key

For the initial implementation, use approach (1): the `from_checkpoint` dict allows
passing `{name: value}` directly.  For the string form, we require the key template to
have no placeholders (or all placeholders have already been resolved).

A future enhancement can add manifest-based checkpoint tracking.

---

## Phase 4: Cleanup & Public API

### 4.1  Remove stale plugin hook invocations

After Phase 2, `check_cache` and `update_cache` hooks are no longer called by the
built-in cache path. However, if a user has registered a `CachePlugin` (deprecated),
it should still fire. The hooks remain in `WorkerSideNodeSpecs` but are documented
as deprecated.

### 4.2  Update `evaluate_workflow()` and `evaluate_workflow_async()`

These must also accept `cache_store` and `from_checkpoint` parameters and thread
them through, mirroring the changes to `evaluate_async()`.

### 4.3  Update `Workflow.run()` / `Workflow.run_async()`

**File**: `src/daglite/workflows.py`

Accept and forward `cache_store` and `from_checkpoint` parameters.

---

## Phase 5: Testing

### 5.1  New unit tests for built-in caching

**File**: `tests/cache/test_builtin_cache.py`

```
test_cache_via_settings             — settings.cache_store enables caching
test_cache_via_run_param            — .run(cache_store="...") enables caching
test_cache_hit_skips_execution      — second call returns cached value
test_cache_miss_computes            — first call executes the function
test_no_store_executes_normally     — cache=True with no store just runs
test_cache_ttl_expiration           — expired entries are not returned
test_cache_none_result              — tasks returning None are cached correctly
test_cache_with_cloudpickle_lambda  — cloudpickle handles lambdas in results
test_async_task_caching             — async tasks work with built-in cache
test_deprecated_cache_plugin_warns  — CachePlugin emits DeprecationWarning
```

### 5.2  New unit tests for checkpointing

**File**: `tests/test_checkpoint.py`

```
test_from_checkpoint_dict              — dict values pre-populate state
test_from_checkpoint_string            — string loads from store
test_from_checkpoint_skips_upstream    — ancestors are not executed
test_from_checkpoint_unknown_name      — raises CheckpointError
test_checkpoint_write_then_resume      — round-trip: save then load
test_checkpoint_with_multiple_sinks    — workflow with multiple checkpoints
```

### 5.3  New integration tests

**File**: `tests/integration/test_builtin_cache_e2e.py`

```
test_e2e_caching_with_settings        — full pipeline
test_e2e_checkpoint_resume            — save checkpoint, re-run from it
test_e2e_caching_multibackend         — threading + inline mixed
```

### 5.4  Update existing tests

- `tests/test_serialization.py` — adapt to simplified cloudpickle API
- `tests/integration/test_cache.py` — update tests to use built-in approach
- `tests/plugins/builtin/test_cache.py` — update for deprecation warnings
- `tests/cache/test_store.py` — update for cloudpickle (should mostly pass as-is)

---

## File Change Summary

### New Files

| File | Purpose |
|------|---------|
| `tests/cache/test_builtin_cache.py` | Unit tests for built-in caching |
| `tests/test_checkpoint.py` | Unit tests for checkpoint resumption |
| `tests/integration/test_builtin_cache_e2e.py` | End-to-end integration tests |

### Modified Files (in implementation order)

| # | File | Changes |
|---|------|---------|
| 1 | `pyproject.toml` | Add `cloudpickle>=3.0` dependency |
| 2 | `src/daglite/serialization.py` | Simplify to cloudpickle + hash strategies |
| 3 | `src/daglite/cache/store.py` | `pickle` → `cloudpickle` |
| 4 | `src/daglite/datasets/store.py` | `pickle` → `cloudpickle` in fallback |
| 5 | `src/daglite/datasets/builtin.py` | `pickle` → `cloudpickle` in PickleDataset |
| 6 | `src/daglite/settings.py` | Add `cache_store` field |
| 7 | `src/daglite/backends/context.py` | Add `_cache_store` context var |
| 8 | `src/daglite/backends/base.py` | Add `cache_store` to `Backend.start()` |
| 9 | `src/daglite/backends/manager.py` | Accept + forward `cache_store` |
| 10 | `src/daglite/backends/impl/local.py` | Thread `cache_store` through all backends |
| 11 | `src/daglite/graph/nodes/_workers.py` | Direct cache store calls (replace hooks) |
| 12 | `src/daglite/engine.py` | `cache_store` + `from_checkpoint` params, `_apply_checkpoint` |
| 13 | `src/daglite/futures/base.py` | `cache_store` + `from_checkpoint` on `.run()` |
| 14 | `src/daglite/futures/task_future.py` | Update overloads |
| 15 | `src/daglite/workflows.py` | Forward new params |
| 16 | `src/daglite/exceptions.py` | Add `CheckpointError` |
| 17 | `src/daglite/plugins/builtin/cache.py` | Deprecation warning |
| 18 | `src/daglite/plugins/hooks/specs.py` | Deprecation notes on `check_cache`/`update_cache` |
| 19 | `src/daglite/__init__.py` | Export `FileCacheStore` |
| 20 | `tests/test_serialization.py` | Adapt to simplified API |
| 21 | `tests/integration/test_cache.py` | Update for built-in approach |
| 22 | `tests/plugins/builtin/test_cache.py` | Update for deprecation |

---

## Migration Guide

### Before (v0.8.x)

```python
from daglite.cache.store import FileCacheStore
from daglite.plugins.builtin.cache import CachePlugin

store = FileCacheStore("/tmp/cache")
plugin = CachePlugin(store=store)
result = my_task(x=5).run(plugins=[plugin])
```

### After (v0.9.0) — Option A: Settings-level

```python
from daglite.settings import set_global_settings, DagliteSettings

set_global_settings(DagliteSettings(cache_store="/tmp/cache"))
result = my_task(x=5).run()  # Just works!
```

### After (v0.9.0) — Option B: Per-run

```python
result = my_task(x=5).run(cache_store="/tmp/cache")
```

### After (v0.9.0) — Option C: Environment variable

```bash
export DAGLITE_CACHE_STORE=/tmp/cache
```

```python
result = my_task(x=5).run()  # Picks up from env
```

### Checkpoint Usage

```python
# Pipeline with checkpoint
@task
def step_a(x: int) -> int:
    return x * 2

@task
def step_b(y: int) -> int:
    return y + 100

result = step_b(y=step_a(x=5).save("step_a_output", save_checkpoint="after_a")).run()

# Later: resume from checkpoint
result = step_b(y=step_a(x=5).save("step_a_output", save_checkpoint="after_a")).run(
    from_checkpoint={"after_a": 10}  # Skip step_a, inject value 10
)
```

---

## Key Design Decisions

1. **Cache store resolution order**: explicit `.run(cache_store=)` → `DagliteSettings.cache_store` → `None` (no caching)

2. **Keep `on_cache_hit` hook**: Logging/observability plugins benefit from cache hit notifications.  Deprecate `check_cache` and `update_cache` only.

3. **`from_checkpoint` semantics**:
   - `str` → load by checkpoint name from dataset store
   - `dict[str, Any]` → inject pre-computed values (power-user escape hatch)
   - DAG pruning marks the checkpoint node and all ancestors as complete

4. **Checkpoints reuse dataset infrastructure**: `.save(save_checkpoint=...)` already writes data via `DatasetStore`/`DatasetNode`.  Checkpoint resumption just reads it back and feeds it into the execution state.

5. **cloudpickle replaces pickle everywhere**: Single serialization strategy for cache + datasets + checkpoint.  Hash strategies remain for cache key computation.

6. **No `CachePlugin` needed for backwards compat**: The deprecated `CachePlugin` still works (its hooks fire), but the built-in cache takes precedence if configured.

---

## Implementation Sequence

Recommended order (respecting dependencies):

1. **Phase 1** (cloudpickle) — Foundation with no breaking changes
2. **Phase 2.1–2.2** (settings + context var) — Infrastructure
3. **Phase 2.3–2.4** (engine + backends) — Core wiring
4. **Phase 2.5** (workers) — The critical built-in cache change
5. **Phase 2.6–2.8** (API surface) — Complete built-in cache
6. **Phase 3** (checkpoint resumption) — DAG pruning + from_checkpoint
7. **Phase 4** (cleanup) — Deprecation, API polish
8. **Phase 5** (testing) — Comprehensive test suite
