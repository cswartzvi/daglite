# Async-First Engine Refactoring Plan

## Goal

Unify `evaluate()` and `evaluate_async()` to use a single async-first internal implementation, eliminating code duplication while giving sync users automatic sibling concurrency.

## Key Decisions

### 1. `evaluate()` wraps `evaluate_async()`

The module-level `evaluate()` function detects whether an event loop is already running and delegates to `evaluate_async()`:

```python
def evaluate(expr, *, plugins=None):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass  # No loop running, safe to proceed
    else:
        raise RuntimeError("Use evaluate_async() in async contexts")
    return asyncio.run(evaluate_async(expr, plugins=plugins))
```

`Engine.evaluate()` similarly wraps `Engine.evaluate_async()`:

```python
def evaluate(self, root: GraphBuilder) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError("Use evaluate_async() in async contexts")
    return asyncio.run(self.evaluate_async(root))
```

### 2. Remove `_run_Inline` and all sync-only execution code

Only keep `_run_async` as the execution path. This removes:

**engine.py:**
- `_run_Inline` (line 366) — sync execution loop
- `_submit_node_sync` (line 499) — sync node submission
- `_collect_result_sync` (line 547) — sync result collection
- `_NodeFutureWrapper` (line 640) — sync future wrapper
- `_MapFutureWrapper` (line 648) — mapped sync future wrapper
- `_materialize_sync` (line 754) — sync generator materialization
- `_validate_sync_compatibility` (line 330) — no longer needed (see decision 3)
- `_validate_async_compatibility` (line 347) — no longer needed (see decision 3)

**nodes.py:**
- `_run_sync_impl` (line 290) — sync node implementation (~148 lines)
- `TaskNode.run()` (line 82) — sync execution method
- `MapTaskNode.run()` (line 245) — sync mapped execution method

**Total: ~450 lines of duplicated code removed.**

### 3. Remove both validation functions

- **`_validate_sync_compatibility`** (line 330): No longer needed because `evaluate()` now delegates to the async path, which handles both sync and async task functions via `_run_async_impl`.

- **`_validate_async_compatibility`** (line 347): Sync functions on the Inline backend are suboptimal (they block the event loop) but not incorrect. Removing this validation simplifies the API — users no longer need to think about backend/function compatibility. If performance matters, they can switch backends.

### 4. `_run_async_impl` already handles sync functions (nodes.py line 522)

No changes needed here. The existing code handles both sync and async task functions:

```python
if inspect.iscoroutinefunction(func):
    result = await func(**resolved_inputs)
else:
    result = func(**resolved_inputs)
```

### 5. Use `FIRST_EXCEPTION` for fail-fast

Switch from `ALL_COMPLETED` (current default) to `FIRST_EXCEPTION` to cancel pending siblings immediately on failure instead of waiting for all to finish:

```python
done, pending = await asyncio.wait(tasks.keys(), return_when=asyncio.FIRST_EXCEPTION)
try:
    for task in done:
        nid = tasks[task]
        result = task.result()
        ready.extend(state.mark_complete(nid, result))
except Exception:
    for t in pending:
        t.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    raise
```

**Behavioral change**: In-flight sibling tasks are cancelled on first failure rather than being allowed to complete. This is the desired fail-fast behavior.

### 6. Remove `is_async` from the hook system

The `is_async` parameter on graph-level hooks (`before_graph_execute`, `after_graph_execute`, `on_graph_error`) becomes meaningless when all execution is async-first. Remove it from:

**Hook specs** (`plugins/hooks/specs.py`):
- `GraphSpec.before_graph_execute` (line 246)
- `GraphSpec.after_graph_execute` (line 265)
- `GraphSpec.on_graph_error` (line 285)

**Builtin plugins** (`plugins/builtin/logging.py`):
- `LifecycleLoggingPlugin.before_graph_execute` (line 300)
- `LifecycleLoggingPlugin.after_graph_execute` (line 314)
- `LifecycleLoggingPlugin.on_graph_error` (line 328)

**Extras plugins** (`extras/rich/src/daglite_rich/`):
- `RichProgressPlugin.before_graph_execute` (progress.py line 81)
- `RichProgressPlugin.after_graph_execute` (progress.py line 161)
- `RichLifecycleLoggingPlugin` (logging.py — inherits from LifecycleLoggingPlugin)

**Test plugins** (`tests/examples/plugins.py`):
- `CounterPlugin` (line 43, 48)
- `ParameterCapturePlugin` (lines 92, 98, 103, 110)
- `OrderTrackingPlugin` (lines 175, 180)
- `ErrorRaisingPlugin` (lines 229, 235)

**Tests** (`tests/evaluation/test_plugins.py`):
- Assertions checking `is_async` value (lines 185, 190)
- Custom hook implementations in tests (line 247)

**Note**: The `Task.is_async` field (`tasks.py` line 402) is a different concept — it tracks whether the task function itself is async. This stays.

### 7. Add nested call guard

Prevent `evaluate()`/`evaluate_async()` from being called within a task function using a `ContextVar`:

```python
_IN_EVALUATION: ContextVar[bool] = ContextVar("in_evaluation", default=False)

async def evaluate_async(expr, *, plugins=None):
    if _IN_EVALUATION.get():
        raise RuntimeError(
            "Cannot call evaluate()/evaluate_async() from within a task. "
            "Compose tasks using the fluent API instead."
        )
    token = _IN_EVALUATION.set(True)
    try:
        engine = Engine(plugins=plugins)
        return await engine.evaluate_async(expr)
    finally:
        _IN_EVALUATION.reset(token)
```

This guard belongs in the module-level entry points (`evaluate` and `evaluate_async`), not in the `Engine` class, since the `Engine` methods are the internal implementation.

### 8. Keep plugin hooks synchronous

All plugin hooks remain synchronous. The rationale:

- Worker-side hooks **cannot** be async (they run in thread/process pool workers with no event loop)
- Graph-level and coordinator-side hooks are quick operations that don't benefit from being async
- pluggy doesn't natively support async hooks well
- Converting would require updating all plugin implementations (builtin, extras, tests) for minimal benefit

If a specific async hook is needed later (e.g., async cache backend), add a targeted async variant rather than converting the whole system.

## Testing Strategy

Since both `evaluate()` and `evaluate_async()` now share a single code path (`_run_async`), parameterizing every test over both modes adds limited value. Instead:

**Entry-point tests** — targeted tests for the thin `evaluate()` wrapper:
```python
def test_evaluate_rejects_running_loop():
    """evaluate() raises when called from async context."""
    async def inner():
        with pytest.raises(RuntimeError, match="Use evaluate_async"):
            evaluate(my_task(x=1))
    asyncio.run(inner())

def test_evaluate_rejects_nested_call():
    """evaluate() raises when called from within a task."""
    @task
    def outer():
        return evaluate(inner_task(x=1))  # Should raise
    with pytest.raises(RuntimeError, match="Cannot call evaluate"):
        evaluate(outer())
```

**Core evaluation tests** — use `evaluate_async()` directly:
```python
async def test_basic_chain():
    result = await evaluate_async(my_task(x=1))
    assert result == expected
```

**Existing `evaluate()` tests** — keep working as-is (they exercise the `asyncio.run` wrapper path). No need to rewrite them unless they tested sync-specific internals.

## Performance

Async overhead is negligible (~microseconds per task) for real workloads. The per-node cost of `asyncio.create_task` + `wrap_future` + `await` is 1-5 microseconds. For a 100-node DAG, that's 0.1-0.5ms total — dwarfed by any actual task execution time. The current sync path also has overhead from `concurrent.futures.wait()`.

The only scenario where this matters is a micro-benchmark with hundreds of trivial no-op tasks. No real workload looks like that. If profiling proves otherwise, optimize then.

## Migration Checklist

1. Add nested call guard (`ContextVar`) to `evaluate()` and `evaluate_async()`
2. Rewrite `Engine.evaluate()` to wrap `Engine.evaluate_async()` with loop detection
3. Remove `_run_Inline` and all sync-only code paths from `engine.py`
4. Remove `_run_sync_impl`, `TaskNode.run()`, `MapTaskNode.run()` from `nodes.py`
5. Remove `_validate_sync_compatibility` and `_validate_async_compatibility`
6. Switch `_run_async` to use `FIRST_EXCEPTION` in `asyncio.wait()`
7. Remove `is_async` parameter from hook specs, all plugin implementations, and tests
8. Update tests: add entry-point-specific tests, remove sync-internal tests
9. Update CLI (`extras/cli/src/daglite_cli/cmd_run.py`) — both paths still work since `evaluate()` wraps `evaluate_async()`
10. Update documentation (`docs/user-guide/evaluation.md`, `docs/api-reference/engine.md`)
