# Daglite Design Review - Findings & Recommendations

**Review Date:** November 20, 2025
**Reviewer:** Deep analysis of codebase structure, API design, and implementation

---

## 🐛 Critical Bugs (Must Fix)

### 1. PartialTask.bind() Ignores Backend Parameter
**Location:** `src/daglite/tasks.py:329`
**Severity:** HIGH - Silent backend selection failure

**Bug:**
```python
def bind(self, *, backend: str | Backend | None = None, **kwargs: Any) -> TaskFuture[R]:
    merged: dict[str, Any] = dict(self.fixed_kwargs)
    backend = _select_backend(backend, self.backend)  # Computed but not used!

    for name, val in kwargs.items():
        if name in merged:
            raise ParameterError(...)
        merged[name] = val
    return TaskFuture(self.task, merged, self.backend)  # ❌ Should be 'backend'
```

**Impact:** When users explicitly pass `backend="threading"` to `partial_task.bind()`, it's silently ignored.

**Fix:**
```python
return TaskFuture(self.task, merged, backend)  # ✅ Use computed backend
```

---

### 2. ThreadBackend Accesses Private Executor Attribute
**Location:** `src/daglite/backends/local.py:97`
**Severity:** HIGH - Fragile implementation

**Bug:**
```python
max_concurrent = min(self._max_workers, executor._max_workers)  # ❌ Private attr
```

**Impact:**
- Accessing `_max_workers` (private attribute) from `ThreadPoolExecutor`
- No guarantee this attribute exists or won't change
- Could break with different Python versions or executor implementations

**Fix Option 1 (Track it ourselves):**
```python
class ThreadBackend(Backend):
    def __init__(self, max_workers: int | None = None):
        self._max_workers = max_workers
        self._global_pool_size: int | None = None

    def run_many(self, fn: Callable[..., T], calls: list[dict[str, Any]]) -> list[T]:
        executor = _get_global_thread_pool()

        # Get global pool size safely
        if self._global_pool_size is None:
            self._global_pool_size = get_global_settings().max_backend_threads or 5  # Default

        if self._max_workers is None:
            futures = [executor.submit(fn, **kw) for kw in calls]
            return [f.result() for f in futures]

        max_concurrent = min(self._max_workers, self._global_pool_size)
        # ... rest of implementation
```

**Fix Option 2 (Simpler - just use self._max_workers):**
```python
# If user set max_workers, respect it; otherwise submit all at once
if self._max_workers is None:
    futures = [executor.submit(fn, **kw) for kw in calls]
    return [f.result() for f in futures]

# Limit concurrency to self._max_workers
max_concurrent = self._max_workers
# ... rest of implementation
```

---

### 3. Global Thread Pool Never Recreated After Settings Change
**Location:** `src/daglite/backends/local.py:17`
**Severity:** MEDIUM - Confusing behavior

**Bug:**
```python
def _get_global_thread_pool() -> ThreadPoolExecutor:
    settings = get_global_settings()
    global _GLOBAL_THREAD_POOL
    if _GLOBAL_THREAD_POOL is None:  # Only creates once!
        max_workers = settings.max_backend_threads if settings else None
        _GLOBAL_THREAD_POOL = ThreadPoolExecutor(max_workers=max_workers)
    return _GLOBAL_THREAD_POOL
```

**Impact:** If user calls `set_global_settings()` after any threading work, the new `max_backend_threads` has no effect.

**Fix Option 1 (Document):**
Add to docstring: "Note: Settings must be configured before first use. Changing settings after thread pool creation has no effect."

**Fix Option 2 (Recreate pool):**
```python
_GLOBAL_THREAD_POOL: ThreadPoolExecutor | None = None
_POOL_MAX_WORKERS: int | None = None  # Track what we created with

def _get_global_thread_pool() -> ThreadPoolExecutor:
    settings = get_global_settings()
    global _GLOBAL_THREAD_POOL, _POOL_MAX_WORKERS

    current_max = settings.max_backend_threads if settings else None

    # Recreate if settings changed
    if _GLOBAL_THREAD_POOL is not None and current_max != _POOL_MAX_WORKERS:
        _GLOBAL_THREAD_POOL.shutdown(wait=True)
        _GLOBAL_THREAD_POOL = None

    if _GLOBAL_THREAD_POOL is None:
        _GLOBAL_THREAD_POOL = ThreadPoolExecutor(max_workers=current_max)
        _POOL_MAX_WORKERS = current_max

    return _GLOBAL_THREAD_POOL
```

---

### 4. Unused `settings` Parameter in evaluate()
**Location:** `src/daglite/engine.py:42`
**Severity:** LOW - Misleading API

**Bug:**
```python
def evaluate(
    expr: GraphBuilder,
    default_backend: str | Backend = "sequential",
    use_async: bool = False,
    settings: DagliteSettings | None = None,  # ❌ Never used!
) -> Any:
    """..."""
    engine = Engine(default_backend=default_backend, use_async=use_async)
    return engine.evaluate(expr)  # settings not passed
```

**Fix Option 1 (Use it):**
```python
def evaluate(
    expr: GraphBuilder,
    default_backend: str | Backend = "sequential",
    use_async: bool = False,
    settings: DagliteSettings | None = None,
) -> Any:
    """..."""
    actual_settings = settings if settings is not None else get_global_settings()
    engine = Engine(
        default_backend=default_backend,
        use_async=use_async,
        settings=actual_settings
    )
    return engine.evaluate(expr)
```

**Fix Option 2 (Remove it):**
```python
def evaluate(
    expr: GraphBuilder,
    default_backend: str | Backend = "sequential",
    use_async: bool = False,
) -> Any:
    """
    ...
    Note: Uses global settings from get_global_settings().
    Set via set_global_settings() before evaluation.
    """
    engine = Engine(default_backend=default_backend, use_async=use_async)
    return engine.evaluate(expr)
```

**Recommendation:** Option 2 (remove) - simpler API, settings are already global.

---

### 5. Typo in Directory Name
**Location:** `src/daglite/backends/disributed/`
**Severity:** LOW - But embarrassing

**Fix:** Rename `disributed` → `distributed`

---

## ⚠️ Design Inconsistencies

### 6. Inconsistent Backend Parameter Naming
**Locations:** Multiple files
**Severity:** MEDIUM - UX confusion

**Issue:**
- `evaluate(default_backend="sequential")`
- `task.bind(backend="threading")`
- `task.extend(backend="threading")`

The `default_backend` name in `evaluate()` is misleading - it's THE backend used unless overridden by nodes, not just a default.

**Fix:**
```python
def evaluate(
    expr: GraphBuilder,
    backend: str | Backend = "sequential",  # ✅ Renamed
    use_async: bool = False,
) -> Any:
    """
    Evaluate a task graph.

    Args:
        backend: Backend for task execution. Individual nodes can override
            this by specifying their own backend in bind()/extend()/zip().
    """
    engine = Engine(default_backend=backend, use_async=use_async)
    return engine.evaluate(expr)
```

Or keep `default_backend` but improve docs:
```python
def evaluate(
    expr: GraphBuilder,
    default_backend: str | Backend = "sequential",
    use_async: bool = False,
) -> Any:
    """
    Evaluate a task graph.

    Args:
        default_backend: Default backend for task execution. Used for any node
            that doesn't specify its own backend via @task(backend=...) or
            .bind(backend=...). This provides the baseline execution strategy
            for the entire DAG.
    """
```

---

### 7. Backend Resolution Documentation Unclear
**Location:** `src/daglite/engine.py:110`
**Severity:** MEDIUM - Developer confusion

**Issue:** The docstring says:

```
Backend Resolution Priority:
    1. Node-specific backend from task/task-future operations (bind, extend, ...)
    2. Default task backend from `@task` decorator
    3. Engine's default_backend_name
```

But this doesn't explain:
- What happens when `backend=None` is explicitly passed?
- That `None` means "use next in chain" not "use no backend"
- The difference between "not specified" and "specified as None"

**Fix:** Improve docstring:
```python
"""
Backend Resolution Priority:
    1. Backend passed to .bind()/.extend()/.zip() (if not None)
    2. Backend from @task() decorator (if not None)
    3. Engine's default_backend parameter

Note: Passing backend=None explicitly is the same as not specifying it -
both mean "use the next backend in the priority chain". There is no way
to "disable" backend selection; every node must execute on some backend.

Example:
    @task(backend="threading")
    def process(x): ...

    # Uses threading (from @task decorator)
    result1 = evaluate(process.bind(x=1))

    # Uses sequential (overridden in bind)
    result2 = evaluate(process.bind(x=1, backend="sequential"))

    # Uses threading (from @task, None means "use default")
    result3 = evaluate(process.bind(x=1, backend=None))
"""
```

---

### 8. Missing Parameter Validation in bind()
**Location:** `src/daglite/tasks.py:153`
**Severity:** MEDIUM - Late error detection

**Issue:** No validation that bound parameters match the function signature:

```python
@task
def add(x: int, y: int) -> int:
    return x + y

# This should fail immediately, but doesn't
future = add.bind(x=1, y=2, z=999)  # 'z' doesn't exist!

# Error happens later during evaluation:
# TypeError: add() got an unexpected keyword argument 'z'
```

**Impact:** Errors are detected late (at evaluation time) with generic TypeErrors instead of early with helpful ParameterErrors.

**Fix:**
```python
def bind(self, *, backend: str | Backend | None = None, **kwargs: Any) -> TaskFuture[R]:
    """Create a TaskFuture by binding parameters to this task."""

    # Validate parameters match function signature
    try:
        sig = inspect.signature(self.fn)
        param_names = set(sig.parameters.keys())

        # Check for unknown parameters
        unknown = set(kwargs.keys()) - param_names
        if unknown:
            raise ParameterError(
                f"Task '{self.name}' received unknown parameters: {sorted(unknown)}. "
                f"Valid parameters are: {sorted(param_names)}"
            )

        # Check for missing required parameters (those without defaults)
        provided = set(kwargs.keys())
        required = {
            name for name, param in sig.parameters.items()
            if param.default is inspect.Parameter.empty
            and param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        }
        missing = required - provided
        if missing:
            raise ParameterError(
                f"Task '{self.name}' missing required parameters: {sorted(missing)}. "
                f"Required: {sorted(required)}, Provided: {sorted(provided)}"
            )

    except (ValueError, TypeError) as e:
        # Can't inspect signature (lambda, C function, etc.) - skip validation
        pass

    backend = _select_backend(backend, self.backend)
    return TaskFuture(task=self, kwargs=dict(kwargs), backend=backend)
```

**Alternative:** Make this validation optional via a setting:
```python
class DagliteSettings:
    max_backend_threads: int | None = None
    validate_parameters: bool = True  # ✅ New setting
```

---

## 🎯 UX/DX Sharp Edges

### 9. TaskFuture.__repr__() Not Helpful for Debugging
**Location:** `src/daglite/tasks.py:477`
**Severity:** LOW - Poor debugging experience

**Current:**
```python
def __repr__(self) -> str:
    return f"<Lazy {id(self):#x}>"
```

**Output:** `<Lazy 0x7f8b4c0a1b80>` (meaningless)

**Better:**
```python
def __repr__(self) -> str:
    params_preview = ', '.join(
        f"{k}={repr(v)[:20]}..." if len(repr(v)) > 20 else f"{k}={repr(v)}"
        for k, v in list(self.kwargs.items())[:3]  # Show first 3 params
    )
    if len(self.kwargs) > 3:
        params_preview += f", ...+{len(self.kwargs) - 3} more"

    backend_info = f" backend={self.backend}" if self.backend else ""
    return f"<TaskFuture[{self.task.name}]({params_preview}){backend_info}>"
```

**Output:** `<TaskFuture[add](x=1, y=2)>` (useful!)

Similar for `MapTaskFuture`:
```python
def __repr__(self) -> str:
    mode_symbol = "×" if self.mode == "extend" else "||"
    n_fixed = len(self.fixed_kwargs)
    n_mapped = len(self.mapped_kwargs)
    return (
        f"<MapTaskFuture[{self.task.name}] "
        f"mode={self.mode} fixed={n_fixed} mapped={n_mapped}>"
    )
```

---

### 10. Confusing Error for map() on Zero-Parameter Functions
**Location:** `src/daglite/tasks.py:656`
**Severity:** LOW - Unclear error message

**Current:**
```python
try:
    signature = inspect.signature(task.fn)
    first_param_name = next(iter(signature.parameters))
except StopIteration:
    raise ParameterError(
        f"Cannot use task '{task.name}' in map/join: function has no parameters."
    )
```

**Issue:** What if the function has parameters but the user forgot to specify which one?

**Better:**
```python
try:
    sig = inspect.signature(task.fn)
    params = list(sig.parameters.keys())

    if not params:
        raise ParameterError(
            f"Cannot use task '{task.name}' in map/join: "
            f"function has no parameters to receive the sequence."
        )

    if len(params) == 1:
        first_param_name = params[0]
    else:
        # Multiple parameters - could be ambiguous
        available = [p for p in params if p not in fixed]
        if len(available) == 1:
            first_param_name = available[0]
        else:
            raise ParameterError(
                f"Cannot determine sequence parameter for task '{task.name}'. "
                f"Function has multiple parameters: {params}. "
                f"Specify explicitly with param='name'. "
                f"Available (not in fixed): {available}"
            )
```

---

### 11. No Way to Introspect TaskFuture Properties
**Location:** `src/daglite/tasks.py:493`
**Severity:** LOW - Poor debuggability

**Issue:** Can't easily access task metadata from a future:

```python
future = my_task.bind(x=1)
# How do I get the task name? future.task.name (verbose)
# How do I see what parameters were bound? future.kwargs (ok)
# How do I see the backend? future.backend (ok)
```

**Suggestion:** Add convenience properties:
```python
@dataclass(frozen=True)
class TaskFuture(BaseTaskFuture, GraphBuilder, Generic[R]):
    """..."""

    task: Task[Any, R]
    kwargs: Mapping[str, Any]
    backend: Backend | None = None

    # Convenience accessors for introspection
    @property
    def name(self) -> str:
        """Task name (convenience accessor for self.task.name)."""
        return self.task.name

    @property
    def description(self) -> str:
        """Task description (convenience accessor for self.task.description)."""
        return self.task.description

    @property
    def bound_parameters(self) -> dict[str, Any]:
        """Return a copy of bound parameters for introspection."""
        return dict(self.kwargs)
```

---

### 12. MapTaskFuture Doesn't Support Common Sequence Operations
**Location:** `src/daglite/tasks.py:456`
**Severity:** LOW - Conceptual inconsistency

**Issue:** `MapTaskFuture` represents a sequence, but:
- Can't index: `result[0]` → TypeError
- Can't iterate: `for x in result:` → TypeError
- Can't check length: `len(result)` → TypeError
- Can't check emptiness: `if result:` → TypeError

The generic `TypeError` messages aren't helpful.

**Better Error Messages:**
```python
class BaseTaskFuture(abc.ABC):
    """Base class for all task futures."""

    def __bool__(self) -> bool:
        raise TypeError(
            f"{type(self).__name__} cannot be used in boolean context. "
            f"TaskFutures must be evaluated first with evaluate(). "
            f"Did you forget to call evaluate()?"
        )

    def __len__(self) -> int:
        raise TypeError(
            f"{type(self).__name__} does not support len(). "
            f"TaskFutures must be evaluated first with evaluate(). "
            f"Did you forget to call evaluate()?"
        )

    def __iter__(self):
        raise TypeError(
            f"{type(self).__name__} is not iterable until evaluated. "
            f"Call evaluate() first to get actual results. "
            f"Example: results = evaluate({self.__repr__()})"
        )

    def __getitem__(self, key):
        raise TypeError(
            f"{type(self).__name__} does not support indexing until evaluated. "
            f"Call evaluate() first to get actual results."
        )
```

---

### 13. Extend/Zip Chaining Limitation Not Clear
**Location:** Design issue across `src/daglite/tasks.py`
**Severity:** LOW - Documentation gap

**Issue:** This pattern doesn't work:
```python
result = (
    task.extend(x=[1, 2, 3])
        .extend(y=[4, 5, 6])  # ❌ AttributeError: MapTaskFuture has no 'extend'
)
```

Once you fan out with `extend()`/`zip()`, you get a `MapTaskFuture`, which only has `.map()` and `.join()`. You can't chain multiple fan-outs.

**Fix:** Add to docs and better error:
```python
@dataclass(frozen=True)
class MapTaskFuture(...):
    """
    Represents a fan-out task invocation producing a sequence of values.

    Once created, MapTaskFutures can only be:
    1. Further transformed with .map() (element-wise operations)
    2. Collapsed with .join() (reduce to single value)
    3. Evaluated with evaluate() (execute the DAG)

    You cannot call .extend() or .zip() on a MapTaskFuture. For nested
    fan-outs, use .map() to apply a task that does extend/zip:

        # ❌ This doesn't work:
        result = task.extend(x=[1,2]).extend(y=[3,4])

        # ✅ This works:
        inner = task.extend(x=[1,2])
        result = inner.map(other_task.extend, y=[3,4])
    """
    ...
```

Add `__getattr__` to provide helpful error:
```python
def __getattr__(self, name: str):
    if name in ('extend', 'zip', 'bind', 'partial'):
        raise AttributeError(
            f"MapTaskFuture does not support .{name}(). "
            f"Once you fan out with extend()/zip(), you can only use "
            f".map() for element-wise operations or .join() to reduce. "
            f"For nested fan-outs, use .map() to apply a task that does extend/zip."
        )
    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
```

---

### 14. Type Hints Lost for Generic Functions
**Location:** General limitation
**Severity:** LOW - Type checker limitation

**Issue:**
```python
from typing import TypeVar

T = TypeVar('T')

@task
def identity(x: T) -> T:
    return x

result = identity.bind(x=5)  # Type: TaskFuture[T] (not TaskFuture[int])
```

Python decorators can't preserve generic type information perfectly.

**Fix:** Document this limitation and provide workarounds:
```python
# Workaround 1: Type annotation
result: TaskFuture[int] = identity.bind(x=5)

# Workaround 2: Type: ignore
result = identity.bind(x=5)  # type: ignore[type-var]

# Workaround 3: Specific task variants
@task
def identity_int(x: int) -> int:
    return x

@task
def identity_str(x: str) -> str:
    return x
```

Add to FAQ or documentation section on type checking.

---

## 🔧 Implementation Improvements

### 15. No Timeout Support for Tasks
**Severity:** MEDIUM - Production readiness gap

**Issue:** Long-running or hanging tasks block entire DAG with no way to recover.

**Suggestion:**
```python
@task(timeout=30.0)  # seconds
def external_api_call(url: str) -> dict:
    """Call external API with 30s timeout."""
    ...
```

Implementation in `Backend.run_single()`:
```python
import signal
from contextlib import contextmanager

@contextmanager
def timeout_context(seconds: float):
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Task exceeded {seconds}s timeout")

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)

# In Backend:
def run_single(self, fn: Callable[..., T], kwargs: dict[str, Any], timeout: float | None = None) -> T:
    if timeout is not None:
        with timeout_context(timeout):
            return fn(**kwargs)
    return fn(**kwargs)
```

---

### 16. No Retry Logic for Flaky Tasks
**Severity:** MEDIUM - Production readiness gap

**Issue:** Network calls, external APIs, and I/O operations can fail transiently.

**Suggestion:**
```python
@task(retries=3, retry_delay=1.0, retry_backoff=2.0)
def fetch_data(url: str) -> dict:
    """Fetch data with exponential backoff: 1s, 2s, 4s."""
    ...
```

Implementation:
```python
@dataclass(frozen=True)
class Task(Generic[P, R]):
    fn: Callable[P, R]
    name: str
    description: str
    backend: Backend | None
    retries: int = 0  # ✅ New
    retry_delay: float = 1.0  # ✅ New
    retry_backoff: float = 1.0  # ✅ New

# In Backend:
def run_single_with_retry(self, task: Task, kwargs: dict[str, Any]) -> Any:
    last_exception = None
    delay = task.retry_delay

    for attempt in range(task.retries + 1):
        try:
            return task.fn(**kwargs)
        except Exception as e:
            last_exception = e
            if attempt < task.retries:
                time.sleep(delay)
                delay *= task.retry_backoff
            else:
                raise ExecutionError(
                    f"Task '{task.name}' failed after {task.retries + 1} attempts"
                ) from last_exception
```

---

### 17. No Progress Reporting / Observability
**Severity:** MEDIUM - Production readiness gap

**Issue:** For large DAGs (100+ tasks), no way to monitor:
- Which tasks are running
- How long each task takes
- How many tasks completed
- Progress percentage

**Suggestion:** Add callback hooks:
```python
from typing import Protocol

class ExecutionCallback(Protocol):
    def on_task_start(self, task_name: str, node_id: UUID, params: dict) -> None: ...
    def on_task_complete(self, task_name: str, node_id: UUID, result: Any, duration: float) -> None: ...
    def on_task_error(self, task_name: str, node_id: UUID, error: Exception) -> None: ...

def evaluate(
    expr: GraphBuilder,
    backend: str | Backend = "sequential",
    use_async: bool = False,
    callbacks: list[ExecutionCallback] | None = None,
) -> Any:
    """..."""
    engine = Engine(
        default_backend=backend,
        use_async=use_async,
        callbacks=callbacks or []
    )
    return engine.evaluate(expr)

# Usage:
class ProgressTracker:
    def __init__(self):
        self.completed = 0
        self.total = 0

    def on_task_start(self, task_name: str, node_id: UUID, params: dict):
        self.total += 1
        print(f"[{self.completed}/{self.total}] Starting {task_name}...")

    def on_task_complete(self, task_name: str, node_id: UUID, result: Any, duration: float):
        self.completed += 1
        print(f"[{self.completed}/{self.total}] ✓ {task_name} ({duration:.2f}s)")

tracker = ProgressTracker()
result = evaluate(my_dag, callbacks=[tracker])
```

---

### 18. No Caching/Memoization
**Severity:** MEDIUM - Performance optimization

**Issue:** Re-running same DAG repeats all work. The commented-out `cache` field in `Engine` suggests this was planned but not implemented.

**Suggestion:**
```python
@task(cache=True, cache_key=lambda x, y: f"{x}_{y}")
def expensive_computation(x: int, y: int) -> int:
    """This will be cached based on (x, y) values."""
    time.sleep(10)
    return x ** y

# First call: takes 10s
result1 = evaluate(expensive_computation.bind(x=2, y=10))

# Second call with same params: instant (cached)
result2 = evaluate(expensive_computation.bind(x=2, y=10))
```

Implementation outline:
```python
from hashlib import sha256
from typing import Callable

class TaskCache:
    def __init__(self):
        self._cache: dict[bytes, Any] = {}

    def get_key(self, task: Task, kwargs: dict[str, Any]) -> bytes:
        """Generate cache key from task and parameters."""
        if task.cache_key_fn:
            key_str = task.cache_key_fn(**kwargs)
        else:
            # Default: use task ID + sorted params
            key_str = f"{task.name}:{sorted(kwargs.items())}"
        return sha256(key_str.encode()).digest()

    def get(self, key: bytes) -> Any:
        return self._cache.get(key)

    def set(self, key: bytes, value: Any) -> None:
        self._cache[key] = value
```

---

### 19. Thread Safety Issues with Global Settings
**Location:** `src/daglite/settings.py`
**Severity:** MEDIUM - Concurrency bug

**Issue:**
```python
def set_global_settings(settings: DagliteSettings) -> None:
    global _GLOBAL_DAGLITE_SETTINGS
    _GLOBAL_DAGLITE_SETTINGS = settings  # ❌ Not thread-safe
```

If two threads call `set_global_settings()` concurrently, race conditions possible.

**Fix:**
```python
import threading

_GLOBAL_DAGLITE_SETTINGS: DagliteSettings | None = None
_SETTINGS_LOCK = threading.RLock()

def get_global_settings() -> DagliteSettings:
    """Get the global daglite settings (thread-safe)."""
    with _SETTINGS_LOCK:
        global _GLOBAL_DAGLITE_SETTINGS
        if _GLOBAL_DAGLITE_SETTINGS is None:
            _GLOBAL_DAGLITE_SETTINGS = DagliteSettings()
        return _GLOBAL_DAGLITE_SETTINGS

def set_global_settings(settings: DagliteSettings) -> None:
    """Set the global daglite settings (thread-safe)."""
    with _SETTINGS_LOCK:
        global _GLOBAL_DAGLITE_SETTINGS
        _GLOBAL_DAGLITE_SETTINGS = settings
```

---

### 20. No Graceful Shutdown for Global Thread Pool
**Location:** `src/daglite/backends/local.py`
**Severity:** LOW - Resource cleanup

**Issue:** Global thread pool never shuts down cleanly on program exit. Threads might be interrupted mid-task.

**Fix:**
```python
import atexit
import threading

_GLOBAL_THREAD_POOL: ThreadPoolExecutor | None = None
_SHUTDOWN_LOCK = threading.Lock()

def _shutdown_global_pool():
    """Shutdown the global thread pool gracefully."""
    with _SHUTDOWN_LOCK:
        global _GLOBAL_THREAD_POOL
        if _GLOBAL_THREAD_POOL is not None:
            _GLOBAL_THREAD_POOL.shutdown(wait=True, cancel_futures=False)
            _GLOBAL_THREAD_POOL = None

# Register cleanup on exit
atexit.register(_shutdown_global_pool)

# Also provide manual shutdown
def shutdown_thread_backend(wait: bool = True) -> None:
    """
    Manually shutdown the global thread pool.

    Args:
        wait: If True, block until all tasks complete. If False, cancel pending tasks.
    """
    with _SHUTDOWN_LOCK:
        global _GLOBAL_THREAD_POOL
        if _GLOBAL_THREAD_POOL is not None:
            _GLOBAL_THREAD_POOL.shutdown(wait=wait)
            _GLOBAL_THREAD_POOL = None
```

---

## 🎨 API Polish Suggestions

### 21. Add `.then()` for Fluent Chaining
**Severity:** LOW - Nice to have

**Current (verbose):**
```python
raw = download.bind(url="https://example.com")
info = parse.bind(raw=raw)
length_val = length.bind(info=info)
result = evaluate(length_val)
```

**Proposed (fluent):**
```python
result = evaluate(
    download.bind(url="https://example.com")
        .then(parse)
        .then(length)
)
```

**Implementation:**
```python
@dataclass(frozen=True)
class TaskFuture(BaseTaskFuture, GraphBuilder, Generic[R]):
    """..."""

    def then(
        self,
        next_task: Task[Any, S],
        *,
        param: str | None = None,
        backend: str | Backend | None = None,
        **extra_kwargs: Any,
    ) -> TaskFuture[S]:
        """
        Chain this future into the next task.

        Args:
            next_task: Task to apply to this future's result
            param: Parameter name in next_task that receives this result.
                If None, uses the first parameter.
            backend: Backend override for next_task execution
            **extra_kwargs: Additional parameters for next_task

        Example:
            result = (
                fetch.bind(url="https://api.example.com")
                    .then(parse, format="json")
                    .then(validate)
                    .then(transform, param="data", output_format="csv")
            )
        """
        # Determine which parameter receives this future
        if param is None:
            sig = inspect.signature(next_task.fn)
            param_names = list(sig.parameters.keys())
            if not param_names:
                raise ParameterError(
                    f"Cannot chain to task '{next_task.name}': "
                    f"function has no parameters"
                )
            if len(param_names) > 1 and any(k in param_names for k in extra_kwargs):
                # Ambiguous - multiple params and some are in extra_kwargs
                available = [p for p in param_names if p not in extra_kwargs]
                if len(available) != 1:
                    raise ParameterError(
                        f"Cannot determine parameter for chaining to '{next_task.name}'. "
                        f"Specify explicitly with param='name'. "
                        f"Available: {available}"
                    )
                param = available[0]
            else:
                param = param_names[0]

        # Build kwargs with this future
        kwargs = dict(extra_kwargs)
        kwargs[param] = self

        return next_task.bind(backend=backend, **kwargs)
```

---

### 22. Add Operator Overloading for Composition
**Severity:** LOW - Syntactic sugar

**Proposal:**
```python
# Pipeline composition
pipeline = download >> parse >> length
result = evaluate(pipeline.bind(url="https://example.com"))

# Or even:
result = evaluate(download.bind(url="https://example.com") >> parse >> length)
```

**Implementation:**
```python
@dataclass(frozen=True)
class Task(Generic[P, R]):
    """..."""

    def __rshift__(self, other: Task[Any, S]) -> Task[P, S]:
        """
        Compose two tasks using >> operator.

        Creates a new task that applies self then other.
        The output of self becomes the first parameter of other.
        """
        if not isinstance(other, Task):
            raise TypeError(f"Cannot compose Task with {type(other)}")

        @task(name=f"{self.name}>>>{other.name}")
        def composed(**kwargs: Any) -> S:
            # Execute self with provided kwargs
            intermediate = self.fn(**kwargs)
            # Pass result to other (assuming it takes one param)
            return other.fn(intermediate)

        return composed
```

Note: This is tricky with type hints and might not be worth the complexity.

---

## 📚 Documentation Needs

### 23. Expand README with Examples
**Current:** Minimal, just installation
**Needed:**
- Quick start example
- Common patterns (map, reduce, fan-out/fan-in)
- Backend comparison table
- When to use async
- Performance tips

---

### 24. Add API Reference Documentation
**Missing:**
- Detailed class/method documentation
- Backend comparison
- Error handling guide
- Type hints guide
- Migration guide if breaking changes

---

### 25. Add Cookbook / Patterns Guide
**Useful patterns:**
- Data pipelines (ETL)
- Parameter sweeps (ML hyperparameter tuning)
- Parallel web scraping
- Conditional execution
- Error handling and retries
- Caching strategies
- Testing DAGs

---

## 🧹 Code Quality

### 26. Add `__all__` to Submodules
**Location:** `src/daglite/graph/*.py`, `src/daglite/backends/*.py`

**Issue:** No `__all__` means `from daglite.graph import *` imports everything.

**Fix:** Add to each module:
```python
# src/daglite/graph/base.py
__all__ = [
    "GraphNode",
    "GraphBuilder",
    "GraphBuildContext",
    "ParamInput",
]

# src/daglite/graph/nodes.py
__all__ = [
    "TaskNode",
    "MapTaskNode",
]

# src/daglite/backends/base.py
__all__ = [
    "Backend",
]

# src/daglite/backends/local.py
__all__ = [
    "SequentialBackend",
    "ThreadBackend",
]
```

---

### 27. Fix Docstring Inconsistencies
**Issues:**
- Some use `daglite.backends.Backend`, others use full import path
- `optinal` typo (line ~215 in tasks.py)
- Inconsistent capitalization in error messages

**Fix:** Run automated docstring formatter and spell checker.

---

### 28. Improve Type Annotations
**Issues:**
- `Task[Any, Any]` used in many places (could be more specific)
- `Backend` conditionally typed as `object` in some files
- `ParamInput.value: Any | None` (could be union based on `kind`)

**Consider:** Stricter typing with `typing.Literal` for discriminated unions:
```python
from typing import Literal, Union

@dataclass(frozen=True)
class ParamInputValue:
    kind: Literal["value"]
    value: Any

@dataclass(frozen=True)
class ParamInputRef:
    kind: Literal["ref"]
    ref: UUID

@dataclass(frozen=True)
class ParamInputSequence:
    kind: Literal["sequence"]
    value: list[Any]

@dataclass(frozen=True)
class ParamInputSequenceRef:
    kind: Literal["sequence_ref"]
    ref: UUID

ParamInput = Union[ParamInputValue, ParamInputRef, ParamInputSequence, ParamInputSequenceRef]
```

This makes the type system enforce correct usage at compile time.

---

## 📊 Summary & Priorities

### Must Fix Before v1.0:
1. ✅ Bug #1: PartialTask.bind() backend selection
2. ✅ Bug #2: ThreadBackend private attribute access
3. ✅ Bug #4: Unused settings parameter
4. ✅ Bug #5: Typo in directory name
5. ✅ Issue #8: Parameter validation in bind()
6. ✅ Issue #19: Thread-safety for global settings

### Should Fix Soon:
7. ✅ Bug #3: Global pool recreation
8. ✅ Issue #6: Backend parameter naming consistency
9. ✅ Issue #7: Backend resolution documentation
10. ✅ UX #9: Better __repr__() for debugging
11. ✅ UX #10: Better error messages for map/join
12. ✅ UX #12: Better error messages for TaskFuture operations

### Nice to Have (v1.1+):
13. Feature #15: Timeout support
14. Feature #16: Retry logic
15. Feature #17: Progress reporting / callbacks
16. Feature #18: Caching / memoization
17. API #21: .then() for fluent chaining
18. Docs #23-25: Comprehensive documentation

### Low Priority Polish:
19. Issue #20: Graceful thread pool shutdown
20. UX #11: Convenience properties on TaskFuture
21. UX #13: Better documentation for extend/zip chaining
22. Quality #26-28: Code quality improvements

---

## 🎯 Recommended Action Plan

**Phase 1: Critical Fixes (1-2 days)**
- Fix all 6 "Must Fix" items
- Add tests for each fix
- Update docstrings where needed

**Phase 2: UX Improvements (2-3 days)**
- Implement better error messages
- Improve __repr__() methods
- Add convenience properties
- Update documentation

**Phase 3: Features (1-2 weeks)**
- Add timeout support
- Add retry logic
- Add basic callback system
- Consider caching strategy

**Phase 4: Documentation (1 week)**
- Expand README
- Create API reference
- Write cookbook with patterns
- Add migration guide

---

## Contact & Questions

If any of these findings need clarification or you want to discuss implementation approaches, happy to dive deeper into specific areas.
