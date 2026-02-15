# Caching and Checkpointing Design

**Status**: Design Complete, Ready for Implementation
**Created**: 2024-12-09
**Target**: daglite v0.5.0

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Implementation Phases](#implementation-phases)
4. [API Reference](#api-reference)
5. [Implementation Details](#implementation-details)
6. [Testing Strategy](#testing-strategy)
7. [Future Enhancements](#future-enhancements)

---

## Overview

### Goals

Add production-ready caching and checkpointing to daglite with:
- **Type-safe**: Leverage Python's type system for automatic serialization
- **Modular**: Plugin-based architecture for extensibility
- **Performant**: Smart hashing strategies for large objects
- **Simple**: Intuitive API with sensible defaults
- **Professional**: Match or exceed Prefect/Dagster/Hamilton capabilities

### Key Concepts

**Caching** (Performance):
- Content-addressable (hash-based) storage
- Transparent and automatic
- Hash = f(function_source, parameters)5/
- Ephemeral (safe to clear)
- Purpose: Skip expensive recomputation

**Checkpointing** (Resilience & Audit):
- Named, versioned artifacts
- Explicit via `.checkpoint(name)`
- Rich metadata (timestamp, git hash, params)
- Persistent (precious audit trail)
- Purpose: Debugging, recovery, compliance

**Side Effects** (Flexibility):
- Non-branching via `.also(callback)`
- Logging, metrics, notifications
- Return value ignored
- Purpose: Observability without affecting flow

### Design Principles

1. **Separation of Concerns**: Cache, checkpoint, and side effects are distinct
2. **Shared Infrastructure**: All use same serialization registry
3. **Hash-Based Keys**: No explicit cache_key needed (content-addressable)
4. **Type Registration**: Register once, works everywhere
5. **Plugin Architecture**: Stores live in extras/ packages

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│         Serialization Registry (Core)               │
│  • Type → Serializer/Deserializer                   │
│  • Type → Hash Strategy                             │
│  • Format preferences (CSV vs Parquet)              │
└─────────────────────────────────────────────────────┘
                    ↑            ↑
          ┌─────────┴────────┐   └────────────┐
          │                  │                │
┌─────────────────┐  ┌───────────────────┐  ┌──────────────┐
│  Cache Store    │  │ Checkpoint Store  │  │ .also()      │
│  (extras/)      │  │  (extras/)        │  │  (core)      │
│                 │  │                   │  │              │
│ • Hash-based    │  │ • Named locations │  │ • Side       │
│ • Transparent   │  │ • Versioned       │  │   effects    │
│ • LRU eviction  │  │ • Rich metadata   │  │ • Logging    │
│ • Git layout    │  │ • S3 support      │  │ • Metrics    │
└─────────────────┘  └───────────────────┘  └──────────────┘
```

### Package Structure

```
daglite/
├── src/daglite/
│   ├── serialization.py         # SerializationRegistry + built-in strategies
│   ├── caching/
│   │   ├── __init__.py
│   │   ├── store.py             # CacheStore Protocol
│   │   ├── hash.py              # default_cache_hash()
│   │   └── eviction.py          # Eviction policies
│   ├── tasks.py                 # @task decorator updates
│   └── ... other files ...
│
├── extras/
│   ├── serialization/
│   │   ├── src/daglite_serialization/  # Single package, multiple extras
│   │   │   ├── __init__.py       # register_all()
│   │   │   ├── numpy.py          # NumPy handlers
│   │   │   ├── pandas.py         # Pandas handlers
│   │   │   └── pillow.py         # Pillow handlers
│   │   └── pyproject.toml        # [numpy], [pandas], [pillow], [all]
│   │
│   ├── cache/
│   │   └── daglite-cache-file/       # Future
│   │
│   └── checkpoint/
│       ├── daglite-checkpoint-file/  # Future
│       └── daglite-checkpoint-s3/    # Future
```

---

## Implementation Phases

### Phase 1: Serialization Registry ✅ COMPLETE

**Goal**: Core infrastructure that everything builds on

**Files created**:

- `src/daglite/serialization.py` - Single module with registry and built-in strategies
- `extras/serialization/daglite-serialization/` - Plugin package
- `tests/test_serialization.py` - 39 tests passing
- `extras/serialization/tests/test_serialization_plugin.py` - 16 tests passing

**Key classes**:

- `SerializationRegistry`: Main registry with module-based error messages
- `SerializationHandler`: Per-format handler
- `HashStrategy`: Per-type hash function

**Success criteria**:

- [x] Register custom types with multiple formats
- [x] Strict TypeError for unregistered types (no repr() fallback)
- [x] Plugin package (`daglite_serialization`) with numpy/pandas/pillow
- [x] Smart hash strategies with start/middle/end sampling (<100ms for 800MB arrays)
- [x] 100% test coverage (39 core + 16 plugin tests)
- [x] Recursive hashing via closures for nested data structures

### Phase 2: Caching Infrastructure

**Goal**: Content-addressable caching with @task integration

**Files to create**:

- `src/daglite/caching/store.py` (Protocol)
- `src/daglite/caching/hash.py`
- `src/daglite/caching/eviction.py`
- `extras/cache/daglite-cache-file/src/daglite/plugins/cache/file.py`
- `extras/cache/daglite-cache-file/src/daglite/plugins/cache/layout.py`

**Success criteria**:

- [ ] `@task(cache=True)` works
- [ ] Hash-based cache keys (no explicit cache_key)
- [ ] Git-style file layout (Windows-safe)
- [ ] LRU eviction with configurable size limits
- [ ] TTL support

### Phase 3: Checkpointing

**Goal**: Named, versioned artifacts for debugging/audit

**Files to create**:
- `src/daglite/checkpointing/store.py` (Protocol)
- `extras/checkpoint/daglite-checkpoint-file/...`
- `extras/checkpoint/daglite-checkpoint-s3/...`

**Success criteria**:

- [ ] `.checkpoint(name)` method on TaskFuture
- [ ] Versioned storage
- [ ] Rich metadata (timestamp, git hash, params)
- [ ] S3 backend support

### Phase 4: Side Effects

**Goal**: `.also()` for non-branching operations

**Files to modify**:
- `src/daglite/task.py` (add `.also()` to TaskFuture)

**Success criteria**:

- [ ] `.also(callback)` runs side effects
- [ ] Doesn't affect data flow
- [ ] Used internally by `.checkpoint()`

### Phase 5: Plugin Registration ✅ COMPLETE (done with Phase 1)

**Goal**: Auto-register pandas, numpy serializers

**Implemented**:

- Single package: `extras/serialization/daglite-serialization/`
- Package name: `daglite_serialization` (flat namespace)
- Optional dependencies: `[numpy]`, `[pandas]`, `[pillow]`, `[all]`
- API: `register_all()` convenience function

---

## API Reference

### @task Decorator Parameters

```python
@task(
    # === CACHING ===
    cache: bool = False,
    # Enable content-addressable caching

    cache_ttl: int | timedelta | None = None,
    # Time-to-live for cache entries
    # None = never expire, int = seconds, timedelta = duration

    cache_hash: Callable[[Callable, dict], str] | None = None,
    # Custom hash function: (function, bound_args) → hash_string
    # None = use default (source + params)

    ....  # existing parameters
)
def my_task(...) -> ...:
    ...
```

**Examples**:

```python
# Simple caching
@task(cache=True)
def compute(x: int, y: int) -> pd.DataFrame:
    # Expensive computation
    ...

# With TTL
@task(cache=True, cache_ttl=3600)  # 1 hour
def fetch_stock_price(ticker: str) -> float:
    return api.get_latest_price(ticker)

# Completely custom hash
def my_hash(func, bound_args):
    return hashlib.sha256(
        func.__name__.encode() +
        str(bound_args['version']).encode()
    ).hexdigest()

@task(cache=True, cache_hash=my_hash)
def train_model(data: pd.DataFrame, version: str) -> Model:
    ...
```

### TaskFuture Methods

#### `.also()` - Side Effects

```python
def also(self, callback: Callable[[T], None]) -> TaskFuture[T]:
    """
    Execute a side effect without affecting data flow.

    The callback receives the task result but its return value
    is ignored. Exceptions in callbacks are logged but don't
    stop the pipeline.

    Args:
        callback: Function that takes result, returns None

    Returns:
        Same TaskFuture (for chaining)
    """
```

**Examples**:

```python
result = (
    compute()
    .also(lambda x: logger.info(f"Result: {x}"))
    .also(lambda x: metrics.record("value", x))
    .also(save_to_audit_log)
    .then(next_task)
)

# Save copy without branching
pipeline = (
    process_data()
    .also(lambda x: x.to_csv("/audit/data.csv"))
    .then(analyze)  # Main flow continues
)
```

#### `.checkpoint()` - Named Artifacts

```python
def checkpoint(
    self,
    name: str,
    store: CheckpointStore | None = None,
    versioning: bool = True,
    metadata: dict[str, Any] | None = None,
) -> TaskFuture[T]:
    """
    Save task result as a named, versioned artifact.

    Args:
        name: Human-readable checkpoint name
        store: Optional custom checkpoint store
        versioning: If True, keep multiple versions
        metadata: Additional metadata to store

    Returns:
        Same TaskFuture (for chaining)
    """
```

**Examples**:

```python
result = (
    expensive_computation()
    .checkpoint("raw_features")
    .then(normalize)
    .checkpoint("normalized_features")
    .then(train_model)
    .checkpoint("trained_model", metadata={'accuracy': 0.95})
)

# Different store (e.g., S3)
from daglite.plugins.checkpoint import S3CheckpointStore

s3_store = S3CheckpointStore(bucket="ml-artifacts")

result = (
    train_model()
    .checkpoint("model_v1", store=s3_store)
    .then(evaluate)
)

# Later: recover from checkpoint
checkpoint_store = FileCheckpointStore(Path("/checkpoints"))
features = checkpoint_store.load("normalized_features", version="latest")
```

### SerializationRegistry

```python
class SerializationRegistry:
    """Central registry for type-based serialization"""

    def register(
        self,
        type_: Type[T],
        serializer: Callable[[T], bytes],
        deserializer: Callable[[bytes], T],
        format: str = 'default',
        file_extension: str = 'bin',
        make_default: bool = False,
    ) -> None:
        """Register a serialization handler"""

    def register_hash_strategy(
        self,
        type_: Type,
        hasher: Callable[[Any], str],
        description: str = "",
    ) -> None:
        """Register how to hash a type for cache keys"""

    def serialize(
        self,
        obj: Any,
        format: str | None = None,
    ) -> tuple[bytes, str]:
        """Serialize object → (bytes, file_extension)"""

    def deserialize(
        self,
        data: bytes,
        type_: Type[T],
        format: str | None = None,
    ) -> T:
        """Deserialize bytes → object"""

    def hash_value(self, obj: Any) -> str:
        """Hash a value using registered strategy"""

    def set_default_format(self, type_: Type, format: str) -> None:
        """Set default format for a type"""

    def get_extension(self, type_: Type, format: str | None = None) -> str:
        """Get file extension for type/format"""

# Global instance
default_registry = SerializationRegistry()
```

**Usage**:

```python
from daglite.serialization import default_registry

# Register custom type
class MyModel:
    def to_bytes(self) -> bytes: ...
    @classmethod
    def from_bytes(cls, data: bytes) -> 'MyModel': ...

default_registry.register(
    MyModel,
    lambda m: m.to_bytes(),
    lambda b: MyModel.from_bytes(b),
    format='default',
    file_extension='model'
)

# Register hash strategy (don't serialize whole model!)
default_registry.register_hash_strategy(
    MyModel,
    lambda m: m.get_version_hash(),
    "Hash model version and config"
)

# Change format preference
default_registry.set_default_format(pd.DataFrame, 'parquet')
```

### CacheStore Protocol

```python
class CacheStore(Protocol):
    """Protocol for cache storage backends"""

    def get(self, hash_key: str, return_type: Type[T]) -> Optional[T]:
        """Retrieve cached value by hash"""

    def put(
        self,
        hash_key: str,
        value: Any,
        format: str | None = None,
        ttl: int | None = None,
    ) -> None:
        """Store value with hash key"""

    def invalidate(self, hash_key: str) -> None:
        """Remove cached entry"""

    def clear(self) -> None:
        """Clear entire cache"""

    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
```

### CheckpointStore Protocol

```python
class CheckpointStore(Protocol):
    """Protocol for checkpoint storage backends"""

    def save(
        self,
        name: str,
        value: Any,
        metadata: dict[str, Any],
        format: str | None = None,
    ) -> str:
        """Save checkpoint, returns version identifier"""

    def load(
        self,
        name: str,
        version: str | None = None,
        return_type: Type[T] | None = None,
    ) -> T:
        """Load checkpoint (latest or specific version)"""

    def list_versions(self, name: str) -> list[CheckpointVersion]:
        """List all versions of a checkpoint"""

    def delete(self, name: str, version: str | None = None) -> None:
        """Delete checkpoint"""

@dataclass
class CheckpointVersion:
    version: str
    timestamp: datetime
    metadata: dict[str, Any]
    size_bytes: int
```

---

## Implementation Details

### Hash Strategies (Performance Critical!)

The default hash function can be **devastating** for large objects:

```python
# BAD: 2 seconds to hash 800MB array
large_array = np.random.rand(10000, 10000)
hash = hashlib.sha256(pickle.dumps(large_array)).hexdigest()
```

**Solution**: Sample-based hashing in `hash_strategies.py`

#### Numpy Arrays

```python
def hash_numpy_array(arr: np.ndarray) -> str:
    """
    Fast hash: metadata + sample of data

    Time: ~1ms for 800MB array (vs ~2000ms for full hash)
    """
    h = hashlib.sha256()

    # Metadata (always fast)
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())

    # Sample data
    if arr.size > 1000:
        flat = arr.flatten()
        sample = np.concatenate([flat[:500], flat[-500:]])
        h.update(sample.tobytes())
    else:
        h.update(arr.tobytes())

    return h.hexdigest()
```

#### Pandas DataFrames

```python
def hash_dataframe(df: pd.DataFrame) -> str:
    """
    Fast hash: schema + sample rows

    Time: ~10ms for 1M rows (vs ~5000ms for full hash)
    """
    h = hashlib.sha256()

    # Schema
    h.update(str(df.shape).encode())
    h.update(str(df.dtypes.to_dict()).encode())
    h.update(str(df.columns.tolist()).encode())

    # Sample rows
    if len(df) > 1000:
        sample = pd.concat([df.head(500), df.tail(500)])
        h.update(pd.util.hash_pandas_object(sample).values.tobytes())
    else:
        h.update(pd.util.hash_pandas_object(df).values.tobytes())

    return h.hexdigest()
```

#### Images

```python
def hash_image(img: Image.Image) -> str:
    """
    Fast hash: downsample to 32x32

    Much faster than full resolution, catches all visible changes
    """
    h = hashlib.sha256()
    h.update(str(img.size).encode())
    h.update(str(img.mode).encode())

    thumb = img.resize((32, 32), Image.Resampling.LANCZOS)
    h.update(thumb.tobytes())

    return h.hexdigest()
```

### Git-Style Cache Layout (Windows-Safe)

**Problem**: Windows path limit (260 chars), filesystem limits on files per directory

**Solution**: Git's proven approach

```python
class GitStyleCacheLayout:
    """
    Git-style hash storage.

    Hash: "abcdef1234567890..."
    Path: base/ab/cd/ef1234567890...

    Benefits:
    - Windows-safe (stays under 260 chars)
    - Balanced tree (256 buckets per level)
    - Fast lookups
    - Familiar pattern
    """

    def __init__(self, base_path: Path, split_pattern: tuple[int, ...] = (2, 2)):
        """
        Args:
            split_pattern: (2, 2) means: ab/cd/remainder
        """
        self.base_path = base_path
        self.split_pattern = split_pattern

    def get_path(self, hash_key: str, extension: str) -> Path:
        """
        Convert hash to path.

        Example: "abcdef123..." → base/ab/cd/ef123....ext
        """
        parts = []
        offset = 0

        for length in self.split_pattern:
            parts.append(hash_key[offset:offset + length])
            offset += length

        filename = hash_key[offset:]
        return self.base_path.joinpath(*parts, f"{filename}.{extension}")
```

**Path length analysis**:
```
C:/Users/Name/AppData/Local/daglite/cache  (47 chars)
+ ab/cd/                                    (6 chars)
+ ef1234567890...parquet                    (68 chars)
= 121 chars total (well under 260!)
```

### Cache Eviction Policies

**LRU (Recommended Default)**:
```python
# Evict least recently used entries
SELECT hash_key, file_path, size_bytes
FROM cache_entries
ORDER BY last_accessed ASC
LIMIT ?
```

**LFU**:
```python
# Evict least frequently used
SELECT hash_key, file_path, size_bytes
FROM cache_entries
ORDER BY access_count ASC, last_accessed ASC
LIMIT ?
```

**SIZE_FIRST**:
```python
# Evict largest files first (maximize freed space)
SELECT hash_key, file_path, size_bytes
FROM cache_entries
ORDER BY size_bytes DESC, last_accessed ASC
LIMIT ?
```

#### Cache Metadata Store

Use SQLite for lightweight metadata tracking:

```python
class CacheMetadataStore:
    """
    Track cache metadata in SQLite.

    Schema:
    CREATE TABLE cache_entries (
        hash_key TEXT PRIMARY KEY,
        file_path TEXT NOT NULL,
        size_bytes INTEGER NOT NULL,
        created_at TEXT NOT NULL,
        last_accessed TEXT NOT NULL,
        access_count INTEGER NOT NULL DEFAULT 1,
        ttl_expires TEXT
    )
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def record_access(self, hash_key: str) -> None:
        """Update last_accessed and increment access_count"""

    def record_put(self, hash_key: str, file_path: Path, size: int, ttl: int | None):
        """Record new cache entry"""

    def get_eviction_candidates(
        self,
        policy: EvictionPolicy,
        target_bytes: int,
    ) -> list[CacheEntry]:
        """Get entries to evict"""
```

#### FileCacheStore with Eviction

```python
class FileCacheStore:
    def __init__(
        self,
        base_path: Path,
        max_size_mb: int | None = None,
        eviction_policy: EvictionPolicy = EvictionPolicy.LRU,
        high_water_mark: float = 0.9,  # Evict at 90% full
        low_water_mark: float = 0.7,   # Evict down to 70%
    ):
        self.metadata = CacheMetadataStore(base_path / ".metadata" / "cache.db")
        self.evictor = CacheEvictor(self.metadata, eviction_policy)
        # ...

    def put(self, hash_key: str, value: Any, format: str | None, ttl: int | None):
        """Store with eviction check"""
        # Serialize
        data, ext = self.registry.serialize(value, format)

        # Check if eviction needed
        if self.max_size:
            current = self._get_total_size()
            if current + len(data) > (self.max_size * self.high_water):
                target = int(self.max_size * self.low_water)
                self.evictor.evict_to_size(target)

        # Write file
        cache_file = self.layout.get_path(hash_key, ext)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_bytes(data)

        # Record metadata
        ttl_expires = datetime.now() + timedelta(seconds=ttl) if ttl else None
        self.metadata.record_put(hash_key, cache_file, len(data), ttl_expires)
```

### Default Cache Hash Function

```python
def default_cache_hash(
    func: Callable,
    bound_args: dict[str, Any],
    registry: SerializationRegistry,
    exclude: list[str] | None = None,
    param_hashers: dict[str, Callable] | None = None,
) -> str:
    """
    Default cache hash: function source + parameter values.

    Uses smart hashing strategies from registry to avoid
    performance issues with large objects.
    """
    h = hashlib.sha256()

    # Hash function source
    try:
        source = inspect.getsource(func)
        h.update(source.encode())
    except (OSError, TypeError):
        # Can't get source - use name
        h.update(func.__name__.encode())

    # Hash each parameter
    exclude = exclude or []
    param_hashers = param_hashers or {}

    for name, value in sorted(bound_args.items()):
        if name in exclude:
            continue

        if name in param_hashers:
            param_hash = param_hashers[name](value)
        else:
            param_hash = registry.hash_value(value)  # Smart!

        h.update(f"{name}={param_hash}".encode())

    return h.hexdigest()
```

---

## Testing Strategy

### Unit Tests

**Serialization Registry** (`test_serialization.py`):
- Register custom types
- Multiple formats per type
- Format preferences
- Hash strategies (speed + correctness)
- Type inheritance
- Edge cases (None, empty collections)

**Hash Functions** (`test_hash.py`):
- Default hash function
- Parameter exclusion
- Custom per-param hashers
- Large object performance
- Hash stability

**Cache Store** (`test_cache.py`):
- Put/get operations
- TTL expiration
- Eviction policies (LRU, LFU, SIZE)
- Git-style layout
- Metadata tracking
- Stats/diagnostics

**Checkpoint Store** (`test_checkpoint.py`):
- Save/load
- Versioning
- Metadata storage
- List versions
- S3 backend

### Integration Tests

**Caching** (`test_task_caching.py`):
```python
@task(cache=True)
def expensive(x: int) -> int:
    return x * 2

# First call - computes
result1 = expensive(5).run()

# Second call - cache hit
result2 = expensive(5).run()

# Different params - cache miss
result3 = expensive(10).run()
```

**Checkpointing** (`test_checkpointing.py`):
```python
result = (
    compute()
    .checkpoint("step1")
    .then(transform)
    .checkpoint("step2")
    .run()
)

# Recover from checkpoint
store = FileCheckpointStore(Path("/checkpoints"))
step1_data = store.load("step1", version="latest")
```

### Performance Tests

**Hash Performance** (`test_performance.py`):
```python
def test_large_array_hash_performance():
    large_array = np.random.rand(10000, 10000)  # 800MB

    start = time.time()
    hash_value = default_registry.hash_value(large_array)
    elapsed = time.time() - start

    assert elapsed < 0.1  # Should be < 100ms
```

### Property-Based Tests

**Hash Stability** (using Hypothesis):
```python
from hypothesis import given, strategies as st

@given(st.integers(), st.integers())
def test_hash_deterministic(x, y):
    """Same inputs always produce same hash"""
    hash1 = default_cache_hash(compute, {'x': x, 'y': y}, default_registry)
    hash2 = default_cache_hash(compute, {'x': x, 'y': y}, default_registry)
    assert hash1 == hash2
```

---

## Future Enhancements

### LSP Integration (Phase 6)

**Diagnostic**: Detect missing serializers
```python
@task(cache=True)
def process(data: CustomClass) -> Result:  # ⚠️ Warning
    ...

# LSP diagnostic:
# "Type 'CustomClass' has no registered serializer.
#  Caching will fail at runtime."
```

**Code Action**: Generate serializer registration
```python
# Quick fix generates:
from daglite.serialization import default_registry
import pickle

default_registry.register(
    CustomClass,
    lambda obj: pickle.dumps(obj),
    lambda data: pickle.loads(data),
    format='pickle',
    file_extension='pkl'
)
```

**Hover Info**: Show serialization details
```python
@task(cache=True)
def process(data: pd.DataFrame) -> Result:
#                  ^^^^^^^^^^^^
# Hover shows:
# Type: pd.DataFrame
# Formats: csv (default), parquet, pickle
# Hash: Sample-based (500 rows)
# Est. time: ~10ms for 1M rows
```

### Additional Store Backends

- Redis cache store
- PostgreSQL checkpoint store
- Azure Blob checkpoint store
- Google Cloud Storage checkpoint store

### Advanced Eviction

- Custom eviction scores (weighted by age + size + access)
- Per-task eviction policies
- Cache warming strategies

### Metrics & Observability

- Cache hit rate tracking
- Size growth monitoring
- Eviction event logging
- Checkpoint creation alerts

---

## Configuration

### Programmatic

```python
from daglite.plugins.cache import FileCacheStore, EvictionPolicy

cache = FileCacheStore(
    Path("/tmp/daglite-cache"),
    max_size_mb=1000,
    eviction_policy=EvictionPolicy.LRU
)

with cache:
    result = dag.run(cache=cache)
```

### CLI (daglite.toml)

```toml
[cache]
type = "file"
path = "/tmp/daglite-cache"
max_size_mb = 1000
eviction_policy = "lru"
high_water_mark = 0.9
low_water_mark = 0.7

[checkpoint]
type = "s3"
bucket = "my-ml-artifacts"
prefix = "experiment-123"
versioning = true
```

---

## Implementation Checklist

### Phase 1: Serialization Registry ✅
- [ ] Create `SerializationRegistry` class
- [ ] Add `SerializationHandler` dataclass
- [ ] Add `HashStrategy` dataclass
- [ ] Implement `register()` method
- [ ] Implement `register_hash_strategy()` method
- [ ] Implement `serialize()` / `deserialize()` methods
- [ ] Implement `hash_value()` method
- [ ] Register built-in types (int, str, dict, list, etc.)
- [ ] Add numpy hash strategy (sample-based)
- [ ] Add pandas hash strategy (schema + sample)
- [ ] Write comprehensive tests
- [ ] Document usage

### Phase 2: Caching ✅
- [ ] Create `CacheStore` Protocol
- [ ] Create `default_cache_hash()` function
- [ ] Create `EvictionPolicy` enum
- [ ] Create `CacheMetadataStore` (SQLite)
- [ ] Create `CacheEvictor` class
- [ ] Create `GitStyleCacheLayout`
- [ ] Create `FileCacheStore`
- [ ] Add cache parameters to `@task`
- [ ] Integrate cache lookup in task execution
- [ ] Write tests (unit + integration)
- [ ] Performance benchmarks

### Phase 3: Checkpointing ✅
- [ ] Create `CheckpointStore` Protocol
- [ ] Create `CheckpointVersion` dataclass
- [ ] Create `FileCheckpointStore`
- [ ] Create `S3CheckpointStore`
- [ ] Add `.checkpoint()` to `TaskFuture`
- [ ] Implement metadata tracking
- [ ] Write tests
- [ ] Document recovery workflow

### Phase 4: Side Effects ✅
- [ ] Add `.also()` to `TaskFuture`
- [ ] Handle exceptions in callbacks
- [ ] Use internally for `.checkpoint()`
- [ ] Write tests

### Phase 5: Plugin Registration ✅
- [ ] Create pandas plugin
- [ ] Create numpy plugin
- [ ] Auto-register on import
- [ ] Document plugin creation

---

## Questions for Implementation

1. **Should cache metadata DB be per-cache or global?**
   - Recommendation: Per-cache (more flexible)

2. **How to handle cache migration/versioning?**
   - Recommendation: Include version in metadata, migrate on load

3. **Should `.checkpoint()` block execution or be async?**
   - Recommendation: Block by default, async option for S3

4. **How to handle git hash detection for metadata?**
   - Recommendation: Try `subprocess.run(['git', 'rev-parse', 'HEAD'])`, catch if fails

5. **Should eviction be synchronous or background task?**
   - Recommendation: Synchronous on put() (simple), async option later

---

## Related Decisions

- **No explicit cache_key**: Use content-addressable hashing (simpler, proven)
- **Separate cache and checkpoint**: Different concerns, separate lifecycle
- **Plugin architecture**: Extensibility without core bloat
- **Git-style layout**: Windows-safe, proven, familiar
- **Sample-based hashing**: Performance critical for large objects

---

## References

- Prefect caching: https://docs.prefect.io/concepts/tasks/#caching
- Dagster assets: https://docs.dagster.io/concepts/assets/software-defined-assets
- Hamilton IO: https://hamilton.apache.org/reference/io/
- Hamilton caching: https://hamilton.apache.org/reference/caching/
- Git object storage: https://git-scm.com/book/en/v2/Git-Internals-Git-Objects
