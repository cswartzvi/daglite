## TL;DR

We're adding production-ready caching and checkpointing to daglite with a type-safe, plugin-based architecture. 

**Key innovations**:
- Content-addressable caching (no explicit cache keys!)
- Shared serialization registry (register types once, use everywhere)
- Smart hash strategies (sample-based for large objects = 100-2000x faster)
- Git-style file layout (Windows-safe)
- Separate concerns: caching (performance) vs checkpointing (audit)

## Start Here: Phase 1 - Serialization Registry

This is the foundation everything builds on. Start with these files:

### 1. Create `daglite-core/src/daglite/serialization/__init__.py`

```python
"""Type-based serialization and hashing."""

from .registry import (
    SerializationRegistry,
    SerializationHandler,
    HashStrategy,
    default_registry,
)

__all__ = [
    'SerializationRegistry',
    'SerializationHandler', 
    'HashStrategy',
    'default_registry',
]
```

### 2. Create `daglite-core/src/daglite/serialization/registry.py`

This is the main registry. Key methods:
- `register()` - register serializer/deserializer for a type
- `register_hash_strategy()` - register hash function for a type
- `serialize()` / `deserialize()` - use registered handlers
- `hash_value()` - hash using registered strategy

See the design doc for full API.

### 3. Create `daglite-core/src/daglite/serialization/hash_strategies.py`

Smart hash functions for common types:
- `hash_numpy_array()` - sample first/last 500 elements (~1ms for 800MB!)
- `hash_dataframe()` - schema + sample rows (~10ms for 1M rows!)
- `hash_image()` - downsample to 32x32

These are **critical** for performance. Without them, hashing large objects kills caching.

### 4. Create `daglite-core/tests/test_serialization.py`

Test everything:
- Type registration
- Multiple formats per type
- Hash strategies (correctness + performance!)
- Type inheritance
- Edge cases

## Implementation Order

**Phase 1** (this is priority!):
1. `serialization/registry.py` - Core registry class
2. `serialization/hash_strategies.py` - Smart hashers
3. `tests/test_serialization.py` - Comprehensive tests

**Phase 2** (after Phase 1):
1. `caching/store.py` - Protocol
2. `caching/hash.py` - default_cache_hash()
3. `caching/eviction.py` - Eviction policies
4. `plugins/cache/file.py` - File-based store
5. Update `@task` decorator with cache parameters

**Phase 3** (after Phase 2):
1. `checkpointing/store.py` - Protocol
2. `plugins/checkpoint/file.py` - File-based store
3. `plugins/checkpoint/s3.py` - S3 store
4. Add `.checkpoint()` to TaskFuture

**Phase 4** (after Phase 3):
1. Add `.also()` to TaskFuture

**Phase 5** (polish):
1. Pandas plugin auto-registration
2. Numpy plugin auto-registration

## Key Design Decisions

1. **No cache_key parameter**: Use content-addressable hashing (hash = function source + params)
2. **Sample-based hashing**: Hash metadata + sample of large objects (massive speedup)
3. **Git-style layout**: `ab/cd/ef123...` (Windows-safe, proven)
4. **Separate cache/checkpoint**: Different purposes, different lifecycle
5. **Plugin architecture**: Core stays lean, features in extras/

## What Makes This Special

Compared to Prefect/Dagster/Hamilton:
- ✅ **More modular**: Plugin-based, not monolithic
- ✅ **Type-friendly**: Auto-serialization via type registry
- ✅ **Faster hashing**: Sample-based strategies (100-2000x speedup)
- ✅ **Simpler API**: No explicit cache keys, smart defaults
- ✅ **Extensible**: Easy to add custom types/stores

## Testing Priorities

1. **Hash performance** - CRITICAL! Must be fast for large objects
2. **Hash stability** - Same inputs always produce same hash
3. **Eviction correctness** - LRU actually evicts least recently used
4. **Serialization round-trip** - Serialize → deserialize → same value
5. **Windows compatibility** - Path lengths, line endings

## Common Pitfalls to Avoid

1. **Don't hash entire large objects** - Use sample-based strategies!
2. **Don't forget Windows** - Test path lengths, git-style layout
3. **Don't couple cache and checkpoint** - Separate concerns
4. **Don't pickle function closures** - Can't reliably hash, use inspect.getsource
5. **Don't block on S3 ops** - Make async where appropriate

## Questions? Check the Design Doc

The full design doc (`caching-and-checkpointing-design.md`) has:
- Complete API reference
- Implementation details for every component
- Performance benchmarks
- Testing strategy
- Future enhancements (LSP integration!)

## Current daglite Code Structure

The existing codebase is well-organized. You'll be adding to:
```
daglite-core/src/daglite/
  task.py          # Update @task decorator here
  dag.py           # TaskFuture lives here
  serialization/   # NEW - create this
  caching/         # NEW - create this
  checkpointing/   # NEW - create this (Phase 3)
```

## Integration Points

**@task decorator** (in `task.py`):
- Add: `cache`, `cache_ttl`, `cache_hash`, `cache_exclude`, `cache_hash_params`
- Integration point in task execution

**TaskFuture** (in `dag.py`):
- Add: `.also(callback)` method
- Add: `.checkpoint(name)` method

## Success Criteria for Phase 1

✅ Can register custom types  
✅ Can serialize/deserialize with multiple formats  
✅ Can hash small types (exact)  
✅ Can hash large types (sample-based, fast!)  
✅ 100% test coverage  
✅ Performance: hash 800MB numpy array in <100ms  

Once Phase 1 is solid, everything else builds on top!

## Let's Go! 🚀

Start with `serialization/registry.py`. The design doc has the complete API and implementation details. Focus on getting Phase 1 working perfectly before moving to Phase 2.

Remember: **Hash performance is CRITICAL**. Without fast hashing, the entire caching system becomes unusable for realistic workloads.
