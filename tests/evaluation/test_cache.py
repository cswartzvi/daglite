"""Tests for the caching system."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from daglite import evaluate
from daglite import evaluate_async
from daglite import task
from daglite.cache.store import FileCacheStore
from daglite.plugins.builtin.cache import CachePlugin


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def cache_plugin(temp_cache_dir):
    """Create a CachePlugin with a temporary FileCacheStore."""
    store = FileCacheStore(temp_cache_dir)
    return CachePlugin(store=store)


def test_cache_hit(cache_plugin):
    """Test that cached results are returned without re-execution."""
    call_count = 0

    @task(cache=True)
    def expensive_task(x: int) -> int:
        nonlocal call_count
        call_count += 1
        return x * 2

    # First call - cache miss, should execute
    result1 = evaluate(expensive_task(x=5), plugins=[cache_plugin])
    assert result1 == 10
    assert call_count == 1

    # Second call - cache hit, should NOT execute
    result2 = evaluate(expensive_task(x=5), plugins=[cache_plugin])
    assert result2 == 10
    assert call_count == 1  # Count should not increase


def test_cache_miss_different_inputs(cache_plugin):
    """Test that different inputs result in cache misses."""
    call_count = 0

    @task(cache=True)
    def add_task(a: int, b: int) -> int:
        nonlocal call_count
        call_count += 1
        return a + b

    # First call
    result1 = evaluate(add_task(a=1, b=2), plugins=[cache_plugin])
    assert result1 == 3
    assert call_count == 1

    # Different inputs - should be cache miss
    result2 = evaluate(add_task(a=2, b=3), plugins=[cache_plugin])
    assert result2 == 5
    assert call_count == 2


def test_cache_disabled(cache_plugin):
    """Test that tasks without cache=True are not cached."""
    call_count = 0

    @task()  # No cache=True
    def regular_task(x: int) -> int:
        nonlocal call_count
        call_count += 1
        return x * 3

    # First call
    result1 = evaluate(regular_task(x=5), plugins=[cache_plugin])
    assert result1 == 15
    assert call_count == 1

    # Second call - should execute again (no caching)
    result2 = evaluate(regular_task(x=5), plugins=[cache_plugin])
    assert result2 == 15
    assert call_count == 2  # Count should increase


def test_cache_different_functions(cache_plugin):
    """Test that different functions don't share cache entries."""
    call_count_1 = 0
    call_count_2 = 0

    @task(cache=True)
    def task_a(x: int) -> int:
        nonlocal call_count_1
        call_count_1 += 1
        return x * 2

    @task(cache=True)
    def task_b(x: int) -> int:
        nonlocal call_count_2
        call_count_2 += 1
        return x * 2

    # Call both with same input
    result1 = evaluate(task_a(x=5), plugins=[cache_plugin])
    result2 = evaluate(task_b(x=5), plugins=[cache_plugin])

    assert result1 == 10
    assert result2 == 10
    assert call_count_1 == 1
    assert call_count_2 == 1  # Different functions, both should execute


def test_cache_ttl_not_expired(cache_plugin):
    """Test that cache entries within TTL are used."""
    call_count = 0

    @task(cache=True, cache_ttl=10)  # 10 second TTL
    def ttl_task(x: int) -> int:
        nonlocal call_count
        call_count += 1
        return x * 4

    # First call
    result1 = evaluate(ttl_task(x=5), plugins=[cache_plugin])
    assert result1 == 20
    assert call_count == 1

    # Second call immediately - should hit cache
    result2 = evaluate(ttl_task(x=5), plugins=[cache_plugin])
    assert result2 == 20
    assert call_count == 1


def test_cache_with_complex_types(cache_plugin):
    """Test caching with complex data types."""

    @task(cache=True)
    def dict_task(data: dict) -> dict:
        return {k: v * 2 for k, v in data.items()}

    input_data = {"a": 1, "b": 2, "c": 3}

    # First call
    result1 = evaluate(dict_task(data=input_data), plugins=[cache_plugin])
    assert result1 == {"a": 2, "b": 4, "c": 6}

    # Second call - should hit cache
    result2 = evaluate(dict_task(data=input_data), plugins=[cache_plugin])
    assert result2 == {"a": 2, "b": 4, "c": 6}


def test_cache_file_structure(cache_plugin, temp_cache_dir):
    """Test that cache files are created with proper structure."""

    @task(cache=True)
    def simple_task(x: int) -> int:
        return x + 1

    # Execute task
    evaluate(simple_task(x=10), plugins=[cache_plugin])

    # Check that cache directory contains files
    cache_path = Path(temp_cache_dir)
    # Cache files don't have extensions, look for directories with files
    shard_dirs = [d for d in cache_path.iterdir() if d.is_dir()]

    assert len(shard_dirs) > 0, "Cache shard directories should be created"

    # Check that at least one shard has cache files
    cache_files = []
    for shard in shard_dirs:
        cache_files.extend([f for f in shard.iterdir() if not f.name.endswith(".meta.json")])

    assert len(cache_files) > 0, "Cache files should be created"


def test_cache_no_plugin():
    """Test that tasks work normally without cache plugin."""
    call_count = 0

    @task(cache=True)
    def task_without_plugin(x: int) -> int:
        nonlocal call_count
        call_count += 1
        return x * 5

    # First call
    result1 = evaluate(task_without_plugin(x=5))
    assert result1 == 25
    assert call_count == 1

    # Second call - should execute again (no cache plugin)
    result2 = evaluate(task_without_plugin(x=5))
    assert result2 == 25
    assert call_count == 2


def test_cache_with_none_result(cache_plugin):
    """Test that None results can be cached."""
    call_count = 0

    @task(cache=True)
    def returns_none(x: int) -> None:
        nonlocal call_count
        call_count += 1
        # Intentionally returns None

    # First call
    result1 = evaluate(returns_none(x=5), plugins=[cache_plugin])
    assert result1 is None
    assert call_count == 1

    # Second call - should hit cache
    # Note: This is tricky because check_cache returns None on miss too
    # We need to verify by checking call_count
    result2 = evaluate(returns_none(x=5), plugins=[cache_plugin])
    assert result2 is None
    assert call_count == 1  # Should still be 1 (cache hit)


def test_cache_with_async_task(cache_plugin):
    """Test that async tasks work with caching."""
    import asyncio

    call_count = 0

    @task(cache=True)
    async def async_expensive_task(x: int) -> int:
        nonlocal call_count
        call_count += 1
        return x * 3

    async def run():
        nonlocal call_count

        # First call - cache miss, should execute
        result1 = await evaluate_async(async_expensive_task(x=7))
        assert result1 == 21
        assert call_count == 1

        # Second call - cache hit, should NOT execute
        result2 = await evaluate_async(async_expensive_task(x=7), plugins=[cache_plugin])
        assert result2 == 21
        assert call_count == 2  # Should execute again (no plugin on first call)

        # Third call with plugin - cache hit now
        result3 = await evaluate_async(async_expensive_task(x=7), plugins=[cache_plugin])
        assert result3 == 21
        assert call_count == 2  # Should still be 2 (cache hit)

    asyncio.run(run())
