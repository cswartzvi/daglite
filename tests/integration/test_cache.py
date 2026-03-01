"""Tests for the built-in caching system."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from daglite import task
from daglite.cache.store import CacheStore


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def cache_store(temp_cache_dir):
    """Create a CacheStore with a temporary directory."""
    return CacheStore(temp_cache_dir)


def test_cache_hit(cache_store):
    """Test that cached results are returned without re-execution."""
    call_count = 0

    @task(cache=True)
    def expensive_task(x: int) -> int:
        nonlocal call_count
        call_count += 1
        return x * 2

    # First call - cache miss, should execute
    result1 = expensive_task(x=5).run(cache_store=cache_store)
    assert result1 == 10

    # Second call - cache hit, should NOT execute
    result2 = expensive_task(x=5).run(cache_store=cache_store)
    assert result2 == 10
    assert call_count == 1  # Should still be 1 (cache hit, not executed again)


def test_cache_miss_different_inputs(cache_store):
    """Test that different inputs result in cache misses."""
    call_count = 0

    @task(cache=True)
    def add_task(a: int, b: int) -> int:
        nonlocal call_count
        call_count += 1
        return a + b

    # First call
    result1 = add_task(a=1, b=2).run(cache_store=cache_store)
    assert result1 == 3
    assert call_count == 1

    # Different inputs - should be cache miss
    result2 = add_task(a=2, b=3).run(cache_store=cache_store)
    assert result2 == 5
    assert call_count == 2


def test_cache_disabled(cache_store):
    """Test that tasks without cache=True are not cached."""
    call_count = 0

    @task()  # No cache=True
    def regular_task(x: int) -> int:
        nonlocal call_count
        call_count += 1
        return x * 3

    # First call
    result1 = regular_task(x=5).run(cache_store=cache_store)
    assert result1 == 15
    assert call_count == 1

    # Second call - should execute again (no caching)
    result2 = regular_task(x=5).run(cache_store=cache_store)
    assert result2 == 15
    assert call_count == 2  # Count should increase


def test_cache_different_functions(cache_store):
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
    result1 = task_a(x=5).run(cache_store=cache_store)
    result2 = task_b(x=5).run(cache_store=cache_store)

    assert result1 == 10
    assert result2 == 10
    # Different functions should not share cache entries
    assert call_count_1 == 1
    assert call_count_2 == 1


def test_cache_ttl_not_expired(cache_store):
    """Test that cache entries within TTL are used."""
    call_count = 0

    @task(cache=True, cache_ttl=10)  # 10 second TTL
    def ttl_task(x: int) -> int:
        nonlocal call_count
        call_count += 1
        return x * 4

    # First call
    result1 = ttl_task(x=5).run(cache_store=cache_store)
    assert result1 == 20
    assert call_count == 1

    # Second call immediately - should hit cache
    result2 = ttl_task(x=5).run(cache_store=cache_store)
    assert result2 == 20
    assert call_count == 1


def test_cache_with_complex_types(cache_store):
    """Test caching with complex data types."""

    @task(cache=True)
    def dict_task(data: dict) -> dict:
        return {k: v * 2 for k, v in data.items()}

    input_data = {"a": 1, "b": 2, "c": 3}

    # First call
    result1 = dict_task(data=input_data).run(cache_store=cache_store)
    assert result1 == {"a": 2, "b": 4, "c": 6}

    # Second call - should hit cache
    result2 = dict_task(data=input_data).run(cache_store=cache_store)
    assert result2 == {"a": 2, "b": 4, "c": 6}


def test_cache_file_structure(cache_store, temp_cache_dir):
    """Test that cache files are created with proper structure."""

    @task(cache=True)
    def simple_task(x: int) -> int:
        return x + 1

    # Execute task
    simple_task(x=10).run(cache_store=cache_store)

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


def test_cache_no_store():
    """Test that tasks work normally without cache store."""
    call_count = 0

    @task(cache=True)
    def task_without_store(x: int) -> int:
        nonlocal call_count
        call_count += 1
        return x * 5

    # First call
    result1 = task_without_store(x=5).run()
    assert result1 == 25
    assert call_count == 1

    # Second call - should execute again (no cache store)
    result2 = task_without_store(x=5).run()
    assert result2 == 25
    assert call_count == 2


def test_cache_with_none_result(cache_store):
    """Test that None results can be cached."""
    call_count = 0

    @task(cache=True)
    def returns_none(x: int) -> None:
        nonlocal call_count
        call_count += 1
        # Intentionally returns None

    # First call
    result1 = returns_none(x=5).run(cache_store=cache_store)
    assert result1 is None

    # Second call - should hit cache
    result2 = returns_none(x=5).run(cache_store=cache_store)
    assert result2 is None
    assert call_count == 1  # Verify cache hit - function not called again


def test_cache_with_async_task(cache_store):
    """Test that async tasks work with caching."""
    import asyncio

    call_count = 0

    @task(cache=True)
    async def async_expensive_task(x: int) -> int:
        nonlocal call_count
        call_count += 1
        return x * 3

    async def run():
        # First call - cache miss, should execute
        result1 = await async_expensive_task(x=7).run_async(cache_store=cache_store)
        assert result1 == 21

        # Second call - cache hit, should NOT execute
        result2 = await async_expensive_task(x=7).run_async(cache_store=cache_store)
        assert result2 == 21
        assert call_count == 1  # Verify cache hit - function not called again

    asyncio.run(run())


def test_cache_with_string_path(temp_cache_dir):
    """Test that cache_store accepts a string path."""
    call_count = 0

    @task(cache=True)
    def counting_task(x: int) -> int:
        nonlocal call_count
        call_count += 1
        return x + 1

    # First call with string path
    result1 = counting_task(x=10).run(cache_store=temp_cache_dir)
    assert result1 == 11
    assert call_count == 1

    # Second call with same string path - should hit cache
    result2 = counting_task(x=10).run(cache_store=temp_cache_dir)
    assert result2 == 11
    assert call_count == 1
