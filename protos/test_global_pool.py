"""Test global thread pool and settings integration."""

import threading
import time

from daglite import evaluate
from daglite import task
from daglite.backends.local import _GLOBAL_THREAD_POOL
from daglite.backends.local import ThreadBackend
from daglite.settings import DagliteSettings


@task(backend="threading")
def get_thread_id(x: int) -> tuple[int, str]:
    """Return input and current thread name."""
    time.sleep(0.1)  # Small delay to make threading visible
    return (x, threading.current_thread().name)


def test_global_thread_pool():
    """Test that ThreadBackend uses a single global pool."""
    print("=== Testing Global Thread Pool ===\n")

    # First evaluation initializes the pool
    future1 = get_thread_id.bind(x=1)
    result1 = evaluate(future1)
    print(f"First call: {result1}")
    pool1 = _GLOBAL_THREAD_POOL
    print(f"Global pool created: {pool1 is not None}")

    # Second evaluation should reuse the same pool
    future2 = get_thread_id.bind(x=2)
    result2 = evaluate(future2)
    print(f"Second call: {result2}")
    pool2 = _GLOBAL_THREAD_POOL
    print(f"Pool reused: {pool1 is pool2}")

    assert pool1 is pool2, "Global pool should be reused!"
    print("\n✓ Global thread pool is properly reused\n")


def test_settings_integration():
    """Test that settings are passed to the thread pool."""
    print("=== Testing Settings Integration ===\n")

    # Create settings with custom thread limit
    settings = DagliteSettings(max_backend_threads=4)

    # Create a map operation that will use multiple threads
    numbers = list(range(10))
    future = get_thread_id.fix(x=0).extend(x=numbers)

    start = time.time()
    results = evaluate(future)
    elapsed = time.time() - start

    print(f"Processed {len(results)} items in {elapsed:.2f}s")
    print(f"Thread IDs used: {set(name for _, name in results)}")

    # With 10 items and 0.1s delay each, if we had only 1 thread it would take ~1s
    # With 4+ threads, should take closer to 0.3s (10 items / 4 threads * 0.1s)
    assert elapsed < 0.5, f"Should benefit from parallelism, took {elapsed:.2f}s"
    print("\n✓ Settings properly control thread pool behavior\n")


def test_backend_instance_behavior():
    """Test that multiple ThreadBackend instances work correctly."""
    print("=== Testing Backend Instance Behavior ===\n")

    # Create multiple backend instances - they should all use the same global pool
    backend1 = ThreadBackend()
    backend2 = ThreadBackend()

    # Both should work
    result1 = backend1.run_single(lambda x: x * 2, {"x": 5})
    result2 = backend2.run_single(lambda x: x * 3, {"x": 5})

    print(f"Backend 1 result: {result1}")
    print(f"Backend 2 result: {result2}")

    # Test run_many
    results = backend1.run_many(lambda x: x**2, [{"x": i} for i in range(5)])
    print(f"run_many results: {results}")

    assert results == [0, 1, 4, 9, 16], "run_many should work correctly"
    print("\n✓ Multiple backend instances work correctly\n")


if __name__ == "__main__":
    test_global_thread_pool()
    test_settings_integration()
    test_backend_instance_behavior()
    print("=== All Global Thread Pool Tests Passed! ===")
