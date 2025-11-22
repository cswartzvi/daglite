"""Test that ThreadBackend respects global pool size when limiting concurrency."""

import threading
import time

from daglite import task
from daglite.backends.local import ThreadBackend
from daglite.settings import DagliteSettings
from daglite.settings import set_global_settings

# Track concurrent executions
concurrent_count = 0
max_concurrent = 0
lock = threading.Lock()


@task(backend="threading")
def track_concurrency(x: int, delay: float = 0.1) -> int:
    """Track how many tasks run concurrently."""
    global concurrent_count, max_concurrent

    with lock:
        concurrent_count += 1
        if concurrent_count > max_concurrent:
            max_concurrent = concurrent_count

    time.sleep(delay)

    with lock:
        concurrent_count -= 1

    return x * 2


def test_respects_pool_size():
    """Test that max_workers respects global pool size."""
    global max_concurrent, concurrent_count
    max_concurrent = 0
    concurrent_count = 0

    print("=== Test: max_workers Respects Global Pool Size ===")

    # Set global pool to 5 threads
    set_global_settings(DagliteSettings(max_backend_threads=5))

    # Try to request 20 concurrent workers (more than pool has)
    backend = ThreadBackend(max_workers=20)

    # Submit 20 tasks
    calls = [{"x": i, "delay": 0.1} for i in range(20)]

    start = time.time()
    results = backend.run_many(track_concurrency.func, calls)
    elapsed = time.time() - start

    print(f"Processed {len(results)} tasks in {elapsed:.2f}s")
    print(f"Max concurrent tasks: {max_concurrent}")
    print("Requested: 20 workers, Pool size: 5")
    print("Expected: Limited to 5 concurrent (respects pool size)")
    print()

    # Should be limited to 5, not 20
    assert max_concurrent <= 6, f"Expected max 5-6 concurrent, got {max_concurrent}"
    assert max_concurrent >= 4, f"Expected at least 4-5 concurrent, got {max_concurrent}"

    # Should take about 0.4s (20 tasks / 5 workers * 0.1s per task)
    assert 0.3 < elapsed < 0.6, f"Expected ~0.4s, got {elapsed:.2f}s"

    print("✅ Correctly limited to global pool size!")


def test_smaller_request_still_works():
    """Test that requesting fewer workers than pool size still works."""
    global max_concurrent, concurrent_count
    max_concurrent = 0
    concurrent_count = 0

    print("=== Test: Smaller max_workers Request ===")

    # Set global pool to 10 threads
    set_global_settings(DagliteSettings(max_backend_threads=10))

    # Request only 3 concurrent workers (less than pool has)
    backend = ThreadBackend(max_workers=3)

    # Submit 10 tasks
    calls = [{"x": i, "delay": 0.1} for i in range(10)]

    start = time.time()
    results = backend.run_many(track_concurrency.func, calls)
    elapsed = time.time() - start

    print(f"Processed {len(results)} tasks in {elapsed:.2f}s")
    print(f"Max concurrent tasks: {max_concurrent}")
    print("Requested: 3 workers, Pool size: 10")
    print("Expected: Limited to 3 concurrent (respects request)")
    print()

    # Should be limited to 3, not 10
    assert max_concurrent <= 4, f"Expected max 3-4 concurrent, got {max_concurrent}"
    assert max_concurrent >= 2, f"Expected at least 2-3 concurrent, got {max_concurrent}"

    print("✅ Correctly limited to requested max_workers!")


if __name__ == "__main__":
    test_respects_pool_size()
    test_smaller_request_still_works()
    print("\n=== All Pool Size Respect Tests Passed! ===")
